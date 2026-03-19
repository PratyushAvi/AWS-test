import modal
import numpy as np
import collections
import time
from heapq import heappush
import random

app = modal.App("neighborhood-compute")
vol = modal.Volume.from_name("billion-dataset")
image = modal.Image.debian_slim().pip_install("numpy")

EFS_PATH = "/dataset/dataset"
NUM_SHARDS = 17

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    volumes={"/dataset": vol},
    cpu=4,
    memory=131072,
    timeout=86400,
    scaledown_window=3600,
)
class Worker:
    shard_id: int = modal.parameter()

    @modal.enter()
    def load(self):
        import numpy as np
        print(f"[Worker {self.shard_id}] Starting load...", flush=True)
        vol.reload()
        allocations = np.load(f"{EFS_PATH}/shards.npy")
        self.start = int(allocations[self.shard_id][0])
        self.end   = int(allocations[self.shard_id][1])
        print(f"[Worker {self.shard_id}] Shard: {self.start} - {self.end} ({self.end - self.start} vectors)", flush=True)

        self.dataset = np.load(f"{EFS_PATH}/vectors.npy",  mmap_mode='r')
        self.norms   = np.load(f"{EFS_PATH}/sq_norms.npy", mmap_mode='r')
        print(f"[Worker {self.shard_id}] mmap files opened", flush=True)

        n = self.end - self.start
        self.X = np.empty((n, 102), dtype=np.float32)
        print(f"[Worker {self.shard_id}] Allocated X: shape={self.X.shape}", flush=True)

        self.X[:, :100] = self.dataset[self.start:self.end]
        print(f"[Worker {self.shard_id}] X vectors filled", flush=True)

        self.X[:, 100] = 1.0
        self.X[:, 101] = self.norms[self.start:self.end]
        print(f"[Worker {self.shard_id}] X fully built", flush=True)

        self.n = n
        self.active_ids       = {}
        self.free_rows        = []
        self.dists_matrix     = None
        self.uncovered_matrix = None
        print(f"[Worker {self.shard_id}] Load complete", flush=True)

    def _alloc_row(self, vec_id):
        row = self.free_rows.pop() if self.free_rows else len(self.active_ids)
        self.active_ids[vec_id] = row
        return row

    @modal.method()
    def message(self, vec_ids, inputs):
        import numpy as np
        print(f"[Worker {self.shard_id}] message() called: {len(vec_ids)} vec_ids, {len(inputs)} inputs", flush=True)

        for vec_id, command, _ in inputs:
            if command == 'INIT':
                print(f"[Worker {self.shard_id}] INIT {vec_id}", flush=True)
                row = self._alloc_row(vec_id)
                if self.dists_matrix is None or row >= self.dists_matrix.shape[0]:
                    capacity = max(64, row + 1)
                    print(f"[Worker {self.shard_id}] Allocating matrices capacity={capacity}", flush=True)
                    new_d = np.full((capacity, self.n), np.inf, dtype=np.float32)
                    new_u = np.zeros((capacity, self.n), dtype=np.uint8)
                    if self.dists_matrix is not None:
                        new_d[:self.dists_matrix.shape[0]] = self.dists_matrix
                        new_u[:self.uncovered_matrix.shape[0]] = self.uncovered_matrix
                    self.dists_matrix    = new_d
                    self.uncovered_matrix = new_u
                self.uncovered_matrix[row] = 1
                if self.start <= vec_id < self.end:
                    self.uncovered_matrix[row][vec_id - self.start] = 0
            elif command == 'KILL':
                print(f"[Worker {self.shard_id}] KILL {vec_id}", flush=True)
                row = self.active_ids.pop(vec_id)
                self.free_rows.append(row)

        print(f"[Worker {self.shard_id}] Computing distances...", flush=True)
        self._compute_dist(vec_ids)
        print(f"[Worker {self.shard_id}] Distances computed, building response...", flush=True)

        response = []
        for vec_id, command, update_vec_id in inputs:
            if command == 'KILL':
                continue
            row = self.active_ids[vec_id]
            if command == 'UPDATE':
                update_row = self.active_ids.get(update_vec_id)
                if update_row is not None:
                    mask = self.dists_matrix[update_row] < self.dists_matrix[row]
                    self.uncovered_matrix[row][mask] = 0
                    self.dists_matrix[row][mask] = np.inf
            if np.any(self.uncovered_matrix[row]):
                rv   = int(np.argmin(self.dists_matrix[row]))
                dist = float(self.dists_matrix[row][rv])
                response.append((vec_id, rv + self.start, dist))
            else:
                response.append((vec_id, None, None))

        print(f"[Worker {self.shard_id}] message() done, {len(response)} responses", flush=True)
        return response

    def _compute_dist(self, vec_ids):
        import numpy as np
        if not len(vec_ids):
            return
        print(f"[Worker {self.shard_id}] _compute_dist: {len(vec_ids)} queries", flush=True)
        V = np.hstack([
            -2 * self.dataset[vec_ids].astype(np.float32),
            self.norms[vec_ids][:, None],
            np.ones((len(vec_ids), 1), dtype=np.float32),
        ])
        D = self.X @ V.T
        for i, vec_id in enumerate(vec_ids):
            row = self.active_ids.get(vec_id)
            if row is not None:
                self.dists_matrix[row] = D[:, i]
        print(f"[Worker {self.shard_id}] _compute_dist done", flush=True)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/dataset": vol},
    memory=16384,
    timeout=86400,
)
def coordinator(num_points: int = 100, batch: int = 50):
    import numpy as np
    import collections, time
    from heapq import heappush

    vol.reload()
    vector_ids = np.load(f"{EFS_PATH}/ids.npy", mmap_mode='r')
    workers    = [Worker(shard_id=i) for i in range(NUM_SHARDS)]

    # --- resume state ---
    computed    = set()
    neighborhoods = {}
    start_times   = {}
    stats         = Stats()

    try:
        with open(f"{EFS_PATH}/computed.txt") as f:
            for line in f:
                computed.add(int(line.strip()))
    except FileNotFoundError:
        pass
    print(f"Resuming from {len(computed)} already computed neighborhoods.", flush=True)

    def send_messages(compute_distances, message):
        # spawn all 17 calls without blocking
        handles = [w.message.spawn(compute_distances, message) for w in workers]
        # wait for ALL workers to finish before processing any results
        all_results = [handle.get() for handle in handles]
        responses = collections.defaultdict(list)
        for worker_responses in all_results:
            for resp in worker_responses:
                if resp[1] is not None:
                    heappush(responses[resp[0]], (resp[2], resp[1]))
        return {vid: responses[vid][0] for vid in responses}

    def write_neighborhood(vec_id):
        with open(f"{EFS_PATH}/computed.txt", 'a') as f:
            f.write(f"{vec_id}\n")
        with open(f"{EFS_PATH}/neighborhoods.txt", 'a') as f:
            f.write(f"{vec_id}: {neighborhoods[vec_id]}\n")
        vol.commit()

    # --- init first batch ---
    active  = set(int(v) for v in np.random.choice(vector_ids, batch, replace=False)
                  if int(v) not in computed)
    message = []
    compute_distances = []
    for vec_id in active:
        message.append((vec_id, 'INIT', None))
        compute_distances.append(vec_id)
        neighborhoods[vec_id] = []
        start_times[vec_id]   = time.time()

    print("Initializing first batch...", flush=True)
    responses = send_messages(compute_distances, message)

    print("Beginning main computation.", flush=True)
    while active:
        compute_distances = []
        message = []

        for vec_id in list(active):
            if vec_id not in responses or responses[vec_id] is None:
                elapsed = time.time() - start_times.pop(vec_id)
                stats.record_completion(vec_id, elapsed)
                message.append((vec_id, 'KILL', None))
                write_neighborhood(vec_id)
                active.remove(vec_id)
                computed.add(vec_id)
                print(f"[Coordinator] Completed {vec_id}, total={len(computed)}", flush=True)

                if (len(active) + len(computed)) < num_points:
                    new_vec = int(np.random.choice(vector_ids))
                    while new_vec in computed:
                        new_vec = int(np.random.choice(vector_ids))
                    active.add(new_vec)
                    message.append((new_vec, 'INIT', None))
                    compute_distances.append(new_vec)
                    neighborhoods[new_vec] = []
                    start_times[new_vec]   = time.time()
                    print(f"[Coordinator] Added new vec {new_vec}", flush=True)
            else:
                _, neighbor = responses[vec_id]
                neighborhoods[vec_id].append(neighbor)
                stats.record_neighbor(vec_id)
                message.append((vec_id, 'UPDATE', neighbor))
                compute_distances.append(vec_id)

        responses = send_messages(compute_distances, message)
        stats.report(active)

    print("All neighborhoods computed.", flush=True)
    stats.report(active)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self):
        self.total_computed   = 0
        self.total_time       = 0.0
        self.times            = []
        self.neighbors_picked = collections.defaultdict(int)
        self.worker_times     = []

    def record_neighbor(self, vec_id):
        self.neighbors_picked[vec_id] += 1

    def record_worker_response(self, elapsed):
        self.worker_times.append(elapsed)

    def record_completion(self, vec_id, elapsed):
        self.times.append(elapsed)
        self.total_time += elapsed
        self.total_computed += 1
        del self.neighbors_picked[vec_id]

    def report(self, active):
        print(f"\n--- Stats at {time.strftime('%H:%M:%S')} ---", flush=True)
        print(f"  Active:          {len(active)}", flush=True)
        print(f"  Total completed: {self.total_computed}", flush=True)
        if self.neighbors_picked:
            counts = list(self.neighbors_picked.values())
            print(f"  Neighbors/active: min={min(counts)} max={max(counts)} avg={np.mean(counts):.1f}", flush=True)
        if self.total_computed:
            avg  = self.total_time / self.total_computed
            p50  = np.percentile(self.times, 50)
            p95  = np.percentile(self.times, 95)
            rate = self.total_computed / self.total_time
            print(f"  Time: avg={avg:.3f}s p50={p50:.3f}s p95={p95:.3f}s rate={rate:.2f}/s", flush=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    coordinator.remote(num_points=10, batch=10)