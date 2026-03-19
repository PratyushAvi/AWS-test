import numpy as np
import collections
import time
from heapq import heappush
import ray

def main():
    ray.init()

    num_shards = 17
    EFS_PATH = "/mnt/efs/dataset"

    workers = [WorkerActor.remote(i, EFS_PATH) for i in range(num_shards)]

    coordinator = CoordinatorActor.remote(
        EFS_PATH=EFS_PATH,
        num_points=100,
        batch=50,
        workers=workers
    )

    ray.get(coordinator.computeNeighborhoods.remote())

    
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
        print(f"  Active points:      {len(active)}", flush=True)
        print(f"  Total completed:    {self.total_computed}", flush=True)

        if self.neighbors_picked:
            counts = list(self.neighbors_picked.values())
            print(f"  Neighbors picked per active point:", flush=True)
            print(f"    min={min(counts)}  max={max(counts)}  avg={np.mean(counts):.1f}", flush=True)

        if self.worker_times:
            recent = self.worker_times[-50:]
            print(f"  Worker response time (last 50 calls):", flush=True)
            print(f"    min={min(recent)*1000:.0f}ms  , flush=True"
                  f"max={max(recent)*1000:.0f}ms  "
                  f"avg={np.mean(recent)*1000:.0f}ms")

        if self.total_computed:
            avg  = self.total_time / self.total_computed
            p50  = np.percentile(self.times, 50)
            p95  = np.percentile(self.times, 95)
            rate = self.total_computed / self.total_time
            print(f"  Neighborhood completion time:\n    avg={avg:.3f}s  p50={p50:.3f}s  p95={p95:.3f}s  rate={rate:.2f}/s\n---", flush=True)


class Coordinator:
    def __init__(self, EFS_PATH, num_points, batch):
        self.EFS_PATH = EFS_PATH
        self.vector_ids = np.load(f"{self.EFS_PATH}/ids.npy", mmap_mode='r')
        self.num_points = num_points
        self.worker_assignments = np.load(f"{self.EFS_PATH}/shards.npy")
        self.neighborhoods = {}
        self.active = set()
        self.batch = batch
        self.computed = set()
        self.stats = Stats()
        self.start_times = {}

        with open(f"{self.EFS_PATH}/computed.txt", "a+") as f:
            for p in f.readlines():
                self.computed.add(int(p.strip()))

        print(f"Resuming from {len(self.computed)} already computed neighborhoods.")

    def computeNeighborhoods(self):
        self.active = set(np.random.choice(self.vector_ids, self.batch, replace=False))
        message = []
        compute_distances = []

        print("Initializing the first batch of vectors...", flush=True)
        for vec_id in self.active:
            message.append((vec_id, 'INIT', None))
            compute_distances.append(vec_id)
            self.neighborhoods[vec_id] = []
            self.start_times[vec_id] = time.time()

        responses = self.sendMessages(compute_distances, message)

        print("Beginning main computation", flush=True)
        while self.active:
            compute_distances = []
            message = []

            for vec_id in list(self.active):
                if not len(responses[vec_id]):
                    elapsed = time.time() - self.start_times.pop(vec_id)
                    self.stats.record_completion(vec_id, elapsed)

                    message.append((vec_id, 'KILL', None))
                    self.writeNeighborhood(vec_id)
                    self.active.remove(vec_id)
                    self.computed.add(vec_id)



                    if (len(self.active) + len(self.computed)) < self.num_points:
                        new_vec = np.random.choice(self.vector_ids)
                        while new_vec in self.computed:
                            new_vec = np.random.choice(self.vector_ids)

                        self.active.add(new_vec)
                        message.append((new_vec, 'INIT', None))
                        compute_distances.append(new_vec)
                        self.neighborhoods[new_vec] = []
                        self.start_times[new_vec] = time.time()
                else:
                    x = responses[vec_id][1]
                    self.neighborhoods[vec_id].append(x)
                    self.stats.record_neighbor(vec_id)
                    message.append((vec_id, 'UPDATE', x))
                    compute_distances.append(vec_id)

            responses = self.sendMessages(compute_distances, message)
            self.stats.report(self.active)

        print("All neighborhoods computed.")
        self.stats.report(self.active)

    def sendMessages(self, compute_distances, message):
        responses = collections.defaultdict(lambda: [])

        for w in self.workers:
            for response in w.message(compute_distances, message):
                if response[1] is not None:
                    heappush(responses[response[0]], (response[2], response[1]))

        for vec_id in responses:
            responses[vec_id] = responses[vec_id][0]

        return responses

    def writeNeighborhood(self, vec_id):
        with open(f"{self.EFS_PATH}/computed.txt", 'a+') as f:
            f.write(f"{vec_id}\n")

        with open(f"{self.EFS_PATH}/neighborhoods.txt", 'a+') as f:
            f.write(f"{vec_id}: {self.neighborhoods[vec_id]}\n")


class Worker:
    def __init__(self, shard_id, EFS_PATH):
        self.id = shard_id
        self.EFS_PATH = EFS_PATH

        allocations = np.load(f"{self.EFS_PATH}/shards.npy")
        self.start = allocations[self.id][0]
        self.end = allocations[self.id][1]

        self.dataset = np.load(f"{self.EFS_PATH}/vectors.npy", mmap_mode='r')
        self.norms = np.load(f"{self.EFS_PATH}/sq_norms.npy", mmap_mode='r')

        self.X = np.hstack([
            self.dataset[self.start:self.end].astype(np.float32),
            np.ones((self.end - self.start, 1), dtype=np.float32),
            self.norms[self.start:self.end][:, None]
        ])

        self.uncovered = {}
        self.dists = {}

    def message(self, vecs, inputs):
        response = []
        start = time.time()
        self.compute_dist(vecs)
        compute_time = time.time() - start

        for vec_id, command, update_vec_id in inputs:
            if command == 'INIT':
                self.uncovered[vec_id] = np.ones(self.X.shape[0])
                if self.start <= vec_id < self.end:
                    self.uncovered[vec_id][vec_id - self.start] = 0

            elif command == 'KILL':
                del self.uncovered[vec_id]
                del self.dists[vec_id]
                continue

            else:
                mask = self.dists[update_vec_id] < self.dists[vec_id]
                self.uncovered[vec_id][mask] = 0
                self.dists[vec_id][mask] = np.inf

            return_vec = None
            dist = None
            if np.any(self.uncovered[vec_id]):
                return_vec = np.argmin(self.dists[vec_id])
                dist = self.dists[vec_id][return_vec]

            response.append((vec_id, return_vec + self.start if return_vec is not None else None, dist))

        return response, compute_time

    def compute_dist(self, vec_ids):
        if not len(vec_ids):
            return
        V = np.hstack([
            -2 * self.dataset[vec_ids].astype(np.float32),
            self.norms[vec_ids][:, None],
            np.ones((len(vec_ids), 1), dtype=np.float32)
        ])

        D = self.X @ V.T

        for i in range(len(vec_ids)):
            self.dists[vec_ids[i]] = D[:, i]


@ray.remote(num_cpus=16)
class WorkerActor(Worker):
    pass


@ray.remote(num_cpus=0)
class CoordinatorActor(Coordinator):
    def __init__(self, EFS_PATH, num_points, batch, workers):
        self.workers = workers
        self.EFS_PATH = EFS_PATH
        self.vector_ids = np.load(f"{EFS_PATH}/ids.npy", mmap_mode='r')
        self.num_points = num_points
        self.worker_assignments = np.load(f"{EFS_PATH}/shards.npy")
        self.neighborhoods = {}
        self.active = set()
        self.batch = batch
        self.computed = set()
        self.stats = Stats()
        self.start_times = {}

        with open(f"{EFS_PATH}/computed.txt", "a+") as f:
            for p in f.readlines():
                self.computed.add(int(p.strip()))

        print(f"Resuming from {len(self.computed)} already computed neighborhoods.")

    def sendMessages(self, compute_distances, message):
        responses = collections.defaultdict(list)

        t0 = time.time()
        futures = [w.message.remote(compute_distances, message)
                   for w in self.workers]
        all_responses = ray.get(futures)
        total_rtt = time.time() - t0
        self.stats.record_worker_response(total_rtt)

        compute_times = [r[1] for r in all_responses]
        print(f"RTT: {total_rtt*1000:.0f}ms  "
            f"Compute: avg={np.mean(compute_times)*1000:.0f}ms  "
            f"max={np.max(compute_times)*1000:.0f}ms"
        )

        for worker_responses, _ in all_responses:
            for response in worker_responses:
                if response[1] is not None:
                    heappush(responses[response[0]], (response[2], response[1]))

        for vec_id in responses:
            responses[vec_id] = responses[vec_id][0]

        return responses


if __name__ == '__main__':
    main()
