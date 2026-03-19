import modal
import random
import time

app = modal.App("worker-benchmark")
vol = modal.Volume.from_name("billion-dataset")
image = modal.Image.debian_slim().pip_install("numpy")

EFS_PATH = "/dataset/dataset"

@app.cls(
    image=image,
    volumes={"/dataset": vol},
    cpu=4,
    memory=131072,
    timeout=3600,
    scaledown_window=3600,
)
class Worker:
    shard_id: int = modal.parameter()

    @modal.enter()
    def load(self):
        import numpy as np
        vol.reload()
        allocations = np.load(f"{EFS_PATH}/shards.npy")
        self.start = int(allocations[self.shard_id][0])
        self.end = int(allocations[self.shard_id][1])
        self.dataset = np.load(f"{EFS_PATH}/vectors.npy", mmap_mode='r')
        self.norms = np.load(f"{EFS_PATH}/sq_norms.npy", mmap_mode='r')
        self.X = np.hstack([
            self.dataset[self.start:self.end].astype(np.float32),
            np.ones((self.end - self.start, 1), dtype=np.float32),
            self.norms[self.start:self.end][:, None]
        ])
        self.n = self.X.shape[0]

        # Store dists as a 2D matrix (n_active x shard_size) instead of a dict of arrays
        # Use a dict mapping vec_id -> row index into the matrix
        self.active_ids = {}       # vec_id -> row index
        self.free_rows = []        # recycled row indices
        self.dists_matrix = None   # allocated on first INIT
        self.uncovered_matrix = None

    def _alloc_row(self, vec_id):
        """Assign a row in the batch matrix to this vec_id."""
        if self.free_rows:
            row = self.free_rows.pop()
        else:
            row = len(self.active_ids)
        self.active_ids[vec_id] = row
        return row

    @modal.method()
    def message(self, vec_ids, inputs):
        import numpy as np

        # --- Process commands first (INIT/KILL/UPDATE metadata) ---
        new_inits = []
        for vec_id, command, update_vec_id in inputs:
            if command == 'INIT':
                row = self._alloc_row(vec_id)
                # Lazily allocate matrices on first use
                if self.dists_matrix is None or row >= self.dists_matrix.shape[0]:
                    capacity = max(64, row + 1)
                    new_d = np.full((capacity, self.n), np.inf, dtype=np.float32)
                    new_u = np.zeros((capacity, self.n), dtype=np.uint8)
                    if self.dists_matrix is not None:
                        new_d[:self.dists_matrix.shape[0]] = self.dists_matrix
                        new_u[:self.uncovered_matrix.shape[0]] = self.uncovered_matrix
                    self.dists_matrix = new_d
                    self.uncovered_matrix = new_u
                self.uncovered_matrix[row] = 1
                if self.start <= vec_id < self.end:
                    self.uncovered_matrix[row][vec_id - self.start] = 0
                new_inits.append(vec_id)
            elif command == 'KILL':
                row = self.active_ids.pop(vec_id)
                self.free_rows.append(row)

        # --- Compute distances for all vec_ids at once ---
        t0 = __import__('time').time()
        self.compute_dist(vec_ids)
        compute_time = __import__('time').time() - t0

        # --- Vectorized UPDATE across all vec_ids in batch ---
        t1 = __import__('time').time()
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
                return_vec = int(np.argmin(self.dists_matrix[row]))
                dist = float(self.dists_matrix[row][return_vec])
                response.append((vec_id, return_vec + self.start, dist))
            else:
                response.append((vec_id, None, None))

        loop_time = __import__('time').time() - t1
        return response, compute_time, loop_time

    def compute_dist(self, vec_ids):
        import numpy as np
        if not len(vec_ids):
            return
        V = np.hstack([
            -2 * self.dataset[vec_ids].astype(np.float32),
            self.norms[vec_ids][:, None],
            np.ones((len(vec_ids), 1), dtype=np.float32)
        ])
        D = self.X @ V.T  # (shard_size x batch_size)
        for i, vec_id in enumerate(vec_ids):
            row = self.active_ids.get(vec_id)
            if row is not None:
                self.dists_matrix[row] = D[:, i]

@app.function(
    image=image,
    memory=16384,
    timeout=3600,
)
def coordinator():
    batch_sizes = [100]
    num_trials = 3

    worker = Worker(shard_id=0)

    print(f"{'Operation':<12} {'Batch Size':<12} {'Trial':<8} {'Wall (s)':<14} {'Compute (s)':<14} {'Loop (s)':<14}")
    print("-" * 76)

    for batch_size in batch_sizes:
        for trial in range(num_trials):
            vec_ids = random.sample(range(1_400_000_000), batch_size)
            inputs = [(i, 'INIT', None) for i in vec_ids]

            wall_start = time.time()
            response, compute_time, loop_time = worker.message.remote(vec_ids, inputs)
            wall_time = time.time() - wall_start
            print(f"{'INIT':<12} {batch_size:<12} {trial:<8} {wall_time:<14.4f} {compute_time:<14.4f} {loop_time:<14.4f}")

            compute_vecs = [i + 1 for i in vec_ids]
            inputs = [(i, 'UPDATE', i + 1) for i in vec_ids]

            wall_start = time.time()
            response, compute_time, loop_time = worker.message.remote(compute_vecs, inputs)
            wall_time = time.time() - wall_start
            print(f"{'UPDATE':<12} {batch_size:<12} {trial:<8} {wall_time:<14.4f} {compute_time:<14.4f} {loop_time:<14.4f}")

            inputs = [(i, 'KILL', None) for i in vec_ids]

            wall_start = time.time()
            response, compute_time, loop_time = worker.message.remote([], inputs)
            wall_time = time.time() - wall_start
            print(f"{'KILL':<12} {batch_size:<12} {trial:<8} {wall_time:<14.4f} {compute_time:<14.4f} {loop_time:<14.4f}")

        print()

@app.local_entrypoint()
def main():
    coordinator.remote()