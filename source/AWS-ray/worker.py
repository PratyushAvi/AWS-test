"""
worker.py — run one instance per machine
Usage: python worker.py <shard_id> <port>
"""
import sys
import time
import numpy as np
import zmq
import pickle

EFS_PATH = "/mnt/efs/dataset"
PORT     = int(sys.argv[2]) if len(sys.argv) > 2 else 5555
SHARD_ID = int(sys.argv[1])


class Worker:
    def __init__(self, shard_id, EFS_PATH):
        self.id       = shard_id
        self.EFS_PATH = EFS_PATH

        allocations  = np.load(f"{self.EFS_PATH}/shards.npy")
        self.start   = allocations[self.id][0]
        self.end     = allocations[self.id][1]

        print(f"[Worker {self.id}] Loading shard [{self.start}, {self.end})...")
        self.dataset = np.load(f"{self.EFS_PATH}/vectors.npy",  mmap_mode='r')
        self.norms   = np.load(f"{self.EFS_PATH}/sq_norms.npy", mmap_mode='r')

        self.X = np.hstack([
            self.dataset[self.start:self.end].astype(np.float32),
            np.ones((self.end - self.start, 1), dtype=np.float32),
            self.norms[self.start:self.end][:, None]
        ])

        self.uncovered = {}
        self.dists     = {}
        print(f"[Worker {self.id}] Ready. X shape: {self.X.shape}")

    def message(self, vecs, inputs):
        response = []

        t0 = time.time()
        self.compute_dist(vecs)
        compute_time = time.time() - t0

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
            dist       = None
            if np.any(self.uncovered[vec_id]):
                return_vec = int(np.argmin(self.dists[vec_id]))
                dist       = float(self.dists[vec_id][return_vec])

            response.append((
                vec_id,
                return_vec + self.start if return_vec is not None else None,
                dist
            ))

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


def serve():
    worker  = Worker(SHARD_ID, EFS_PATH)
    context = zmq.Context()
    socket  = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"[Worker {SHARD_ID}] Listening on port {PORT}")

    while True:
        raw                  = socket.recv()
        compute_distances, inputs = pickle.loads(raw)
        response, compute_time   = worker.message(compute_distances, inputs)
        socket.send(pickle.dumps((response, compute_time)))


if __name__ == '__main__':
    serve()
