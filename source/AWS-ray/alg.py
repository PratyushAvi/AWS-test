import numpy as np
import collections
import time
import os
from heapq import heappush

class Stats:
    def __init__(self):
        self.total_computed    = 0
        self.total_time        = 0.0
        self.times             = []
        self.start_time        = None

    def start_neighborhood(self):
        self.start_time = time.time()

    def end_neighborhood(self):
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        self.total_time  += elapsed
        self.total_computed += 1
        self.start_time  = None

    def report(self):
        if not self.total_computed:
            print("No neighborhoods computed yet.")
            return
        avg  = self.total_time / self.total_computed
        p50  = np.percentile(self.times, 50)
        p95  = np.percentile(self.times, 95)
        rate = self.total_computed / self.total_time
        print(f"  Computed:      {self.total_computed}")
        print(f"  Total time:    {self.total_time:.1f}s")
        print(f"  Avg time:      {avg:.3f}s")
        print(f"  Median time:   {p50:.3f}s")
        print(f"  p95 time:      {p95:.3f}s")
        print(f"  Rate:          {rate:.2f} neighborhoods/sec")


class Coordinator:
    def __init__(self, EFS_PATH, num_points, batch):
        self.EFS_PATH = EFS_PATH
        self.vector_ids = np.load(f"{self.EFS_PATH}/ids.npy", mmap_mode='r')
        self.num_points = num_points
        self.worker_assignments = np.load(f"{self.EFS_PATH}/shards.npy")
        self.workers = []

        for i in range(self.worker_assignments.shape[0]):
            worker = Worker(i, self.EFS_PATH)
            self.workers.append(worker)

        self.neighborhoods = {}
        self.active = set()
        self.batch = batch
        self.computed = set()
        self.stats = Stats()

        # track when each vec_id started being processed
        self.start_times = {}

        with open(f"{self.EFS_PATH}/computed.txt", "a+") as f:
            for p in f.readlines():
                self.computed.add(int(p.strip()))

        print(f"Resuming from {len(self.computed)} already computed neighborhoods.")

    def computeNeighborhoods(self):
        self.active = set(np.random.choice(self.vector_ids, self.batch, replace=False))
        message = []
        compute_distances = []

        for vec_id in self.active:
            message.append((vec_id, 'INIT', None))
            compute_distances.append(vec_id)
            self.neighborhoods[vec_id] = []
            self.start_times[vec_id] = time.time()

        responses = self.sendMessages(compute_distances, message)

        report_every = 10  # print stats every N completions

        while self.active:
            compute_distances = []
            message = []

            for vec_id in list(self.active):
                if not len(responses[vec_id]):
                    # neighborhood complete
                    elapsed = time.time() - self.start_times.pop(vec_id)
                    self.stats.times.append(elapsed)
                    self.stats.total_time += elapsed
                    self.stats.total_computed += 1

                    message.append((vec_id, 'KILL', None))
                    self.writeNeighborhood(vec_id)
                    self.active.remove(vec_id)
                    self.computed.add(vec_id)

                    if self.stats.total_computed % report_every == 0:
                        self.stats.report()

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
                    x = min(responses[vec_id])[1]
                    self.neighborhoods[vec_id].append(x)
                    message.append((vec_id, 'UPDATE', x))
                    compute_distances.append(vec_id)

            responses = self.sendMessages(compute_distances, message)

        print("All neighborhoods computed.")
        self.stats.report()

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
        self.compute_dist(vecs)

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

        return response

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