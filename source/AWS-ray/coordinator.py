"""
coordinator.py — run on the head node
Usage: python coordinator.py
"""
import numpy as np
import collections
import time
import pickle
import threading
from heapq import heappush
import zmq

EFS_PATH    = "/mnt/efs/dataset"
NUM_SHARDS  = 17
WORKER_PORT = 5556
NUM_POINTS  = 100
BATCH       = 50

# list of (ip, port) for each worker — fill in after ray up
WORKER_ADDRESSES = [
    ("172.31.25.172", WORKER_PORT),
    ("172.31.31.16", WORKER_PORT),
    ("172.31.23.98", WORKER_PORT),
    ("172.31.21.20", WORKER_PORT),
    ("172.31.20.37", WORKER_PORT),
    ("172.31.17.255", WORKER_PORT),
    ("172.31.16.43", WORKER_PORT),
    ("172.31.16.250", WORKER_PORT),
    ("172.31.19.104", WORKER_PORT),
    ("172.31.25.86", WORKER_PORT),
    ("172.31.17.182", WORKER_PORT),
    ("172.31.30.198", WORKER_PORT),
    ("172.31.16.99", WORKER_PORT),
    ("172.31.25.75", WORKER_PORT),
    ("172.31.29.172", WORKER_PORT),
    ("172.31.24.143", WORKER_PORT),
    ("172.31.24.32", WORKER_PORT),
]


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
        print(f"  Active points:   {len(active)}", flush=True)
        print(f"  Total completed: {self.total_computed}", flush=True)

        if self.neighbors_picked:
            counts = list(self.neighbors_picked.values())
            print(f"  Neighbors picked: min={min(counts)}  max={max(counts)}  avg={np.mean(counts):.1f}", flush=True)

        if self.worker_times:
            recent = self.worker_times[-50:]
            print(f"  Worker RTT (last 50): min={min(recent)*1000:.0f}ms  "
                  f"max={max(recent)*1000:.0f}ms  avg={np.mean(recent)*1000:.0f}ms", flush=True)

        if self.total_computed:
            avg  = self.total_time / self.total_computed
            p50  = np.percentile(self.times, 50)
            p95  = np.percentile(self.times, 95)
            rate = self.total_computed / self.total_time
            print(f"  Completion: avg={avg:.3f}s  p50={p50:.3f}s  "
                  f"p95={p95:.3f}s  rate={rate:.2f}/s", flush=True)
        print("---", flush=True)


class Coordinator:
    def __init__(self):
        self.EFS_PATH   = EFS_PATH
        self.vector_ids = np.load(f"{EFS_PATH}/ids.npy", mmap_mode='r')
        self.num_points = NUM_POINTS
        self.batch      = BATCH
        self.neighborhoods = {}
        self.active     = set()
        self.computed   = set()
        self.stats      = Stats()
        self.start_times = {}

        # set up ZMQ sockets — one per worker
        self.context = zmq.Context()
        self.sockets = []
        for ip, port in WORKER_ADDRESSES:
            s = self.context.socket(zmq.REQ)
            s.connect(f"tcp://{ip}:{port}")
            self.sockets.append(s)
        print(f"Connected to {len(self.sockets)} workers.")

        with open(f"{EFS_PATH}/computed.txt", "a+") as f:
            for p in f.readlines():
                self.computed.add(int(p.strip()))
        print(f"Resuming from {len(self.computed)} already computed neighborhoods.")

    def sendMessages(self, compute_distances, message):
        """Fan out to all workers in parallel using threads."""
        responses     = collections.defaultdict(list)
        results       = [None] * len(self.sockets)
        payload       = pickle.dumps((compute_distances, message))

        def call_worker(i, socket):
            socket.send(payload)
            results[i] = pickle.loads(socket.recv())

        t0      = time.time()
        threads = [threading.Thread(target=call_worker, args=(i, s))
                   for i, s in enumerate(self.sockets)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rtt = time.time() - t0

        self.stats.record_worker_response(rtt)
        compute_times = [r[1] for r in results]
        print(f"  RTT: {rtt*1000:.0f}ms  "
              f"Compute: avg={np.mean(compute_times)*1000:.0f}ms  "
              f"max={np.max(compute_times)*1000:.0f}ms", flush=True)

        for worker_responses, _ in results:
            for response in worker_responses:
                if response[1] is not None:
                    heappush(responses[response[0]], (response[2], response[1]))

        for vec_id in responses:
            responses[vec_id] = responses[vec_id][0]

        return responses

    def computeNeighborhoods(self):
        self.active = set(int(x) for x in
                          np.random.choice(self.vector_ids, self.batch, replace=False))
        message           = []
        compute_distances = []

        print("Initializing first batch...", flush=True)
        for vec_id in self.active:
            message.append((vec_id, 'INIT', None))
            compute_distances.append(vec_id)
            self.neighborhoods[vec_id] = []
            self.start_times[vec_id]   = time.time()

        responses = self.sendMessages(compute_distances, message)

        print("Beginning main computation", flush=True)
        while self.active:
            compute_distances = []
            message           = []

            for vec_id in list(self.active):
                if not len(responses[vec_id]):
                    elapsed = time.time() - self.start_times.pop(vec_id)
                    self.stats.record_completion(vec_id, elapsed)

                    message.append((vec_id, 'KILL', None))
                    self.writeNeighborhood(vec_id)
                    self.active.discard(vec_id)
                    self.computed.add(vec_id)

                    if (len(self.active) + len(self.computed)) < self.num_points:
                        new_vec = int(np.random.choice(self.vector_ids))
                        while new_vec in self.computed:
                            new_vec = int(np.random.choice(self.vector_ids))
                        self.active.add(new_vec)
                        message.append((new_vec, 'INIT', None))
                        compute_distances.append(new_vec)
                        self.neighborhoods[new_vec] = []
                        self.start_times[new_vec]   = time.time()
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

    def writeNeighborhood(self, vec_id):
        with open(f"{self.EFS_PATH}/computed.txt", 'a+') as f:
            f.write(f"{vec_id}\n")
        with open(f"{self.EFS_PATH}/neighborhoods.txt", 'a+') as f:
            f.write(f"{vec_id}: {self.neighborhoods[vec_id]}\n")


if __name__ == '__main__':
    c = Coordinator()
    c.computeNeighborhoods()
