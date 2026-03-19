# main.py
import ray
from alg import *

@ray.remote
class WorkerActor(Worker):
    pass

@ray.remote
class CoordinatorActor(Coordinator):
    def __init__(self, EFS_PATH, num_points, batch):
        # initialize workers as Ray actors instead
        self.workers = [WorkerActor.remote(i, EFS_PATH) 
                        for i in range(17)]
        super().__init__(EFS_PATH, num_points, batch)

    def sendMessages(self, compute_distances, message):
        import collections
        from heapq import heappush

        responses = collections.defaultdict(list)

        # fan out in parallel
        futures = [w.message.remote(compute_distances, message) 
                   for w in self.workers]
        all_responses = ray.get(futures)

        for worker_responses in all_responses:
            for response in worker_responses:
                if response[1] is not None:
                    heappush(responses[response[0]], (response[2], response[1]))

        for vec_id in responses:
            responses[vec_id] = responses[vec_id][0]

        return responses