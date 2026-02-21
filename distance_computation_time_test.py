import numpy as np
from tqdm import tqdm
from time import time
from utils import *
import argparse
from timeit import timeit

def main():
    parser = argparse.ArgumentParser(description='Benchmark batched float32 vector operations')
    parser.add_argument('--batch', type=int, default=2, help='Batch size (default: 5)')
    parser.add_argument('--cap', type=int, default=50, help='Dataset capacity (default: 50)')
    parser.add_argument('--dim', type=int, default=100, help='Vector dimensions (default: 100)')
    parser.add_argument('--tests', type=int, default=10, help='Number of tests to run (default: 10)')
    parser.add_argument('--skip', type=bool, default=False, help='Skip non-essential tests') 
    args = parser.parse_args()

    BATCH = args.batch
    CAP = args.cap
    DIM = args.dim
    TESTS = args.tests
    SKIPS = args.skip

    print(f"------------\nBATCHED float32 TEST\n------------")
    print(f"BATCH SIZE: {BATCH}, TESTS: {TESTS}")
    ram, used_ram, num_vectors, dataset = fill_ram_with_float32_vectors(CAP, DIM, with_stats=True)

    setup_stats = [ram, used_ram, CAP, num_vectors, 'float32', DIM, BATCH]
    results = []

    start = time()
    norms = pre_compute_norms(dataset)
    results.append(time() - start)
    print(f"All L2 Norm Computation Time: {results[-1]}")

    points = np.random.choice(num_vectors, BATCH, replace=False)

    results.append(timeit(
        lambda: compute_distances_using_precomputed_norms(dataset, norms, points),
        number = TESTS
    ))
    print(f"Avg. Distance Computation Time (standard): {results[-1]}")

    dataset = np.hstack([dataset, np.ones((num_vectors, 1), dtype=np.float32), norms[:, None]])

    results.append(timeit(
        lambda: compute_distances_using_matmul_trick(dataset, norms, points),
        number = TESTS
    ))

    print(f"Avg. Distance Computation Time (stack and multiply): {results[-1]}")
    print(f"\nAll results:\n{setup_stats + results}\n")

def pre_compute_norms(M):
    return np.einsum('ij,ij->i', M, M)

def compute_distances_using_precomputed_norms(X, norms, points):
    V = X[points]

    dot = V @ X.T
    
    return norms[points][:, None] + norms[None, :] - 2 * dot

def compute_distances_using_matmul_trick(X, norms, points):
    V = np.hstack([-2 * X[points][:,:-2], norms[points][:, None], np.ones((len(points), 1), dtype=np.float32)])
    return X @ V.T


if __name__ == '__main__':
    main()


# TODO: see how long memory allocations take
