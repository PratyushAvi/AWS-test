import numpy as np
from tqdm import tqdm
from time import time
import argparse
from utils import *

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
    
    setup_stats = [ram, used_ram, num_vectors, 'float32', DIM, BATCH]
    results = []
    
    # Test 1: L2 Norm - linalg.norm
    times = []
    for i in tqdm(range(min(TESTS, dataset.shape[0]))):
        vecs = dataset[i:i + BATCH]
        start = time()
        norm = np.linalg.norm(vecs, axis=1) ** 2
        times.append(time() - start)
    results.append(sum(times)/len(times))
        
    # Test 2: L2 Norm - einsum
    times = []
    for i in tqdm(range(min(TESTS, dataset.shape[0]))):
        vecs = dataset[i:i + BATCH]
        start = time()
        norm_einsum = np.einsum('ij,ij->i', vecs, vecs)
        times.append(time() - start)
    results.append(sum(times)/len(times))
    
    # Test 3: L2 Norm - sum of squares
    times = []
    for i in tqdm(range(min(TESTS, dataset.shape[0]))):
        vecs = dataset[i:i + BATCH]
        start = time()
        norm_squared = np.sum(vecs * vecs, axis=1)
        times.append(time() - start)
    results.append(sum(times)/len(times))
    
    print(f"Avg. L2 Norm Computation Time (linalg.norm): {results[0]}")
    print(f"Avg. L2 Norm Computation Time (einsum): {results[1]}")
    print(f"Avg. L2 Norm Computation Time (sum of squares): {results[2]}")
    
    # Test 4: Matrix-Matrix Product - matmul
    times = []
    for i in tqdm(range(min(TESTS, dataset.shape[0]))):
        vecs = dataset[i:i + BATCH]
        start = time()
        matvecs = dataset @ vecs.T
        times.append(time() - start)
    results.append(sum(times)/len(times))
    
    if SKIPS:
       results.append(None)
    else:
        # Test 5: Matrix-Matrix Product - einsum
        times = []
        for i in tqdm(range(min(TESTS, dataset.shape[0]))):
            vecs = dataset[i:i + BATCH]
            start = time()
            matvecs_einsum = np.einsum('ij,kj->ik', dataset, vecs)
            times.append(time() - start)
        results.append(sum(times)/len(times))
    
    print(f"Avg. Matrix-Matrix Product Computation Time (matmul): {results[3]}")
    print(f"Avg. Matrix-Matrix Product Computation Time (einsum): {results[4]}")
    
    if SKIPS:
        results.append(None)
        results.append(None)
    else:
        # Test 6: Distance - linalg.norm
        times = []
        for i in tqdm(range(min(TESTS, dataset.shape[0]))):
            vecs = dataset[i: i + BATCH]
            start = time()
            norms = (np.linalg.norm(vecs, axis=1)**2).astype(np.float32)
            dot_prod = dataset @ vecs.T
            distances = norms - 2 * dot_prod
            times.append(time() - start)
        results.append(sum(times)/len(times))
    
        # Test 7: Distance - einsum
        times = []
        for i in tqdm(range(min(TESTS, dataset.shape[0]))):
            vecs = dataset[i: i + BATCH]
            start = time()
            norms = np.einsum('ij,ij->i', vecs, vecs, dtype=np.float32)
            dot_prod = dataset @ vecs.T
            distances_einsum = norms - 2 * dot_prod
            times.append(time() - start)
        results.append(sum(times)/len(times))
    
    # Test 8: Distance - sum of squares
    times = []
    for i in tqdm(range(min(TESTS, dataset.shape[0]))):
        vecs = dataset[i: i + BATCH]
        start = time()
        norms = np.sum(vecs * vecs, axis=1, dtype=np.float32)
        dot_prod = dataset @ vecs.T 
        distances_sum = norms - 2 * dot_prod
        times.append(time() - start)
    results.append(sum(times)/len(times))
    
    print(f"Avg. Distance Computation Time (linalg.norm): {results[5]}")
    print(f"Avg. Distance Computation Time (einsum): {results[6]}")
    print(f"Avg. Distance Computation Time (sum of squares): {results[7]}")
    
    
    print(f"\nAll results:\n{setup_stats + results}\n")


if __name__ == '__main__':
    main()
