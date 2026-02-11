import numpy as np
from tqdm import tqdm
from time import time
import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Benchmark batched float32 vector operations')
    parser.add_argument('--batch', type=int, default=5, help='Batch size (default: 5)')
    parser.add_argument('--cap', type=int, default=50, help='Dataset capacity (default: 50)')
    parser.add_argument('--dim', type=int, default=100, help='Vector dimensions (default: 100)')
    parser.add_argument('--tests', type=int, default=10, help='Number of tests to run (default: 10)')
    
    args = parser.parse_args()
    
    BATCH = args.batch
    CAP = args.cap
    DIM = args.dim
    TESTS = args.tests
   
    print(f"------------\nBATCHED float32 TEST\n------------")
    print(f"BATCH SIZE: {BATCH}, TESTS: {TESTS}")
    dataset = fill_ram_with_float32_vectors(CAP, DIM)
    
    norm_times = []
    norm_einsum_times = []
    norm_squared_times = []
    matvec_times = []
    matvec_einsum_times = []
    distance_times = []
    for i in tqdm(range(min(TESTS, dataset.shape[0]))):
        vecs = dataset[i:i + BATCH]
        start = time()
        norm = np.linalg.norm(vecs, axis=1) ** 2
        norm_times.append(time() - start)
        
        start = time()
        norm_einsum = np.einsum('ij,ij->i', vecs, vecs)
        norm_einsum_times.append(time() - start)
        
        start = time()
        norm_squared = np.sum(vecs * vecs, axis=1)
        norm_squared_times.append(time() - start)
        
    print(f"Avg. L2 Norm Computation Time (linalg.norm): {sum(norm_times)/len(norm_times)}")
    print(f"Avg. L2 Norm Computation Time (einsum): {sum(norm_einsum_times)/len(norm_einsum_times)}")
    print(f"Avg. L2 Norm Computation Time (sum of squares): {sum(norm_squared_times)/len(norm_squared_times)}")
    matvec_einsum_times = []
    for i in tqdm(range(min(TESTS, dataset.shape[0]))):
        vecs = dataset[i:i + BATCH]
        start = time()
        matvecs = dataset @ vecs.T
        matvec_times.append(time() - start)
        
        start = time()
        matvecs_einsum = np.einsum('ij,kj->ik', dataset, vecs)
        matvec_einsum_times.append(time() - start)
    
    print(f"Avg. Matrix-Matrix Product Computation Time (matmul): {sum(matvec_times)/len(matvec_times)}")
    print(f"Avg. Matrix-Matrix Product Computation Time (einsum): {sum(matvec_einsum_times)/len(matvec_einsum_times)}")
    
    distance_einsum_times = []
    distance_sum_times = []
    for i in tqdm(range(min(TESTS, dataset.shape[0]))):
        vecs = dataset[i: i + BATCH]
        start = time()
        distances = np.linalg.norm(vecs, axis=1)**2 - 2 * dataset @ vecs.T
        distance_times.append(time() - start)
        
        start = time()
        distances_einsum = np.einsum('ij,ij->i', vecs, vecs) - 2 * dataset @ vecs.T
        distance_einsum_times.append(time() - start)
        
        start = time()
        distances_sum = np.sum(vecs * vecs, axis=1) - 2 * dataset @ vecs.T
        distance_sum_times.append(time() - start)
        
    print(f"Avg. Distance Computation Time (linalg.norm): {sum(distance_times)/len(distance_times)}")
    print(f"Avg. Distance Computation Time (einsum): {sum(distance_einsum_times)/len(distance_einsum_times)}")
    print(f"Avg. Distance Computation Time (sum of squares): {sum(distance_sum_times)/len(distance_sum_times)}")
        
if __name__ == '__main__':
    main()
