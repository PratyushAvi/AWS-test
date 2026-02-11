import numpy as np
from tqdm import tqdm
from time import time
import psutil
from utils import *

def main():
    cap = 50
    n = 100
    tests = 10

    print("------------\nfloat32 test\n------------")
    dataset = fill_ram_with_float32_vectors(cap, n)

    inner_prod_times = []
    matvec_times = []
    distance_times = []

    for i in tqdm(range(min(tests, dataset.shape[0]))):
        vec = dataset[i]
        start = time()
        prod = vec @ vec
        inner_prod_times.append(time() - start)

    print(f"Avg. Inner Product Computation Time: {sum(inner_prod_times)/len(inner_prod_times)}")

    for i in tqdm(range(min(tests, dataset.shape[0]))):
        vec = dataset[i]
        start = time()
        matvec = dataset @ vec
        matvec_times.append(time() - start)

    print(f"Avg. Matvec Computation Time: {sum(matvec_times)/len(matvec_times)}")

    for i in tqdm(range(min(tests, dataset.shape[0]))):
        vec = dataset[i]
        start = time()
        dist = vec @ vec - 2 * (dataset @ vec)
        distance_times.append(time() - start)

    print(f"Avg. Distance Computation Time: {sum(distance_times)/len(distance_times)}")

if __name__ == '__main__':
    main()
