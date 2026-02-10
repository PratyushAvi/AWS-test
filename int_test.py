import numpy as np
from tqdm import tqdm
from time import time
import psutil
from scipy.spatial.distance import cdist

def main():
    cap = 50
    n = 100
    tests = 10

    print("------------\nint8 test\n------------")
    dataset = fill_ram_with_int8_vectors(cap, n)
    
    inner_prod_times = []
    matvec_times = []
    distance_times = []
    scipy_distance_times = []
 
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
    ''' 
    for i in tqdm(range(min(tests, dataset.shape[0]))):
        vec = dataset[i:i+1]
        start = time()
        cdist(dataset, vec, metric='sqeuclidean')
        scipy_distance_times.append(time() - start)

    print(f"Avg. Scipy Distance Computation Time: {sum(scipy_distance_times)/len(scipy_distance_times)}")
    print()
    '''
def fill_ram_with_float32_vectors(percent_to_fill, n_dimensions):
    """Fill RAM with n-dimensional float32 vectors."""
    total_ram = psutil.virtual_memory().total
    print(f"Total RAM: {total_ram / (1024**3):.2f} GB")
    
    target_memory = total_ram * (percent_to_fill / 100)
    print(f"Target memory: {target_memory / (1024**3):.2f} GB ({percent_to_fill}%)")
    
    bytes_per_vector = n_dimensions * 4
    num_vectors = int(target_memory / bytes_per_vector)
    
    print(f"Creating {num_vectors:,} vectors of dimension {n_dimensions}")
    
    vectors = np.random.rand(num_vectors, n_dimensions).astype(np.float32)
    return vectors

def fill_ram_with_int8_vectors(percent_to_fill, n_dimensions):
    """Fill RAM with n-dimensional int8 vectors."""
    total_ram = psutil.virtual_memory().total
    print(f"Total RAM: {total_ram / (1024**3):.2f} GB")
    
    target_memory = total_ram * (percent_to_fill / 100)
    print(f"Target memory: {target_memory / (1024**3):.2f} GB ({percent_to_fill}%)")
    
    bytes_per_vector = n_dimensions * 1  # int8 = 1 byte
    num_vectors = int(target_memory / bytes_per_vector)
    
    print(f"Creating {num_vectors:,} vectors of dimension {n_dimensions}")
    
    vectors = np.random.randint(-128, 128, size=(num_vectors, n_dimensions), dtype=np.int8)
    return vectors

if __name__ == '__main__':
    main()
