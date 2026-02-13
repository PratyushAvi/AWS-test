import numpy as np
import psutil

def fill_ram_with_float32_vectors(percent_to_fill, n_dimensions, with_stats=False):
    """Fill RAM with n-dimensional float32 vectors."""
    total_ram = psutil.virtual_memory().total
    print(f"Total RAM: {total_ram / (1024**3):.2f} GB")

    target_memory = total_ram * (percent_to_fill / 100)
    print(f"Target memory: {target_memory / (1024**3):.2f} GB ({percent_to_fill}%)")

    bytes_per_vector = n_dimensions * 4
    num_vectors = int(target_memory / bytes_per_vector)

    print(f"Creating {num_vectors:,} vectors of dimension {n_dimensions}")
    rng = np.random.default_rng()
    vectors = rng.random((num_vectors, n_dimensions), dtype=np.float32)
    
    if with_stats:
        return total_ram, target_memory, num_vectors, vectors
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
