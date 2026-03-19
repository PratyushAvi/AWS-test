import modal

app = modal.App("combine-chunks")
vol = modal.Volume.from_name("billion-dataset")
image = modal.Image.debian_slim().pip_install("numpy", "tqdm")

@app.function(
    volumes={"/dataset": vol},
    image=image,
    timeout=86400,
    memory=32768,
)
def combine_chunks():
    import os
    import numpy as np
    from tqdm import tqdm

    chunks_dir = "/dataset/dataset/vector_chunks"
    output_path = "/dataset/dataset/vectors.npy"

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted existing {output_path}")

    chunks = sorted([
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
    ])
    print(f"Found {len(chunks)} chunks: {[os.path.basename(c) for c in chunks]}")

    total_bytes = sum(os.path.getsize(c) for c in chunks)

    with open(output_path, "wb") as out_f, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Combining chunks",
    ) as pbar:
        for chunk_path in chunks:
            print(f"Writing {os.path.basename(chunk_path)}...")
            with open(chunk_path, "rb") as in_f:
                while True:
                    buf = in_f.read(512 * 1024 * 1024)  # 512 MB at a time
                    if not buf:
                        break
                    out_f.write(buf)
                    pbar.update(len(buf))

    arr = np.load(output_path, mmap_mode="r")
    print(f"Done! Shape: {arr.shape}, dtype: {arr.dtype}")

    vol.commit()

@app.local_entrypoint()
def main():
    import numpy as np
    combine_chunks.remote()