import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent/"data"

def read_fvecs(path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)

    count = raw[0]

    records = raw.reshape(-1, count + 1)

    data = records[:, 1:]

    data = data.view(np.float32)

    assert np.all(records[:, 0] == count), "inconsistent record sizes"

    return data.copy()

def read_ivecs(path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    
    count = raw[0]

    records = raw.reshape(-1, count + 1)

    data = records[:, 1:]

    assert np.all(records[:, 0] == count), "inconsistent record sizes"

    return data.copy()

def load_siftsmall():
    d = DATA_DIR/"siftsmall"

    return read_fvecs(d/"siftsmall_base.fvecs"), read_fvecs(d/"siftsmall_query.fvecs"), read_ivecs(d/"siftsmall_groundtruth.ivecs")

def load_sift1m():
    d = DATA_DIR/"sift"

    return read_fvecs(d/"sift_base.fvecs"), read_fvecs(d/"sift_query.fvecs"), read_ivecs(d/"sift_groundtruth.ivecs")

if __name__ == "__main__":
    base, queries, gt = load_siftsmall()
    print(f"base:    {base.shape}  {base.dtype}")
    print(f"queries: {queries.shape}  {queries.dtype}")
    print(f"gt:      {gt.shape}  {gt.dtype}")
    print(f"contiguous: {base.flags['C_CONTIGUOUS']}")
    print(f"first vector, first 8 dims: {base[0][:8]}")