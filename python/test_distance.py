import ctypes
import numpy as np
from pathlib import Path
import time

from data import load_siftsmall

LIB_PATH = Path(__file__).resolve().parent.parent / "build" / "libvindex.so"
print(LIB_PATH, LIB_PATH.exists())
lib = ctypes.CDLL(str(LIB_PATH))


FLOAT_VEC = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
lib.l2sq_distance.argtypes = [FLOAT_VEC, FLOAT_VEC, ctypes.c_int]
lib.l2sq_distance.restype = ctypes.c_float

def reference_l2sq(a, b):
    """Squared L2 distance in float64. The independent reference."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    return np.sum(diff * diff)


def test_correctness(base, queries, n_base=50):
    """Compare C against numpy on real SIFT pairs."""
    dim = base.shape[1]
    max_rel_error = 0.0
    worst = None

    for q in queries:
        for i in range(n_base):
            mine = lib.l2sq_distance(q, base[i], dim)
            ref = reference_l2sq(q, base[i])

            rel_error = abs(mine - ref)/abs(ref)
            if rel_error > max_rel_error:
                max_rel_error = rel_error
                worst = (mine, ref)

    n_pairs = len(queries) * n_base
    print(f"pairs compared: {n_pairs}")
    print(f"max relative error: {max_rel_error:.2e}")
    if worst:
        print(f"  worst case: C={worst[0]:.4f}  ref={worst[1]:.4f}")

    assert max_rel_error < 1e-4, f"too large: {max_rel_error:.2e}"
    print("correctness: PASS")


    

    print(f"max relative error: {max_rel_error:.2e}")
    assert max_rel_error < 1e-4, f"too large: {max_rel_error:.2e}"


def test_zero_distance(base):
    """A vector against itself must be exactly zero, even at dim=128."""
    # TODO: assert this for a handful of real vectors, using ==.
    #       No arithmetic error is possible when every difference is 0.
    dim = base.shape[1]
    for i in [0, 1, 42, 999, 9999]:
        d = lib.l2sq_distance(base[i], base[i], dim)
        assert d == 0.0, f"self-distance of vector {i} was {d}, not 0"
    print("zero-distance: PASS")


def benchmark(base, queries):
    """How fast is C vs numpy for one query against all base vectors?"""
    dim = base.shape[1]
    n = len(base)
    q = queries[0]

    start = time.perf_counter()
    c_dists = np.empty(n, dtype=np.float64)
    for i in range(n):
        c_dists[i] = lib.l2sq_distance(q, base[i], dim)

    c_time = time.perf_counter() - start

    start = time.perf_counter()
    diff = base - q                      # broadcasts q across all n rows
    np_dists = np.sum(diff * diff, axis=1)
    np_time = time.perf_counter() - start

    print(f"\nC   (Python loop, {n} ctypes calls): {c_time*1000:8.2f} ms")
    print(f"numpy (single vectorised call):      {np_time*1000:8.2f} ms")
    print(f"ratio: numpy is {c_time/np_time:.1f}x faster")
    print(f"per-call overhead: {c_time/n*1e6:.2f} us")

    assert np.allclose(c_dists, np_dists, rtol=1e-4), "C and numpy disagree in bulk"
    print("bulk agreement: PASS")

def test_correctness_random(dim=128, n_trials=2000, seed=0):
    """Same check on arbitrary floats, where float32 rounding actually occurs."""
    rng = np.random.default_rng(seed)
    max_rel_error = 0.0
    for _ in range(n_trials):
        a = rng.standard_normal(dim).astype(np.float32)
        b = rng.standard_normal(dim).astype(np.float32)
        mine = lib.l2sq_distance(a, b, dim)
        ref = reference_l2sq(a, b)
        max_rel_error = max(max_rel_error, abs(mine - ref) / abs(ref))
    print(f"random-float max relative error: {max_rel_error:.2e}")
    assert max_rel_error < 1e-4




if __name__ == "__main__":
    base, queries, gt = load_siftsmall()
    test_zero_distance(base)
    test_correctness(base, queries)
    test_correctness_random()
    benchmark(base, queries)