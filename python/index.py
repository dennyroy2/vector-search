import ctypes
import numpy as np
from pathlib import Path
from data import load_siftsmall
import time
from evaluate import report
LIB_PATH = Path(__file__).resolve().parent.parent / "build" / "libvindex.so"
lib = ctypes.CDLL(str(LIB_PATH))

FLOAT_VEC = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
FLOAT_MAT = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")

# --- VectorStore lifecycle ---------------------------------------------
# C: VectorStore *vs_create(const float *data, int n, int dim);
lib.vs_create.argtypes = [FLOAT_MAT, ctypes.c_int, ctypes.c_int]
lib.vs_create.restype = ctypes.c_void_p      # opaque handle — Python never inspects it

# C: void vs_free(VectorStore *vs);
lib.vs_free.argtypes = [ctypes.c_void_p]
lib.vs_free.restype = None

# --- Search -------------------------------------------------------------
# C: void bruteforce_nn(const VectorStore *vs, const float *query,
#                       int *out_id, float *out_dist);
lib.bruteforce_nn.argtypes = [
    ctypes.c_void_p,
    FLOAT_VEC,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
]
lib.bruteforce_nn.restype = None

INT_VEC = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

# C: int bruteforce_topk(const VectorStore *vs, const float *query, int k,
#                        int *out_ids, float *out_dists);
lib.bruteforce_topk.argtypes = [
    ctypes.c_void_p, FLOAT_VEC, ctypes.c_int, INT_VEC, FLOAT_VEC,
]
lib.bruteforce_topk.restype = ctypes.c_int


class BruteForceIndex:
    """Exhaustive-search index. Ground truth reference."""

    def __init__(self, vectors: np.ndarray):
        if vectors.dtype != np.float32:
            raise TypeError(f"expected float32, got {vectors.dtype}")
        if not vectors.flags["C_CONTIGUOUS"]:
            raise ValueError("vectors must be C-contiguous")

        # CRITICAL: keep a reference to the array. C holds a bare pointer into
        # this memory and will not keep it alive. If Python garbage-collects
        # `vectors` while the store exists, C reads freed memory — a
        # use-after-free that manifests as garbage results or a segfault
        # at an unpredictable later moment.
        self._vectors = vectors

        self.n, self.dim = vectors.shape
        self._handle = lib.vs_create(vectors, self.n, self.dim)
        if not self._handle:
            raise MemoryError("vs_create failed")

    def search_1(self, query: np.ndarray):
        """Nearest neighbour to `query`. Returns (id, squared_distance)."""
        out_id = ctypes.c_int()
        out_dist = ctypes.c_float()
        lib.bruteforce_nn(self._handle, query,
                          ctypes.byref(out_id), ctypes.byref(out_dist))
        return out_id.value, out_dist.value

    def search(self, query: np.ndarray, k: int = 10):
        """k nearest neighbours. Returns (ids, distances), nearest first."""
        # Preallocate the output arrays. C fills them in place —
        # same zero-copy mechanism as step 4, just in the other direction.
        ids = np.empty(k, dtype=np.int32)
        dists = np.empty(k, dtype=np.float32)

        count = lib.bruteforce_topk(self._handle, query, k, ids, dists)

        # Trim if the store had fewer than k vectors.
        return ids[:count], dists[:count]

    def search_batch(self, queries: np.ndarray, k: int = 10):
        """Search many queries. Returns (n_queries, k) arrays."""
        all_ids = np.empty((len(queries), k), dtype=np.int32)
        all_dists = np.empty((len(queries), k), dtype=np.float32)
        for i, q in enumerate(queries):
            ids, dists = self.search(q, k)
            all_ids[i, :len(ids)] = ids
            all_dists[i, :len(dists)] = dists
        return all_ids, all_dists

    def close(self):
        if self._handle:
            lib.vs_free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

# C: Graph *graph_create(int n, int M);
lib.graph_create.argtypes = [ctypes.c_int, ctypes.c_int]
lib.graph_create.restype = ctypes.c_void_p

lib.graph_free.argtypes = [ctypes.c_void_p]
lib.graph_free.restype = None

# C: int graph_fill_random(Graph *g, int seed);
lib.graph_fill_random.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.graph_fill_random.restype = ctypes.c_int

lib.visited_create.argtypes = [ctypes.c_int]
lib.visited_create.restype = ctypes.c_void_p

lib.visited_free.argtypes = [ctypes.c_void_p]
lib.visited_free.restype = None

# C: int graph_greedy_search(const Graph *g, const VectorStore *vs,
#                            const float *query, int entry,
#                            VisitedSet *visited, float *out_dist,
#                            int *out_ndists);
lib.graph_greedy_search.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, FLOAT_VEC, ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int, ),ctypes.POINTER(ctypes.c_int)
]
lib.graph_greedy_search.restype = ctypes.c_int

lib.graph_get_neighbours_copy.argtypes = [ctypes.c_void_p, ctypes.c_int, INT_VEC]
lib.graph_get_neighbours_copy.restype = ctypes.c_int

class RandomGraphIndex:
    """Greedy search over a randomly-connected graph. The control."""

    def __init__(self, vectors, M=16, seed=42):
        self._vectors = vectors          # keep alive — C borrows the pointer
        self.n, self.dim = vectors.shape
        self.M = M

        self._vs = lib.vs_create(vectors, self.n, self.dim)
        self._g = lib.graph_create(self.n, M)
        self._v = lib.visited_create(self.n)

        if not lib.graph_fill_random(self._g, seed):
            raise ValueError(f"cannot build random graph with M={M}, n={self.n}")

    def search_1(self, query, entry=0):
        """Returns (id, squared_distance, n_distance_computations)."""
        out_dist = ctypes.c_float()
        out_ndists = ctypes.c_int()
        out_hops = ctypes.c_int()
        found = lib.graph_greedy_search(
            self._g, self._vs, query, entry, self._v,
            ctypes.byref(out_dist), ctypes.byref(out_ndists), ctypes.byref(out_hops)
        )
        return found, out_dist.value, out_ndists.value, out_hops.value

    def neighbours(self, node):
        out = np.empty(self.M, dtype=np.int32)
        count = lib.graph_get_neighbours_copy(self._g, node, out)
        return out[:count]

    def close(self):
        if self._v: lib.visited_free(self._v); self._v = None
        if self._g: lib.graph_free(self._g); self._g = None
        if self._vs: lib.vs_free(self._vs); self._vs = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def __del__(self): self.close()

def main():
    base, queries, gt = load_siftsmall()

    print(f"base {base.shape}, queries {queries.shape}\n")

    with BruteForceIndex(base) as idx:

        # --- Check 1: a vector from the store finds itself at distance 0 ---
        for i in [0, 42, 9999]:
            found_id, dist = idx.search_1(base[i])
            assert found_id == i, f"self-search on {i} returned {found_id}"
            assert dist == 0.0, f"self-distance was {dist}, not 0"
        print("self-search: PASS")

        # --- Check 2: agreement with numpy on all 100 queries -------------
        mismatches = 0
        for qi, q in enumerate(queries):
            mine, _ = idx.search_1(q)
            # argmin over squared distances, computed in float64
            d = base.astype(np.float64) - q.astype(np.float64)
            theirs = int(np.argmin(np.sum(d * d, axis=1)))
            if mine != theirs:
                mismatches += 1
                print(f"  query {qi}: mine={mine} numpy={theirs}")
        print(f"numpy agreement: {100 - mismatches}/100")
        assert mismatches == 0

        # --- Check 3: agreement with SIFT's shipped ground truth ----------
        # gt[i][0] is the true nearest neighbour of queries[i].
        correct = sum(idx.search_1(q)[0] == gt[i][0] for i, q in enumerate(queries))
        print(f"ground-truth agreement: {correct}/100")
        assert correct == 100

        # --- Check 4: timing, now that the loop lives in C ----------------
        # Warm up once so the first call's page faults don't skew the result.
        idx.search_1(queries[0])

        reps = 20
        start = time.perf_counter()
        for _ in range(reps):
            idx.search_1(queries[0])
        c_time = (time.perf_counter() - start) / reps

        start = time.perf_counter()
        for _ in range(reps):
            d = base - queries[0]
            np.argmin(np.sum(d * d, axis=1))
        np_time = (time.perf_counter() - start) / reps

        print(f"\nC (loop inside C):  {c_time*1000:6.3f} ms")
        print(f"numpy vectorised:   {np_time*1000:6.3f} ms")
        print(f"step 9 (loop in Python): 20.24 ms  <- 10,000 boundary crossings")

                # --- Check 5: top-k against SIFT ground truth ---------------------
        k = 10
        found_ids, found_dists = idx.search_batch(queries, k=k)

        mean = report(found_ids, gt, k=k, label="brute force")
        assert mean == 1.0, f"brute force recall is {mean}, must be exactly 1.0"

        # --- Check 6: distances are sorted ascending, every query ---------

        assert np.all(np.diff(found_dists, axis=1) >= 0)
        # --- Check 7: verify the distances themselves, not just the ids ---
        # An id can be right while the distance reported alongside it is
        # wrong — they travel separately through the heap.
       
        q = queries[3]
        for slot in range(k):
            vid = found_ids[3][slot]
            reported = found_dists[3][slot]
            
            d = base[vid].astype(np.float64) - q.astype(np.float64)
            actual = np.sum(d * d)
            
            assert np.isclose(reported, actual, rtol=1e-4)

if __name__ == '__main__':
    main()