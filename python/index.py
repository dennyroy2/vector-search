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

lib.graph_beam_search.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, FLOAT_VEC, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, INT_VEC, FLOAT_VEC
        , ctypes.POINTER(ctypes.c_int, ), ctypes.c_void_p, ctypes.c_void_p,
]

lib.graph_beam_search.restype = ctypes.c_int
'''
int graph_beam_search(const Graph *g, const VectorStore *vs,
                      const float *query, int entry, int ef, int k,
                      VisitedSet *visited,
                      int *out_ids, float *out_dists, int *out_ndists)
int graph_greedy_search(const Graph *g, const VectorStore *vs, const float *query, int entry, VisitedSet *visited, 
                        float *out_dist, int *out_ndists, int * out_hops)
                      '''

lib.graph_get_neighbours_copy.argtypes = [ctypes.c_void_p, ctypes.c_int, INT_VEC]
lib.graph_get_neighbours_copy.restype = ctypes.c_int

lib.heap_free.argtypes = [ctypes.c_void_p]
lib.heap_free.restype = None

# C: Heap *heap_create(int capacity, int is_max);
lib.heap_create.argtypes = [ctypes.c_int, ctypes.c_int]
lib.heap_create.restype = ctypes.c_void_p

MAX_EF = 2000

class RandomGraphIndex:
    """Greedy search over a randomly-connected graph. The control."""

    def __init__(self, vectors, M=16, seed=42):
        self._vs = self._g = self._v = None
        self._candidates = self._results = None
        self._vectors = vectors          # keep alive — C borrows the pointer
        self.n, self.dim = vectors.shape
        self.M = M

        self._vs = lib.vs_create(vectors, self.n, self.dim)
        self._g = lib.graph_create(self.n, M)
        self._v = lib.visited_create(self.n)
        self._candidates = lib.heap_create(self.n, 0)      # 0 = min-heap
        self._results    = lib.heap_create(MAX_EF + 1, 1)  # 1 = max-heap

        if not lib.graph_fill_random(self._g, seed):
            raise ValueError(f"cannot build random graph with M={M}, n={self.n}")
        if not (self._v and self._candidates and self._results):
            raise MemoryError("allocation failed")

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

    def search(self, query, ef, k, entry = 0):
        if ef > MAX_EF:
            raise ValueError(f"ef={ef} exceeds MAX_EF={MAX_EF}")
        out_dist = np.empty(k, dtype=np.float32)
        out_ndists = ctypes.c_int()
        out_ids = np.empty(k, dtype=np.int32)
        
        count = lib.graph_beam_search(
            self._g, self._vs, query, entry, ef, k,  self._v,
            out_ids, out_dist, ctypes.byref(out_ndists), self._candidates, self._results
        )
        '''int graph_beam_search(const Graph *g, const VectorStore *vs,
                      const float *query, int entry, int ef, int k,
                      VisitedSet *visited,
                      int *out_ids, float *out_dists, int *out_ndists)'''
        return out_ids[:count], out_dist[:count], out_ndists.value 

    def neighbours(self, node):
        out = np.empty(self.M, dtype=np.int32)
        count = lib.graph_get_neighbours_copy(self._g, node, out)
        return out[:count]

    def close(self):
        if self._v: lib.visited_free(self._v); self._v = None
        if self._g: lib.graph_free(self._g); self._g = None
        if self._vs: lib.vs_free(self._vs); self._vs = None
        if self._candidates: lib.heap_free(self._candidates); self._candidates = None
        if self._results: lib.heap_free(self._results); self._results = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def __del__(self): self.close()

lib.graph_build.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
]

lib.graph_build.restype = ctypes.c_int
MAX_EF = 2000

class BuiltGraphIndex:
    """Proximity-built graph. Edges connect actually-nearby vectors."""
    def __init__(self, vectors, M=16, ef_construction=100):
        self._vs = self._g = self._v = None
        self._candidates = self._results = None
        self._vectors = vectors          # keep alive — C borrows the pointer
        self.n, self.dim = vectors.shape
        self.M = M
        self.ef_construction = ef_construction

        self._vs = lib.vs_create(vectors, self.n, self.dim)
        self._g = lib.graph_create(self.n, M)
        self._v = lib.visited_create(self.n)
        self._candidates = lib.heap_create(self.n, 0)      # 0 = min-heap
        self._results    = lib.heap_create(MAX_EF + 1, 1)  # 1 = max-heap

        # Build happens ONCE, here. This is the expensive part.
        start = time.perf_counter()
        if not lib.graph_build(self._g, self._vs, ef_construction):
            raise MemoryError("graph_build failed")
        if not (self._v and self._candidates and self._results):
            raise MemoryError("allocation failed")
        self.build_seconds = time.perf_counter() - start

    def search(self, query, ef=10, k=10, entry=0):
        if ef > MAX_EF:
            raise ValueError(f"ef={ef} exceeds MAX_EF={MAX_EF}")
            
        """Returns (ids, distances, n_distance_computations)."""
        ids = np.empty(k, dtype=np.int32)
        dists = np.empty(k, dtype=np.float32)
        out_ndists = ctypes.c_int()

        count = lib.graph_beam_search(
            self._g, self._vs, query, entry, ef, k, self._v,
            ids, dists, ctypes.byref(out_ndists), self._candidates, self._results
        )
        return ids[:count], dists[:count], out_ndists.value

    def neighbours(self, node):
        out = np.empty(self.M, dtype=np.int32)
        count = lib.graph_get_neighbours_copy(self._g, node, out)
        return out[:count]

    def close(self):
        if self._v: lib.visited_free(self._v); self._v = None
        if self._g: lib.graph_free(self._g); self._g = None
        if self._vs: lib.vs_free(self._vs); self._vs = None
        if self._candidates: lib.heap_free(self._candidates); self._candidates = None
        if self._results: lib.heap_free(self._results); self._results = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def __del__(self): self.close()

lib.hnsw_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.hnsw_create.restype = ctypes.c_void_p

lib.hnsw_free.argtypes = [ctypes.c_void_p]
lib.hnsw_free.restype = None

lib.hnsw_build.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.hnsw_build.restype = ctypes.c_int

lib.hnsw_search.argtypes = [
    ctypes.c_void_p, FLOAT_VEC, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    INT_VEC, FLOAT_VEC, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
]
lib.hnsw_search.restype = ctypes.c_int

lib.hnsw_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.hnsw_save.restype = ctypes.c_int

lib.hnsw_load.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
lib.hnsw_load.restype = ctypes.c_void_p


class HNSWIndex:
    """Hierarchical navigable small world index."""

    def __init__(self, vectors, M=16, ef_construction=100, seed=42,
                 max_ef=2000):
        self._vs = self._h = self._v = None
        self._candidates = self._results = None

        self._vectors = vectors
        self.n, self.dim = vectors.shape
        self.M = M
        self.max_ef = max_ef

        self._vs = lib.vs_create(vectors, self.n, self.dim)
        self._h = lib.hnsw_create(self._vs, M)
        self._v = lib.visited_create(self.n)
        self._candidates = lib.heap_create(self.n, 0)
        self._results = lib.heap_create(max_ef + 1, 1)

        start = time.perf_counter()
        if not lib.hnsw_build(self._h, ef_construction, seed):
            raise MemoryError("hnsw_build failed")
        self.build_seconds = time.perf_counter() - start

    def search(self, query, ef=10, k=10):
        if ef > self.max_ef:
            raise ValueError(f"ef={ef} exceeds max_ef={self.max_ef}")
        ids = np.empty(k, dtype=np.int32)
        dists = np.empty(k, dtype=np.float32)
        nd = ctypes.c_int()
        descent = ctypes.c_int()
        count = lib.hnsw_search(
            self._h, query, k, ef,
            self._v, self._candidates, self._results,
            ids, dists, ctypes.byref(nd), ctypes.byref(descent),
        )
        self.last_descent_ndists = descent.value
        return ids[:count], dists[:count], nd.value
    
    def save(self, path):
        """Write the index to disk. Vectors are not saved — only the graph."""
        # C wants bytes, not a str. Path objects need str() first.
        ok = lib.hnsw_save(self._h, str(path).encode("utf-8"))
        if not ok:
            raise IOError(f"hnsw_save failed writing {path}")

    @classmethod
    def load(cls, path, vectors, max_ef=2000):
        """Build an index from a saved file.

        `vectors` must be the same vectors, in the same order, the index
        was built over — the file stores node IDs, which are indices into
        this array. C validates n and dim but cannot check the contents.
        """
        if vectors.dtype != np.float32:
            raise TypeError(f"expected float32, got {vectors.dtype}")
        if not vectors.flags["C_CONTIGUOUS"]:
            raise ValueError("vectors must be C-contiguous")

        # __new__ makes an instance without running __init__ — which is
        # what we want, since __init__ builds the index and we're loading
        # one instead. Every attribute has to be set by hand here.
        self = cls.__new__(cls)
        self._vs = self._h = self._v = None
        self._candidates = self._results = None

        self._vectors = vectors          # keep alive — C borrows the pointer
        self.n, self.dim = vectors.shape
        self.max_ef = max_ef
        self.build_seconds = 0.0         # loaded, not built

        self._vs = lib.vs_create(vectors, self.n, self.dim)
        if not self._vs:
            raise MemoryError("vs_create failed")

        self._h = lib.hnsw_load(str(path).encode("utf-8"), self._vs)
        if not self._h:
            self.close()
            raise IOError(f"hnsw_load failed reading {path} "
                          f"(wrong file, version mismatch, or vector "
                          f"store shape mismatch)")

        self._v = lib.visited_create(self.n)
        self._candidates = lib.heap_create(self.n, 0)
        self._results = lib.heap_create(max_ef + 1, 1)
        if not (self._v and self._candidates and self._results):
            self.close()
            raise MemoryError("scratch allocation failed")

        return self

    def close(self):
        if self._h: lib.hnsw_free(self._h); self._h = None
        if self._v: lib.visited_free(self._v); self._v = None
        if self._candidates: lib.heap_free(self._candidates); self._candidates = None
        if self._results: lib.heap_free(self._results); self._results = None
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