# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Build and test the C core:
- `make` — builds `build/libvindex.so`
- `make test` — builds and runs every `tests/test_*.c` as its own binary (`build/test_*`), stopping at the first failure
- Run a single test after `make test`: `./build/test_hnsw_insert` (or whichever `build/test_*` binary)
- `make clean` — do NOT run without asking (see rules below); rebuilds are slow at 1M vectors
- Memory check via valgrind (no Apple Silicon support, so it runs in a container):
  `docker build -f Dockerfile.test -t vindex-test .`
  `docker run --rm -it vindex-test bash`, then inside: `make test && for t in build/test_*; do valgrind --leak-check=full "$t"; done`

Python / benchmarks (`pip install numpy matplotlib faiss-cpu`):
- `python python/index.py` — smoke-runs the ctypes wrapper (see `main()`)
- `python benchmarks/sweep_sift1m.py` — builds 6 indexes at different M/efConstruction (~20 min), sweeps efSearch; indexes cache in `data/indexes/` and are skipped if already present
- `python benchmarks/faiss_compare.py` — same sweep against FAISS
- `python benchmarks/plot.py` — writes plots/tables into `benchmarks/results/`

After any change under `src/` or `include/`, run `make` before testing from Python — `python/index.py` loads `build/libvindex.so` directly and does not rebuild it for you.

## Architecture

Two-language system: C does the numeric/graph work as a shared library, Python orchestrates and owns the data, and the two share vector memory without copying.

**`src/` + `include/`** — the HNSW implementation. Each module is a matching `.h`/`.c` pair with one test binary in `tests/`:
- `vectors.*` — `VectorStore`, a contiguous float32 block. Owned by whoever allocated the backing buffer (numpy, when called from Python) — C never copies or frees it.
- `distance.*` — squared-L2 kernel, the innermost loop of every search; relies on compiler auto-vectorization rather than hand-written SIMD (see Known limitations in README.md).
- `heap.*` — `MaxHeap`; used both as a fixed-size top-k result buffer and, ordered as a min-heap, as the candidate queue during beam search.
- `visited.*` — `VisitedSet`, dedupes nodes across a single traversal.
- `graph.*` — `Graph`, one layer's adjacency as a flat `n * M` int array of neighbour slots. `graph_greedy_search` does hill-climbing (used above layer 0); `graph_beam_search` is the ef-bounded search (used at layer 0, and the one that trades recall for speed).
- `hnsw.*` — `HNSW`: an array of `Graph *` layers over one shared `VectorStore`, plus per-node level assignment and an entry point. `hnsw_insert`/`hnsw_build` construct it, `hnsw_search` queries it, `hnsw_save`/`hnsw_load` (de)serialize the graph structure only — the caller supplies the vectors separately on load, since they were never owned by C.
- `bruteforce.*` — exact search; the correctness/recall baseline everything else is measured against.

**`python/index.py`** — the ctypes boundary. Declares `argtypes`/`restype` for every exported C function (numpy arrays passed via `np.ctypeslib.ndpointer`, typed as `FLOAT_MAT`/`FLOAT_VEC`/`INT_VEC`), then wraps them in classes of increasing capability: `BruteForceIndex` → `RandomGraphIndex` → `BuiltGraphIndex` → `HNSWIndex`. Each wrapper class holds a reference to its backing numpy array on purpose — C only borrows a pointer into that buffer, so letting Python garbage-collect the array while C still holds the pointer is a use-after-free (see decisions.md).

This FFI boundary is the one place a mismatch fails silently instead of raising: wrong `argtypes`, wrong dtype/contiguity, or a stale `.so` after editing `src/` all tend to produce garbage results rather than an error.

**Other python/ files** — `data.py` loads the SIFT-1M / siftsmall benchmark sets (TEXMEX format) from `data/`; `embed.py` turns text into unit-norm float32 vectors via sentence-transformers for the deployed search service and is currently a skeleton (Phase 6 work).

**`benchmarks/`** — `bench.py`/`sweep.py`/`sweep_sift1m.py` are sweep harnesses, `faiss_compare.py` runs the same sweep against FAISS, `plot.py` renders `benchmarks/results/`.

### Design decisions that constrain future changes

Full writeups with numbers are in `decisions.md`; the ones that will bite if reversed casually:
- **C, not C++** — chosen so exported function names aren't mangled and `python/index.py` can find them by name via ctypes. Revisit only if adding SIMD intrinsics or multi-threading, where C++ genuinely helps.
- **Vectors are borrowed, not owned** — `VectorStore` never copies or frees the vector buffer. Changing this reintroduces the two crash modes documented in decisions.md (C freeing borrowed memory; Python GC'ing memory C still points at).
- **Neighbour selection heuristic** (`graph_select_neighbours`) rejects a candidate that's closer to an already-picked neighbour than to the new node, instead of just taking the M nearest. Removing it silently reintroduces the reachability collapse in decisions.md (50.3% → 99.9% reachable).

Known limitations (detailed in README.md): no deletion, single-threaded construction and search, no `keepPrunedConnections` backfill, no hand-written SIMD.

---

# How to work with me on this project

## What this is
An HNSW approximate nearest neighbour index written from scratch in C with a
ctypes Python binding, benchmarked against FAISS on SIFT-1M. Currently adding
a deployed FastAPI search service. See README.md for results.

## The rule that matters most
I am building this to be able to defend every line in an interview. Do not
hand me working code I don't understand.

## Write it for me (plumbing — low learning value, high time cost)
- Dockerfiles, docker-compose, build configuration
- FastAPI app scaffolding, dependency wiring, CORS, startup handlers
- The embedding script
- Frontend HTML/CSS/JS
- Load test scripts
- Anything boilerplate I'd otherwise copy from docs

For these: write it, then give me a short explanation of what each part
does and why it's there.

## Do NOT write for me (I write these; give me a skeleton with TODOs)
- The search endpoint body — query embedding, index call, ID-to-document mapping
- Rate limiting logic
- Latency instrumentation and percentile computation
- Anything touching the C code or the ctypes wrapper

For these: give me function signatures, control flow, and comments marking
what belongs in each blank. Name the exact library/function/data structure
to use — vagueness doesn't help. Then review what I write.

## Always
- Before a decision (instance type, auth model, framework), tell me the
  alternatives, the tradeoff, and what would change the answer.
- Define jargon in plain English the first time it appears.
- Tell me how to test each piece as I build it.
- Push back if I'm about to do something that will hurt me later.
- Append decisions to decisions.md as we make them, in the format used there.

## Never
- Don't run `make clean` without asking — rebuilds are slow at 1M vectors.
- Don't modify anything in src/, include/, or tests/ without asking first.
- Don't commit. I'll do that.
- Don't touch data/ — it holds multi-gigabyte datasets.

## Where I am
Phase 6 (deployment), step 41. The full roadmap is in ROADMAP.md —
read it if you need context on what came before or what's next.
