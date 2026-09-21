# Vector Index from Scratch — Full Roadmap

Approximate nearest neighbour search index in C, wrapped for Python,
benchmarked against FAISS, then served as a deployed search API.

**Status as of this document:** Steps 1–11 complete. Currently on step 12.

---

## The high-level idea

You have a large collection of things — documents, images, product
descriptions — each turned into a **vector** (a list of numbers, also called
an *embedding*). Similar things have vectors that sit close together. Given a
new vector, which of the stored ones are closest?

**Brute force** compares against everything. Correct, but the work grows
linearly with the data.

**Approximate nearest neighbour (ANN)** gives up on being exactly right.
Accept an answer that's usually right — say 95% of the time — in exchange for
being 100x faster. The whole field is a trade: accuracy for speed.

Two numbers measure that trade:

- **Recall** — of the 10 true nearest neighbours, how many did I return?
  9 of 10 → recall 0.9.
- **QPS** — queries per second.

Every parameter setting gives one (recall, QPS) pair. Plot many and you get a
curve. "Better" means your curve sits above and to the right of someone
else's. That curve is the **Pareto frontier**, and it is the single most
important artifact in this project.

**HNSW** (Hierarchical Navigable Small World) is the algorithm. It turns the
vectors into a **graph** — each vector connected to a few near neighbours.
Search means starting somewhere and repeatedly hopping to whichever neighbour
is closer to the query. You walk downhill instead of checking everything.

The *hierarchical* part: a single flat graph needs many small steps to cross
the space, so HNSW stacks layers. The top layer holds few points with very
long connections — big jumps. Each layer down holds more points with shorter
connections. Start at the top, jump coarsely, drop a layer, refine, and do the
fine-grained search in the bottom layer that contains everything.

*The airport analogy:* to get from a small town in Alberta to a small town in
Portugal you don't drive. You fly to a hub, hub to hub across the ocean, then
take progressively smaller connections down to the town. Top layer =
intercontinental hubs. Bottom layer = local roads.

---

## Phase 0 — Make the skeleton work before there's anything in it
*4–6 hours · COMPLETE*

Three moving parts — C code, a compiled library, Python calling into it. You
do not want to debug the connection between them at the same time as debugging
a graph algorithm. So prove the connection works while there's nothing in it.

| # | Step | Status |
|---|------|--------|
| 1 | Repo and folder layout, `.gitignore`, `.gitkeep` placeholders | done |
| 2 | Makefile compiling one C file into a shared library | done |
| 3 | Call that library from Python, get a number back | done |
| 4 | Pass a numpy array into C and modify it in place | done |
| 5 | Download `siftsmall`, write the `.fvecs` loader | done |
| 6 | Start the decisions log | outstanding |

**Phase test:** `make` produces a `.so`, Python calls it, SIFT data loads with
the right shape and dtype.

**Key results so far:** same `a.ctypes.data` address before and after the C
call — proof of zero copy. That fact is what the whole design rests on.

---

## Phase 1 — Brute force, the thing you grade everything against
*8–12 hours · IN PROGRESS*

Without this there is no ground truth. Every recall number you ever report is
measured against it. It's also the speed baseline — "47x faster than brute
force" only means something if you built the brute force.

| # | Step | Status |
|---|------|--------|
| 7 | Vector store in C — one contiguous block, not array-of-pointers | done |
| 8 | Squared Euclidean distance (no square root) | done |
| 9 | Validate against numpy on real SIFT data | done |
| 10 | Brute-force nearest neighbour, k=1 | done |
| 11 | Binary heap (priority queue) | done |
| 12 | Brute-force top-k using the heap | **current** |
| 13 | Wrap in Python, verify against SIFT ground truth — recall must be 1.0 | |
| 14 | Measure and record baseline QPS | |

**Phase test:** perfect recall against provided ground truth, heap has its own
test file, no leaks under the sanitizer.

**Key results so far:**
- Moving the loop from Python into C: 20.24 ms → 0.29 ms (70x), identical
  arithmetic, purely from eliminating 9,999 of 10,000 ctypes boundary crossings.
- C brute force beats numpy's vectorised equivalent 0.29 ms vs 0.81 ms.
  Decomposed: ~1.3x from single-pass fusion (no 5 MB intermediates), ~2.2x from
  Clang auto-vectorisation (verified by rebuilding with `-fno-vectorize`).
- float32 noise floor measured at 6.67e-07 max relative error.

---

## Phase 2 — One flat graph
*15–20 hours*

Hierarchy is an optimisation on top of graph search. Get traversal working
flat first, or you're debugging two hard things at once.

| # | Step |
|---|------|
| 15 | Design the adjacency storage — how "node 5 connects to 12, 88, 3" is recorded |
| 16 | Build a graph with random connections |
| 17 | Greedy search — hop to whichever neighbour is closest, stop when none is |
| 18 | Measure recall. Watch it be terrible. |
| 19 | Upgrade to beam search with the `ef` parameter |
| 20 | Plot recall against `ef` — your first real curve |
| 21 | Build the graph properly: insert nodes one at a time, connect to `M` nearest |
| 22 | Add the neighbour selection heuristic |

**Step 18 is deliberate.** Greedy search gets stuck in a **local minimum** — a
node where every neighbour looks worse, but the real answer is two hops away
behind a hill. Seeing that failure is what makes step 19 make sense.

**Step 19 is the knob.** Instead of tracking one current-best, track the `ef`
most promising unexplored nodes plus a separate list of best results found.
Two heaps: a min-heap of candidates to explore, a max-heap of results. Bigger
`ef` = search more broadly = higher recall, lower speed.

**Step 22 matters more than it looks.** Naively connecting each node to its M
closest gives clusters that connect densely inside themselves and barely to
each other — the graph fragments. The HNSW paper's heuristic deliberately
keeps some longer, more diverse edges. Implement both, plot both, keep the
comparison. Excellent interview story.

**Phase test:** recall improves monotonically with `ef`, graph is connected,
measurable speedup over brute force.

> ### MINIMUM RESUME-VIABLE MILESTONE — step 22
>
> You can honestly write: *"Implemented a graph-based approximate nearest
> neighbour index in C with a Python binding; achieved 95% recall@10 at ~20x
> the throughput of exact search on SIFT-1M."*
>
> You built a heap, a graph, a beam search, and a foreign-function interface,
> and you can defend all of it. Everything after this makes the bullet
> stronger, but this is the point where you have something real.
>
> Apply before you reach it anyway, and update as you go.

---

## Phase 3 — Full HNSW
*15–25 hours*

| # | Step |
|---|------|
| 23 | Random layer assignment from a decaying distribution (`mL` controls decay) |
| 24 | Extend storage to multiple layers — layer 0 holds everything, gets 2M connections |
| 25 | Multi-layer insertion |
| 26 | Multi-layer search: greedy with `ef`=1 through upper layers, full beam search at layer 0 |
| 27 | Verify recall recovered and search got faster than the flat graph at equal recall |
| 28 | Clean under valgrind and AddressSanitizer — every malloc freed |
| 29 | Save and load the index to disk |

**Step 29 is not optional.** Rebuilding a million-vector index every time you
benchmark will destroy your patience.

---

## Phase 4 — The benchmark study
*10–15 hours*

| # | Step |
|---|------|
| 30 | Scale to SIFT1M — 100x the data, expect things to break |
| 31 | Measure memory usage and reason about where it goes |
| 32 | Sweep harness: loop over `M`, `efConstruction`, `efSearch` → CSV |
| 33 | Plot the Pareto frontier — recall@10 vs QPS, log scale |
| 34 | Install FAISS, benchmark `IndexHNSWFlat` on identical data |
| 35 | Overlay the curves and explain the gap honestly |
| 36 | Write the README with plots embedded |

**Step 35:** you will be slower. FAISS uses hand-written SIMD and years of
tuning. Explaining *precisely why* you're 3x slower is a better interview
answer than being fast for reasons you can't articulate.

**Step 36 is what a recruiter actually looks at.**

---

## Phase 5 — Linear algebra extensions
*15–20 hours*

Sits directly on top of the Linear Algebra 2 course.

| # | Step |
|---|------|
| 37 | Demonstrate the curse of dimensionality empirically |
| 38 | Implement PCA — centre, covariance, eigenvectors, project |
| 39 | Index the reduced vectors, plot dimensions vs recall vs speed |
| 40 | Implement k-means, then product quantization |

**Step 37:** generate random points in 2, 10, 50, 200, 1000 dimensions and plot
the ratio of farthest to nearest distance. It collapses toward 1 — in high
dimensions everything is roughly equidistant, which is *why* this problem is
hard at all.

**Step 40:** chop each vector into chunks, cluster each chunk's values into 256
centroids, store one byte per chunk. 128 floats (512 bytes) becomes 16 bytes.
Plot memory against recall.

---

## Phase 6 — The deployed search service
*25–35 hours*

> **SEQUENCING CONSTRAINT — from your own project brief:**
>
> Do not start this phase until steps 1–36 are done. The index is the part
> nobody else has; the service layer is the part you could build in two
> weekends once the hard thing works. A deployed shell around a half-finished
> index is worth much less than a finished index with no deployment.
>
> If you try to jump ahead because deployment feels more exciting or more
> "employable," the answer is no.

| # | Step |
|---|------|
| 41 | Embed a text corpus with a sentence transformer (`all-MiniLM-L6-v2`) |
| 42 | Index the embeddings, verify semantic search works locally |
| 43 | FastAPI service exposing a search endpoint that calls the C index |
| 44 | API-key authentication |
| 45 | Per-key rate limiting |
| 46 | Set a latency budget; instrument p50 and p99 |
| 47 | Dockerfile — C build, Python wrapper, and service in one image |
| 48 | Deploy to a small cloud VM (EC2 or GCE) at a real URL, running continuously |
| 49 | Measure p50/p99 on the live service, not just locally |
| 50 | Minimal frontend: one search box, one results list |
| 51 | Load test — find where it breaks and why |
| 52 | Write up the cost, the latency budget, and what breaks at 100x traffic |

**Why API keys and not user login/JWT:** the auth should fit what the service
actually is. This is a machine-to-machine search API, not a consumer app with
user accounts. Picking the auth model that matches the product is itself a
defensible decision — and being able to explain why you *didn't* build JWT
login is stronger than having built it without thinking.

> ### SECOND MILESTONE — end of Phase 6
>
> The resume bullet stops being "implemented an algorithm" and becomes
> "implemented and operated a service." Covers backend and cloud alongside the
> systems and ML work.
>
> "Deployed on AWS" on a resume often means "followed a tutorial, there's an
> EC2 instance," and an interviewer can tell in one follow-up. Step 52 exists
> so you can answer that follow-up.

---

## The tech stack, and why

**C, not C++.** C++ hands you `std::priority_queue` and `std::vector` for
free — which is the problem, because writing the heap yourself is a chunk of
the learning and one of the better things to talk about. C also has a stable,
unmangled ABI that `ctypes` can call directly; C++ mangles function names and
would need `extern "C"` wrappers at the boundary, paying a cost to disable the
feature you switched languages for. *Would change if:* the focus moved to SIMD
optimization or multi-threaded construction, where C++'s tooling is better.

**ctypes, not pybind11 or Cython.** Standard library, no build step for the
wrapper, works against a plain C `.so`. Its weakness is ~2 µs per-call
overhead — irrelevant *because* you cross the boundary once per query with a
pointer, not once per number. Measured: 10,000 crossings cost 20 ms; one
crossing costs 0.002 ms.

**Contiguous float array, not array-of-pointers.** All vectors in one malloc,
laid end to end. CPUs read memory in 64-byte cache lines and prefetch
sequentially; pointer-chasing defeats that and costs a cache miss (~100
cycles) instead of an L1 hit (~4). Also matches numpy's own layout, so C
borrows the buffer with zero copies.

**Fixed-capacity adjacency arrays, not linked lists.** HNSW caps connections
at `M` (and `2M` in layer 0). Known cap means one flat block of neighbour IDs
plus a count per node, indexed arithmetically. A pointer hop per neighbour
would be the worst possible access pattern in the innermost loop.

**float32, not float64.** Half the memory, half the memory bandwidth.
Embeddings are noisy well above float32 precision. It's what every ANN library
and embedding model uses.

**Normalize vectors, then use squared L2.** For unit-length vectors, squared
Euclidean and cosine similarity rank identically. Normalize once at load, then
implement exactly one distance function.

**SIFT1M as the dataset.** Standard ANN benchmark, ships with precomputed
ground truth, comparable against published numbers. Random Gaussian vectors
would be a mistake — no cluster structure, so ANN performs unrealistically
badly and results mean nothing.

**`all-MiniLM-L6-v2` for demo embeddings.** 384 dimensions, CPU-only, seconds
to run, no API key, no cost, fully reproducible.

**Makefile, not CMake.** One library and a few test binaries. CMake solves
multi-platform multi-target problems you don't have.

**Assert-based tests, plus valgrind and AddressSanitizer.** A C test framework
is overhead at this size. The memory checkers are non-negotiable — they catch
errors that otherwise surface as inexplicable wrong answers at 1M vectors.

---

## Decisions painful to reverse

Ranked by how much it hurts to change late.

1. **Who owns the vector memory.** numpy owns, C borrows, C owns only the
   graph. Reversing changes every function signature and is the classic source
   of crashes at this boundary.
2. **Contiguous vs pointer-per-vector layout.** Touches every loop.
3. **C vs C++.** Full rewrite.
4. **The ID scheme.** Internal IDs are dense integers 0..n-1 so they double as
   array indices; any mapping to real-world IDs lives in Python.
5. **Whether deletion is supported.** Recommendation: no. Say so in the README
   as a known limitation with a sentence on how you'd approach it. "I scoped it
   out and here's what it would take" is a strong answer.
6. **Distance convention.** Normalize at load, one distance function, decided
   once.

---

## Working principles

- **Isolate one new thing at a time.** k=1 before top-k. Flat graph before
  layers. A two-line function before real code. Slower per step, much faster
  overall.
- **Every recall number traces back to ground truth.** If the oracle is wrong,
  everything downstream is plausible-looking garbage.
- **Suspect the build first.** `undefined reference` and `dlsym: symbol not
  found` mean the code isn't in the object file. Check the build before reading
  the C.
- **Suspect the boundary second.** Wrong numbers from C usually mean a type
  mismatch between the C signature and `argtypes`, not a logic bug.
- **Log the decision the day you make it.** A week later you'll remember the
  choice and not the alternative you rejected, and the alternative is the
  interesting part.
