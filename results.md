from benchmarks/bench.py

## Baseline: brute force, siftsmall (10K × 128)

Hardware: Apple Silicon, macOS
Build: -O2, auto-vectorisation enabled
Method: median of 20 passes over 100 queries, 10 warm-up passes discarded

| k   | median ms/query | QPS   | noise |
|-----|----------------:|------:|------:|
| 1   |           0.277 |  3613 |  0.37% |
| 10  |           0.280 |  3572 |  0.34% |
| 100 |           0.307 |  3254 |  0.55% |

k has almost no effect (11% from k=1 to k=100) because all 10,000 distances
are computed regardless; only heap depth changes. Contrast with HNSW, where
k and ef determine how many vectors are visited at all.

dataset: 10000 vectors, 128 dims, 100 queries

median_ms: 0.28022416576277465

min_ms: 0.2787170803640038

qps: 3568.5716015176067

noise_pct: 0.5407222968906605

median (1, 10, 100): 0.2752552047604695, 0.28022416576277465, 0.30747812532354146

min (1, 10, 100): 0.27489749947562814, 0.2787170803640038, 0.30616584001109004

qps (1, 10, 100): 3632.9921567521765, 3568.5716015176067, 3252.26387713844

noise_pct (1, 10, 100): 0.1301231497280594, 0.5407222968906605, 0.42861911453082707

## SIFT1M (1,000,000 × 128), M=16

Hardware: Apple Silicon, plugged in, low power mode OFF
Build: -O2, auto-vectorisation enabled
Method: 1,000 queries subsampled from the 10,000 provided
Brute force baseline: 27.5 ms/query, 36 QPS

| efC | build | ef  | recall@10 | QPS   | vs exact | ndists | scanned |
|----:|------:|----:|----------:|------:|---------:|-------:|--------:|
| 100 |  164s |  50 |     0.934 | 8,265 |     230x |  1,069 |  0.107% |
| 100 |  164s | 100 |     0.974 | 4,933 |     137x |  1,813 |  0.181% |
| 200 |  326s |  50 |     0.949 | 8,038 |     221x |  1,178 |  0.118% |
| 200 |  326s | 100 |     0.980 | 4,694 |     129x |  2,004 |  0.200% |

efConstruction=200 chosen: ~19% less search work at matched recall, for
2x the build time. Build is paid once; query cost is paid per request.

## Parameter sweep, SIFT-1M, 1000 queries, k=10
Brute force baseline: 37 QPS, 1,000,000 distances/query

| config          | r=0.90            | r=0.95            | r=0.99            | memory |
|-----------------|-------------------|-------------------|-------------------|-------:|
| M=8  efC=100    | 8,352 (227x)      | 4,550 (124x)      | not reached       | 360 MB |
| M=8  efC=200    | 9,227 (251x)      | 5,006 (136x)      | not reached       | 360 MB |
| M=16 efC=100    | 11,725 (319x)     | 7,504 (204x)      | 2,948 (80x)       | 544 MB |
| M=16 efC=200    | 11,777 (320x)     | 7,526 (205x)      | 2,947 (80x)       | 544 MB |
| M=32 efC=100    | 11,327 (308x)     | 7,772 (211x)      | 3,504 (95x)       | 792 MB |
| M=32 efC=200    | 11,622 (316x)     | 8,042 (219x)      | 3,838 (104x)      | 792 MB |

**M crosses over between recall 0.90 and 0.95.** M=16 does less work below;
M=32 does less above (2,686 vs 3,344 distance computations at 0.99). Higher M
costs more per hop but covers more ground per node expanded — the coverage
only pays once you need thorough exploration.

**M=8 cannot reach recall 0.99** at any efSearch swept.

**efConstruction saturates at M=16** — efC=100 and efC=200 produced identical
graphs. The neighbour heuristic rejects candidates that sit behind an
already-selected one, so beyond ~100 candidates the additional ones are all
farther and all redundant. M=8 and M=32 do show an efC effect.

**Default: M=16, efC=200.** At the 0.95 operating point M=32 buys 7% more
throughput for 45% more memory. At 0.99 it buys 30%, which changes the answer
if that recall is a requirement.

The vector-search algorithm is 1.81-2.38 times slower than FAISS's algorithm at matched recall, single-threaded on identical data. At M= 16, 0.95 recall, my algorithm has QPS = 7,526 while FAISS has 16,089. Compared to brute force, vector-search is 205 times faster.

At M = 32, my algorithm has an nd of 899 vs 950 for FAISS (0.90 recall), 1278 vs 1270 (0.95 recall), and 2686 vs 2738 (0.99 recall). The work done is roughly equivalent while QPS differs by a factor of ~2. This can be attributed to cost of each individual distance computation rather than the algorithm. FAISS has hand-written SIMD intrinsics, prefetching of the next neighbour's vector boosting QPS. Vector-search relies on Clang's compiler auto vectorization (2.2 times scalar, measured by compiling with -fno-vectorize).

At M = 16 and 0.99 recall, my nd is 3344 vs 2438 for FAISS, a 37% difference. This extra computational cost can be minimized by adding a keepPrunedConnections function, which backfills the empty neighbour slots from rejected candidates. I measured mean layer 0 degree as 21 against a cap of 32. The additional function would help ensure each node is close to their M0 cap of 32 neighbours. Both implementations were compared single-threaded, since mine has no parallelism. Measured separately, FAISS's batch API with 10 threads reaches 51,046 QPS at recall 0.977 against its own single-threaded per-query 10,644 — a 4.7× gain, of which only ~2% comes from batching itself. Multi-threaded search is the largest single advantage FAISS holds and the one I did not implement; the sub-linear scaling (4.7× on 10 cores) reflects memory bandwidth contention, since HNSW search is bandwidth-bound over a 656 MB index.

per-query, 1 thread:   10,644 QPS   ← the fair baseline
batched,   1 thread:   10,887 QPS   ← +2%   from batching
batched,  10 threads:  51,046 QPS   ← 4.7x  from threading

mine, per-query, 1 thread:          ~7,500 QPS      (interpolated)
FAISS, per-query, 1 thread:         10,644 QPS      1.4x
FAISS, batched, 10 threads:         51,046 QPS      6.8x