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