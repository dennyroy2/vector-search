# benchmarks/faiss_compare.py
import csv
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import faiss
from data import load_sift1m

import os
faiss.omp_set_num_threads(os.cpu_count())
print(faiss.omp_get_max_threads())

RESULTS_DIR = Path(__file__).resolve().parent / "results"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "indexes"

M_VALUES = [16]
EFC_VALUES = [200]
EFS_VALUES = [10, 20, 40, 80, 160, 320]
N_QUERIES = 1000
K = 10


def recall_at_k(found, truth, k):
    return len(set(found[:k]) & set(truth[:k])) / k


def bench_faiss(base, queries, gt, threads=1, batched=False):
    print(f"bench_faiss: threads={threads}, batched={batched}", flush=True)
    # CRITICAL: single-threaded, or we measure core count rather than
    # implementation quality. FAISS parallelises search by default.
    faiss.omp_set_num_threads(threads)
    suffix = "" if threads == 1 else f"-mt"
    rows = []
    for M in M_VALUES:
        for efc in EFC_VALUES:
            path = INDEX_DIR / f"faiss_sift1m_M{M}_efc{efc}.index"

            if path.exists():
                print(f"FAISS M={M} efc={efc}: loading")
                index = faiss.read_index(str(path))
                build_s = 0.0
            else:
                print(f"FAISS M={M} efc={efc}: building...", flush=True)
                index = faiss.IndexHNSWFlat(base.shape[1], M)
                index.hnsw.efConstruction = efc
                t0 = time.perf_counter()
                index.add(base)
                build_s = time.perf_counter() - t0
                print(f"  {build_s:.0f}s", flush=True)
                faiss.write_index(index, str(path))

            index_bytes = path.stat().st_size

            for efs in EFS_VALUES:
                index.hnsw.efSearch = efs

                # Warm up.
                index.search(queries[:100], K)

                # Correctness. FAISS searches in batch; pass one at a time
                # so the comparison matches your per-query API.
                # Distance computations. FAISS keeps a global counter,
                # so reset it immediately before the pass we want to measure.
                faiss.cvar.hnsw_stats.reset()
                total_recall = 0.0
                for i in range(len(queries)):
                    _, ids = index.search(queries[i:i+1], K)
                    total_recall += recall_at_k(ids[0], gt[i], K)

                ndists = faiss.cvar.hnsw_stats.ndis / len(queries)
                # Timing — median of 3 full passes, matching your methodology.
               
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    if batched:
                        index.search(queries, K)
                    else:
                        for i in range(len(queries)):
                            index.search(queries[i:i+1], K)
                    times.append(time.perf_counter() - t0)
                per_query = np.median(times) / len(queries)
            
                rows.append({
                    "label": f"faiss-M{M}-efc{efc}{suffix}",
                    "M": M, "ef_construction": efc, "ef_search": efs, "k": K,
                    "recall": total_recall / len(queries),
                    "qps": 1.0 / per_query,
                    "ms_per_query": per_query * 1000,
                    "ndists": ndists,           # FAISS doesn't expose this
                    "pct_scanned": 100.0 * ndists / len(base),
                    "descent_ndists": -1,
                    "build_seconds": build_s,
                    "index_bytes": index_bytes,
                    "n_queries": len(queries),
                })
                print(f"  M={M} efc={efc} ef={efs:<4} "
                      f"recall={rows[-1]['recall']:.4f} "
                      f"qps={rows[-1]['qps']:.0f}", flush=True)
    return rows


def main():
    base, queries, gt = load_sift1m()
    queries, gt = queries[:N_QUERIES], gt[:N_QUERIES]

    rows  = bench_faiss(base, queries, gt, threads=1, batched=False)  # baseline
    rows += bench_faiss(base, queries, gt, threads=1, batched=True)   # + batching
    rows += bench_faiss(base, queries, gt, threads=os.cpu_count(), batched=True)   # + threading

    out = RESULTS_DIR / "faiss_sift1m.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()