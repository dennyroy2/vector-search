import csv
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from data import load_sift1m
from index import BruteForceIndex, HNSWIndex

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def sift1m_smoke_test():
    base, queries, gt = load_sift1m()
    queries = queries[:1000]
    gt = gt[:1000]

    print(f"base:    {base.shape}  {base.nbytes / 1e6:.0f} MB")
    print(f"queries: {queries.shape}")

    # Brute force baseline. This is now genuinely slow — 1M distances
    # per query — so use fewer queries for the timing.
    with BruteForceIndex(base) as bf:
        t0 = time.perf_counter()
        for q in queries[:100]:
            bf.search(q, k=10)
        bf_per_query = (time.perf_counter() - t0) / 100
        print(f"brute force: {bf_per_query*1000:.1f} ms/query, "
              f"{1/bf_per_query:.0f} QPS")

    # HNSW build. Time it — this is the number you don't have yet.
    t0 = time.perf_counter()
    idx = HNSWIndex(base, M=16, ef_construction=100, seed=42)
    print(f"build: {idx.build_seconds:.0f}s")
    idx.save("data/indexes/sift1m_M16_efc200.bin")
    print(f"index file: {Path('data/indexes/sift1m_M16_efc200.bin').stat().st_size/1e6:.0f} MB")

    # And the payoff.
    for ef in [10, 50, 100, 200]:
        total_recall, total_nd = 0.0, 0
        t0 = time.perf_counter()
        for i, q in enumerate(queries):
            ids, _, nd = idx.search(q, ef=ef, k=10)
            total_recall += len(set(ids) & set(gt[i][:10])) / 10
            total_nd += nd
        elapsed = time.perf_counter() - t0
        qps = len(queries) / elapsed
        print(f"  ef={ef:<4} recall={total_recall/len(queries):.3f}  "
              f"qps={qps:.0f}  ({qps*bf_per_query:.0f}x brute force)  "
              f"ndists={total_nd/len(queries):.0f} "
              f"({100*total_nd/len(queries)/len(base):.3f}%)")

    idx.close()

def main():
    sift1m_smoke_test()

if __name__ == "__main__":
    main()
