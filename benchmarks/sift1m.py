import csv
import sys
import time
import numpy as np
from pathlib import Path
import resource

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from data import load_sift1m
from index import BruteForceIndex, HNSWIndex

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def peak_mb():
    # macOS reports bytes, Linux reports kilobytes. Annoying but real.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3

def memory_report(idx, base):
    """Where the memory goes, and how much of it is wasted."""
    n, M, M0 = idx.n, idx.M, 2 * idx.M
    total = idx.memory()

    print(f"\nmemory at n={n:,}, M={M}")
    print(f"  vectors (numpy-owned):  {base.nbytes / 1e6:>7.0f} MB")
    print(f"  index total (C-owned):  {total / 1e6:>7.0f} MB")
    print(f"  combined:               {(base.nbytes + total) / 1e6:>7.0f} MB")
    print(f"  bytes per vector:       {total / n:>7.0f} "
          f"(vectors alone: {base.nbytes / n:.0f})")

    # Per-layer breakdown, and how much of each layer is actually occupied.
    print(f"\n  {'layer':>5} {'width':>6} {'allocated':>11} {'members':>9} "
          f"{'used':>10} {'wasted':>10}")
    total_wasted = 0
    for l in range(idx.max_level + 1):
        width = M0 if l == 0 else M
        allocated = n * width * 4 + n * 4          # neighbours + degrees
        members = idx.layer_members(l)             # needs a C accessor
        used = members * width * 4 + n * 4
        wasted = allocated - used
        total_wasted += wasted
        print(f"  {l:>5} {width:>6} {allocated/1e6:>8.0f} MB "
              f"{members:>9,} {used/1e6:>7.0f} MB {wasted/1e6:>7.0f} MB")

    print(f"\n  wasted on empty upper-layer slots: {total_wasted/1e6:.0f} MB "
          f"({100*total_wasted/total:.0f}% of the index)")


def sift1m_smoke_test():
    before = peak_mb()
    base, queries, gt = load_sift1m()
    after_load = peak_mb()
    print(f"vectors: {after_load - before:.0f} MB")
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
    after_build = peak_mb()
    print(f"index:   {after_build - after_load:.0f} MB")
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

def memory_analysis(index_path="data/indexes/sift1m_M16_efc200.bin"):
    """Load a saved index and report where its memory goes."""
    base, _, _ = load_sift1m()

    with HNSWIndex.load(index_path, base) as idx:
        memory_report(idx, base)

def main():
    #sift1m_smoke_test()
    memory_analysis()

if __name__ == "__main__":
    main()
