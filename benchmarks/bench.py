import sys
import time
import numpy as np
from pathlib import Path

# benchmarks/ and python/ are siblings, so Python won't find `index` on its
# own. 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from index import BruteForceIndex
from data import load_siftsmall
from evaluate import recall_at_k


def measure_qps(index, queries, k=10, warmup=10, repeats=20):
    """
    Queries per second for a single-query search API.

    Returns a dict of timing statistics.
    """

    for i in range(warmup):
        for q in queries:
            index.search(q, k)

    pass_times = []
    for i in range(repeats):
        start = time.perf_counter()
        index.search_batch(queries, k)
        pass_times.append(time.perf_counter() - start)

    pass_times = np.array(pass_times)
    n_items = len(queries)
    per_query_times = [i/n_items for i in pass_times]
    # Per-item time, derived from the median pass.
    median_per_item = np.median(per_query_times)
    min_per_item    = np.min(per_query_times)

    # TODO: return a dict with median_ms, min_ms, qps, noise_pct
    return {
        "median_ms": median_per_item * 1000,
        "min_ms": min_per_item * 1000,
        "qps": 1.0/median_per_item,
        "noise_pct": (median_per_item - min_per_item)/min_per_item * 100
    }


def main():
    base, queries, gt = load_siftsmall()
    print(f"dataset: {base.shape[0]} vectors, {base.shape[1]} dims, "
          f"{len(queries)} queries\n")

    with BruteForceIndex(base) as idx:
        #verify recall is still 1.0 before trusting any timing.
        #       A fast wrong answer is not a result.
        found_ids, found_dists = idx.search_batch(queries)
        mean, per_query_array = recall_at_k(found_ids, gt, k=10)
        assert mean == 1.0, f"brute force recall is {mean}, must be exactly 1.0"
        # measure QPS at k=10 
        results_10 = measure_qps(idx, queries, 10)
        for i in results_10.keys():
            print(f"{i}: {results_10[i]}\n")
        results_1 = measure_qps(idx, queries, 1)
        results_100 = measure_qps(idx, queries, 100)

        print(f"median (1, 10, 100): {results_1['median_ms']}, {results_10['median_ms']}, {results_100['median_ms']}\n")
        print(f"min (1, 10, 100): {results_1['min_ms']}, {results_10['min_ms']}, {results_100['min_ms']}\n")
        print(f"qps (1, 10, 100): {results_1['qps']}, {results_10['qps']}, {results_100['qps']}\n")
        print(f"noise_pct (1, 10, 100): {results_1['noise_pct']}, {results_10['noise_pct']}, {results_100['noise_pct']}\n")


if __name__ == "__main__":
    main()