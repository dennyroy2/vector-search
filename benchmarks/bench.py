import sys
import time
import numpy as np
from pathlib import Path

# benchmarks/ and python/ are siblings, so Python won't find `index` on its
# own. 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from index import BruteForceIndex, RandomGraphIndex
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


def greedy_vs_bruteforce(base, queries, gt, M=16, seed=4):
    with RandomGraphIndex(base, M=M, seed=seed) as idx:
        correct = 0
        total_ndists = 0

        for i, q in enumerate(queries):
            found, dist, ndists, hops = idx.search_1(q)
            total_ndists += ndists
            if found == gt[i][0]:
                correct += 1

        n = len(queries)
        print(f"random graph, M={M}, seed={seed}")
        print(f"  recall@1:        {correct}/{n} = {correct/n:.2f}")
        print(f"  avg distances:   {total_ndists/n:.0f}  (brute force: {len(base)})")
        print(f"  fraction scanned: {total_ndists/n/len(base)*100:.2f}%")

def analyse_greedy(base, queries, gt, M=16, seed=4):
    """Full characterisation of greedy search on a random graph."""
    with RandomGraphIndex(base, M=M, seed=seed) as idx, \
         BruteForceIndex(base) as truth:

        recall_1 = 0
        ndists_all, hops_all, ratios = [], [], []

        for i, q in enumerate(queries):
            found, dist, ndists, hops = idx.search_1(q)

            # The true nearest distance, for the ratio.
            _, true_dists = truth.search(q, k=1)
            true_dist = true_dists[0]

            if found == gt[i][0]:
                recall_1 += 1

            ndists_all.append(ndists)
            hops_all.append(hops)
            # 1.0 means it found the true nearest. 40.0 means it's lost.
            ratios.append(dist / true_dist if true_dist > 0 else 1.0)

        n = len(queries)
        ndists_all = np.array(ndists_all)
        hops_all = np.array(hops_all)
        ratios = np.array(ratios)

        print(f"greedy on random graph (M={M}, seed={seed})")
        print(f"  recall@1:          {recall_1}/{n} = {recall_1/n:.3f}")
        print(f"  distances/query:   mean {ndists_all.mean():.1f}, "
              f"median {np.median(ndists_all):.0f}, max {ndists_all.max()}")
        print(f"  hops/query:        mean {hops_all.mean():.1f}, "
              f"max {hops_all.max()}")
        print(f"  scanned:           {ndists_all.mean()/len(base)*100:.2f}% "
              f"of the collection")
        print(f"  dist ratio vs true: median {np.median(ratios):.2f}x, "
              f"p90 {np.percentile(ratios, 90):.2f}x, "
              f"worst {ratios.max():.2f}x")

        # Which query went worst? Useful input to trace_one_query.
        print(f"  worst query index: {int(np.argmax(ratios))}")


def trace_one_query(base, queries, gt, qi=0, M=16, seed=4):
    """Print every hop of one greedy search, and what it missed."""
    q = queries[qi]

    with RandomGraphIndex(base, M=M, seed=seed) as idx:
        # Recompute distances in numpy so we can inspect freely.
        def dist_to(node_id):
            d = base[node_id].astype(np.float64) - q.astype(np.float64)
            return float(np.sum(d * d))

        true_id = int(gt[qi][0])
        true_dist = dist_to(true_id)

        print(f"\ntrace: query {qi}, true nearest is node {true_id} "
              f"at dist {true_dist:.0f}")

        current = 0
        current_dist = dist_to(current)
        visited = {current}
        hop = 0

        while True:
            print(f"  hop {hop}: node {current:5d}  dist {current_dist:10.0f}")

            nbrs = idx.neighbours(current)

            best, best_dist = -1, current_dist
            for nb in nbrs:
                nb = int(nb)
                if nb in visited:
                    continue
                visited.add(nb)
                d = dist_to(nb)
                if d < best_dist:
                    best, best_dist = nb, d

            if best == -1:
                # This is the local minimum. Show what it settled for.
                nbr_dists = sorted(dist_to(int(nb)) for nb in nbrs)
                print(f"    STOPPED. All {len(nbrs)} neighbours are worse.")
                print(f"    closest neighbour was {nbr_dists[0]:.0f} "
                      f"(current is {current_dist:.0f})")
                break

            current, current_dist = best, best_dist
            hop += 1

        print(f"  final:  node {current} at {current_dist:.0f}")
        print(f"  truth:  node {true_id} at {true_dist:.0f}")
        print(f"  ratio:  {current_dist/true_dist:.2f}x worse than optimal")
        print(f"  visited {len(visited)} nodes of {len(base)}")

def main():
    base, queries, gt = load_siftsmall()
    print(f"dataset: {base.shape[0]} vectors, {base.shape[1]} dims, "
          f"{len(queries)} queries\n")

    '''with BruteForceIndex(base) as idx:
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
        print(f"noise_pct (1, 10, 100): {results_1['noise_pct']}, {results_10['noise_pct']}, {results_100['noise_pct']}\n")'''

    greedy_vs_bruteforce(base, queries, gt)
    rng = np.random.default_rng(0)
    hits = 0
    for i, q in enumerate(queries):
        ids = rng.choice(len(base), size=41, replace=False)
        d = base[ids].astype(np.float64) - q.astype(np.float64)
        best = ids[np.argmin(np.sum(d * d, axis=1))]
        if best == gt[i][0]:
            hits += 1
    print(f"random sampling of 41: {hits}/100")

    analyse_greedy(base, queries, gt)
    trace_one_query(base, queries, gt, qi=0)


if __name__ == "__main__":
    main()