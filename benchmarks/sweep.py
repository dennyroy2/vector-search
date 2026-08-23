import csv
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from data import load_siftsmall
from index import RandomGraphIndex, BruteForceIndex, BuiltGraphIndex

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def recall_at_k(found_ids, true_ids, k):
    return len(set(found_ids[:k]) & set(true_ids[:k])) / k


def sweep(index, queries, gt, efs, k=10, label="", warmup=5, repeats=10):
    """Sweep ef, measuring recall, QPS, and distance computations."""
    rows = []

    for ef in efs:
        # Warm-up: discard. Same methodology as step 14.
        for _ in range(warmup):
            for q in queries:
                index.search(q, ef=ef, k=k)

        # Correctness pass — measured once, not timed.
        total_recall = 0.0
        total_ndists = 0
        for i, q in enumerate(queries):
            ids, _, nd = index.search(q, ef=ef, k=k)
            total_recall += recall_at_k(ids, gt[i], k)
            total_ndists += nd

        # Timing pass — median of `repeats` full passes over all queries.
        pass_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            for q in queries:
                index.search(q, ef=ef, k=k)
            pass_times.append(time.perf_counter() - start)

        per_query = np.median(pass_times) / len(queries)

        rows.append({
            "label": label,
            "ef": ef,
            "k": k,
            "recall": total_recall / len(queries),
            "qps": 1.0 / per_query,
            "ms_per_query": per_query * 1000,
            "ndists": total_ndists / len(queries),
            "pct_scanned": total_ndists / len(queries) / index.n * 100,
        })
        print(f"  ef={ef:<5} recall={rows[-1]['recall']:.3f}  "
              f"qps={rows[-1]['qps']:.0f}  ndists={rows[-1]['ndists']:.0f}")

    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")

def bfs_reachable(index, start=0):
    """How many nodes are reachable from `start`? Caps achievable recall."""
    from collections import deque
    visited = set([start])
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nb in index.neighbours(node):
            nb = int(nb)
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited)


def main():
    base, queries, gt = load_siftsmall()
    efs = [10, 20, 50, 100, 200, 500, 1000]
    rows = []
    print("random graph, M=16")
    with RandomGraphIndex(base, M=16, seed=4) as idx:
        rows += sweep(idx, queries, gt, efs, label="random-M16")

    print("\nbuilt graph, M=16, efConstruction=100")
    with BuiltGraphIndex(base, M=16, ef_construction=100) as idx:
        print(f"  build took {idx.build_seconds:.1f}s")
        rows += sweep(idx, queries, gt, efs, label="built-M16")

        # Connectivity: what fraction is even reachable from node 0?
        reachable = bfs_reachable(idx)
        print(f"  reachable from node 0: {reachable}/{idx.n} "
              f"({100*reachable/idx.n:.1f}%)")

    # Brute force as a reference point: recall 1.0 at a known QPS.
    print("\nbrute force baseline")
    with BruteForceIndex(base) as bf:
        for _ in range(3):
            for q in queries:
                bf.search(q, k=10)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            for q in queries:
                bf.search(q, k=10)
            times.append(time.perf_counter() - start)
        per_query = np.median(times) / len(queries)
        rows.append({
            "label": "bruteforce", "ef": -1, "k": 10, "recall": 1.0,
            "qps": 1.0 / per_query, "ms_per_query": per_query * 1000,
            "ndists": len(base), "pct_scanned": 100.0,
        })
        print(f"  recall=1.000  qps={1.0/per_query:.0f}  ndists={len(base)}")

    write_csv(rows, RESULTS_DIR / "sweep_random.csv")


if __name__ == "__main__":
    main()