import csv
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from data import load_sift1m
from index import BruteForceIndex, HNSWIndex

RESULTS_DIR = Path(__file__).resolve().parent / "results"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "indexes"
RESULTS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

M_VALUES = [8, 16, 32]
EFC_VALUES = [100, 200]
EFS_VALUES = [10, 20, 40, 80, 160, 320]
N_QUERIES = 1000
K = 10


def index_path(M, efc):
    return INDEX_DIR / f"sift1m_M{M}_efc{efc}.bin"


def build_all(base):
    """Phase A: build and save every (M, efConstruction) combination.
    Skips anything already on disk, so this is safe to re-run."""
    meta = {}
    for M in M_VALUES:
        for efc in EFC_VALUES:
            path = index_path(M, efc)
            if path.exists():
                print(f"M={M} efc={efc}: already built, skipping")
                # Load briefly to record its size and memory.
                with HNSWIndex.load(path, base) as idx:
                    meta[(M, efc)] = (0.0, idx.memory())
                continue

            print(f"M={M} efc={efc}: building...", flush=True)
            t0 = time.perf_counter()
            idx = HNSWIndex(base, M=M, ef_construction=efc, seed=42)
            elapsed = time.perf_counter() - t0
            idx.save(path)
            meta[(M, efc)] = (elapsed, idx.memory())
            print(f"  {elapsed:.0f}s, {idx.memory()/1e6:.0f} MB, "
                  f"max_level={idx.max_level}", flush=True)
            idx.close()
    return meta


def measure(idx, queries, gt, ef, k=K, warmup=2, repeats=3):
    """One configuration: recall, throughput, work done."""
    for _ in range(warmup):
        for q in queries[:100]:
            idx.search(q, ef=ef, k=k)

    # Correctness pass — not timed.
    total_recall, total_nd, total_desc = 0.0, 0, 0
    for i, q in enumerate(queries):
        ids, _, nd = idx.search(q, ef=ef, k=k)
        total_recall += len(set(ids) & set(gt[i][:k])) / k
        total_nd += nd
        total_desc += idx.last_descent_ndists

    # Timing pass — median of full passes, per step 14's methodology.
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for q in queries:
            idx.search(q, ef=ef, k=k)
        times.append(time.perf_counter() - t0)
    per_query = np.median(times) / len(queries)

    n = len(queries)
    return {
        "recall": total_recall / n,
        "qps": 1.0 / per_query,
        "ms_per_query": per_query * 1000,
        "ndists": total_nd / n,
        "descent_ndists": total_desc / n,
        "pct_scanned": 100.0 * (total_nd / n) / idx.n,
    }


def sweep_all(base, queries, gt, meta):
    """Phase B: load each index, sweep efSearch."""
    rows = []
    for M in M_VALUES:
        for efc in EFC_VALUES:
            path = index_path(M, efc)
            build_s, mem = meta[(M, efc)]

            with HNSWIndex.load(path, base) as idx:
                for efs in EFS_VALUES:
                    r = measure(idx, queries, gt, efs)
                    r.update({
                        "label": f"hnsw-M{M}-efc{efc}",
                        "M": M, "ef_construction": efc, "ef_search": efs,
                        "k": K, "build_seconds": build_s,
                        "index_bytes": mem, "n_queries": len(queries),
                    })
                    rows.append(r)
                    print(f"  M={M} efc={efc} ef={efs:<4} "
                          f"recall={r['recall']:.4f} qps={r['qps']:.0f} "
                          f"nd={r['ndists']:.0f}", flush=True)
    return rows


def brute_force_row(base, queries, gt, n_timed=50):
    """The baseline every speedup is measured against."""
    with BruteForceIndex(base) as bf:
        for q in queries[:5]:
            bf.search(q, k=K)
        t0 = time.perf_counter()
        for q in queries[:n_timed]:
            bf.search(q, k=K)
        per_query = (time.perf_counter() - t0) / n_timed

    return {
        "label": "bruteforce", "M": -1, "ef_construction": -1,
        "ef_search": -1, "k": K, "recall": 1.0,
        "qps": 1.0 / per_query, "ms_per_query": per_query * 1000,
        "ndists": len(base), "descent_ndists": 0, "pct_scanned": 100.0,
        "build_seconds": 0.0, "index_bytes": 0, "n_queries": n_timed,
    }


def main():
    base, queries, gt = load_sift1m()
    queries, gt = queries[:N_QUERIES], gt[:N_QUERIES]
    print(f"base {base.shape}, {len(queries)} queries\n")

    meta = build_all(base)
    print()
    rows = sweep_all(base, queries, gt, meta)
    rows.append(brute_force_row(base, queries, gt))

    out = RESULTS_DIR / "sweep_sift1m.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    for efc in [100, 200]:
        p = index_path(16, efc)
        print(p, p.exists())
        with HNSWIndex.load(p, base) as idx:
            ids, _, nd = idx.search(queries[0], ef=10, k=10)
            print(f"  efc={efc}: nd={nd}, first id={ids[0]}")

    with HNSWIndex.load("data/indexes/sift1m_M16_efc200.bin", base) as idx:
        degrees = np.array([idx.degree(i, 0) for i in range(0, idx.n, 1000)])
        print(f"layer 0 degree over 1000 sampled nodes:")
        print(f"  mean {degrees.mean():.1f}, median {np.median(degrees):.0f}, "
            f"min {degrees.min()}, max {degrees.max()}, M0={2*idx.M}")
        print(f"  nodes with degree < 5: {(degrees < 5).sum()}")
        print(f"  nodes at full M0:      {(degrees == 2*idx.M).sum()}")


if __name__ == "__main__":
    main()