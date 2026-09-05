import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"

NUMERIC = ("M", "ef_construction", "ef_search", "k", "recall", "qps",
           "ms_per_query", "ndists", "descent_ndists", "pct_scanned",
           "build_seconds", "index_bytes", "n_queries")


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in NUMERIC:
            if key in r:
                r[key] = float(r[key])
    return rows


def curves(rows):
    """Group HNSW rows into one curve per (M, efConstruction)."""
    hnsw = [r for r in rows if r["label"] != "bruteforce"]
    keys = sorted({(int(r["M"]), int(r["ef_construction"])) for r in hnsw})
    out = []
    for M, efc in keys:
        pts = sorted([r for r in hnsw
                      if int(r["M"]) == M and int(r["ef_construction"]) == efc],
                     key=lambda r: r["recall"])
        out.append(((M, efc), pts))
    return out


def plot_pareto(rows, out_path, log_x=False):
    """The headline chart. Up and to the right is better."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for (M, efc), pts in curves(rows):
        x = [r["recall"] for r in pts]
        y = [r["qps"] for r in pts]
        if log_x:
            # 1 - recall on a log axis spreads out the high-recall region,
            # where the differences between configurations actually matter.
            x = [max(1 - v, 1e-4) for v in x]
        if M == 16 and efc == 100:
            label = "M=16, efC=100/200 (identical)"
            # thick and faded so the solid efC=200 line sits visibly on top
            ax.plot(x, y, marker="o", ms=4, linewidth=5, alpha=0.35, label=label)
        elif M == 16 and efc == 200:
            # already covered by the combined label above — draw, don't label
            ax.plot(x, y, marker="o", ms=4)
        else:
            ax.plot(x, y, marker="o", ms=4, label=f"M={M}, efC={efc}")

    bf = [r for r in rows if r["label"] == "bruteforce"]
    if bf:
        ax.axhline(bf[0]["qps"], ls="--", c="gray", lw=1,
                   label=f"exact search ({bf[0]['qps']:.0f} QPS)")

    ax.set_ylabel("queries per second")
    ax.set_yscale("log")
    if log_x:
        ax.set_xlabel("1 − recall@10  (lower is better)")
        ax.set_xscale("log")
        ax.invert_xaxis()          # so better is still to the right
    else:
        ax.set_xlabel("recall@10")
        ax.set_xlim(0.5, 1.005)

    ax.set_title("HNSW on SIFT-1M — recall vs throughput")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_work(rows, out_path):
    """Hardware-independent view: recall vs distance computations."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for (M, efc), pts in curves(rows):
        ax.plot([r["recall"] for r in pts], [r["ndists"] for r in pts],
                marker="o", ms=4, label=f"M={M}, efC={efc}")

    bf = [r for r in rows if r["label"] == "bruteforce"]
    if bf:
        ax.axhline(bf[0]["ndists"], ls="--", c="gray", lw=1,
                   label="exact search (1,000,000)")

    ax.set_xlabel("recall@10")
    ax.set_ylabel("distance computations per query")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1.005)
    ax.set_title("Work done per query (hardware-independent)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_memory(rows, out_path):
    """What each configuration costs in memory, and what it buys."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for (M, efc), pts in curves(rows):
        mem_mb = pts[0]["index_bytes"] / 1e6
        best = max(r["recall"] for r in pts)
        # QPS at the highest-recall point on this curve.
        qps_at_best = [r["qps"] for r in pts if r["recall"] == best][0]
        ax.scatter(mem_mb, qps_at_best, s=80)
        ax.annotate(f"M={M}, efC={efc}\n(recall {best:.3f})",
                    (mem_mb, qps_at_best), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")

    ax.set_xlabel("index memory (MB)")
    ax.set_ylabel("QPS at highest measured recall")
    ax.set_title("Memory cost per configuration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def qps_at_recall(pts, target):
    """Interpolate QPS and ndists at a target recall."""
    pts = sorted(pts, key=lambda r: r["recall"])
    rs = [r["recall"] for r in pts]
    if target > rs[-1]:
        return None, None, "not reached"
    if target < rs[0]:
        return pts[0]["qps"], pts[0]["ndists"], "exceeds"
    return (float(np.interp(target, rs, [r["qps"] for r in pts])),
            float(np.interp(target, rs, [r["ndists"] for r in pts])),
            "interp")


def matched_recall_table(rows, targets=(0.90, 0.95, 0.99)):
    bf = [r for r in rows if r["label"] == "bruteforce"]
    bf_qps = bf[0]["qps"] if bf else None
    print(f"\nbrute force: {bf_qps:.0f} QPS, 1,000,000 distances/query\n")

    print(f"{'config':>16} " + "".join(f"{f'recall {t}':>26}" for t in targets))
    print("-" * (16 + 26 * len(targets)))

    for (M, efc), pts in curves(rows):
        line = f"{f'M={M} efC={efc}':>16} "
        for t in targets:
            qps, nd, status = qps_at_recall(pts, t)
            if qps is None:
                line += f"{'not reached':>26}"
            else:
                mark = "*" if status == "exceeds" else " "
                line += (f"{mark}{qps:>9.0f} QPS "
                         f"{qps/bf_qps:>5.0f}x {nd:>6.0f}nd")
        print(line)
    print("\n* cheapest swept setting already exceeds this recall")

def plot_comparison(mine, faiss_rows, out_path, y="qps"):
    """Both implementations, same axes. Solid = mine, dashed = FAISS."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Colour by M so the pairs line up visually.
    colours = {8: "tab:blue", 16: "tab:red", 32: "tab:green"}

    for rows, style, name in [(mine, "-", "mine"),
                              (faiss_rows, "--", "FAISS")]:
        for (M, efc), pts in curves(rows):
            if efc != 200:          # one efC per M keeps the chart readable
                continue
            ax.plot([r["recall"] for r in pts], [r[y] for r in pts],
                    marker="o", ms=4, linestyle=style, color=colours[M],
                    label=f"{name}, M={M}")

    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second" if y == "qps"
                  else "distance computations per query")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1.005)
    ax.set_title(f"Mine vs FAISS on SIFT-1M "
                 f"({'throughput' if y == 'qps' else 'work done'}, "
                 f"single-threaded)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

def comparison_table(mine, faiss_rows, targets=(0.90, 0.95, 0.99)):
    print(f"{'config':>12} {'recall':>8} {'mine QPS':>10} {'FAISS QPS':>11} "
          f"{'ratio':>7} {'mine nd':>9} {'FAISS nd':>10} {'nd ratio':>9}")
    print("-" * 82)

    for M in [8, 16, 32]:
        m_pts = [r for r in mine
                 if int(r["M"]) == M and int(r["ef_construction"]) == 200]
        f_pts = [r for r in faiss_rows
                 if int(r["M"]) == M and int(r["ef_construction"]) == 200]

        for t in targets:
            mq, mn, ms = qps_at_recall(m_pts, t)
            fq, fn, fs = qps_at_recall(f_pts, t)
            if mq is None or fq is None:
                print(f"{f'M={M}':>12} {t:>8.2f} {'not reached':>50}")
                continue
            print(f"{f'M={M}':>12} {t:>8.2f} {mq:>10.0f} {fq:>11.0f} "
                  f"{fq/mq:>6.2f}x {mn:>9.0f} {fn:>10.0f} {fn/mn:>8.2f}x")


def main():
    mine = read_csv(RESULTS_DIR / "sweep_sift1m.csv")
    faiss_rows = read_csv(RESULTS_DIR / "faiss_sift1m.csv")

    # Existing single-implementation plots.
    plot_pareto(mine, RESULTS_DIR / "pareto_sift1m.png")
    plot_pareto(mine, RESULTS_DIR / "pareto_sift1m_log.png", log_x=True)
    plot_work(mine, RESULTS_DIR / "work_sift1m.png")
    plot_memory(mine, RESULTS_DIR / "memory_sift1m.png")

    # The comparison.
    plot_comparison(mine, faiss_rows, RESULTS_DIR / "vs_faiss_qps.png", y="qps")
    plot_comparison(mine, faiss_rows, RESULTS_DIR / "vs_faiss_work.png", y="ndists")

    matched_recall_table(mine)
    print()
    comparison_table(mine, faiss_rows)


if __name__ == "__main__":
    main()