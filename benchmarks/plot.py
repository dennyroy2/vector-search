import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")           # no display needed — write files directly
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("ef", "k", "recall", "qps", "ms_per_query",
                    "ndists", "pct_scanned"):
            r[key] = float(r[key])
    return rows


def plot_pareto(all_rows, out_path, title):
    """The headline plot: recall vs QPS. Up and to the right is better."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = sorted({r["label"] for r in all_rows if r["label"] != "bruteforce"})
    for label in labels:
        rows = sorted([r for r in all_rows if r["label"] == label],
                      key=lambda r: r["recall"])
        ax.plot([r["recall"] for r in rows], [r["qps"] for r in rows],
                marker="o", label=label)

        # Annotate each point with its ef so the curve is readable.
        for r in rows:
            ax.annotate(f"{int(r['ef'])}", (r["recall"], r["qps"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points",
                        alpha=0.7)

    bf = [r for r in all_rows if r["label"] == "bruteforce"]
    if bf:
        ax.axhline(bf[0]["qps"], ls="--", c="gray", lw=1)
        ax.plot(1.0, bf[0]["qps"], marker="*", ms=14, c="black",
                label="brute force (exact)")

    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def plot_diagnostics(all_rows, out_path):
    """Two diagnostic panels: recall vs ef, and recall vs work done."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = sorted({r["label"] for r in all_rows if r["label"] != "bruteforce"})

    for label in labels:
        rows = sorted([r for r in all_rows if r["label"] == label],
                      key=lambda r: r["ef"])
        ax1.plot([r["ef"] for r in rows], [r["recall"] for r in rows],
                 marker="o", label=label)
        ax2.plot([r["pct_scanned"] for r in rows], [r["recall"] for r in rows],
                 marker="o", label=label)

    ax1.set_xlabel("ef (beam width)")
    ax1.set_ylabel("recall@10")
    ax1.set_xscale("log")
    ax1.set_title("recall vs beam width")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    # The y=x line: recall equal to fraction scanned means the graph
    # is contributing nothing beyond random sampling.
    ax2.plot([0, 100], [0, 1], ls=":", c="red", lw=1,
             label="random sampling")
    ax2.set_xlabel("% of collection scanned")
    ax2.set_ylabel("recall@10")
    ax2.set_title("recall vs work done")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

def qps_at_recall(rows, target_recall):
    """QPS and ndists at a target recall.

    Returns (qps, ndists, status) where status is:
      "interp"  — interpolated between two measured points
      "exceeds" — the cheapest setting already beats the target, so this
                  is the ef=10 row: an upper bound on what's needed
      "below"   — the index never reaches this recall at any ef swept
    """
    rows = sorted(rows, key=lambda r: r["recall"])
    recalls = [r["recall"] for r in rows]
    qpss    = [r["qps"] for r in rows]
    ndists  = [r["ndists"] for r in rows]

    if target_recall > recalls[-1]:
        return None, None, "below"

    if target_recall < recalls[0]:
        # Cheapest measured setting already exceeds the target. Report it —
        # the true QPS at exactly this recall would be HIGHER, since a
        # smaller ef would be faster. This is a conservative bound.
        return qpss[0], ndists[0], "exceeds"

    return (float(np.interp(target_recall, recalls, qpss)),
            float(np.interp(target_recall, recalls, ndists)),
            "interp")


def matched_recall_table(all_rows, targets=(0.80, 0.90, 0.95, 0.99)):
    """Compare indexes at equal recall rather than equal parameters."""
    labels = sorted({r["label"] for r in all_rows if r["label"] != "bruteforce"})
    bf = [r for r in all_rows if r["label"] == "bruteforce"]
    bf_qps = bf[0]["qps"] if bf else None

    if bf_qps:
        print(f"brute force baseline: {bf_qps:.0f} QPS\n")

    header = f"{'recall':>8} " + "".join(f"{l:>26}" for l in labels)
    print(header)
    print("-" * len(header))

    saw_exceeds = False

    for t in targets:
        line = f"{t:>8.2f} "
        for label in labels:
            rows = [r for r in all_rows if r["label"] == label]
            qps, nd, status = qps_at_recall(rows, t)

            if status == "below":
                line += f"{'not reached':>26}"
            else:
                mark = "*" if status == "exceeds" else " "
                if status == "exceeds":
                    saw_exceeds = True
                speed = f"{qps/bf_qps:.0f}x" if bf_qps else ""
                line += f"{mark}{qps:>10.0f} QPS {speed:>5} {nd:>5.0f}nd"
        print(line)

    print()
    if saw_exceeds:
        print("* cheapest swept setting already exceeds this recall;")
        print("  true QPS at exactly this recall would be higher.")



def main():
    rows = read_csv(RESULTS_DIR / "sweep_random.csv")

    plot_pareto(rows, RESULTS_DIR / "pareto.png",
                "recall vs throughput — siftsmall")
    plot_diagnostics(rows, RESULTS_DIR / "diagnostics.png")

    print()
    matched_recall_table(rows)


if __name__ == "__main__":
    main()