import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")           # no display needed — write files directly
import matplotlib.pyplot as plt

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


def main():
    rows = read_csv(RESULTS_DIR / "sweep_random.csv")
    plot_pareto(rows, RESULTS_DIR / "pareto_random.png",
                "Random graph — recall vs throughput (siftsmall)")
    plot_diagnostics(rows, RESULTS_DIR / "diagnostics_random.png")


if __name__ == "__main__":
    main()