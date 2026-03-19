"""01 — Summary statistics and overview of all census results.

Prints a formatted table of total pairs, percentages, and distribution stats
for each threshold. Generates a grouped bar chart comparing all thresholds.
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt


def main(output_dir="output"):
    setup_style()
    data = load_all(output_dir)
    meta = load_metadata(output_dir)

    total_possible = N * N

    print(f"{'Threshold':<28} {'Total Pairs':>18} {'%':>8} {'Min':>10} {'Max':>10} {'Mean':>12} {'Median':>12} {'Std':>12}")
    print("=" * 120)

    names = []
    totals = []
    pcts = []
    means = []
    medians = []

    for name, counts in sorted(data.items()):
        total = int(counts.sum(dtype=np.uint64))
        pct = total / total_possible * 100
        mn = int(counts.min())
        mx = int(counts.max())
        mean = counts.mean()
        med = float(np.median(counts))
        std = counts.std()

        print(f"{name:<28} {total:>18,} {pct:>7.2f}% {mn:>10,} {mx:>10,} {mean:>12,.1f} {med:>12,.1f} {std:>12,.1f}")

        names.append(name)
        totals.append(total)
        pcts.append(pct)
        means.append(mean)
        medians.append(med)

    # Bar chart: % of all pairs passing each threshold
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors_apca = ["#58a6ff", "#388bfd", "#1f6feb"]
    colors_wcag = ["#f0883e", "#db6d28", "#bd561d"]
    colors_both = ["#8b949e", "#6e7681", "#484f58"]

    bar_colors = []
    for n in names:
        if n.startswith("apca"):
            bar_colors.append(colors_apca.pop(0) if colors_apca else "#58a6ff")
        elif n.startswith("wcag"):
            bar_colors.append(colors_wcag.pop(0) if colors_wcag else "#f0883e")
        else:
            bar_colors.append(colors_both.pop(0) if colors_both else "#8b949e")

    ax1.barh(names, pcts, color=bar_colors)
    ax1.set_xlabel("% of all 281T pairs passing")
    ax1.set_title("Passing Rate by Threshold")
    ax1.invert_yaxis()

    ax2.barh(names, means, color=bar_colors)
    ax2.set_xlabel("Mean accessible backgrounds per text color")
    ax2.set_title("Mean Accessible Pairs per Color")
    ax2.invert_yaxis()

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/01_summary_stats.png", bbox_inches="tight")
    print(f"\nSaved figures/01_summary_stats.png")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(d)
