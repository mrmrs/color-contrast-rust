"""05 — Hue-dependent accessibility bias.

Groups all 16M colors by hue and saturation to reveal which hues are
systematically advantaged or disadvantaged under each algorithm.

Key insight: sRGB luminance coefficients (R=0.2126, G=0.7152, B=0.0722)
mean green dominates. Blue text is notoriously problematic — this quantifies
exactly how much worse it is.
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main(output_dir="output"):
    setup_style()
    data = load_all(output_dir)

    print("Computing HSL for all 16M colors (vectorized)...")
    h, s, l = hex_to_hsl_fast(ALL_HEX)

    # Filter out near-gray colors (s < 0.1) — hue is meaningless for grays
    chromatic = s >= 0.1
    print(f"  Chromatic colors (S >= 0.1): {chromatic.sum():,} ({chromatic.sum()/N*100:.1f}%)")

    # 36 hue bins of 10 degrees each
    hue_bins = np.linspace(0, 360, 37)
    hue_centers = (hue_bins[:-1] + hue_bins[1:]) / 2
    hue_idx = np.clip(np.digitize(h, hue_bins) - 1, 0, 35)

    # Hue labels
    hue_names = {
        0: "Red", 30: "Orange", 60: "Yellow", 90: "Chartreuse",
        120: "Green", 150: "Spring", 180: "Cyan", 210: "Azure",
        240: "Blue", 270: "Violet", 300: "Magenta", 330: "Rose",
    }

    thresholds = [n for n in sorted(data) if not n.startswith("both")]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), subplot_kw={"projection": "polar"})
    axes = axes.flatten()

    for ax_idx, name in enumerate(thresholds[:6]):
        counts = data[name]
        ax = axes[ax_idx]

        # Mean count per hue bin (chromatic only)
        bin_means = np.array([
            counts[chromatic & (hue_idx == i)].mean()
            if (chromatic & (hue_idx == i)).sum() > 0 else 0
            for i in range(36)
        ])

        # Normalize for comparison across thresholds
        if bin_means.max() > 0:
            bin_norm = bin_means / bin_means.max()
        else:
            bin_norm = bin_means

        # Color each bar by its actual hue
        theta = np.radians(hue_centers)
        width = np.radians(10)
        bar_colors = [mcolors.hsv_to_rgb((hc / 360, 0.8, 0.9)) for hc in hue_centers]

        bars = ax.bar(theta, bin_norm, width=width, color=bar_colors, alpha=0.85, edgecolor="#30363d")
        ax.set_title(name, pad=15, fontsize=12, color="#c9d1d9")
        ax.set_yticklabels([])
        ax.set_thetagrids(np.arange(0, 360, 30),
                          [hue_names.get(a, "") for a in range(0, 360, 30)],
                          fontsize=8)

    plt.suptitle("Mean Accessible Pairs by Hue (chromatic colors only)", fontsize=15, y=1.02)
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/05_hue_polar.png", bbox_inches="tight")
    print(f"Saved figures/05_hue_polar.png")

    # Print the raw numbers
    print(f"\n{'Hue':>6} {'Name':>12}", end="")
    for name in thresholds[:6]:
        print(f" {name:>14}", end="")
    print()
    print("-" * (20 + 15 * 6))

    for i in range(36):
        hc = hue_centers[i]
        label = hue_names.get(int(round(hc / 30) * 30) % 360, "")
        print(f"{hc:>5.0f}° {label:>12}", end="")
        for name in thresholds[:6]:
            mask = chromatic & (hue_idx == i)
            mean = data[name][mask].mean() if mask.sum() > 0 else 0
            print(f" {mean:>14,.0f}", end="")
        print()

    # Saturation analysis: how does saturation affect accessibility?
    print(f"\n\n{'Saturation':>12}", end="")
    for name in thresholds[:6]:
        print(f" {name:>14}", end="")
    print()

    sat_bins = np.linspace(0, 1, 11)
    sat_centers = (sat_bins[:-1] + sat_bins[1:]) / 2
    sat_idx = np.clip(np.digitize(s, sat_bins) - 1, 0, 9)

    for i in range(10):
        mask = sat_idx == i
        print(f"  {sat_centers[i]:>8.1f}  ", end="")
        for name in thresholds[:6]:
            mean = data[name][mask].mean() if mask.sum() > 0 else 0
            print(f" {mean:>14,.0f}", end="")
        print()


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(d)
