"""06 — Distribution shape analysis.

Analyzes the histogram shape of accessible-pair counts for each threshold.
Computes Shannon entropy, skewness, kurtosis. Tests for bimodality.

If APCA produces a more bimodal distribution (clearer pass/fail separation)
while WCAG is smoother, that's an argument for one over the other.
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt
from scipy import stats as sp_stats


def shannon_entropy(counts: np.ndarray, num_bins: int = 1000) -> float:
    """Shannon entropy of the count distribution (in bits)."""
    hist, _ = np.histogram(counts, bins=num_bins)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def main(output_dir="output"):
    setup_style()
    data = load_all(output_dir)

    thresholds = [n for n in sorted(data) if not n.startswith("both")]

    print(f"{'Threshold':<16} {'Entropy':>10} {'Skewness':>10} {'Kurtosis':>10} {'%_zero':>10} {'%_max':>10}")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, name in enumerate(thresholds[:6]):
        counts = data[name]
        c64 = counts.astype(np.float64)

        ent = shannon_entropy(counts)
        skew = float(sp_stats.skew(c64))
        kurt = float(sp_stats.kurtosis(c64))
        pct_zero = (counts == 0).sum() / N * 100
        pct_max = (counts == counts.max()).sum() / N * 100

        print(f"{name:<16} {ent:>10.3f} {skew:>10.3f} {kurt:>10.3f} {pct_zero:>9.3f}% {pct_max:>9.5f}%")

        # Histogram
        ax = axes[i]
        ax.hist(c64, bins=500, color="#58a6ff" if "apca" in name else "#f0883e",
                alpha=0.85, log=True)
        ax.set_xlabel("Accessible background count")
        ax.set_ylabel("Colors (log scale)")
        ax.set_title(f"{name}  (H={ent:.2f} bits, skew={skew:.1f})")

        # Mark mean and median
        mean_val = c64.mean()
        med_val = float(np.median(c64))
        ymin, ymax = ax.get_ylim()
        ax.axvline(mean_val, color="#3fb950", linestyle="-", linewidth=1.5,
                   label=f"mean={mean_val:,.0f}")
        ax.axvline(med_val, color="#f0883e", linestyle="--", linewidth=1.5,
                   label=f"median={med_val:,.0f}")
        ax.legend(fontsize=8)

    plt.suptitle("Distribution of Accessible Pair Counts", fontsize=15, y=1.02)
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/06_distributions.png", bbox_inches="tight")
    print(f"\nSaved figures/06_distributions.png")

    # Bimodality analysis using Hartigan's dip test approximation
    # We'll use a simpler metric: the ratio of the two tallest peaks
    print("\nBimodality indicator (valley-to-peak ratio in smoothed histogram):")
    for name in thresholds[:6]:
        counts = data[name]
        hist, edges = np.histogram(counts, bins=200)
        # Simple smoothing
        kernel = np.ones(5) / 5
        smoothed = np.convolve(hist, kernel, mode="same")

        peaks = []
        for j in range(1, len(smoothed) - 1):
            if smoothed[j] > smoothed[j-1] and smoothed[j] > smoothed[j+1]:
                peaks.append((smoothed[j], j))
        peaks.sort(reverse=True)

        if len(peaks) >= 2:
            # Find valley between two tallest peaks
            p1_idx = min(peaks[0][1], peaks[1][1])
            p2_idx = max(peaks[0][1], peaks[1][1])
            valley = smoothed[p1_idx:p2_idx+1].min()
            peak_min = min(peaks[0][0], peaks[1][0])
            ratio = valley / peak_min if peak_min > 0 else 1.0
            bimodal = "likely bimodal" if ratio < 0.5 else "unimodal"
            print(f"  {name:<16}: valley/peak = {ratio:.3f}  ({bimodal})")
        else:
            print(f"  {name:<16}: single peak detected (unimodal)")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(d)
