"""04 — APCA polarity asymmetry analysis.

APCA treats text and background differently (different exponents).
This script compares the normal census (indexed by text) with the
swapped census (indexed by background) to quantify the asymmetry.

Requires: census run with --swap (output_bg/ directory)

For WCAG, the two runs should produce identical counts (sanity check).
For APCA, the difference reveals polarity asymmetry.
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt


def main(txt_dir="output", bg_dir="output_bg"):
    setup_style()

    txt_data = load_all(txt_dir)
    try:
        bg_data = load_all(bg_dir)
    except (FileNotFoundError, AssertionError):
        print(f"ERROR: '{bg_dir}/' not found. Run census with --swap first:")
        print(f"  ./target/release/census --swap")
        return

    # Sanity check: WCAG should be identical
    print("=== WCAG Symmetry Check ===")
    for name in sorted(txt_data):
        if not name.startswith("wcag"):
            continue
        if name not in bg_data:
            continue
        diff = (txt_data[name].astype(np.int64) - bg_data[name].astype(np.int64))
        max_diff = np.abs(diff).max()
        print(f"  {name}: max |txt - bg| = {max_diff}  {'✓ PASS' if max_diff == 0 else '✗ FAIL'}")

    # APCA asymmetry
    print("\n=== APCA Polarity Asymmetry ===")
    apca_names = [n for n in sorted(txt_data) if n.startswith("apca")]

    fig, axes = plt.subplots(1, len(apca_names), figsize=(6 * len(apca_names), 6))
    if len(apca_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, apca_names):
        if name not in bg_data:
            print(f"  {name}: not in bg data, skipping")
            continue

        as_txt = txt_data[name].astype(np.int64)
        as_bg = bg_data[name].astype(np.int64)
        diff = as_txt - as_bg  # positive = more options when used as text

        print(f"\n  {name}:")
        print(f"    Mean diff (txt − bg):   {diff.mean():>+12.1f}")
        print(f"    Median diff:            {np.median(diff):>+12.1f}")
        print(f"    Std of diff:            {diff.std():>12.1f}")
        print(f"    Max (favors text):      {diff.max():>+12,}")
        print(f"    Min (favors bg):        {diff.min():>+12,}")
        print(f"    Colors where txt > bg:  {(diff > 0).sum():>10,}  ({(diff > 0).sum()/N*100:.1f}%)")
        print(f"    Colors where txt < bg:  {(diff < 0).sum():>10,}  ({(diff < 0).sum()/N*100:.1f}%)")
        print(f"    Colors where txt == bg: {(diff == 0).sum():>10,}  ({(diff == 0).sum()/N*100:.1f}%)")

        # Histogram of differences
        pct = np.percentile(diff, [0.5, 99.5])
        ax.hist(diff, bins=200, range=(pct[0], pct[1]), color="#58a6ff", alpha=0.8)
        ax.axvline(0, color="#f0883e", linestyle="--", linewidth=1.5)
        ax.axvline(diff.mean(), color="#3fb950", linestyle="-", linewidth=1.5,
                   label=f"mean={diff.mean():+.0f}")
        ax.set_xlabel("Count as text − Count as background")
        ax.set_ylabel("Colors")
        ax.set_title(f"{name} Polarity Asymmetry")
        ax.legend()

        # Asymmetry vs luminance
    fig2, axes2 = plt.subplots(1, len(apca_names), figsize=(6 * len(apca_names), 5))
    if len(apca_names) == 1:
        axes2 = [axes2]

    lum = srgb_luminance(ALL_HEX)

    for ax, name in zip(axes2, apca_names):
        if name not in bg_data:
            continue
        diff = txt_data[name].astype(np.int64) - bg_data[name].astype(np.int64)

        # Bin by luminance, show mean asymmetry per bin
        lum_bins = np.linspace(0, 1, 101)
        bin_idx = np.digitize(lum, lum_bins) - 1
        bin_means = np.array([
            diff[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else 0
            for i in range(len(lum_bins) - 1)
        ])
        bin_centers = (lum_bins[:-1] + lum_bins[1:]) / 2

        ax.bar(bin_centers, bin_means, width=0.01, color="#58a6ff", alpha=0.8)
        ax.axhline(0, color="#f0883e", linestyle="--")
        ax.set_xlabel("Relative Luminance")
        ax.set_ylabel("Mean (txt count − bg count)")
        ax.set_title(f"{name}: Asymmetry by Luminance")

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/04_polarity_histogram.png", bbox_inches="tight")
    fig2.savefig("figures/04_polarity_by_luminance.png", bbox_inches="tight")
    print(f"\nSaved figures/04_polarity_histogram.png")
    print(f"Saved figures/04_polarity_by_luminance.png")


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "output"
    b = sys.argv[2] if len(sys.argv) > 2 else "output_bg"
    main(t, b)
