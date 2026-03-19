"""08 — Most and least accessible colors.

Ranks all 16M colors by their accessible pair count. Shows the top 50
and bottom 50 for each threshold, with their hex codes and RGB values.

Generates a visual swatch grid of the most and least accessible colors.
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main(output_dir="output"):
    setup_style()
    data = load_all(output_dir)
    lum = srgb_luminance(ALL_HEX)

    thresholds = [n for n in sorted(data) if not n.startswith("both")]

    for name in thresholds[:6]:
        counts = data[name]

        # Sort indices
        sorted_idx = np.argsort(counts)
        top50 = sorted_idx[-50:][::-1]
        bottom50 = sorted_idx[:50]

        # Filter bottom to only non-zero (zero means no accessible pairs at all)
        nonzero_bottom = sorted_idx[counts[sorted_idx] > 0][:50]

        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        print(f"\n  Top 25 most accessible (as text):")
        print(f"  {'Rank':>4} {'Hex':>8} {'Count':>12} {'Luminance':>10} {'R':>4} {'G':>4} {'B':>4}")
        for rank, idx in enumerate(top50[:25]):
            h = int(idx)
            r, g, b = (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF
            print(f"  {rank+1:>4} #{h:06x} {counts[idx]:>12,} {lum[idx]:>10.4f} {r:>4} {g:>4} {b:>4}")

        print(f"\n  Bottom 25 least accessible (nonzero, as text):")
        print(f"  {'Rank':>4} {'Hex':>8} {'Count':>12} {'Luminance':>10} {'R':>4} {'G':>4} {'B':>4}")
        for rank, idx in enumerate(nonzero_bottom[:25]):
            h = int(idx)
            r, g, b = (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF
            print(f"  {rank+1:>4} #{h:06x} {counts[idx]:>12,} {lum[idx]:>10.4f} {r:>4} {g:>4} {b:>4}")

        zero_count = (counts == 0).sum()
        print(f"\n  Colors with ZERO accessible backgrounds: {zero_count:,} ({zero_count/N*100:.2f}%)")

    # Visual swatch for APCA 75 and WCAG 4.5
    for name in ["apca_75", "wcag_4_5"]:
        if name not in data:
            continue
        counts = data[name]
        sorted_idx = np.argsort(counts)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4))

        # Top 100
        top100 = sorted_idx[-100:][::-1]
        for i, idx in enumerate(top100):
            h = int(idx)
            r, g, b = ((h >> 16) & 0xFF) / 255, ((h >> 8) & 0xFF) / 255, (h & 0xFF) / 255
            rect = Rectangle((i, 0), 1, 1, facecolor=(r, g, b))
            ax1.add_patch(rect)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        ax1.set_title(f"{name}: Top 100 most accessible text colors", fontsize=11)
        ax1.set_xlabel("Rank")

        # Bottom 100 (nonzero)
        nonzero = sorted_idx[counts[sorted_idx] > 0][:100]
        for i, idx in enumerate(nonzero):
            h = int(idx)
            r, g, b = ((h >> 16) & 0xFF) / 255, ((h >> 8) & 0xFF) / 255, (h & 0xFF) / 255
            rect = Rectangle((i, 0), 1, 1, facecolor=(r, g, b))
            ax2.add_patch(rect)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.set_title(f"{name}: Bottom 100 least accessible text colors (nonzero)", fontsize=11)
        ax2.set_xlabel("Rank")

        plt.tight_layout()
        Path("figures").mkdir(exist_ok=True)
        plt.savefig(f"figures/08_swatches_{name}.png", bbox_inches="tight")
        print(f"Saved figures/08_swatches_{name}.png")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(d)
