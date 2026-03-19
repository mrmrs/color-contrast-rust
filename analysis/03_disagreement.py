"""03 — Disagreement analysis: colors that pass one algorithm but fail the other.

Focuses on the most-debated pair: WCAG 4.5 vs APCA 75.
Identifies colors in the "disagreement set" and characterizes them by
luminance, hue, and saturation.
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt


def main(output_dir="output"):
    setup_style()
    data = load_all(output_dir)

    # Primary comparison: WCAG 4.5 vs APCA 75
    pairs = [
        ("wcag_3_0", "apca_60"),
        ("wcag_4_5", "apca_75"),
        ("wcag_7_0", "apca_90"),
    ]

    for wcag_name, apca_name in pairs:
        if wcag_name not in data or apca_name not in data:
            continue

        wcag = data[wcag_name]
        apca = data[apca_name]

        has_wcag = wcag > 0  # at least 1 accessible bg under WCAG
        has_apca = apca > 0  # at least 1 accessible bg under APCA

        both = has_wcag & has_apca
        wcag_only = has_wcag & ~has_apca
        apca_only = ~has_wcag & has_apca
        neither = ~has_wcag & ~has_apca

        print(f"\n{'='*60}")
        print(f"  {wcag_name} vs {apca_name}")
        print(f"{'='*60}")
        print(f"  Both pass:   {both.sum():>10,}  ({both.sum()/N*100:>5.2f}%)")
        print(f"  WCAG only:   {wcag_only.sum():>10,}  ({wcag_only.sum()/N*100:>5.2f}%)")
        print(f"  APCA only:   {apca_only.sum():>10,}  ({apca_only.sum()/N*100:>5.2f}%)")
        print(f"  Neither:     {neither.sum():>10,}  ({neither.sum()/N*100:>5.2f}%)")

        # Characterize disagreement by luminance
        lum = srgb_luminance(ALL_HEX)

        for label, mask in [("WCAG-only", wcag_only), ("APCA-only", apca_only)]:
            if mask.sum() == 0:
                print(f"\n  {label}: no colors in this set")
                continue
            subset_lum = lum[mask]
            print(f"\n  {label} luminance: mean={subset_lum.mean():.4f}  "
                  f"median={np.median(subset_lum):.4f}  "
                  f"std={subset_lum.std():.4f}  "
                  f"range=[{subset_lum.min():.4f}, {subset_lum.max():.4f}]")

    # Detailed plot for WCAG 4.5 vs APCA 75
    if "wcag_4_5" in data and "apca_75" in data:
        wcag = data["wcag_4_5"]
        apca = data["apca_75"]
        lum = srgb_luminance(ALL_HEX)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top left: count scatter (subsample for speed)
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=200_000, replace=False)
        ax = axes[0, 0]
        ax.scatter(wcag[idx], apca[idx], s=0.1, alpha=0.15, c="#58a6ff")
        ax.set_xlabel("WCAG 4.5 count")
        ax.set_ylabel("APCA 75 count")
        ax.set_title("Count Correlation (200K sample)")
        ax.plot([0, max(wcag[idx].max(), apca[idx].max())],
                [0, max(wcag[idx].max(), apca[idx].max())],
                "--", color="#f0883e", alpha=0.5, label="y=x")
        ax.legend()

        # Top right: disagreement by luminance
        ax = axes[0, 1]
        has_wcag = wcag > 0
        has_apca = apca > 0
        wcag_only = has_wcag & ~has_apca
        apca_only = ~has_wcag & has_apca

        bins = np.linspace(0, 1, 101)
        if wcag_only.sum() > 0:
            ax.hist(lum[wcag_only], bins=bins, alpha=0.7, label=f"WCAG-only ({wcag_only.sum():,})",
                    color="#f0883e")
        if apca_only.sum() > 0:
            ax.hist(lum[apca_only], bins=bins, alpha=0.7, label=f"APCA-only ({apca_only.sum():,})",
                    color="#58a6ff")
        ax.set_xlabel("Relative Luminance")
        ax.set_ylabel("Color count")
        ax.set_title("Disagreement Set by Luminance")
        ax.legend()

        # Bottom left: count difference histogram
        ax = axes[1, 0]
        diff = apca.astype(np.int64) - wcag.astype(np.int64)
        pct = np.percentile(diff, [1, 99])
        ax.hist(diff, bins=200, range=(pct[0], pct[1]), color="#8b949e", alpha=0.8)
        ax.axvline(0, color="#f0883e", linestyle="--", alpha=0.7)
        ax.set_xlabel("APCA 75 count − WCAG 4.5 count")
        ax.set_ylabel("Colors")
        ax.set_title("Per-Color Count Difference (1st–99th percentile)")

        # Bottom right: Venn-style summary
        ax = axes[1, 1]
        both = (has_wcag & has_apca).sum()
        wo = wcag_only.sum()
        ao = apca_only.sum()
        ne = (~has_wcag & ~has_apca).sum()

        labels = ["Both pass", "WCAG 4.5 only", "APCA 75 only", "Neither"]
        sizes = [both, wo, ao, ne]
        bar_colors = ["#3fb950", "#f0883e", "#58a6ff", "#484f58"]
        bars = ax.barh(labels, sizes, color=bar_colors)
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_width() + N * 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{size:,} ({size/N*100:.1f}%)", va="center", fontsize=10, color="#c9d1d9")
        ax.set_xlabel("Number of colors")
        ax.set_title("WCAG 4.5 vs APCA 75: Agreement")

        plt.suptitle("Disagreement Analysis: WCAG 4.5 vs APCA 75", fontsize=15, y=1.02)
        plt.tight_layout()
        Path("figures").mkdir(exist_ok=True)
        plt.savefig("figures/03_disagreement.png", bbox_inches="tight")
        print(f"\nSaved figures/03_disagreement.png")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(d)
