"""07 — Luminance vs accessible pair count.

Plots each color's relative luminance against its accessible pair count.
Reveals the fundamental relationship between brightness and accessibility
options. Extreme luminances (near black/white) should have the most options.
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt


def main(output_dir="output"):
    setup_style()
    data = load_all(output_dir)

    lum = srgb_luminance(ALL_HEX)

    thresholds = [n for n in sorted(data) if not n.startswith("both")]

    # Subsample for plotting (16M points would overwhelm matplotlib)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=500_000, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, name in enumerate(thresholds[:6]):
        counts = data[name]
        ax = axes[i]

        # Color each point by its actual RGB
        r, g, b = hex_to_rgb_float(ALL_HEX[idx])
        rgb_colors = np.stack([r, g, b], axis=1)

        ax.scatter(lum[idx], counts[idx], s=0.05, c=rgb_colors, alpha=0.3)
        ax.set_xlabel("Relative Luminance")
        ax.set_ylabel("Accessible backgrounds")
        ax.set_title(name)

    plt.suptitle("Luminance vs Accessible Pair Count (500K sample, true color)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/07_luminance_scatter.png", bbox_inches="tight")
    print(f"Saved figures/07_luminance_scatter.png")

    # Binned luminance means (exact, no sampling)
    print(f"\n{'Luminance':>12}", end="")
    for name in thresholds[:6]:
        print(f" {name:>14}", end="")
    print()
    print("-" * (14 + 15 * 6))

    lum_bins = np.linspace(0, 1, 21)
    lum_idx = np.clip(np.digitize(lum, lum_bins) - 1, 0, 19)
    for bi in range(20):
        mask = lum_idx == bi
        center = (lum_bins[bi] + lum_bins[bi + 1]) / 2
        print(f"  {center:>8.2f}    ", end="")
        for name in thresholds[:6]:
            mean = data[name][mask].mean() if mask.sum() > 0 else 0
            print(f" {mean:>14,.0f}", end="")
        print()

    # APCA vs WCAG curve shape comparison
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    lum_fine = np.linspace(0, 1, 201)
    lum_fine_idx = np.clip(np.digitize(lum, lum_fine) - 1, 0, 199)

    colors_line = {
        "apca_60": "#58a6ff", "apca_75": "#388bfd", "apca_90": "#1f6feb",
        "wcag_3_0": "#f0883e", "wcag_4_5": "#db6d28", "wcag_7_0": "#bd561d",
    }
    for name in thresholds[:6]:
        means = np.array([
            data[name][lum_fine_idx == bi].mean()
            if (lum_fine_idx == bi).sum() > 0 else 0
            for bi in range(200)
        ])
        centers = (lum_fine[:-1] + lum_fine[1:]) / 2
        # Normalize to [0,1] for shape comparison
        if means.max() > 0:
            means_norm = means / means.max()
        else:
            means_norm = means
        ax2.plot(centers, means_norm, label=name, color=colors_line.get(name, "#8b949e"),
                 linewidth=2, alpha=0.85)

    ax2.set_xlabel("Relative Luminance")
    ax2.set_ylabel("Normalized mean count")
    ax2.set_title("APCA vs WCAG: Accessibility Curve Shape by Luminance")
    ax2.legend()

    plt.tight_layout()
    fig2.savefig("figures/07_luminance_curves.png", bbox_inches="tight")
    print(f"Saved figures/07_luminance_curves.png")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(d)
