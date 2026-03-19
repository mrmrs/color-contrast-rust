"""10 — RGB cube visualization data export.

Exports downsampled (64x64x64) versions of the count data for 3D
visualization. Each voxel averages the counts of the 4x4x4 block of
full-resolution colors it contains.

Also generates 2D slice visualizations through the RGB cube.
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def downsample_cube(counts: np.ndarray, factor: int = 4) -> np.ndarray:
    """Downsample 256^3 flat array to (256/factor)^3 cube by averaging."""
    side = 256 // factor
    cube = np.zeros((side, side, side), dtype=np.float64)

    # Reshape into 3D
    full = counts.reshape(256, 256, 256).astype(np.float64)

    for ri in range(side):
        for gi in range(side):
            for bi in range(side):
                block = full[
                    ri * factor:(ri + 1) * factor,
                    gi * factor:(gi + 1) * factor,
                    bi * factor:(bi + 1) * factor,
                ]
                cube[ri, gi, bi] = block.mean()

    return cube


def main(output_dir="output"):
    setup_style()
    data = load_all(output_dir)

    thresholds = [n for n in sorted(data) if not n.startswith("both")]

    Path("figures").mkdir(exist_ok=True)
    export_dir = Path("figures/cube_data")
    export_dir.mkdir(exist_ok=True)

    # Export downsampled cubes
    print("Downsampling to 64x64x64 cubes...")
    for name in thresholds[:6]:
        print(f"  {name}...", end="", flush=True)
        cube = downsample_cube(data[name], factor=4)
        np.save(export_dir / f"{name}_64.npy", cube.astype(np.float32))
        print(f" saved ({cube.shape}, {cube.nbytes / 1024:.0f}KB)")

    # 2D slice visualizations: slices through the RGB cube
    # R-axis slices at R=0, R=128, R=255
    for name in ["apca_75", "wcag_4_5"]:
        if name not in data:
            continue

        counts = data[name].reshape(256, 256, 256)

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        slices = [
            ("R=0", counts[0, :, :], "Red=0"),
            ("R=128", counts[128, :, :], "Red=128"),
            ("R=255", counts[255, :, :], "Red=255"),
            ("G=0", counts[:, 0, :], "Green=0"),
            ("G=128", counts[:, 128, :], "Green=128"),
            ("G=255", counts[:, 255, :], "Green=255"),
            ("B=0", counts[:, :, 0], "Blue=0"),
            ("B=128", counts[:, :, 128], "Blue=128"),
            ("B=255", counts[:, :, 255], "Blue=255"),
        ]

        vmax = max(s[1].max() for s in slices)

        for ax, (label, slice_data, title) in zip(axes.flatten(), slices):
            im = ax.imshow(slice_data, cmap="inferno", aspect="equal",
                           origin="lower", norm=Normalize(vmin=0, vmax=vmax))
            ax.set_title(f"{title}", fontsize=10)

            # Axis labels depend on which dimension was sliced
            if label.startswith("R"):
                ax.set_xlabel("Blue")
                ax.set_ylabel("Green")
            elif label.startswith("G"):
                ax.set_xlabel("Blue")
                ax.set_ylabel("Red")
            else:
                ax.set_xlabel("Green")
                ax.set_ylabel("Red")

        fig.colorbar(im, ax=axes, shrink=0.6, label="Accessible backgrounds")
        plt.suptitle(f"{name}: RGB Cube Cross-Sections", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(f"figures/10_cube_slices_{name}.png", bbox_inches="tight")
        print(f"Saved figures/10_cube_slices_{name}.png")

    # Lightness-sorted heatmap: reshape by luminance bands
    print("\nGenerating luminance-sorted heatmaps...")
    lum = srgb_luminance(ALL_HEX)

    for name in ["apca_75", "wcag_4_5"]:
        if name not in data:
            continue

        counts = data[name]
        sort_idx = np.argsort(lum)

        # Reshape sorted data into a 4096x4096 image
        side = 4096
        sorted_counts = counts[sort_idx].reshape(side, side)

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(sorted_counts, cmap="inferno", aspect="equal", origin="lower")
        ax.set_xlabel("Position (within luminance band)")
        ax.set_ylabel("Luminance rank (dark → light)")
        ax.set_title(f"{name}: All 16M colors sorted by luminance")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Accessible backgrounds")

        plt.tight_layout()
        plt.savefig(f"figures/10_luminance_sorted_{name}.png", bbox_inches="tight")
        print(f"Saved figures/10_luminance_sorted_{name}.png")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(d)
