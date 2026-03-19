"""02 — APCA ↔ WCAG equivalence mapping.

For every APCA/WCAG threshold pair, measures the overlap of their passing sets
to find which thresholds are functionally equivalent. Outputs a correlation
matrix and Jaccard similarity heatmap.

Key question: at what APCA level does the accessible set best match WCAG 4.5?
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two boolean arrays."""
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return intersection / union if union > 0 else 0.0


def main(output_dir="output"):
    setup_style()
    data = load_all(output_dir)

    apca_names = [n for n in sorted(data) if n.startswith("apca")]
    wcag_names = [n for n in sorted(data) if n.startswith("wcag")]

    # For each color: does it have ANY accessible backgrounds? (count > 0)
    # This defines the "accessible set" for that threshold.
    apca_sets = {n: data[n] > 0 for n in apca_names}
    wcag_sets = {n: data[n] > 0 for n in wcag_names}

    # Jaccard similarity matrix
    print("Jaccard Similarity (color has at least 1 accessible pair):")
    print(f"{'':>12}", end="")
    for wn in wcag_names:
        print(f"{wn:>14}", end="")
    print()

    jacc_matrix = np.zeros((len(apca_names), len(wcag_names)))
    for i, an in enumerate(apca_names):
        print(f"{an:>12}", end="")
        for j, wn in enumerate(wcag_names):
            j_val = jaccard(apca_sets[an], wcag_sets[wn])
            jacc_matrix[i, j] = j_val
            print(f"{j_val:>14.4f}", end="")
        print()

    # Pearson correlation of count vectors
    print("\nPearson Correlation (count vectors):")
    print(f"{'':>12}", end="")
    for wn in wcag_names:
        print(f"{wn:>14}", end="")
    print()

    corr_matrix = np.zeros((len(apca_names), len(wcag_names)))
    for i, an in enumerate(apca_names):
        print(f"{an:>12}", end="")
        for j, wn in enumerate(wcag_names):
            c = np.corrcoef(data[an].astype(np.float64), data[wn].astype(np.float64))[0, 1]
            corr_matrix[i, j] = c
            print(f"{c:>14.4f}", end="")
        print()

    # Count-weighted overlap: for the "both" files
    both_names = [n for n in sorted(data) if n.startswith("both")]
    if both_names:
        print("\nCross-threshold overlap (from 'both_*' counters):")
        for bn in both_names:
            total_both = int(data[bn].sum(dtype=np.uint64))
            # Parse which APCA and WCAG thresholds this corresponds to
            print(f"  {bn}: {total_both:>18,} pairs passing both thresholds")

    # Find best equivalences
    print("\nBest WCAG match for each APCA threshold (by Jaccard):")
    for i, an in enumerate(apca_names):
        best_j = int(np.argmax(jacc_matrix[i]))
        print(f"  {an} ↔ {wcag_names[best_j]} (J={jacc_matrix[i, best_j]:.4f})")

    print("\nBest APCA match for each WCAG threshold (by Jaccard):")
    for j, wn in enumerate(wcag_names):
        best_i = int(np.argmax(jacc_matrix[:, j]))
        print(f"  {wn} ↔ {apca_names[best_i]} (J={jacc_matrix[best_i, j]:.4f})")

    # Heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(jacc_matrix, cmap="viridis", aspect="auto",
                      norm=Normalize(vmin=0, vmax=1))
    ax1.set_xticks(range(len(wcag_names)), wcag_names, rotation=45, ha="right")
    ax1.set_yticks(range(len(apca_names)), apca_names)
    ax1.set_title("Jaccard Similarity")
    for i in range(len(apca_names)):
        for j in range(len(wcag_names)):
            ax1.text(j, i, f"{jacc_matrix[i,j]:.3f}", ha="center", va="center",
                     color="white" if jacc_matrix[i,j] < 0.5 else "black", fontsize=10)
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(corr_matrix, cmap="magma", aspect="auto",
                      norm=Normalize(vmin=0, vmax=1))
    ax2.set_xticks(range(len(wcag_names)), wcag_names, rotation=45, ha="right")
    ax2.set_yticks(range(len(apca_names)), apca_names)
    ax2.set_title("Pearson Correlation")
    for i in range(len(apca_names)):
        for j in range(len(wcag_names)):
            ax2.text(j, i, f"{corr_matrix[i,j]:.3f}", ha="center", va="center",
                     color="white" if corr_matrix[i,j] < 0.5 else "black", fontsize=10)
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    plt.suptitle("APCA ↔ WCAG 2.1 Threshold Equivalence", fontsize=15, y=1.02)
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/02_equivalence_map.png", bbox_inches="tight")
    print(f"\nSaved figures/02_equivalence_map.png")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(d)
