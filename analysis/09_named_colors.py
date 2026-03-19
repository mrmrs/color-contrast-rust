"""09 — CSS named colors accessibility ranking.

For each of the ~140 CSS named colors, shows the exact accessible pair
count under each threshold. Produces a ranked table and comparison chart.

Practical value: tells designers exactly how much of the color space is
available when they pick a named color for text.
"""

import sys
sys.path.insert(0, ".")
from read_census import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main(output_dir="output"):
    setup_style()
    data = load_all(output_dir)
    lum = srgb_luminance(ALL_HEX)

    thresholds = [n for n in sorted(data) if not n.startswith("both")]

    # Build table
    rows = []
    for name, hex_val in sorted(CSS_NAMED_COLORS.items()):
        row = {"name": name, "hex": hex_val, "luminance": float(lum[hex_val])}
        for tn in thresholds[:6]:
            row[tn] = int(data[tn][hex_val])
        rows.append(row)

    # Print ranked by APCA 75 count
    print(f"\nCSS Named Colors ranked by APCA 75 accessible background count:")
    print(f"{'Rank':>4} {'Name':<20} {'Hex':>8} {'Lum':>6}", end="")
    for tn in thresholds[:6]:
        print(f" {tn:>14}", end="")
    print()
    print("-" * (42 + 15 * 6))

    rows_sorted = sorted(rows, key=lambda r: r.get("apca_75", 0), reverse=True)
    for rank, row in enumerate(rows_sorted):
        print(f"{rank+1:>4} {row['name']:<20} #{row['hex']:06x} {row['luminance']:>5.3f}", end="")
        for tn in thresholds[:6]:
            print(f" {row.get(tn, 0):>14,}", end="")
        print()

    # Percentage of color space accessible
    print(f"\n\n% of color space accessible as background:")
    print(f"{'Name':<20} {'Hex':>8}", end="")
    for tn in thresholds[:6]:
        print(f" {tn:>10}", end="")
    print()

    for row in rows_sorted[:30]:
        print(f"{row['name']:<20} #{row['hex']:06x}", end="")
        for tn in thresholds[:6]:
            pct = row.get(tn, 0) / N * 100
            print(f" {pct:>9.2f}%", end="")
        print()

    # Visual chart: APCA 75 vs WCAG 4.5 for named colors
    if "apca_75" in data and "wcag_4_5" in data:
        fig, ax = plt.subplots(figsize=(14, 10))

        for row in rows:
            h = row["hex"]
            r = ((h >> 16) & 0xFF) / 255
            g = ((h >> 8) & 0xFF) / 255
            b = (h & 0xFF) / 255
            ax.scatter(row.get("wcag_4_5", 0), row.get("apca_75", 0),
                       c=[(r, g, b)], s=60, edgecolors="#30363d", linewidth=0.5,
                       zorder=3)

        # Label some notable colors
        notable = ["black", "white", "red", "blue", "green", "gray", "navy",
                    "yellow", "purple", "orange", "cyan", "magenta"]
        for row in rows:
            if row["name"] in notable:
                ax.annotate(row["name"],
                            (row.get("wcag_4_5", 0), row.get("apca_75", 0)),
                            fontsize=8, ha="left", va="bottom",
                            xytext=(5, 5), textcoords="offset points",
                            color="#c9d1d9")

        # y=x reference
        max_val = max(max(r.get("wcag_4_5", 0) for r in rows),
                      max(r.get("apca_75", 0) for r in rows))
        ax.plot([0, max_val], [0, max_val], "--", color="#484f58", alpha=0.5, label="y=x")

        ax.set_xlabel("WCAG 4.5 accessible backgrounds")
        ax.set_ylabel("APCA 75 accessible backgrounds")
        ax.set_title("CSS Named Colors: WCAG 4.5 vs APCA 75")
        ax.legend()

        plt.tight_layout()
        Path("figures").mkdir(exist_ok=True)
        plt.savefig("figures/09_named_colors.png", bbox_inches="tight")
        print(f"\nSaved figures/09_named_colors.png")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(d)
