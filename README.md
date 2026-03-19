# Color Contrast Census

Exhaustive brute-force analysis of color contrast accessibility across the entire 24-bit sRGB gamut.

For every one of 16,777,216 text colors, tests **every** one of 16,777,216 background colors — **281,474,976,710,656 contrast evaluations** — under both WCAG 2.1 and APCA algorithms at multiple thresholds.

## What this produces

For each text color (hex `#000000` through `#ffffff`), the number of background colors that meet each contrast threshold:

| File | Algorithm | Threshold | Meaning |
|------|-----------|-----------|---------|
| `apca_60.bin` | APCA | \|Lc\| ≥ 60 | Large text minimum |
| `apca_75.bin` | APCA | \|Lc\| ≥ 75 | Body text minimum |
| `apca_90.bin` | APCA | \|Lc\| ≥ 90 | Enhanced contrast |
| `wcag_3_0.bin` | WCAG 2.1 | ratio ≥ 3.0 | Large text (AA) |
| `wcag_4_5.bin` | WCAG 2.1 | ratio ≥ 4.5 | Normal text (AA) |
| `wcag_7_0.bin` | WCAG 2.1 | ratio ≥ 7.0 | Enhanced (AAA) |
| `both_apca60_wcag3.bin` | Both | APCA 60 AND WCAG 3.0 | Cross-threshold overlap |
| `both_apca75_wcag45.bin` | Both | APCA 75 AND WCAG 4.5 | Cross-threshold overlap |
| `both_apca90_wcag7.bin` | Both | APCA 90 AND WCAG 7.0 | Cross-threshold overlap |

Each `.bin` file is a flat little-endian `u32[16777216]` array, indexed by hex value. To look up color `#1a2b3c` → read the u32 at byte offset `0x1a2b3c × 4`.

## Building

```
cargo build --release --bin census
```

## Running

```bash
# Normal: for each TEXT color, count accessible backgrounds → output/
./target/release/census

# Swap: for each BACKGROUND color, count accessible text colors → output_bg/
./target/release/census --swap

# Custom output directory
./target/release/census --output-dir my_output
```

The run is **resumable**. If interrupted (Ctrl+C, crash, power loss), it picks up from the last completed chunk via `progress.dat`.

### Target hardware

Designed for high core-count machines. On a Threadripper 3990X (64C/128T), rayon will saturate all threads. Each chunk processes 1,024 text colors in parallel, each iterating all 16M backgrounds.

### Optimizations

- **sRGB LUT**: 256-entry linearization table replaces `powf(2.4)` per channel
- **Precomputed power arrays**: APCA exponents (`^0.56`, `^0.57`, `^0.62`, `^0.65`) computed once for all 16M colors, eliminating `powf` from the inner loop
- **Packed struct**: All per-color data in a single `ColorData` struct (32 bytes) for cache-friendly sequential access — one memory stream instead of four
- **Nested thresholds**: `if lc >= 60 { if lc >= 75 { if lc >= 90 }}` avoids redundant comparisons

## Analysis

Python scripts for analyzing the census output. Run from the repo root.

```bash
pip install -r analysis/requirements.txt

# Run all analyses
cd analysis
python 01_summary_stats.py ../output
python 02_equivalence_map.py ../output
python 03_disagreement.py ../output
python 04_polarity.py ../output ../output_bg   # needs --swap run
python 05_hue_analysis.py ../output
python 06_distribution.py ../output
python 07_luminance_scatter.py ../output
python 08_top_bottom.py ../output
python 09_named_colors.py ../output
python 10_rgb_cube_export.py ../output
```

Figures are saved to `figures/`.

### What the analyses reveal

| Script | Question |
|--------|----------|
| `01_summary_stats` | How many pairs pass each threshold? What's the distribution? |
| `02_equivalence_map` | Which APCA threshold is functionally equivalent to WCAG 4.5? |
| `03_disagreement` | Which colors pass WCAG but fail APCA (and vice versa)? What characterizes them? |
| `04_polarity` | How asymmetric is APCA? Does the same color have different options as text vs background? |
| `05_hue_analysis` | Which hues are systematically advantaged/disadvantaged? How bad is blue? |
| `06_distribution` | Are the distributions normal, bimodal, power-law? What's the entropy? |
| `07_luminance_scatter` | How does luminance relate to accessibility options? How do APCA and WCAG curves differ? |
| `08_top_bottom` | What are the most and least accessible colors? What do they look like? |
| `09_named_colors` | For each CSS named color, exactly how much of the color space is accessible? |
| `10_rgb_cube_export` | 3D visualization data + RGB cube cross-sections + luminance-sorted heatmaps |

## Library

`src/lib.rs` contains Rust ports of six contrast algorithms from [Color.js](https://colorjs.io/):

- `contrast_wcag21` — WCAG 2.1 luminance ratio
- `contrast_michelson` — Michelson luminance contrast
- `contrast_weber` — Weber contrast
- `contrast_lstar` — CIE L\* difference
- `contrast_delta_phi` — Delta Phi Star perceptual contrast
- `contrast_apca_from_luminance` — APCA (Accessible Perceptual Contrast Algorithm)

All verified against Color.js output via test fixtures (`tests/contrast_fixtures.json`).

## Output format

### Binary files

Each `.bin` file: 64 MB, flat array of 16,777,216 little-endian `u32` values.

```python
import numpy as np
counts = np.fromfile("output/apca_75.bin", dtype=np.uint32)
# counts[0x000000] = accessible backgrounds for black text
# counts[0xFF0000] = accessible backgrounds for red text
# counts[0xFFFFFF] = accessible backgrounds for white text
```

### Histogram CSVs

1000-bin histograms of the count distributions. Columns: `bin_min,bin_max,count,percent`.

### metadata.json

Summary statistics, elapsed time, and configuration for the run.

## License

MIT
