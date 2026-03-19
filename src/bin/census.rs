//! Census: brute-force count of accessible color combinations for every 24-bit hex color.
//!
//! For each of 16,777,216 text colors, iterates ALL 16,777,216 backgrounds,
//! computes contrast, and increments the count if it meets the threshold.
//! That's 281,474,976,710,656 contrast evaluations per threshold.
//!
//! Optimizations:
//! - sRGB linearization lookup table (256 entries, no powf(2.4) per channel)
//! - Precomputed luminance + fclamp'd luminance + APCA powers, packed into a
//!   single contiguous array-of-structs for cache-friendly sequential access
//! - rayon into_par_iter across text colors (scales to 128 threads on TR 3990X)
//! - Resumable: saves progress.dat after each chunk of 1024 text colors
//!
//! Usage:
//!   census                          # normal: index by text color → output/
//!   census --swap                   # swap: index by bg color → output_bg/
//!   census --output-dir my_output   # custom output directory
//!
//! Outputs (9 .bin files, 9 histogram CSVs, metadata):
//!   {apca_60,apca_75,apca_90}.bin         - APCA |Lc| >= threshold
//!   {wcag_3_0,wcag_4_5,wcag_7_0}.bin      - WCAG ratio >= threshold
//!   {both_apca60_wcag3,both_apca75_wcag45,both_apca90_wcag7}.bin
//!                                          - cross-threshold (pass BOTH)
//!   *_histogram.csv                        - 1000-bin histogram per file
//!   metadata.json                          - summary stats + config

use rayon::prelude::*;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const N: usize = 16_777_216; // 256^3
const CHUNK: usize = 1024; // text colors per checkpoint
const NUM_COUNTERS: usize = 9;

// ---- sRGB linearization LUT ----

fn build_srgb_lut() -> [f64; 256] {
    let mut lut = [0.0f64; 256];
    for i in 0..256 {
        let v = i as f64 / 255.0;
        lut[i] = if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        };
    }
    lut
}

fn luminance(hex: u32, lut: &[f64; 256]) -> f64 {
    let r = ((hex >> 16) & 0xFF) as usize;
    let g = ((hex >> 8) & 0xFF) as usize;
    let b = (hex & 0xFF) as usize;
    0.2126 * lut[r] + 0.7152 * lut[g] + 0.0722 * lut[b]
}

// ---- APCA constants (matches lib.rs exactly) ----

const BLK_THRS: f64 = 0.022;
const BLK_CLMP: f64 = 1.414;
const NORM_BG: f64 = 0.56;
const NORM_TXT: f64 = 0.57;
const REV_TXT: f64 = 0.62;
const REV_BG: f64 = 0.65;
const SCALE_BOW: f64 = 1.14;
const SCALE_WOB: f64 = 1.14;
const LO_BOW_OFFSET: f64 = 0.027;
const LO_WOB_OFFSET: f64 = 0.027;
const LO_CLIP: f64 = 0.1;
const DELTA_Y_MIN: f64 = 0.0005;

#[inline(always)]
fn fclamp(y: f64) -> f64 {
    if y >= BLK_THRS {
        y
    } else {
        y + (BLK_THRS - y).powf(BLK_CLMP)
    }
}

/// WCAG 2.1 ratio from raw luminances
#[inline(always)]
fn wcag_ratio(y1: f64, y2: f64) -> f64 {
    let (hi, lo) = if y1 >= y2 { (y1, y2) } else { (y2, y1) };
    (hi + 0.05) / (lo + 0.05)
}

// ---- Packed per-color data for cache-friendly inner loop ----

#[derive(Clone, Copy)]
#[repr(C)]
struct ColorData {
    raw_lum: f64,  // relative luminance (for WCAG)
    fc_lum: f64,   // fclamp'd luminance (for APCA delta check)
    pow_a: f64,    // inner-variable power for BOW path
    pow_b: f64,    // inner-variable power for WOB path
}

// ---- Threshold names ----

const NAMES: [&str; NUM_COUNTERS] = [
    "apca_60", "apca_75", "apca_90",
    "wcag_3_0", "wcag_4_5", "wcag_7_0",
    "both_apca60_wcag3", "both_apca75_wcag45", "both_apca90_wcag7",
];

// ---- Progress / I/O ----

fn load_progress(path: &Path) -> usize {
    match fs::read(path) {
        Ok(bytes) if bytes.len() >= 8 => {
            usize::from_le_bytes(bytes[..8].try_into().unwrap())
        }
        _ => 0,
    }
}

fn save_progress(path: &Path, chunk_idx: usize) {
    fs::write(path, &chunk_idx.to_le_bytes()).unwrap();
}

fn load_bin_u32(path: &Path) -> Option<Vec<u32>> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() != N * 4 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

fn save_bin_u32(path: &Path, data: &[u32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).unwrap();
}

// ---- CLI ----

struct Config {
    swap: bool,
    output_dir: PathBuf,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let swap = args.iter().any(|a| a == "--swap");
    let output_dir = args
        .iter()
        .position(|a| a == "--output-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if swap {
                PathBuf::from("output_bg")
            } else {
                PathBuf::from("output")
            }
        });
    Config { swap, output_dir }
}

// ---- Main ----

fn main() {
    let cfg = parse_args();
    let out = &cfg.output_dir;
    fs::create_dir_all(out).unwrap();

    let mode_label = if cfg.swap {
        "SWAP (indexing by BACKGROUND color)"
    } else {
        "NORMAL (indexing by TEXT color)"
    };
    println!("Mode: {}", mode_label);
    println!("Output: {}/", out.display());

    let start = Instant::now();

    // 1. Build sRGB lookup table
    println!("Building sRGB lookup table...");
    let lut = build_srgb_lut();

    // 2. Precompute all luminances
    println!("Computing luminances for all {} colors...", N);
    let raw_lum: Vec<f64> = (0..N as u32)
        .into_par_iter()
        .map(|h| luminance(h, &lut))
        .collect();
    let fc_lum: Vec<f64> = raw_lum.par_iter().map(|&y| fclamp(y)).collect();

    // 3. Build packed ColorData with mode-dependent exponents.
    //    In normal mode (outer=txt, inner=bg):
    //      inner needs bg exponents: ^0.56 (BOW), ^0.65 (WOB)
    //    In swap mode (outer=bg, inner=txt):
    //      inner needs txt exponents: ^0.57 (BOW), ^0.62 (WOB)
    let (inner_exp_a, inner_exp_b) = if cfg.swap {
        (NORM_TXT, REV_TXT) // 0.57, 0.62
    } else {
        (NORM_BG, REV_BG) // 0.56, 0.65
    };

    println!(
        "Building packed color data (inner exponents: {:.2}, {:.2})...",
        inner_exp_a, inner_exp_b,
    );
    let colors: Vec<ColorData> = (0..N)
        .into_par_iter()
        .map(|i| ColorData {
            raw_lum: raw_lum[i],
            fc_lum: fc_lum[i],
            pow_a: fc_lum[i].powf(inner_exp_a),
            pow_b: fc_lum[i].powf(inner_exp_b),
        })
        .collect();

    // Outer-variable power arrays (opposite exponents from inner).
    let (outer_exp_a, outer_exp_b) = if cfg.swap {
        (NORM_BG, REV_BG) // 0.56, 0.65
    } else {
        (NORM_TXT, REV_TXT) // 0.57, 0.62
    };
    let outer_pow_a: Vec<f64> = fc_lum.par_iter().map(|&y| y.powf(outer_exp_a)).collect();
    let outer_pow_b: Vec<f64> = fc_lum.par_iter().map(|&y| y.powf(outer_exp_b)).collect();

    println!(
        "Precomputation done in {:.2}s  ({} bytes/color, {:.0}MB packed array)",
        start.elapsed().as_secs_f64(),
        std::mem::size_of::<ColorData>(),
        (N * std::mem::size_of::<ColorData>()) as f64 / 1_048_576.0,
    );

    // 4. Load progress
    let progress_path = out.join("progress.dat");
    let start_chunk = load_progress(&progress_path);
    let total_chunks = (N + CHUNK - 1) / CHUNK;

    // 5. Load or initialize count arrays
    let mut counts: Vec<Vec<u32>> = NAMES
        .iter()
        .map(|name| {
            if start_chunk > 0 {
                let path = out.join(format!("{}.bin", name));
                if let Some(data) = load_bin_u32(&path) {
                    return data;
                }
            }
            vec![0u32; N]
        })
        .collect();

    if start_chunk > 0 {
        println!(
            "Resuming from chunk {} / {} ({} colors already done)",
            start_chunk, total_chunks, start_chunk * CHUNK,
        );
    }

    let colors_remaining = N - (start_chunk * CHUNK).min(N);
    println!(
        "Brute-forcing {} outer colors × {} inner colors × 6 thresholds + 3 cross",
        colors_remaining, N,
    );

    // In swap mode the subtraction order in the APCA formula reverses.
    // Normal: inner=bg, outer=txt → BOW when inner.fc > outer.fc
    //   c = (inner.pow_a - outer_pow_a) * SCALE
    // Swap: inner=txt, outer=bg → BOW when outer.fc > inner.fc
    //   c = (outer_pow_a - inner.pow_a) * SCALE
    let swap = cfg.swap;

    // 6. Brute-force
    for ci in start_chunk..total_chunks {
        let chunk_start = ci * CHUNK;
        let chunk_end = (chunk_start + CHUNK).min(N);
        let chunk_time = Instant::now();

        let chunk_results: Vec<[u32; NUM_COUNTERS]> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|outer_hex| {
                let outer = &colors[outer_hex];
                let o_fc = outer.fc_lum;
                let o_raw = outer.raw_lum;
                let o_pa = outer_pow_a[outer_hex];
                let o_pb = outer_pow_b[outer_hex];

                let mut apca_60: u32 = 0;
                let mut apca_75: u32 = 0;
                let mut apca_90: u32 = 0;
                let mut wcag_3: u32 = 0;
                let mut wcag_45: u32 = 0;
                let mut wcag_7: u32 = 0;
                let mut both_60_3: u32 = 0;
                let mut both_75_45: u32 = 0;
                let mut both_90_7: u32 = 0;

                for inner_hex in 0..N {
                    let inner = &colors[inner_hex];

                    // APCA contrast.
                    // Determine bg_fc/txt_fc and the correct power subtraction
                    // based on mode. The branch on `swap` is 100% predicted.
                    let (bg_fc, txt_fc, bow_diff, wob_diff) = if swap {
                        // outer=bg, inner=txt
                        (o_fc, inner.fc_lum,
                         o_pa - inner.pow_a,  // bg^0.56 - txt^0.57
                         o_pb - inner.pow_b)  // bg^0.65 - txt^0.62
                    } else {
                        // outer=txt, inner=bg
                        (inner.fc_lum, o_fc,
                         inner.pow_a - o_pa,  // bg^0.56 - txt^0.57
                         inner.pow_b - o_pb)  // bg^0.65 - txt^0.62
                    };

                    let c = if (bg_fc - txt_fc).abs() < DELTA_Y_MIN {
                        0.0
                    } else if bg_fc > txt_fc {
                        bow_diff * SCALE_BOW
                    } else {
                        wob_diff * SCALE_WOB
                    };

                    let sapc = if c.abs() < LO_CLIP {
                        0.0
                    } else if c > 0.0 {
                        c - LO_BOW_OFFSET
                    } else {
                        c + LO_WOB_OFFSET
                    };
                    let lc = (sapc * 100.0).abs();

                    // WCAG (symmetric, no swap needed)
                    let ratio = wcag_ratio(o_raw, inner.raw_lum);

                    // Threshold checks
                    let a60 = lc >= 60.0;
                    let a75 = lc >= 75.0;
                    let a90 = lc >= 90.0;
                    let w3 = ratio >= 3.0;
                    let w45 = ratio >= 4.5;
                    let w7 = ratio >= 7.0;

                    // Individual counters (nested for branch-prediction friendliness)
                    if a60 {
                        apca_60 += 1;
                        if a75 {
                            apca_75 += 1;
                            if a90 { apca_90 += 1; }
                        }
                    }
                    if w3 {
                        wcag_3 += 1;
                        if w45 {
                            wcag_45 += 1;
                            if w7 { wcag_7 += 1; }
                        }
                    }

                    // Cross-threshold counters
                    if a60 && w3 { both_60_3 += 1; }
                    if a75 && w45 { both_75_45 += 1; }
                    if a90 && w7 { both_90_7 += 1; }
                }

                [apca_60, apca_75, apca_90, wcag_3, wcag_45, wcag_7,
                 both_60_3, both_75_45, both_90_7]
            })
            .collect();

        // Write results
        for (i, r) in chunk_results.into_iter().enumerate() {
            let hex = chunk_start + i;
            for t in 0..NUM_COUNTERS {
                counts[t][hex] = r[t];
            }
        }

        // Checkpoint
        for (t, name) in NAMES.iter().enumerate() {
            save_bin_u32(&out.join(format!("{}.bin", name)), &counts[t]);
        }
        save_progress(&progress_path, ci + 1);

        // Progress
        let elapsed_total = start.elapsed().as_secs_f64();
        let chunks_done = (ci + 1 - start_chunk) as f64;
        let chunks_left = (total_chunks - ci - 1) as f64;
        let secs_per_chunk = elapsed_total / chunks_done;
        let eta = secs_per_chunk * chunks_left;
        let colors_done = (ci + 1) * CHUNK;
        let pct = colors_done as f64 / N as f64 * 100.0;

        println!(
            "[{:>8.1}s] chunk {:>5}/{} | {:>8} colors ({:>5.1}%) | chunk: {:.1}s | ETA: {:.0}s ({:.1}h)",
            elapsed_total,
            ci + 1,
            total_chunks,
            colors_done.min(N),
            pct,
            chunk_time.elapsed().as_secs_f64(),
            eta,
            eta / 3600.0,
        );
    }

    // 7. Summary
    println!("\n=== RESULTS ({}) ===", mode_label);
    let mut stats_json = Vec::new();
    for (t, name) in NAMES.iter().enumerate() {
        let total: u64 = counts[t].iter().map(|&c| c as u64).sum();
        let max = *counts[t].iter().max().unwrap();
        let min = *counts[t].iter().min().unwrap();
        let mean = total as f64 / N as f64;
        let total_possible = N as u64 * N as u64;
        let pct = total as f64 / total_possible as f64 * 100.0;

        println!(
            "{:>24}: {:>15} pairs ({:>5.2}%)  min={:<8} max={:<8} mean={:.1}",
            name, total, pct, min, max, mean,
        );

        stats_json.push(serde_json::json!({
            "name": name,
            "file": format!("{}.bin", name),
            "format": "little-endian u32[16777216], indexed by hex 0x000000..0xFFFFFF",
            "total_passing_pairs": total,
            "total_possible_pairs": total_possible,
            "percent_passing": pct,
            "min_count": min,
            "max_count": max,
            "mean_count": mean,
        }));
    }

    // 8. Metadata
    let meta = serde_json::json!({
        "description": "Brute-force color contrast census. Every single combination tested.",
        "mode": if cfg.swap { "swap (indexed by background)" } else { "normal (indexed by text)" },
        "num_colors": N,
        "num_possible_pairs": N as u64 * N as u64,
        "chunk_size": CHUNK,
        "elapsed_seconds": start.elapsed().as_secs_f64(),
        "counters": stats_json,
    });
    fs::write(
        out.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();

    // 9. Histogram CSVs
    println!("\nGenerating histograms...");
    for (t, name) in NAMES.iter().enumerate() {
        write_histogram_csv(
            &out.join(format!("{}_histogram.csv", name)),
            &counts[t],
            name,
        );
    }

    // Clean up progress
    let _ = fs::remove_file(&progress_path);

    println!(
        "\nCompleted in {:.1}s ({:.1}h). Output: {}/",
        start.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64() / 3600.0,
        out.display(),
    );
}

// ---- Histogram CSV ----

fn write_histogram_csv(path: &Path, counts: &[u32], name: &str) {
    let max_count = *counts.iter().max().unwrap_or(&0) as usize;

    let num_bins: usize = 1000;
    let bin_width = if max_count == 0 {
        1.0
    } else {
        (max_count as f64 + 1.0) / num_bins as f64
    };

    let mut bins = vec![0u64; num_bins];
    for &c in counts {
        let bin = ((c as f64) / bin_width).min((num_bins - 1) as f64) as usize;
        bins[bin] += 1;
    }

    let mut csv = String::with_capacity(num_bins * 40);
    writeln!(csv, "bin_min,bin_max,count,percent").unwrap();
    for (i, &freq) in bins.iter().enumerate() {
        let bin_min = (i as f64 * bin_width) as u64;
        let bin_max = (((i + 1) as f64 * bin_width) - 1.0).max(bin_min as f64) as u64;
        let pct = freq as f64 / N as f64 * 100.0;
        writeln!(csv, "{},{},{},{:.6}", bin_min, bin_max, freq, pct).unwrap();
    }

    fs::write(path, csv).unwrap();
    println!(
        "  {} -> {} ({} bins, width={:.1})",
        name,
        path.display(),
        num_bins,
        bin_width,
    );
}
