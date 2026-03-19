//! Census: brute-force count of accessible color combinations for every 24-bit hex color.
//!
//! For each of 16,777,216 text colors, iterates ALL 16,777,216 backgrounds,
//! computes contrast, and increments the count if it meets the threshold.
//! That's 281,474,976,710,656 contrast evaluations per threshold.
//!
//! Optimizations:
//! - sRGB linearization lookup table (256 entries, no powf(2.4) per channel)
//! - Precomputed luminance + fclamp'd luminance arrays for all 16M colors
//! - rayon into_par_iter across text colors (scales to 128 threads on TR 3990X)
//! - Resumable: saves progress.dat after each chunk of 1024 text colors
//! - Atomic u64 aggregation across threads within each chunk
//!
//! Outputs:
//!   output/apca_60.bin   - u32[16777216] LE, count per text hex at |Lc| >= 60
//!   output/apca_75.bin   - u32[16777216] LE, count per text hex at |Lc| >= 75
//!   output/apca_90.bin   - u32[16777216] LE, count per text hex at |Lc| >= 90
//!   output/wcag_3_0.bin  - u32[16777216] LE, count per text hex at ratio >= 3.0
//!   output/wcag_4_5.bin  - u32[16777216] LE, count per text hex at ratio >= 4.5
//!   output/wcag_7_0.bin  - u32[16777216] LE, count per text hex at ratio >= 7.0
//!   output/apca_60_histogram.csv  - histogram (1000 bins) of count distribution
//!   output/apca_75_histogram.csv  - "
//!   output/apca_90_histogram.csv  - "
//!   output/wcag_3_0_histogram.csv - "
//!   output/wcag_4_5_histogram.csv - "
//!   output/wcag_7_0_histogram.csv - "
//!   output/metadata.json - summary stats
//!   output/progress.dat  - resume checkpoint (deleted on completion)

use rayon::prelude::*;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const N: usize = 16_777_216; // 256^3
const CHUNK: usize = 1024; // text colors per checkpoint

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

// ---- APCA (inlined for speed, matches lib.rs exactly) ----

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

/// WCAG 2.1 ratio from raw luminances (same as lib.rs contrast_wcag21)
#[inline(always)]
fn wcag_ratio(y1: f64, y2: f64) -> f64 {
    let (hi, lo) = if y1 >= y2 { (y1, y2) } else { (y2, y1) };
    (hi + 0.05) / (lo + 0.05)
}

// ---- Optimized inner loop: precomputed fclamp values ----
// By precomputing fclamp for all 16M colors, we avoid calling fclamp
// inside the hot 281T iteration loop. The APCA powf calls on the
// fclamp'd values are the real bottleneck.

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

// ---- Threshold definitions ----

struct Threshold {
    name: &'static str,
    is_apca: bool,
    value: f64,
}

const THRESHOLDS: [Threshold; 6] = [
    Threshold { name: "apca_60",  is_apca: true,  value: 60.0 },
    Threshold { name: "apca_75",  is_apca: true,  value: 75.0 },
    Threshold { name: "apca_90",  is_apca: true,  value: 90.0 },
    Threshold { name: "wcag_3_0", is_apca: false, value: 3.0 },
    Threshold { name: "wcag_4_5", is_apca: false, value: 4.5 },
    Threshold { name: "wcag_7_0", is_apca: false, value: 7.0 },
];

// ---- Main ----

fn main() {
    let out = PathBuf::from("output");
    fs::create_dir_all(&out).unwrap();

    let start = Instant::now();

    // 1. Build sRGB lookup table (256 entries, replaces per-channel powf(2.4))
    println!("Building sRGB lookup table...");
    let lut = build_srgb_lut();

    // 2. Precompute raw luminance for all 16M colors
    println!("Computing luminances for all {} colors...", N);
    let raw_lum: Vec<f64> = (0..N as u32)
        .into_par_iter()
        .map(|h| luminance(h, &lut))
        .collect();

    // 3. Precompute fclamp'd luminance for APCA (avoids fclamp in inner loop)
    let fc_lum: Vec<f64> = raw_lum.par_iter().map(|&y| fclamp(y)).collect();

    // 4. Precompute APCA power arrays — eliminates all powf from the inner loop.
    //    Background needs ^NORM_BG (0.56) for BOW and ^REV_BG (0.65) for WOB.
    //    Text needs ^NORM_TXT (0.57) for BOW and ^REV_TXT (0.62) for WOB.
    //    Text powers are only used once per outer iteration, but precomputing
    //    them too keeps the hot loop completely powf-free.
    println!("Precomputing APCA power tables (4 × 16M)...");
    let pow_norm_bg: Vec<f64> = fc_lum.par_iter().map(|&y| y.powf(NORM_BG)).collect();
    let pow_rev_bg: Vec<f64>  = fc_lum.par_iter().map(|&y| y.powf(REV_BG)).collect();
    let pow_norm_txt: Vec<f64> = fc_lum.par_iter().map(|&y| y.powf(NORM_TXT)).collect();
    let pow_rev_txt: Vec<f64>  = fc_lum.par_iter().map(|&y| y.powf(REV_TXT)).collect();

    println!(
        "Precomputation done in {:.2}s  (lum range: {:.8}..{:.8})",
        start.elapsed().as_secs_f64(),
        raw_lum.iter().cloned().fold(f64::INFINITY, f64::min),
        raw_lum.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // 4. Load progress
    let progress_path = out.join("progress.dat");
    let start_chunk = load_progress(&progress_path);
    let total_chunks = (N + CHUNK - 1) / CHUNK; // 16384

    // 5. Load or initialize count arrays (one per threshold)
    let mut counts: Vec<Vec<u32>> = THRESHOLDS
        .iter()
        .map(|th| {
            if start_chunk > 0 {
                let path = out.join(format!("{}.bin", th.name));
                if let Some(data) = load_bin_u32(&path) {
                    return data;
                }
            }
            vec![0u32; N]
        })
        .collect();

    if start_chunk > 0 {
        println!(
            "Resuming from chunk {} / {} ({} text colors already done)",
            start_chunk,
            total_chunks,
            start_chunk * CHUNK,
        );
    }

    let colors_remaining = N - (start_chunk * CHUNK).min(N);
    println!(
        "Brute-forcing {} text colors × {} backgrounds × 6 thresholds",
        colors_remaining, N,
    );
    println!(
        "Total contrast evaluations remaining: {}",
        colors_remaining as u128 * N as u128 * 6,
    );

    // 6. Brute-force: for each text color, iterate ALL backgrounds
    for ci in start_chunk..total_chunks {
        let chunk_start = ci * CHUNK;
        let chunk_end = (chunk_start + CHUNK).min(N);
        let chunk_time = Instant::now();

        // Each text color in this chunk processed in parallel via rayon.
        // For each text color, we iterate all 16M backgrounds sequentially
        // (the parallelism is across text colors in the chunk).
        let chunk_results: Vec<[u32; 6]> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|txt_hex| {
                let y_txt = raw_lum[txt_hex];
                let y_txt_fc = fc_lum[txt_hex];
                // Text-side powers (looked up from precomputed arrays)
                let txt_pnorm = pow_norm_txt[txt_hex]; // y_txt^0.57
                let txt_prev  = pow_rev_txt[txt_hex];  // y_txt^0.62

                let mut apca_60: u32 = 0;
                let mut apca_75: u32 = 0;
                let mut apca_90: u32 = 0;
                let mut wcag_3: u32 = 0;
                let mut wcag_45: u32 = 0;
                let mut wcag_7: u32 = 0;

                for bg_hex in 0..N {
                    // APCA: all lookups, no powf
                    let y_bg_fc = fc_lum[bg_hex];
                    let c = if (y_bg_fc - y_txt_fc).abs() < DELTA_Y_MIN {
                        0.0
                    } else if y_bg_fc > y_txt_fc {
                        // BOW: bg^0.56 - txt^0.57
                        (pow_norm_bg[bg_hex] - txt_pnorm) * SCALE_BOW
                    } else {
                        // WOB: bg^0.65 - txt^0.62
                        (pow_rev_bg[bg_hex] - txt_prev) * SCALE_WOB
                    };
                    let sapc = if c.abs() < LO_CLIP {
                        0.0
                    } else if c > 0.0 {
                        c - LO_BOW_OFFSET
                    } else {
                        c + LO_WOB_OFFSET
                    };
                    let lc = (sapc * 100.0).abs();
                    if lc >= 60.0 {
                        apca_60 += 1;
                        if lc >= 75.0 {
                            apca_75 += 1;
                            if lc >= 90.0 {
                                apca_90 += 1;
                            }
                        }
                    }

                    // WCAG: use raw luminances
                    let ratio = wcag_ratio(y_txt, raw_lum[bg_hex]);
                    if ratio >= 3.0 {
                        wcag_3 += 1;
                        if ratio >= 4.5 {
                            wcag_45 += 1;
                            if ratio >= 7.0 {
                                wcag_7 += 1;
                            }
                        }
                    }
                }

                [apca_60, apca_75, apca_90, wcag_3, wcag_45, wcag_7]
            })
            .collect();

        // Write results into count arrays
        for (i, r) in chunk_results.into_iter().enumerate() {
            let hex = chunk_start + i;
            counts[0][hex] = r[0];
            counts[1][hex] = r[1];
            counts[2][hex] = r[2];
            counts[3][hex] = r[3];
            counts[4][hex] = r[4];
            counts[5][hex] = r[5];
        }

        // Save checkpoint after every chunk
        for (t, th) in THRESHOLDS.iter().enumerate() {
            save_bin_u32(&out.join(format!("{}.bin", th.name)), &counts[t]);
        }
        save_progress(&progress_path, ci + 1);

        // Progress reporting
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
    println!("\n=== RESULTS ===");
    let mut stats_json = Vec::new();
    for (t, th) in THRESHOLDS.iter().enumerate() {
        let total: u64 = counts[t].iter().map(|&c| c as u64).sum();
        let max = *counts[t].iter().max().unwrap();
        let min = *counts[t].iter().min().unwrap();
        let mean = total as f64 / N as f64;
        let total_possible = N as u64 * N as u64;
        let pct = total as f64 / total_possible as f64 * 100.0;

        println!(
            "{:>8}: {:>15} passing pairs ({:>5.2}%)  min={:<8} max={:<8} mean={:.1}",
            th.name, total, pct, min, max, mean,
        );

        stats_json.push(serde_json::json!({
            "name": th.name,
            "algorithm": if th.is_apca { "APCA" } else { "WCAG 2.1" },
            "threshold": th.value,
            "file": format!("{}.bin", th.name),
            "format": "little-endian u32[16777216], indexed by hex 0x000000..0xFFFFFF",
            "total_passing_pairs": total,
            "total_possible_pairs": total_possible,
            "percent_passing": pct,
            "min_count": min,
            "max_count": max,
            "mean_count": mean,
        }));
    }

    // 8. Save metadata
    let meta = serde_json::json!({
        "description": "Brute-force color contrast census: for each 24-bit sRGB hex (as text), count of ALL 16,777,216 background colors meeting the contrast threshold. Every single combination tested.",
        "num_colors": N,
        "num_possible_pairs": N as u64 * N as u64,
        "chunk_size": CHUNK,
        "elapsed_seconds": start.elapsed().as_secs_f64(),
        "thresholds": stats_json,
    });
    fs::write(
        out.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();

    // 9. Generate histogram CSVs
    println!("\nGenerating histograms...");
    for (t, th) in THRESHOLDS.iter().enumerate() {
        write_histogram_csv(&out.join(format!("{}_histogram.csv", th.name)), &counts[t], th.name);
    }

    // Clean up progress on completion
    let _ = fs::remove_file(&progress_path);

    println!(
        "\nCompleted in {:.1}s ({:.1}h). Output: {}/",
        start.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64() / 3600.0,
        out.display(),
    );
}

// ---- Histogram CSV generation ----

fn write_histogram_csv(path: &Path, counts: &[u32], name: &str) {
    // Collect frequency: how many hex colors have exactly `count` accessible pairs
    let max_count = *counts.iter().max().unwrap_or(&0) as usize;

    // Use 1000 bins spanning 0..=max_count for a manageable CSV
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
    println!("  {} -> {} ({} bins, bin_width={:.1})", name, path.display(), num_bins, bin_width);
}
