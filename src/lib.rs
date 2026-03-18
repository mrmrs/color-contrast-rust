/// WCAG 2.1 contrast ratio.
///
/// Takes two relative luminances \(Y_1, Y_2\) (non‑negative, where 1.0 is white)
/// and returns the contrast ratio, with the same formula as Color.js:
/// \((Y_\text{lighter} + 0.05) / (Y_\text{darker} + 0.05)\).
pub fn contrast_wcag21(y1: f64, y2: f64) -> f64 {
    let mut y1 = y1.max(0.0);
    let mut y2 = y2.max(0.0);
    if y2 > y1 {
        std::mem::swap(&mut y1, &mut y2);
    }
    (y1 + 0.05) / (y2 + 0.05)
}

/// Michelson luminance contrast.
///
/// Relation between the spread and the sum of the two luminances:
/// \((Y_\text{max} - Y_\text{min}) / (Y_\text{max} + Y_\text{min})\).
pub fn contrast_michelson(y1: f64, y2: f64) -> f64 {
    let mut y1 = y1.max(0.0);
    let mut y2 = y2.max(0.0);
    if y2 > y1 {
        std::mem::swap(&mut y1, &mut y2);
    }
    let denom = y1 + y2;
    if denom == 0.0 {
        0.0
    }
    else {
        (y1 - y2) / denom
    }
}

/// Weber luminance contrast.
///
/// Difference between the two luminances divided by the lower luminance:
/// \((Y_\text{max} - Y_\text{min}) / Y_\text{min}\).
/// Division by zero is clamped to the same maximum as Color.js (50000).
pub fn contrast_weber(y1: f64, y2: f64) -> f64 {
    const MAX: f64 = 50000.0;

    let mut y1 = y1.max(0.0);
    let mut y2 = y2.max(0.0);
    if y2 > y1 {
        std::mem::swap(&mut y1, &mut y2);
    }

    if y2 == 0.0 {
        MAX
    }
    else {
        (y1 - y2) / y2
    }
}

/// CIE L\* lightness difference (Lstar contrast).
///
/// Takes two CIE L\* values \(L_1, L_2\) in the same space used by Color.js
/// and returns \(|L_1 - L_2|\).
pub fn contrast_lstar(l1: f64, l2: f64) -> f64 {
    (l1 - l2).abs()
}

/// Delta Phi Star perceptual lightness contrast.
///
/// Same math as Color.js `contrastDeltaPhi`, assuming the inputs are
/// CIE L\* values from the D65‑adapted Lab space used there.
pub fn contrast_delta_phi(l1: f64, l2: f64) -> f64 {
    // phi = (sqrt(5) / 2) + 0.5
    let phi = 5.0_f64.sqrt() * 0.5 + 0.5;

    let delta_phi_star = (l1.powf(phi) - l2.powf(phi)).abs();
    let contrast = delta_phi_star.powf(1.0 / phi) * std::f64::consts::SQRT_2 - 40.0;

    if contrast < 7.5 {
        0.0
    }
    else {
        contrast
    }
}

// ---- APCA helpers ----

// Exponents
const NORM_BG: f64 = 0.56;
const NORM_TXT: f64 = 0.57;
const REV_TXT: f64 = 0.62;
const REV_BG: f64 = 0.65;

// Clamps
const BLK_THRS: f64 = 0.022;
const BLK_CLMP: f64 = 1.414;
const LO_CLIP: f64 = 0.1;
const DELTA_Y_MIN: f64 = 0.0005;

// Scalers
const SCALE_BOW: f64 = 1.14;
const LO_BOW_OFFSET: f64 = 0.027;
const SCALE_WOB: f64 = 1.14;
const LO_WOB_OFFSET: f64 = 0.027;

fn fclamp(y: f64) -> f64 {
    if y >= BLK_THRS {
        y
    }
    else {
        y + (BLK_THRS - y).powf(BLK_CLMP)
    }
}

/// APCA contrast given precomputed, flare‑adjusted luminances.
///
/// This mirrors the tail of Color.js `contrastAPCA`, assuming that `y_bg`
/// and `y_txt` are the flare‑adjusted "screen luminances" after its
/// `linearize` and `fclamp` steps.
pub fn contrast_apca_from_luminance(y_bg: f64, y_txt: f64) -> f64 {
    let y_txt = fclamp(y_txt);
    let y_bg = fclamp(y_bg);

    let bow = y_bg > y_txt;
    let c = if (y_bg - y_txt).abs() < DELTA_Y_MIN {
        0.0
    }
    else if bow {
        // dark text on light background
        let s = y_bg.powf(NORM_BG) - y_txt.powf(NORM_TXT);
        s * SCALE_BOW
    }
    else {
        // light text on dark background
        let s = y_bg.powf(REV_BG) - y_txt.powf(REV_TXT);
        s * SCALE_WOB
    };

    let sapc = if c.abs() < LO_CLIP {
        0.0
    }
    else if c > 0.0 {
        // Woffset is `loBoWoffset`/`loWoBoffset` (same value in Color.js)
        c - LO_BOW_OFFSET
    }
    else {
        c + LO_WOB_OFFSET
    };

    sapc * 100.0
}

