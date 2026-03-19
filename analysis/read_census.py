"""Shared utilities for reading census binary output files."""

import json
import colorsys
import numpy as np
from pathlib import Path

N = 16_777_216  # 256^3

# ---- Binary file I/O ----

def load_bin(path: str | Path) -> np.ndarray:
    """Load a .bin file as a uint32 numpy array of length 16,777,216."""
    data = np.fromfile(path, dtype=np.uint32)
    assert data.shape == (N,), f"Expected {N} entries, got {data.shape}"
    return data


def load_all(output_dir: str | Path = "output") -> dict[str, np.ndarray]:
    """Load all .bin files from an output directory into a dict."""
    d = Path(output_dir)
    result = {}
    for p in sorted(d.glob("*.bin")):
        result[p.stem] = load_bin(p)
    return result


def load_metadata(output_dir: str | Path = "output") -> dict:
    """Load metadata.json from an output directory."""
    return json.loads((Path(output_dir) / "metadata.json").read_text())


# ---- Color helpers ----

def hex_to_rgb(hexvals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert hex integers to (R, G, B) arrays, each 0-255."""
    r = (hexvals >> 16) & 0xFF
    g = (hexvals >> 8) & 0xFF
    b = hexvals & 0xFF
    return r, g, b


def hex_to_rgb_float(hexvals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert hex integers to (R, G, B) arrays, each 0.0-1.0."""
    r, g, b = hex_to_rgb(hexvals)
    return r / 255.0, g / 255.0, b / 255.0


def hex_to_hsl(hexvals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert hex integers to (H, S, L) arrays. H in [0,360), S,L in [0,1]."""
    r, g, b = hex_to_rgb_float(hexvals)
    n = len(hexvals)
    h = np.empty(n, dtype=np.float32)
    s = np.empty(n, dtype=np.float32)
    l = np.empty(n, dtype=np.float32)
    for i in range(n):
        hi, li, si = colorsys.rgb_to_hls(r[i], g[i], b[i])
        h[i] = hi * 360.0
        s[i] = si
        l[i] = li
    return h, s, l


def hex_to_hsl_fast(hexvals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized HSL conversion (no Python loop). H in [0,360), S,L in [0,1]."""
    r, g, b = hex_to_rgb_float(hexvals)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    l = (cmax + cmin) / 2.0

    s = np.where(
        delta == 0, 0.0,
        delta / (1.0 - np.abs(2.0 * l - 1.0)).clip(min=1e-10)
    )

    h = np.zeros_like(r)
    mask_r = (delta > 0) & (cmax == r)
    mask_g = (delta > 0) & (cmax == g) & ~mask_r
    mask_b = (delta > 0) & ~mask_r & ~mask_g

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)
    h = h % 360.0

    return h.astype(np.float32), s.astype(np.float32), l.astype(np.float32)


def srgb_luminance(hexvals: np.ndarray) -> np.ndarray:
    """Compute relative luminance for hex color values (matching the Rust code)."""
    r, g, b = hex_to_rgb_float(hexvals)

    def linearize(v):
        return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def hex_to_css(h: int) -> str:
    """Convert integer hex to CSS color string, e.g. 0xFF0000 -> '#ff0000'."""
    return f"#{h:06x}"


# ---- All hex values ----

ALL_HEX = np.arange(N, dtype=np.uint32)


# ---- Plotting style ----

def setup_style():
    """Apply consistent plot styling."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "font.family": "monospace",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "text.color": "#c9d1d9",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
    })


# ---- Named CSS colors ----

CSS_NAMED_COLORS = {
    "black": 0x000000, "white": 0xFFFFFF, "red": 0xFF0000, "lime": 0x00FF00,
    "blue": 0x0000FF, "yellow": 0xFFFF00, "cyan": 0x00FFFF, "magenta": 0xFF00FF,
    "silver": 0xC0C0C0, "gray": 0x808080, "maroon": 0x800000, "olive": 0x808000,
    "green": 0x008000, "purple": 0x800080, "teal": 0x008080, "navy": 0x000080,
    "orange": 0xFFA500, "orangered": 0xFF4500, "tomato": 0xFF6347,
    "coral": 0xFF7F50, "salmon": 0xFA8072, "lightsalmon": 0xFFA07A,
    "crimson": 0xDC143C, "firebrick": 0xB22222, "darkred": 0x8B0000,
    "pink": 0xFFC0CB, "hotpink": 0xFF69B4, "deeppink": 0xFF1493,
    "gold": 0xFFD700, "khaki": 0xF0E68C, "darkkhaki": 0xBDB76B,
    "plum": 0xDDA0DD, "violet": 0xEE82EE, "orchid": 0xDA70D6,
    "mediumpurple": 0x9370DB, "blueviolet": 0x8A2BE2, "darkviolet": 0x9400D3,
    "indigo": 0x4B0082, "slateblue": 0x6A5ACD, "darkslateblue": 0x483D8B,
    "greenyellow": 0xADFF2F, "chartreuse": 0x7FFF00, "lawngreen": 0x7CFC00,
    "limegreen": 0x32CD32, "springgreen": 0x00FF7F, "mediumseagreen": 0x3CB371,
    "seagreen": 0x2E8B57, "forestgreen": 0x228B22, "darkgreen": 0x006400,
    "darkolivegreen": 0x556B2F, "olivedrab": 0x6B8E23, "yellowgreen": 0x9ACD32,
    "aquamarine": 0x7FFFD4, "turquoise": 0x40E0D0, "mediumturquoise": 0x48D1CC,
    "darkturquoise": 0x00CED1, "cadetblue": 0x5F9EA0, "steelblue": 0x4682B4,
    "lightsteelblue": 0xB0C4DE, "powderblue": 0xB0E0E6, "lightblue": 0xADD8E6,
    "skyblue": 0x87CEEB, "deepskyblue": 0x00BFFF, "dodgerblue": 0x1E90FF,
    "cornflowerblue": 0x6495ED, "royalblue": 0x4169E1, "mediumblue": 0x0000CD,
    "darkblue": 0x00008B, "midnightblue": 0x191970,
    "cornsilk": 0xFFF8DC, "blanchedalmond": 0xFFEBCD, "bisque": 0xFFE4C4,
    "navajowhite": 0xFFDEAD, "wheat": 0xF5DEB3, "burlywood": 0xDEB887,
    "tan": 0xD2B48C, "rosybrown": 0xBC8F8F, "sandybrown": 0xF4A460,
    "goldenrod": 0xDAA520, "darkgoldenrod": 0xB8860B, "peru": 0xCD853F,
    "chocolate": 0xD2691E, "saddlebrown": 0x8B4513, "sienna": 0xA0522D,
    "brown": 0xA52A2A, "snow": 0xFFFAFA, "honeydew": 0xF0FFF0,
    "mintcream": 0xF5FFFA, "azure": 0xF0FFFF, "aliceblue": 0xF0F8FF,
    "ghostwhite": 0xF8F8FF, "whitesmoke": 0xF5F5F5, "seashell": 0xFFF5EE,
    "beige": 0xF5F5DC, "oldlace": 0xFDF5E6, "floralwhite": 0xFFFAF0,
    "ivory": 0xFFFFF0, "antiquewhite": 0xFAEBD7, "linen": 0xFAF0E6,
    "lavenderblush": 0xFFF0F5, "mistyrose": 0xFFE4E1, "lavender": 0xE6E6FA,
    "gainsboro": 0xDCDCDC, "lightgray": 0xD3D3D3, "darkgray": 0xA9A9A9,
    "dimgray": 0x696969, "lightslategray": 0x778899, "slategray": 0x708090,
    "darkslategray": 0x2F4F4F,
}
