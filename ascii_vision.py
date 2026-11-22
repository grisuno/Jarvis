"""Turn a Kinect depth frame into a compact ASCII picture a text-only LLM can read.

Jarvis only sends real JPEG frames when the LLM server advertises vision support
(see jarvis.LLM._detect_vision). A text-only GGUF (e.g. Qwen3) reports no vision,
so the assistant is blind except for the numeric depth telemetry. This module
renders the depth map as a small block of characters that can be dropped straight
into the text prompt, so the model "sees" the silhouette, posture and position of
whatever is in front of the sensor. Distances in metres still come from the
telemetry JSON, not from this drawing.

Convention (same as ascii_depth.py): dense glyphs = near, light glyphs = far,
space = background or no depth reading. The left column of the drawing is the
sensor's left, matching the "izquierda/derecha" used in the telemetry.
"""
import numpy as np

RAMP = " .:-=+*#%@"           # index 0 = space (background); last = nearest
NEAR_MM, FAR_MM = 400, 3000   # depth window the drawing spans
DEFAULT_COLS, DEFAULT_ROWS = 52, 20


def render_depth_ascii(depth, cols=DEFAULT_COLS, rows=DEFAULT_ROWS,
                       near=NEAR_MM, far=FAR_MM, ramp=RAMP):
    """Render a depth_mm frame as `rows` lines of `cols` characters.

    Each output cell summarises a block of source pixels by the median of the
    valid depths inside it (ignoring zeros and anything outside [near, far]).
    A cell with too few valid pixels is drawn as background (a space), which
    keeps a person in the foreground crisp against the empty room behind them.

    Args:
        depth: uint16 depth array in millimetres, shape (H, W).
        cols: width of the drawing in characters.
        rows: height of the drawing in characters.
        near: millimetre distance mapped to the densest glyph.
        far: millimetre distance mapped to the lightest non-space glyph.
        ramp: glyph ramp from background/far to near; ramp[0] is the space.

    Returns:
        A newline-joined string of `rows` lines, each exactly `cols` characters,
        or None when the frame is unusable.
    """
    if depth is None or depth.ndim != 2:
        return None
    h, w = depth.shape
    if h < rows or w < cols:
        return None
    y_edges = np.linspace(0, h, rows + 1).astype(int)
    x_edges = np.linspace(0, w, cols + 1).astype(int)
    span = float(far - near)
    levels = len(ramp)
    lines = []
    for r in range(rows):
        band = depth[y_edges[r]:y_edges[r + 1]]
        chars = []
        for c in range(cols):
            cell = band[:, x_edges[c]:x_edges[c + 1]]
            valid = cell[(cell >= near) & (cell <= far)]
            if valid.size < max(4, cell.size // 20):
                chars.append(ramp[0])
                continue
            dist = float(np.median(valid))
            nearness = 1.0 - (dist - near) / span
            k = 1 + int(nearness * (levels - 2) + 0.5)
            chars.append(ramp[min(k, levels - 1)])
        lines.append("".join(chars))
    return "\n".join(lines)


# --------------------------------------------------------------- RGB -> letters
# Index order matters: it must line up with the conditions in _classify.
PAL_LETTERS = "kwgpnroyvcalm"
COLOR_LEGEND = ("k=negro g=gris w=blanco r=rojo o=naranja y=amarillo "
                "v=verde c=cian a=azul l=lila m=rosa p=piel n=marron")


def _block_mean_rgb(rgb, rows, cols):
    """Average each block of the RGB frame down to a (rows, cols, 3) float array."""
    h, w = rgb.shape[:2]
    y_edges = np.linspace(0, h, rows + 1).astype(int)
    x_edges = np.linspace(0, w, cols + 1).astype(int)
    out = np.zeros((rows, cols, 3), np.float32)
    for r in range(rows):
        band = rgb[y_edges[r]:y_edges[r + 1]].astype(np.float32)
        for c in range(cols):
            out[r, c] = band[:, x_edges[c]:x_edges[c + 1]].reshape(-1, 3).mean(0)
    return out


def _classify(cell_rgb):
    """Map each cell's mean RGB to a palette index (0..12) by hue/saturation/value."""
    r, g, b = cell_rgb[..., 0], cell_rgb[..., 1], cell_rgb[..., 2]
    mx, mn = cell_rgb.max(-1), cell_rgb.min(-1)
    v = mx / 255.0
    s = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1.0), 0.0)
    d = np.where((mx - mn) > 0, mx - mn, 1.0)
    rc, gc, bc = (mx - r) / d, (mx - g) / d, (mx - b) / d
    h = np.where(mx == r, bc - gc, np.where(mx == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0 * 360.0
    h = np.where((mx - mn) > 0, h, 0.0)
    skin = ((s >= 0.10) & (s <= 0.55) & (r > g) & (g >= b) & (g >= 0.5 * r)
            & (b >= 0.30 * r) & (r > 95) & (v > 0.45) & (h < 50))
    cond = [
        v < 0.16,                               # 0  k negro
        (s < 0.15) & (v >= 0.72),               # 1  w blanco
        s < 0.15,                               # 2  g gris
        skin,                                   # 3  p piel
        (h < 50) & (v < 0.45) & (s > 0.3),      # 4  n marron
        (h < 15) | (h >= 345),                  # 5  r rojo
        h < 40,                                 # 6  o naranja
        h < 70,                                 # 7  y amarillo
        h < 170,                                # 8  v verde
        h < 200,                                # 9  c cian
        h < 265,                                # 10 a azul
        h < 295,                                # 11 l lila
    ]
    return np.select(cond, list(range(12)), default=12).astype(int)


def render_rgb_ascii(rgb, cols=DEFAULT_COLS, rows=DEFAULT_ROWS):
    """Render an RGB frame as a grid of color letters plus a legend string.

    Each character names the dominant colour of that cell (see COLOR_LEGEND), so a
    text-only model reads a coarse colour picture: skin tones for a face and hands,
    the colour of clothing, the wall behind, and so on.

    Args:
        rgb: uint8 array shape (H, W, 3) in RGB order.
        cols: width of the grid in characters.
        rows: height of the grid in characters.

    Returns:
        (grid, legend) where grid is a newline-joined block of `rows` lines of
        `cols` letters, and legend maps each letter to its Spanish colour name.
        grid is None when the frame is unusable.
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return None, COLOR_LEGEND
    h, w = rgb.shape[:2]
    if h < rows or w < cols:
        return None, COLOR_LEGEND
    idx = _classify(_block_mean_rgb(rgb, rows, cols))
    grid = "\n".join("".join(PAL_LETTERS[i] for i in row) for row in idx)
    return grid, COLOR_LEGEND


if __name__ == "__main__":
    synth = np.full((480, 640), 5000, np.uint16)
    synth[120:440, 240:400] = 1100
    synth[90:160, 280:360] = 900
    print("depth:")
    print(render_depth_ascii(synth))
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[:] = (120, 125, 130)              # gris wall
    rgb[120:440, 240:400] = (200, 40, 40)  # camiseta roja
    rgb[90:160, 280:360] = (225, 175, 150)  # cara (piel)
    print("\ncolor:")
    grid, legend = render_rgb_ascii(rgb)
    print(grid)
    print(legend)
