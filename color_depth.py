#!/usr/bin/env python3
"""Kinect 360 -> truecolor half-block visualizer in the terminal.

Depth mode: thermal colormap (near = hot). RGB mode: real camera colors.
Keys: m = toggle depth/rgb, c = cycle colormap, q = quit, Ctrl-C = quit
"""
import sys, time, ctypes, os, select, termios, tty, subprocess, shutil

_FN_LIB = os.environ.get("KINECT_FREENECT_LIB", os.path.expanduser("~/.kinect/lib"))
if not os.path.isdir(_FN_LIB):
    _FN_LIB = "/tmp/opencode/fn-install/lib"
if os.path.isdir(_FN_LIB):
    ctypes.CDLL(os.path.join(_FN_LIB, "libfreenect.so.0"), mode=ctypes.RTLD_GLOBAL)

import numpy as np
import freenect

TILT_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinect_tilt")
NEAR, FAR = 500, 3000  # mm

# thermal palettes: (stops) -> LUT 256x3 via piecewise-linear interp
PALETTES = {
    "inferno": [(0, (0, 0, 4)), (0.25, (87, 16, 110)), (0.5, (188, 55, 84)),
                (0.75, (249, 142, 9)), (1.0, (252, 255, 164))],
    "viridis": [(0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
                (0.75, (94, 201, 98)), (1.0, (253, 231, 37))],
    "turbo":   [(0, (48, 18, 59)), (0.25, (55, 90, 255)), (0.5, (39, 255, 88)),
                (0.75, (255, 246, 0)), (1.0, (122, 4, 3))],
    "mono":    [(0, (5, 5, 5)), (1.0, (250, 250, 250))],
    "fire":    [(0, (10, 0, 20)), (0.3, (120, 0, 40)), (0.6, (255, 80, 0)),
                (0.85, (255, 200, 0)), (1.0, (255, 255, 220))],
}
LUTS = {}
for name, stops in PALETTES.items():
    xs = np.array([s[0] for s in stops])
    ys = np.array([s[1] for s in stops], dtype=np.float32)
    t = np.linspace(0, 1, 256)
    LUTS[name] = np.stack([np.interp(t, xs, ys[:, ch]) for ch in range(3)], axis=1).astype(np.uint8)


def term_size():
    cols, rows = shutil.get_terminal_size((100, 40))
    rows -= 1  # status line
    return cols, rows


def downscale(img, cols, rows2):
    h, w = img.shape[:2]
    ys = (np.linspace(0, h - 1, rows2)).astype(np.int32)
    xs = (np.linspace(0, w - 1, cols)).astype(np.int32)
    return img[ys][:, xs]


def frame_depth(cols, rows2, lut):
    d, _ = freenect.sync_get_depth()
    if d is None:
        return None
    d = downscale(d, cols, rows2).astype(np.float32)
    np.clip(d, NEAR, FAR, out=d)
    nearness = 1.0 - (d - NEAR) / (FAR - NEAR)
    idx = (nearness * 255 + 0.5).astype(np.uint8)
    return lut[idx]


def frame_rgb(cols, rows2):
    c, _ = freenect.sync_get_video(freenect.VIDEO_RGB)
    if c is None:
        return None
    return downscale(c, cols, rows2)


def render(img):
    """img: (rows2, cols, 3) uint8 -> ANSI half-block string"""
    top = img[0::2]
    bot = img[1:img.shape[0] - img.shape[0] % 2:2]
    out = []
    for y in range(top.shape[0]):
        t, b = top[y], bot[y]
        buf = []
        last = None
        for x in range(t.shape[0]):
            cur = (tuple(t[x]), tuple(b[x]))
            if cur != last:
                buf.append(f"\x1b[38;2;{cur[0][0]};{cur[0][1]};{cur[0][2]}"
                           f";48;2;{cur[1][0]};{cur[1][1]};{cur[1][2]}m▀")
                last = cur
            else:
                buf.append("▀")
        buf.append("\x1b[0m")
        out.append("".join(buf))
    return "\n".join(out)


def poll_key(timeout=0.0):
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def tilt(angle):
    if os.path.exists(TILT_BIN):
        subprocess.Popen([TILT_BIN, str(int(angle)), "1000"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    if not sys.stdin.isatty():
        print("run from a terminal", file=sys.stderr)
        return 1
    print("\x1b[2J")
    lut_name = "inferno"
    rgb_mode = False
    angle = 0.0
    last_key_ts = 0.0
    fps = 0.0
    try:
        while True:
            cols, rows = term_size()
            rows2 = rows * 2
            t0 = time.time()
            img = frame_rgb(cols, rows2) if rgb_mode else frame_depth(cols, rows2, LUTS[lut_name])
            if img is None:
                time.sleep(0.2)
                continue
            frame = render(img)
            dt = time.time() - t0
            fps = fps * 0.9 + (1.0 / dt if dt > 0 else 0) * 0.1
            mode = f"RGB cam" if rgb_mode else f"depth [{lut_name}]"
            sys.stdout.write(f"\x1b[H\x1b[2J{mode}  {fps:4.1f} fps  tilt {angle:+.0f}deg"
                             f"  (m mode, c colormap, w/s tilt, x level, q quit)\n{frame}\n")
            sys.stdout.flush()

            key = poll_key(0.005)
            now = time.time()
            if key == "m":
                rgb_mode = not rgb_mode
                freenect.sync_stop()
                time.sleep(0.3)
            elif key == "c" and not rgb_mode:
                names = list(PALETTES)
                lut_name = names[(names.index(lut_name) + 1) % len(names)]
            elif key in ("w", "s") and now - last_key_ts > 0.5:
                angle = float(int(np.clip(angle + (10 if key == "w" else -10), -30, 30)))
                tilt(angle)
                last_key_ts = now
            elif key == "x":
                angle = 0.0
                tilt(0)
            elif key == "q":
                break
            spare = 0.033 - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            tilt(0)
            freenect.sync_stop()
        except Exception:
            pass
        print("\x1b[0m\x1b[?25h", end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
