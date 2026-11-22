#!/usr/bin/env python3
"""Kinect 360 -> live ASCII depth art in the terminal.

Keys: w/s = tilt up/down, x = level, q = quit, Ctrl-C = quit
Requires: libfreenect (patched 0.7.5 at /tmp/opencode/fn-install) + pyfreenect in .venv
"""
import sys, time, ctypes, os, select, termios, tty, subprocess

_FN_LIB = os.environ.get("KINECT_FREENECT_LIB", os.path.expanduser("~/.kinect/lib"))
if not os.path.isdir(_FN_LIB):
    _FN_LIB = "/tmp/opencode/fn-install/lib"
if os.path.isdir(_FN_LIB):
    ctypes.CDLL(os.path.join(_FN_LIB, "libfreenect.so.0"), mode=ctypes.RTLD_GLOBAL)

import numpy as np
import freenect

TILT_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinect_tilt")


def tilt(angle):
    if os.path.exists(TILT_BIN):
        subprocess.Popen([TILT_BIN, str(int(angle)), "1000"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

RAMP = " .:-=+*#%@"       # near = dense, far = sparse
COLS, ROWS = 100, 40      # ~2:1 char aspect vs 58.4x46.5 deg depth FOV
NEAR, FAR = 500, 2500     # mm depth window


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


def render():
    depth, _ = freenect.sync_get_depth()
    if depth is None:
        return None
    d = depth[::480 // ROWS, ::640 // COLS][:ROWS, :COLS].astype(np.float32)
    np.clip(d, NEAR, FAR, out=d)
    nearness = 1.0 - (d - NEAR) / (FAR - NEAR)
    idx = (nearness * (len(RAMP) - 1) + 0.5).astype(np.uint8)
    return "\n".join("".join(RAMP[i] for i in row) for row in idx)


def main():
    if not sys.stdin.isatty():
        print("run from a terminal", file=sys.stderr)
        return 1
    print("\x1b[2J", end="")  # clear
    angle = 0.0
    last_key_ts = 0.0
    fps = 0.0
    try:
        while True:
            t0 = time.time()
            art = render()
            if art is None:
                print("\x1b[H\x1b[2Jno depth stream", flush=True)
                time.sleep(0.2)
                continue
            dt = time.time() - t0
            fps = fps * 0.9 + (1.0 / dt if dt > 0 else 0) * 0.1
            sys.stdout.write(
                f"\x1b[H\x1b[2Jtilt {angle:+.0f}deg  {fps:4.1f} fps  (w/s tilt, x level, q quit)\n{art}\n")
            sys.stdout.flush()

            key = poll_key(0.01)
            now = time.time()
            if key in ("w", "s") and now - last_key_ts > 0.5:
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
        print("\x1b[?25h", end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
