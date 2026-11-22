"""Client for the kinect_hub daemon: audio from its stdout, frames/state from /dev/shm."""
import mmap
import os
import struct
import subprocess
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HUB_BIN = os.path.join(HERE, "kinect_hub")
SHM_PATH = "/dev/shm/kinect_hub"
MAGIC = 0x4B48554A
W, H = 640, 480
DEPTH_OFF = 4096
RGB_OFF = DEPTH_OFF + W * H * 2
AUDIO_RATE = 16000
AUDIO_CH = 4


class HubState:
    __slots__ = ("tilt_deg", "tilt_target", "accel", "tilt_status", "last_move_ts",
                 "state_ts", "mode")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def moving(self):
        return self.tilt_status == 4 or (time.time() - self.last_move_ts) < 1.5

    @property
    def pitch_deg(self):
        """Pitch of the sensor head from gravity (positive = looking up)."""
        ax, ay, az = self.accel
        if abs(ay) < 1e-3 and abs(az) < 1e-3:
            return float(self.tilt_deg)
        return float(np.degrees(np.arctan2(az, ay)))

    @property
    def roll_deg(self):
        ax, ay, _ = self.accel
        if abs(ay) < 1e-3:
            return 0.0
        return float(np.degrees(np.arctan2(ax, ay)))


class KinectHub:
    def __init__(self, audio_cb=None, audio=True, log_path=None, frame_ms=20):
        self.audio_cb = audio_cb
        self.frame_ms = frame_ms
        args = [HUB_BIN] + ([] if audio else ["--no-audio"])
        if os.path.exists(SHM_PATH):
            try:
                os.unlink(SHM_PATH)
            except OSError:
                pass
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0,
                                     start_new_session=True)
        self._log = open(log_path, "a") if log_path else None
        self._lock = threading.Lock()
        self.ready = threading.Event()
        threading.Thread(target=self._stderr_loop, daemon=True).start()
        if audio:
            threading.Thread(target=self._audio_loop, daemon=True).start()
        if not self.ready.wait(15):
            raise RuntimeError("kinect_hub did not become ready (see log)")
        f = open(SHM_PATH, "rb")
        self.m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        f.close()
        if struct.unpack_from("<I", self.m, 0)[0] != MAGIC:
            raise RuntimeError("bad shm magic")

    def _stderr_loop(self):
        for raw in self.proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            if "strange length" in line:
                continue
            if self._log:
                self._log.write(f"{time.strftime('%H:%M:%S')} {line}\n")
                self._log.flush()
            if "hub: ready" in line:
                self.ready.set()
        self.ready.set()

    def _audio_loop(self):
        n = AUDIO_RATE * self.frame_ms // 1000
        nbytes = n * AUDIO_CH * 2
        out = self.proc.stdout
        buf = b""
        while True:
            chunk = out.read(nbytes - len(buf))
            if not chunk:
                return
            buf += chunk
            if len(buf) < nbytes:
                continue
            frame = np.frombuffer(buf, np.int16).reshape(n, AUDIO_CH)
            buf = b""
            if self.audio_cb:
                self.audio_cb(frame)

    def cmd(self, line):
        with self._lock:
            try:
                self.proc.stdin.write((line.strip() + "\n").encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def tilt(self, deg):
        self.cmd(f"tilt {max(-27, min(27, int(round(deg))))}")

    def _seq(self, off):
        return struct.unpack_from("<I", self.m, off)[0]

    def _read_locked(self, seq_off, fn, tries=50):
        for _ in range(tries):
            s1 = self._seq(seq_off)
            if s1 & 1:
                time.sleep(0.001)
                continue
            data = fn()
            if self._seq(seq_off) == s1:
                return s1, data
        return self._seq(seq_off), fn()

    def depth(self):
        """Returns (seq, ts, depth_mm uint16[480,640]) copy."""
        def grab():
            ts = struct.unpack_from("<d", self.m, 24)[0]
            arr = np.frombuffer(self.m, np.uint16, W * H, DEPTH_OFF).reshape(H, W).copy()
            return ts, arr
        seq, (ts, arr) = self._read_locked(8, grab)
        return seq, ts, arr

    def rgb(self):
        def grab():
            ts = struct.unpack_from("<d", self.m, 32)[0]
            arr = np.frombuffer(self.m, np.uint8, W * H * 3, RGB_OFF).reshape(H, W, 3).copy()
            return ts, arr
        seq, (ts, arr) = self._read_locked(12, grab)
        return seq, ts, arr

    def state(self):
        def grab():
            mode = struct.unpack_from("<I", self.m, 20)[0]
            st_ts, tilt, target, ax, ay, az = struct.unpack_from("<6d", self.m, 40)
            status, _rate = struct.unpack_from("<ii", self.m, 88)
            last_move = struct.unpack_from("<d", self.m, 96)[0]
            return HubState(tilt_deg=tilt, tilt_target=target, accel=(ax, ay, az),
                            tilt_status=status, last_move_ts=last_move, state_ts=st_ts,
                            mode=mode)
        return self._read_locked(16, grab)[1]

    def snapshot_rgb(self, warmup_frames=6, timeout=2.0):
        """Briefly enable RGB next to depth, return a well-exposed frame, go back to depth."""
        start_seq = self._seq(12)
        self.cmd("mode both")
        t0 = time.time()
        img = None
        while time.time() - t0 < timeout:
            if self._seq(12) - start_seq >= 2 * warmup_frames:
                img = self.rgb()[2]
                break
            time.sleep(0.02)
        self.cmd("mode depth")
        return img

    def close(self):
        try:
            self.cmd("quit")
            self.proc.stdin.close()
            self.proc.wait(5)
        except Exception:
            self.proc.kill()
