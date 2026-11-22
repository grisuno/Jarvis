"""Kinect 4-mic front end: energy VAD with adaptive noise floor, GCC-PHAT DOA, Whisper STT."""
import collections
import queue
import threading
import time

import numpy as np

RATE = 16000
# Kinect 360 linear array, mic x positions in metres (left -> right, sensor's view)
MIC_X = np.array([-0.113, 0.036, 0.076, 0.113])
SOUND_C = 343.0


class VAD:
    def __init__(self, frame_ms=20, start_ratio=3.0, stop_ratio=1.8, min_abs=0.004,
                 hang_ms=700, preroll_ms=300, max_s=15.0, min_s=0.35):
        self.fpms = frame_ms
        self.start_ratio, self.stop_ratio, self.min_abs = start_ratio, stop_ratio, min_abs
        self.hang = hang_ms // frame_ms
        self.pre = collections.deque(maxlen=preroll_ms // frame_ms)
        self.max_frames = int(max_s * 1000 / frame_ms)
        self.min_frames = int(min_s * 1000 / frame_ms)
        self.floor = None
        self.active = False
        self.buf = []
        self.loud_run = 0
        self.quiet_run = 0
        self.level = 0.0

    def reset(self):
        self.active = False
        self.buf = []
        self.loud_run = self.quiet_run = 0
        self.pre.clear()

    def push(self, frame_f32):
        """frame: float32 [n,4] in -1..1. Returns a finished utterance [N,4] or None."""
        mono = frame_f32.mean(1)
        mono = mono - mono.mean()
        rms = float(np.sqrt((mono ** 2).mean()) + 1e-9)
        self.level = rms
        if self.floor is None:
            self.floor = rms
        if not self.active:
            self.floor = 0.97 * self.floor + 0.03 * rms if rms < self.floor * 2 else self.floor * 1.002
            self.pre.append(frame_f32)
            if rms > max(self.floor * self.start_ratio, self.min_abs):
                self.loud_run += 1
                if self.loud_run >= 3:
                    self.active = True
                    self.buf = list(self.pre)
                    self.quiet_run = 0
            else:
                self.loud_run = 0
            return None
        self.buf.append(frame_f32)
        if rms < max(self.floor * self.stop_ratio, self.min_abs * 0.6):
            self.quiet_run += 1
        else:
            self.quiet_run = 0
        if self.quiet_run >= self.hang or len(self.buf) >= self.max_frames:
            frames = self.buf[:len(self.buf) - self.quiet_run + 5] if self.quiet_run else self.buf
            ok = len(self.buf) - self.quiet_run >= self.min_frames
            self.reset()
            return np.concatenate(frames) if ok and frames else None
        return None


def doa_deg(audio4, flip=False, fmin=300.0, fmax=4000.0):
    """Steered-response power with PHAT weighting over all 6 mic pairs.
    Returns azimuth in degrees (+ = right as seen by the sensor), and a confidence 0..1."""
    x = audio4 - audio4.mean(0)
    n = 1 << int(np.ceil(np.log2(len(x))))
    X = np.fft.rfft(x, n=n, axis=0)
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    band = (freqs >= fmin) & (freqs <= fmax)
    X, freqs = X[band], freqs[band]
    angles = np.radians(np.arange(-90, 91, 2))
    power = np.zeros(len(angles))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for i, j in pairs:
        G = X[:, i] * np.conj(X[:, j])
        G /= np.abs(G) + 1e-12
        tau = (MIC_X[j] - MIC_X[i]) * np.sin(angles) / SOUND_C
        steer = np.exp(-2j * np.pi * np.outer(tau, freqs))
        power += np.real(steer @ G)
    k = int(np.argmax(power))
    conf = float((power[k] - np.median(power)) / (np.abs(power).max() + 1e-9))
    az = float(np.degrees(angles[k]))
    return (-az if flip else az), max(0.0, min(1.0, conf))


class STT:
    HALLUCINATIONS = ("subtítulos", "subtitulos", "amara.org", "gracias por ver",
                      "suscríbete", "thanks for watching")

    def __init__(self, model="small", language="es", device="cpu", compute="int8", threads=8):
        from faster_whisper import WhisperModel
        self.language = language or None
        self.model = WhisperModel(model, device=device, compute_type=compute, cpu_threads=threads)

    def transcribe(self, mono_f32, hint="Jarvis"):
        segs, _info = self.model.transcribe(
            mono_f32, language=self.language, beam_size=1, vad_filter=False,
            initial_prompt=hint, condition_on_previous_text=False)
        parts = []
        for s in segs:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -0.8:
                continue
            parts.append(s.text.strip())
        text = " ".join(parts).strip()
        low = text.lower()
        if any(h in low for h in self.HALLUCINATIONS):
            return ""
        return text


class AudioFront:
    """Glue: receives hub audio frames, gates them, emits (audio4, t_end) utterances."""

    def __init__(self, gain=8.0, is_muted=lambda: False, **vad_kw):
        self.gain = gain
        self.is_muted = is_muted
        self.vad = VAD(**vad_kw)
        self.utterances = queue.Queue()
        self.muted_now = False
        self.on_speech_start = None
        self._was_active = False

    def on_frame(self, frame_i16):
        f = frame_i16.astype(np.float32) * (self.gain / 32768.0)
        np.clip(f, -1, 1, out=f)
        if self.is_muted():
            if not self.muted_now:
                self.vad.reset()
            self.muted_now = True
            return
        self.muted_now = False
        utt = self.vad.push(f)
        if self.vad.active and not self._was_active and self.on_speech_start:
            self.on_speech_start()
        self._was_active = self.vad.active
        if utt is not None:
            self.utterances.put((utt, time.time()))
