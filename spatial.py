"""Turn a depth_mm frame + sensor pose into a compact spatial summary for an LLM."""
import time
from collections import deque

import cv2
import numpy as np

W, H = 640, 480
F_DEPTH = 580.0           # px, Kinect 360 depth focal length (approx)
CX, CY = W / 2, H / 2
RGB_HFOV = 62.0           # deg, approx RGB horizontal FOV
DS = 4                    # downsample factor for analysis (160x120)
MIN_MM, MAX_MM = 400, 6000


def pixel_angles(u, v):
    """Azimuth (+ = right of image) and elevation (+ = up) in degrees, camera frame."""
    az = np.degrees(np.arctan2(u - CX, F_DEPTH))
    el = np.degrees(np.arctan2(CY - v, F_DEPTH))
    return float(az), float(el)


def side_word(az):
    if az < -15:
        return "izquierda"
    if az > 15:
        return "derecha"
    if az < -5:
        return "centro-izquierda"
    if az > 5:
        return "centro-derecha"
    return "centro"


class SpatialAnalyzer:
    """Keeps a little history so it can report motion as well as static structure."""

    def __init__(self):
        self.prev = deque(maxlen=12)       # (ts, small depth) ~1 s at 10 Hz
        self.events = deque(maxlen=20)     # (ts, text)
        self.last_motion_ts = 0.0
        self.motion_level = 0.0

    @staticmethod
    def small(depth):
        d = depth[::DS, ::DS].astype(np.float32)
        d[(d < MIN_MM) | (d > MAX_MM)] = 0
        return d

    def feed(self, ts, depth, moving_sensor=False):
        """Call at ~10 Hz from the sensor loop. Returns a motion event dict or None."""
        d = self.small(depth)
        ev = None
        if self.prev and not moving_sensor:
            _, old = self.prev[0]
            both = (d > 0) & (old > 0)
            diff = np.abs(d - old) * both
            changed = diff > np.maximum(120.0, 0.05 * d)
            changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN,
                                       np.ones((3, 3), np.uint8))
            frac = changed.sum() / max(1, both.sum())
            self.motion_level = 0.7 * self.motion_level + 0.3 * frac
            if frac > 0.02:
                ys, xs = np.nonzero(changed)
                u, v = xs.mean() * DS, ys.mean() * DS
                dist = float(np.median(d[ys, xs]))
                az, _ = pixel_angles(u, v)
                ev = {"ts": ts, "frac": float(frac), "az": az, "dist_m": dist / 1000}
                if ts - self.last_motion_ts > 3.0:
                    self.events.append((ts, f"movimiento {side_word(az)} a {dist / 1000:.1f} m"))
                self.last_motion_ts = ts
        if moving_sensor:
            self.prev.clear()
        self.prev.append((ts, d))
        return ev

    def regions(self, depth, pitch_deg=0.0, max_regions=4):
        """Foreground regions: connected areas clearly nearer than what is behind them."""
        d = self.small(depth)
        valid = d > 0
        if valid.sum() < 200:
            return []
        # grey closing = max-filter then min-filter: fills near blobs with the far background
        bg = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        near_layer = np.percentile(d[valid], 12) + 600
        mask = valid & ((d < near_layer) | (bg - d > 400))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 0.008 * d.size:
                continue
            ys, xs = np.nonzero(lab == i)
            z = d[ys, xs]
            zm = float(np.median(z))
            u, v = xs.mean() * DS, ys.mean() * DS
            az, el = pixel_angles(u, v)
            w_m = w * DS * zm / F_DEPTH / 1000
            h_m = h * DS * zm / F_DEPTH / 1000
            # 3D centroid height relative to the sensor, corrected by head pitch
            yc = (CY - v) * zm / F_DEPTH
            top_y = (CY - y * DS) * zm / F_DEPTH
            height_rel = (yc * cp + zm * sp) / 1000
            top_rel = (top_y * cp + zm * sp) / 1000
            r = {
                "dist_m": round(zm / 1000, 2),
                "az_deg": round(az, 1),
                "el_deg": round(el, 1),
                "lado": side_word(az),
                "ancho_m": round(w_m, 2),
                "alto_m": round(h_m, 2),
                "altura_centro_vs_sensor_m": round(height_rel, 2),
                "altura_tope_vs_sensor_m": round(top_rel, 2),
                "cortado_arriba": bool(y <= 1),
                "cortado_abajo": bool(y + h >= d.shape[0] - 1),
                "area": int(area),
                "_uv": (u, v),
            }
            r["posible_persona"] = bool(0.25 < w_m < 1.2 and (h_m > 0.9 or (r["cortado_arriba"] and h_m > 0.5)))
            out.append(r)
        out.sort(key=lambda r: (-r["posible_persona"], r["dist_m"]))
        return out[:max_regions]

    @staticmethod
    def grid(depth):
        d = SpatialAnalyzer.small(depth)
        rows = ["arriba", "medio", "abajo"]
        cols = ["izq", "centro", "der"]
        hh, ww = d.shape[0] // 3, d.shape[1] // 3
        g = {}
        for i, rn in enumerate(rows):
            for j, cn in enumerate(cols):
                cell = d[i * hh:(i + 1) * hh, j * ww:(j + 1) * ww]
                v = cell[cell > 0]
                g[f"{rn}-{cn}"] = round(float(np.median(v)) / 1000, 2) if v.size > 20 else None
        return g

    def summary(self, depth, state, doa_deg=None, now=None):
        now = now or time.time()
        pitch = state.pitch_deg if state else 0.0
        regs = self.regions(depth, pitch)
        center = depth[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20]
        cv = center[(center > MIN_MM) & (center < MAX_MM)]
        info = {
            "sensor": {
                "tilt_motor_deg": round(state.tilt_deg, 1) if state else None,
                "pitch_por_gravedad_deg": round(pitch, 1),
                "roll_deg": round(state.roll_deg, 1) if state else None,
                "acelerometro_ms2": [round(a, 2) for a in state.accel] if state else None,
                "nota": "pitch>0 = el cabezal mira hacia arriba; rango motor +-27 grados",
            },
            "distancia_centro_m": round(float(np.median(cv)) / 1000, 2) if cv.size > 30 else None,
            "mapa_3x3_m": self.grid(depth),
            "regiones_primer_plano": [{k: v for k, v in r.items() if not k.startswith("_")} for r in regs],
            "movimiento_reciente": [
                f"hace {int(now - ts)} s: {txt}" for ts, txt in self.events if now - ts < 60
            ],
        }
        if doa_deg is not None:
            info["voz"] = {"direccion_deg": round(doa_deg, 1), "lado": side_word(doa_deg)}
            if regs:
                best = min(regs, key=lambda r: abs(r["az_deg"] - doa_deg))
                if abs(best["az_deg"] - doa_deg) < 20:
                    info["voz"]["region_probable_hablante"] = {
                        "dist_m": best["dist_m"], "lado": best["lado"]}
        return info, regs


def annotate_rgb(rgb, regions):
    """Draws distance labels on the RGB frame so a VLM can tie pixels to metres."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    f_rgb = (W / 2) / np.tan(np.radians(RGB_HFOV / 2))
    for i, r in enumerate(regions):
        u = CX + f_rgb * np.tan(np.radians(r["az_deg"]))
        v = CY - f_rgb * np.tan(np.radians(r["el_deg"]))
        p = (int(np.clip(u, 10, W - 90)), int(np.clip(v, 20, H - 10)))
        label = f"#{i + 1} {r['dist_m']:.1f}m"
        cv2.putText(img, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes() if ok else None
