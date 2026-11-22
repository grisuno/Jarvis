#!/usr/bin/env python3
"""JARVIS on a Kinect 360: hears (4 mics + direction), sees (depth + RGB), knows its pose
(tilt motor + accelerometer) and can move its head.

Threads:
  hub audio   -> AudioFront (VAD, muted while motor moves or TTS speaks) -> utterance queue
  sensors     -> latest depth/state at 10 Hz, motion events
  brain       -> STT -> wake word -> snapshot (depth summary [+ RGB]) -> LLM (stream) -> TTS
  tts         -> speaks sentences as they stream in

LLM: any OpenAI-compatible endpoint (llama-server by default, Ollama's /v1 also works).
Real JPEG frames are only sent when the server reports vision support (llama-server /props)
or --vision. For a text-only model (no vision) the scene is instead rendered as an ASCII
"vista" (depth silhouette and/or colour-as-letters) and written into the text prompt, so the
model can still see and describe who is in front of it. Control it with --vista / --ascii-*.

Usage:
  .venv/bin/python jarvis.py                     # voice, wake word "jarvis"
  .venv/bin/python jarvis.py --text              # type instead of talking (mics unused)
  .venv/bin/python jarvis.py --sensors           # print telemetry + live ASCII vista
  .venv/bin/python jarvis.py --vista both        # depth silhouette + colour letters
   .venv/bin/python jarvis.py --sensors --vista color   # watch what Jarvis "sees"

Agentes: di "jarvis ejecuta: TAREA" para delegar TAREA a opencode en segundo plano;
"¿cómo va mi tarea?" consulta el estado. Opciones: --agent-dir, --agent, --agent-model,
--agent-auto (aprueba permisos automáticamente) y --no-agent.
"""
import argparse
import base64
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request

import numpy as np

from agent_bridge import AgentRunner
from ascii_vision import render_depth_ascii, render_rgb_ascii
from hub_client import KinectHub
from spatial import SpatialAnalyzer, annotate_rgb

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = """Eres JARVIS, un asistente de voz que vive dentro de un sensor Kinect 360 \
conectado a un PC. Respondes en español, breve y natural (1-3 frases), porque tu respuesta \
se lee en voz alta: nada de markdown, listas ni emojis.

Cada turno recibes un bloque TELEMETRIA en JSON medido por el sensor en el momento en que el \
usuario terminó de hablar:
- distancias reales en metros desde el mapa de profundidad (mapa 3x3 y regiones en primer plano \
con lado, distancia, tamaño y si probablemente es una persona; "izquierda/derecha" es desde el \
punto de vista del sensor),
- la dirección de la que vino la voz (micrófonos),
- el ángulo del motor y el pitch medido por el acelerómetro,
- movimientos recientes.
Si además recibes una imagen, las etiquetas amarillas "#n X.Xm" marcan las regiones con su \
distancia. Usa la telemetría cuando sea relevante y no inventes distancias que no estén ahí. \
No recites el JSON.

Normalmente, en vez de una foto, recibes una VISTA en ASCII de lo que ves en este instante, \
como una rejilla de caracteres. Puede traer dos capas:
- PROFUNDIDAD: los caracteres densos (@ # % *) están MÁS CERCA del sensor y los ligeros \
(. : - =) más lejos; un espacio es fondo o sin dato. Te da forma, postura y posición (una \
persona sentada frente a ti es un bloque denso y vertical).
- COLOR: cada letra es el color de esa celda según la leyenda que viene en el propio mensaje \
(p = piel, por lo que una zona de 'p' arriba suele ser una cara o unas manos; el resto son \
ropa, pared, etc.). Úsalo para describir a la persona (color de ropa, pelo) y la escena.
En las dos capas la columna izquierda del dibujo es tu izquierda y la fila superior es arriba \
(mismo criterio que la telemetría). Las distancias en metros tómalas siempre de la TELEMETRIA, \
no del dibujo. Sí puedes decir con naturalidad que ves a la persona y describir lo que ves.

Puedes mover la cabeza (solo inclinación vertical, -27 a +27 grados, 0 = horizontal). El \
ángulo que la cabeza está manteniendo ahora mismo está en TELEMETRIA.sensor.tilt_motor_deg: \
no lo olvides. Para CAMBIARLO, añade al final de tu respuesta una etiqueta exacta [[tilt:N]] \
con el ángulo absoluto nuevo, por ejemplo [[tilt:15]]. Emite esa etiqueta SOLO cuando el \
usuario pida explícitamente mirar arriba/abajo, volver al centro o ponerse nivel (recto/\
horizontal), o cuando una persona aparezca cortada arriba/abajo y quieras encuadrarla. Si no \
se pide mover la cabeza, NO escribas ninguna etiqueta [[tilt:...]]: la cabeza conserva su \
ángulo actual. En particular NUNCA escribas [[tilt:0]] por defecto; 0 significa horizontal y \
solo se usa si te lo piden de forma explícita."""

WAKE_WORDS = ("jarvis", "yarvis", "jarbis", "yarbis", "charvis", "harvis", "jervis", "llarvis")


def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def log(tag, msg):
    print(f"\x1b[2m{time.strftime('%H:%M:%S')}\x1b[0m \x1b[1m{tag:>6}\x1b[0m {msg}", flush=True)


# --------------------------------------------------------------------------- LLM
class LLM:
    def __init__(self, base_url, model, api_key=None, vision=None, think=False, max_tokens=300):
        self.base = base_url.rstrip("/")
        self.model = model
        self.key = api_key
        self.think = think
        self.max_tokens = max_tokens
        self.vision = self._detect_vision() if vision is None else vision

    def _detect_vision(self):
        root = self.base[:-3] if self.base.endswith("/v1") else self.base
        try:
            with urllib.request.urlopen(root + "/props", timeout=3) as r:
                props = json.load(r)
            return bool(props.get("modalities", {}).get("vision"))
        except Exception:
            return False

    def stream(self, messages):
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": self.max_tokens, "temperature": 0.6}
        if not self.think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(self.base + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        with urllib.request.urlopen(req, timeout=120) as r:
            in_think = False
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content") or ""
                # some templates leak <think> blocks into content
                if "<think>" in piece:
                    in_think = True
                    piece = piece.split("<think>")[0]
                if in_think:
                    if "</think>" in piece:
                        in_think = False
                        piece = piece.split("</think>", 1)[1]
                    else:
                        continue
                if piece:
                    yield piece


# --------------------------------------------------------------------------- TTS
class TTS:
    """piper (if PIPER_MODEL set) > espeak-ng > espeak > spd-say > console only."""

    def __init__(self, engine="auto", voice="es"):
        self.voice = voice
        self.q = queue.Queue()
        self.speaking = False
        self.last_end = 0.0
        self.engine = self._pick(engine)
        threading.Thread(target=self._loop, daemon=True).start()

    def _pick(self, engine):
        if engine != "auto":
            return engine
        if shutil.which("piper") and os.environ.get("PIPER_MODEL"):
            return "piper"
        for e in ("espeak-ng", "espeak", "spd-say"):
            if shutil.which(e):
                return e
        return "none"

    def say(self, text):
        text = text.strip()
        if text:
            self.q.put(text)

    def busy(self):
        return self.speaking or not self.q.empty() or time.time() - self.last_end < 0.4

    def _speak(self, text):
        e = self.engine
        if e == "piper":
            p1 = subprocess.Popen(["piper", "--model", os.environ["PIPER_MODEL"], "--output_raw"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
            p2 = subprocess.Popen(["aplay", "-q", "-r", os.environ.get("PIPER_RATE", "22050"),
                                   "-f", "S16_LE", "-c", "1"], stdin=p1.stdout)
            p1.stdin.write(text.encode())
            p1.stdin.close()
            p2.wait()
        elif e in ("espeak-ng", "espeak"):
            subprocess.run([e, "-v", self.voice, "-s", "165", text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif e == "spd-say":
            subprocess.run(["spd-say", "-w", "-l", self.voice, text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _loop(self):
        while True:
            text = self.q.get()
            self.speaking = True
            try:
                if self.engine != "none":
                    self._speak(text)
            except Exception as ex:
                log("tts", f"error: {ex}")
            finally:
                self.speaking = self.q.qsize() > 0
                self.last_end = time.time()


# --------------------------------------------------------------------------- Jarvis
class Jarvis:
    def __init__(self, args):
        self.a = args
        self.ascii_vision = not args.no_ascii_vision
        self.spatial = SpatialAnalyzer()
        self.tts = TTS(args.tts, args.voice)
        self.system_prompt = SYSTEM_PROMPT
        self.agents = None
        if not args.no_agent:
            self.agents = AgentRunner(cmd=args.agent_cmd, cwd=args.agent_dir,
                                      agent=args.agent, model=args.agent_model,
                                      auto=args.agent_auto,
                                      log_path=os.path.join(HERE, "agent_jobs.log"),
                                      on_finish=self._agent_finished)
            if self.agents.available():
                log("agent", f"opencode listo ({args.agent_dir}"
                             f"{' --auto' if args.agent_auto else ''})")
            else:
                log("agent", f"aviso: no encuentro '{args.agent_cmd}' en PATH")
        self.history = []
        self.awake_until = 0.0
        self.latest_depth = None
        self.latest_state = None
        self.lock = threading.Lock()
        self.busy = False
        self.audio = None
        if not args.text and not args.sensors:
            from audio_front import AudioFront
            self.audio = AudioFront(gain=args.gain, is_muted=self._mic_muted)
            self.audio.on_speech_start = lambda: log("mic", "voz detectada...")
        self.hub = KinectHub(audio_cb=self.audio.on_frame if self.audio else None,
                             audio=self.audio is not None,
                             log_path=os.path.join(HERE, "kinect_hub.log"))
        log("hub", "kinect_hub listo")
        self.llm = None
        if not args.sensors:
            self.llm = LLM(args.llm_url, args.model, os.environ.get("OPENAI_API_KEY"),
                           vision=True if args.vision else (False if args.no_vision else None),
                           think=args.think)
            log("llm", f"{args.llm_url} modelo={args.model} vision={self.llm.vision}")
            if not self.llm.vision:
                log("vista", f"ASCII {'ON modo=' + args.vista if self.ascii_vision else 'OFF'} "
                             f"({args.ascii_cols}x{args.ascii_rows})")
            log("tts", f"motor de voz: {self.tts.engine}")
        self.stt = None
        if self.audio:
            from audio_front import STT
            compute = "int8" if args.stt_device == "cpu" else "float16"
            log("stt", f"cargando whisper '{args.whisper}' ({args.stt_device}, {compute})...")
            self.stt = STT(args.whisper, args.lang, device=args.stt_device, compute=compute)
            log("stt", "listo")
        threading.Thread(target=self._sensor_loop, daemon=True).start()
        t0 = time.time()
        while self.latest_depth is None and time.time() - t0 < 8:
            time.sleep(0.05)
        if self.latest_depth is None:
            log("hub", "aviso: todavía no llega profundidad")

    # ---- gating
    def _mic_muted(self):
        st = self.latest_state
        return self.tts.busy() or (st is not None and st.moving)

    # ---- sensors
    def _sensor_loop(self):
        last_seq = -1
        while True:
            seq, ts, depth = self.hub.depth()
            st = self.hub.state()
            fresh = seq != last_seq and ts > 0
            with self.lock:
                self.latest_state = st
                if fresh:
                    self.latest_depth = (ts, depth)
            if fresh:
                last_seq = seq
                self.spatial.feed(ts, depth, moving_sensor=st.moving)
            time.sleep(0.1)

    def snapshot(self, doa=None):
        """Return (telemetry, regions, jpg, vista).

        jpg is a real annotated frame when the LLM has native vision; otherwise
        vista is an ASCII rendering (depth and/or colour) the text model can read.
        """
        with self.lock:
            depth = self.latest_depth[1] if self.latest_depth else None
            st = self.latest_state
        if depth is None:
            return None, None, None, None
        info, regions = self.spatial.summary(depth, st, doa)
        if self.llm and self.llm.vision:
            jpg = None
            rgb = self.hub.snapshot_rgb()
            if rgb is not None:
                jpg = annotate_rgb(rgb, regions)
            return info, regions, jpg, None
        vista = self._build_vista(depth) if self.ascii_vision else None
        return info, regions, None, vista

    def _build_vista(self, depth):
        """Render the ASCII vista (depth silhouette, colour letters, or both)."""
        mode = self.a.vista
        cols, rows = self.a.ascii_cols, self.a.ascii_rows
        parts = []
        if mode in ("depth", "both"):
            art = render_depth_ascii(depth, cols, rows)
            if art:
                parts.append("PROFUNDIDAD (densos @#% = cerca, ligeros .:- = lejos, "
                             "' ' = fondo):\n" + art)
        if mode in ("color", "both"):
            rgb = self.hub.snapshot_rgb()
            grid, legend = render_rgb_ascii(rgb, cols, rows)
            if grid:
                parts.append("COLOR (" + legend + "):\n" + grid)
            elif mode == "color":
                art = render_depth_ascii(depth, cols, rows)
                if art:
                    parts.append("PROFUNDIDAD (densos @#% = cerca, ' ' = fondo):\n" + art)
        return "\n\n".join(parts) if parts else None

    # ---- conversation
    def wake_check(self, text):
        t = norm(text)
        words = re.findall(r"[a-z0-9]+", t)
        if not self.a.wake:
            return True, text
        if time.time() < self.awake_until:
            return True, text
        for i, w in enumerate(words[:5]):
            if w in WAKE_WORDS or w.startswith(self.a.wake):
                return True, text
        return False, text

    def _agent_finished(self, job):
        tag = f"Tarea {job['id']}"
        if job["status"] == "done":
            res = " ".join(job["text"].split())[:160] or "sin resumen"
            msg = f"{tag} terminada. {res}"
        else:
            msg = f"{tag} falló. {job['error'] or 'sin detalle'}"
        log("agent", msg)
        self.tts.say(msg)

    def _run_agent(self, task):
        try:
            jid = self.agents.submit(task)
            log("agent", f"#{jid} en marcha: {task}")
            msg = f"Ejecutando la tarea {jid}."
        except RuntimeError as ex:
            log("agent", f"error: {ex}")
            msg = "No puedo lanzar el agente, opencode no está disponible."
        self.tts.say(msg)
        self.awake_until = time.time() + self.a.followup

    def handle(self, text, doa=None, doa_conf=0.0):
        if self.agents:
            task = self.agents.extract_command(text)
            if task:
                self._run_agent(task)
                return
            if self.agents.wants_status(text):
                msg = self.agents.spoken_status()
                log("agent", msg)
                self.tts.say(msg)
                self.awake_until = time.time() + self.a.followup
                return
        info, regions, jpg, vista = self.snapshot(doa if doa_conf > 0.15 else None)
        tele = json.dumps(info, ensure_ascii=False) if info else "{}"
        cur = (info or {}).get("sensor", {}).get("tilt_motor_deg")
        motors = (f"\n\nMOTOR ACTUAL: {cur}° (conserva este ángulo salvo que te pidan mover la "
                  f"cabeza; no lo pongas a 0 por defecto).") if cur is not None else ""
        user_content = f"{text}\n\nTELEMETRIA:\n{tele}{motors}"
        if jpg:
            content = [{"type": "text", "text": user_content},
                       {"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}}]
        elif vista:
            content = f"{user_content}\n\nVISTA (lo que ves ahora mismo):\n{vista}"
        else:
            content = user_content
        msgs = [{"role": "system", "content": self.system_prompt}] + self.history[-12:] + \
               [{"role": "user", "content": content}]
        if self.a.debug:
            log("tele", tele)
        t0 = time.time()
        full, pending, first = "", "", None
        try:
            for piece in self.llm.stream(msgs):
                if first is None:
                    first = time.time() - t0
                full += piece
                pending += piece
                # speak complete sentences as soon as they arrive (but never the action tag)
                while True:
                    m = re.search(r"[.!?¡¿…](\s|$)", pending)
                    if not m or "[[" in pending[:m.end()]:
                        break
                    self.tts.say(pending[:m.end()])
                    pending = pending[m.end():]
        except (urllib.error.URLError, ConnectionError, TimeoutError) as ex:
            log("llm", f"error: {ex}")
            self.tts.say("No puedo contactar con mi modelo.")
            return
        actions = re.findall(r"\[\[\s*tilt\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\]\]", full)
        pending = re.sub(r"\[\[.*?\]\]", "", pending)
        self.tts.say(pending)
        spoken = re.sub(r"\[\[.*?\]\]", "", full).strip()
        log("jarvis", f"{spoken}  \x1b[2m(1er token {first or 0:.1f}s, total {time.time() - t0:.1f}s)\x1b[0m")
        self.history += [{"role": "user", "content": user_content},
                         {"role": "assistant", "content": full}]
        for a in actions[-1:]:
            deg = float(a)
            if cur is not None and abs(deg - cur) < 0.5:
                continue
            log("motor", f"tilt -> {deg:+.0f}°")
            while self.tts.busy():
                time.sleep(0.05)
            self.hub.tilt(deg)
        self.awake_until = time.time() + self.a.followup

    # ---- loops
    def run_voice(self):
        log("ready", f"escuchando. Di '{self.a.wake or '(sin palabra clave)'} ...'  (Ctrl-C para salir)")
        self.tts.say("Sistemas en línea.")
        last_level = 0
        while True:
            try:
                utt, t_end = self.audio.utterances.get(timeout=1.0)
            except queue.Empty:
                if self.a.debug and time.time() - last_level > 2:
                    v = self.audio.vad
                    log("lvl", f"rms={v.level:.4f} floor={(v.floor or 0):.4f} muted={self.audio.muted_now}")
                    last_level = time.time()
                continue
            from audio_front import doa_deg
            doa, conf = doa_deg(utt, flip=self.a.doa_flip)
            mono = utt.mean(1).astype(np.float32)
            peak = np.abs(mono).max()
            if peak > 0:
                mono = mono * min(0.9 / peak, 30.0)
            t0 = time.time()
            text = self.stt.transcribe(mono)
            if not text:
                continue
            log("oído", f"\"{text}\"  \x1b[2m({len(utt) / 16000:.1f}s audio, stt {time.time() - t0:.1f}s, "
                         f"voz {doa:+.0f}° conf {conf:.2f})\x1b[0m")
            ok, text = self.wake_check(text)
            if not ok:
                continue
            self.handle(text, doa, conf)

    def run_text(self):
        log("ready", "modo texto: escribe y pulsa Enter (Ctrl-D para salir)")
        while True:
            try:
                text = input("\x1b[1mtú>\x1b[0m ").strip()
            except EOFError:
                break
            if text:
                self.handle(text)
                while self.tts.busy():
                    time.sleep(0.05)

    def run_sensors(self):
        while True:
            info, _, _, vista = self.snapshot()
            if vista:
                print("\x1b[H\x1b[2J" + vista, flush=True)
            if info:
                print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)
            time.sleep(1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm-url", default=os.environ.get("JARVIS_LLM_URL", "http://127.0.0.1:8080/v1"))
    p.add_argument("--model", default=os.environ.get("JARVIS_MODEL", "local-model"))
    p.add_argument("--vision", action="store_true", help="force sending RGB frames")
    p.add_argument("--no-vision", action="store_true", help="never send RGB frames")
    p.add_argument("--vista", choices=["depth", "color", "both"], default="color",
                   help="ASCII vision when the model has no native vision: depth silhouette, "
                        "color letters (RGB written as text), or both")
    p.add_argument("--no-ascii-vision", action="store_true",
                   help="disable ASCII vision entirely (send telemetry only)")
    p.add_argument("--ascii-cols", type=int, default=52, help="ASCII vista width in characters")
    p.add_argument("--ascii-rows", type=int, default=20, help="ASCII vista height in characters")
    p.add_argument("--think", action="store_true", help="allow Qwen3-style thinking (slower)")
    p.add_argument("--whisper", default=os.environ.get("JARVIS_WHISPER", "small"),
                   help="faster-whisper model: tiny/base/small/medium/large-v3/turbo")
    p.add_argument("--stt-device", default="cuda", help="cuda|cuda:N|cpu (default: GPU; CPU can be slow/unstable)")
    p.add_argument("--lang", default="es")
    p.add_argument("--wake", default="jarvis", help="wake word; '' disables it")
    p.add_argument("--followup", type=float, default=20.0, help="seconds awake after answering")
    p.add_argument("--gain", type=float, default=8.0, help="mic digital gain")
    p.add_argument("--doa-flip", action="store_true", help="swap left/right of voice direction")
    p.add_argument("--tts", default="auto", help="auto|piper|espeak-ng|espeak|spd-say|none")
    p.add_argument("--voice", default="es")
    p.add_argument("--text", action="store_true", help="keyboard input instead of microphones")
    p.add_argument("--sensors", action="store_true", help="only print spatial telemetry")
    p.add_argument("--no-agent", action="store_true",
                   help="disable delegation to opencode")
    p.add_argument("--agent-cmd", default=os.environ.get("JARVIS_AGENT_CMD", "opencode"),
                   help="opencode executable")
    p.add_argument("--agent-dir", default=os.environ.get("JARVIS_AGENT_DIR", HERE),
                   help="working directory for delegated tasks")
    p.add_argument("--agent", default=os.environ.get("JARVIS_AGENT"),
                   help="opencode agent name")
    p.add_argument("--agent-model", default=os.environ.get("JARVIS_AGENT_MODEL"),
                   help="model for opencode (provider/model)")
    p.add_argument("--agent-auto", action="store_true",
                   default=os.environ.get("JARVIS_AGENT_AUTO", "").lower() in ("1", "true", "yes"),
                   help="auto-approve agent permissions (dangerous)")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    a.wake = norm(a.wake)

    j = None
    try:
        j = Jarvis(a)
        if a.sensors:
            j.run_sensors()
        elif a.text:
            j.run_text()
        else:
            j.run_voice()
    except KeyboardInterrupt:
        pass
    finally:
        if j:
            j.hub.tilt(0)
            time.sleep(0.3)
            j.hub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
