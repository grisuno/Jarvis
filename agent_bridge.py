"""Async bridge from JARVIS to `opencode run`."""
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _norm(s):
    import unicodedata
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _clip(s, n=200):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


class AgentRunner:
    def __init__(self, cmd="opencode", cwd=None, agent=None, model=None, auto=False,
                 log_path=None, on_finish=None, keep=12):
        self.cmd = cmd
        self.cwd = cwd or os.getcwd()
        self.agent = agent
        self.model = model
        self.auto = auto
        self.log_path = log_path
        self.on_finish = on_finish
        self.keep = keep
        self.jobs = {}
        self.order = []
        self.events = queue.Queue()
        self.lock = threading.Lock()
        self.seq = 0

    def available(self):
        return shutil.which(self.cmd) is not None

    def submit(self, task, cwd=None):
        task = " ".join(task.split())
        if not task:
            return None
        if not self.available():
            raise RuntimeError(f"no encuentro '{self.cmd}' en PATH")
        directory = cwd or self.cwd
        cmd = [self.cmd, "run", "--format", "json", "--dir", directory,
               "--title", "JARVIS: " + _clip(task, 60)]
        if self.agent:
            cmd += ["--agent", self.agent]
        if self.model:
            cmd += ["--model", self.model]
        if self.auto:
            cmd += ["--auto"]
        cmd += [task]
        with self.lock:
            self.seq += 1
            jid = self.seq
            self.jobs[jid] = {
                "id": jid, "task": task, "status": STATUS_RUNNING,
                "started": time.time(), "finished": None, "text": "",
                "error": "", "session": "", "rc": None, "blocked": False,
                "cwd": directory,
            }
            self.order.append(jid)
            if len(self.order) > self.keep:
                old = self.order.pop(0)
                self.jobs.pop(old, None)
        threading.Thread(target=self._monitor, args=(jid, cmd, directory),
                         daemon=True).start()
        return jid

    def _monitor(self, jid, cmd, directory):
        job = self.jobs.get(jid)
        if job is None:
            return
        log = None
        try:
            if self.log_path:
                log = open(self.log_path, "a", encoding="utf-8")
                log.write(f"\n--- #{jid} {time.strftime('%H:%M:%S')} {job['task']}\n")
                log.flush()
            p = subprocess.Popen(cmd, cwd=directory, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", bufsize=1)
        except Exception as ex:
            self._finish(jid, STATUS_FAILED, str(ex))
            if log:
                log.close()
            return
        for line in p.stdout:
            if log:
                log.write(line)
                log.flush()
            self._event(jid, line)
        rc = p.wait()
        if log:
            log.write(f"--- fin #{jid} rc={rc}\n")
            log.close()
        status = STATUS_DONE if rc == 0 else STATUS_FAILED
        self._finish(jid, status, None, rc)

    def _event(self, jid, line):
        line = line.strip()
        if not line.startswith("{"):
            return
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            return
        texts = []
        seen = set()

        def walk(o):
            if isinstance(o, dict):
                t = o.get("type")
                if isinstance(t, str) and t in ("text", "assistant"):
                    v = o.get("text")
                    if isinstance(v, str) and v.strip() and v not in seen:
                        seen.add(v)
                        texts.append(v)
                for k, v in o.items():
                    if k in ("sessionID", "session_id") and isinstance(v, str):
                        with self.lock:
                            j = self.jobs.get(jid)
                            if j:
                                j["session"] = v
                    if k == "error" and isinstance(v, dict) and v.get("message"):
                        with self.lock:
                            j = self.jobs.get(jid)
                            if j:
                                j["error"] = _clip(v["message"], 160)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(ev)
        typ = ev.get("type", "") if isinstance(ev, dict) else ""
        with self.lock:
            j = self.jobs.get(jid)
            if j is None:
                return
            if "permission" in typ or "question" in typ:
                j["blocked"] = "replied" not in typ and "resolved" not in typ
            for text in texts:
                prev = j["text"]
                if prev.endswith(text):
                    continue
                if text.startswith(prev) and prev:
                    j["text"] = text
                else:
                    j["text"] = (prev + " " + text).strip()

    def _finish(self, jid, status, error=None, rc=None):
        with self.lock:
            j = self.jobs.get(jid)
            if j is None:
                return
            j["status"] = status
            j["finished"] = time.time()
            j["rc"] = rc
            if error:
                j["error"] = _clip(error, 160)
        job = self.snapshot(jid)
        self.events.put(job)
        if self.on_finish:
            try:
                self.on_finish(job)
            except Exception:
                pass

    def snapshot(self, jid=None):
        with self.lock:
            if jid is None:
                running = [i for i in reversed(self.order)
                           if self.jobs.get(i, {}).get("status") == STATUS_RUNNING]
                jid = running[0] if running else (self.order[-1] if self.order else None)
            j = self.jobs.get(jid)
            return dict(j) if j else None

    def brief(self):
        with self.lock:
            ids = list(self.order[-4:])
            jobs = [dict(self.jobs[i]) for i in ids if i in self.jobs]
        lines = []
        for j in jobs:
            if j["status"] == STATUS_RUNNING:
                label = f"#{j['id']} en curso desde hace {int(time.time() - j['started'])}s"
                lines.append(label + ": " + _clip(j["task"], 90))
            elif j["status"] == STATUS_DONE:
                res = _clip(j["text"], 160) or "sin resumen"
                lines.append(f"#{j['id']} terminada: {_clip(j['task'], 70)} RESUMEN: {res}")
            else:
                why = j["error"] or "sin detalle"
                lines.append(f"#{j['id']} fallida: {why}")
            if j["blocked"] and j["status"] == STATUS_RUNNING:
                lines[-1] += " (necesita permisos)"
        return "\n".join(lines)

    def spoken_status(self):
        brief = self.brief()
        if not brief:
            return "No tienes ninguna tarea encargada al agente."
        return "Estado de las tareas. " + " ".join(brief.splitlines()[:3])

    def extract_command(self, text):
        prefix = (r"(?:(?:jarvis|yarvis|jarbis|yarbis|charvis|harvis|jervis|llarvis"
                  r"|hey|oye)\b[\s,:]+){0,3}(?:por favor[\s,]+)?")
        m = re.match(prefix + r"ejecuta\b\s*[:,-]?\s*(.+)$", text.strip(), re.I | re.S)
        return " ".join(m.group(1).split()) if m else None

    def wants_status(self, text):
        t = _norm(text)
        return bool(re.search(
            r"(como va|estado|progreso|como va|termino|sabes algo|noticias)"
            r".{0,40}(tarea|peticion|pedido|agente|trabajo)"
            r"|(tarea|peticion|pedido|agente|trabajo).{0,40}"
            r"(como va|estado|progreso|termino|listo)", t))
