"""The other processes the deck can see and control.

Each service is described once: which port it answers on, how to recognise
it in the process table, where its log goes, and how to start it. Anything
without a start command is still watched — it just cannot be started from
the dashboard, which is the right behaviour for something installed
elsewhere on the machine.
"""

import os
import subprocess
import sys
from pathlib import Path

import psutil

from . import config, procs


def _python() -> str:
    return sys.executable or "python"


def _whisper_cmd() -> list:
    """whisper.cpp's own server, with an OpenAI-compatible route."""
    if not os.path.isfile(config.WHISPER_BIN):
        return []
    if not os.path.isfile(config.WHISPER_MODEL):
        return []
    return [config.WHISPER_BIN,
            "-m", config.WHISPER_MODEL,
            "--host", "0.0.0.0",
            "--port", str(config.WHISPER_PORT),
            "--inference-path", "/v1/audio/transcriptions",
            "--convert"]


def registry() -> dict:
    """name -> descriptor, in the order the dashboard should list them."""
    out = {
        "proxy": {
            "label": "multimodal proxy",
            "port": config.PROXY_PORT,
            "needle": "neuraldeck.proxy",
            "log": config.PROXY_LOG,
            # Started as a module so its command line is unmistakable in the
            # process table — and so it can never match the deck's own.
            "cmd": [_python(), "-m", "neuraldeck.proxy"],
        },
        "whisper": {
            "label": "whisper-server",
            "port": config.WHISPER_PORT,
            "names": procs.WHISPER_NAMES,
            "needle": "whisper-server",
            "log": config.WHISPER_LOG,
            "cmd": _whisper_cmd(),
        },
    }
    # The whole command line is the needle: its last word alone is often
    # something generic ("--listen", "main.py") that other processes share.
    if config.TTS_CMD:
        out["tts"] = {"label": "TTS server", "port": config.TTS_PORT,
                      "needle": " ".join(config.TTS_CMD), "log": config.TTS_LOG,
                      "cmd": config.TTS_CMD}
    if config.COMFY_CMD:
        out["comfy"] = {"label": "ComfyUI", "port": config.COMFY_PORT,
                        "needle": " ".join(config.COMFY_CMD),
                        "log": config.COMFY_LOG, "cmd": config.COMFY_CMD}
    return out


def find(name: str):
    svc = registry().get(name)
    if svc is None:
        return None
    proc = procs.find_proc(svc["needle"])
    if proc is None and svc.get("names"):
        proc = procs.find_by_name(svc["names"])
    return proc


def snapshot() -> dict:
    """{name: {up, port, pid, cpu, rss}} for the Servers panel."""
    out = {}
    for name, svc in registry().items():
        proc = find(name)
        out[name] = {**(procs.metrics(proc)), "up": proc is not None,
                     "port": svc["port"], "label": svc["label"],
                     "can_start": bool(svc["cmd"])}
    return out


def start(name: str) -> dict:
    svc = registry().get(name)
    if svc is None:
        raise LookupError(f"unknown service '{name}'")
    if not svc["cmd"]:
        raise LookupError(
            f"'{name}' has no start command on this machine — set "
            f"NEURALDECK_{name.upper()}_CMD, or start it yourself")
    if find(name) is not None:
        raise RuntimeError(f"{name} is already running")
    log = Path(svc["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ}
    # The proxy is part of this package: make sure a child process can
    # import it even when the deck was started from a checkout.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent.parent)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    kwargs = {"stdin": subprocess.DEVNULL, "env": env}
    if config.IS_WINDOWS:
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                  | getattr(subprocess, "DETACHED_PROCESS", 0))
    else:
        kwargs["start_new_session"] = True
    with open(log, "ab", buffering=0) as fh:
        proc = subprocess.Popen(svc["cmd"], stdout=fh, stderr=subprocess.STDOUT,
                                **kwargs)
    (config.RUN_DIR / f"{name}.pid").write_text(str(proc.pid), encoding="utf-8")
    return {"starting": name, "pid": proc.pid, "log": str(log),
            "cmd": svc["cmd"]}


def _pidfile_pid(name: str, svc: dict):
    """The pid we recorded at start — trusted only while that pid is still
    the service. Pids are reused; a stale file must never point stop() at
    whatever unrelated process (and its children) got the number since."""
    pidfile = config.RUN_DIR / f"{name}.pid"
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
        p = psutil.Process(pid)
        if svc["needle"] in " ".join(p.cmdline() or []) \
                or (p.name() or "") in (svc.get("names") or ()):
            return pid
    except Exception:
        pass
    return None


def stop(name: str) -> dict:
    """Stop a service by the pid we can identify, never by pattern."""
    svc = registry().get(name)
    if svc is None:
        raise LookupError(f"unknown service '{name}'")
    proc = find(name)
    pid = proc.pid if proc else _pidfile_pid(name, svc)
    if pid is None:
        raise LookupError(f"{name} does not appear to be running")
    gone = procs.stop_pid(pid)
    if gone:
        try:
            (config.RUN_DIR / f"{name}.pid").unlink()
        except OSError:
            pass
    return {"stopped": name, "pid": pid, "gone": gone}


def log_path(name: str):
    if name == "llama":
        return procs.newest_llama_log()
    svc = registry().get(name)
    return str(svc["log"]) if svc else None


def log_names() -> list:
    """Which log tabs the dashboard should offer."""
    return ["llama"] + list(registry().keys()) + ["launch"]
