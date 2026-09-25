"""Process and port helpers.

Everything here goes through psutil rather than shelling out to pkill or
lsof: those are absent on Windows, and a pattern-matching kill is a hazard
anyway — `pkill -f llama` cheerfully matches the dashboard that ran it.
Stopping something means signalling a pid we identified, never a pattern.
"""

import os
import re
import socket
from pathlib import Path

import psutil

from . import config

LLAMA_NAMES = {"llama-server", "llama-server.exe"}
WHISPER_NAMES = {"whisper-server", "whisper-server.exe"}
# Shells and process-listing tools whose command lines mention a daemon
# without being one.
_WRAPPERS = {"bash", "sh", "zsh", "dash", "grep", "pgrep", "tail", "nohup",
             "cmd.exe", "powershell.exe", "pwsh.exe", "conhost.exe"}

_proc_cache: dict = {}   # needle -> psutil.Process, so cpu_percent() deltas work
_llama_cache: dict = {}  # pid -> psutil.Process, same reason


def _exe_of(p):
    try:
        return p.exe()
    except Exception:
        return None


def _cmdline(p) -> str:
    try:
        return " ".join(p.cmdline() or [])
    except Exception:
        return ""


def find_proc(needle: str):
    """A daemon identified by a substring of its command line, cached.

    Never returns this process or its own children: the deck's command line
    mentions the package name, and matching itself would report the
    dashboard as the service it is looking for.
    """
    proc = _proc_cache.get(needle)
    if proc is not None:
        try:
            if proc.is_running() and needle in _cmdline(proc):
                return proc
        except Exception:
            pass
        _proc_cache.pop(needle, None)
    me = os.getpid()
    try:
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                if p.pid == me:
                    continue
                if (p.info["name"] or "") in _WRAPPERS:
                    continue
                if needle in " ".join(p.info["cmdline"] or []):
                    p.cpu_percent(None)  # prime the delta
                    _proc_cache[needle] = p
                    return p
            except Exception:
                continue
    except Exception:
        pass
    return None


def find_by_name(names: set):
    me = os.getpid()
    for p in psutil.process_iter(["name"]):
        try:
            if p.pid != me and (p.info["name"] or "") in names:
                return p
        except Exception:
            continue
    return None


def metrics(proc) -> dict:
    """pid / cpu% (normalised per core) / rss for a service row."""
    if proc is None:
        return {}
    try:
        ncpu = psutil.cpu_count() or 1
        with proc.oneshot():
            return {"pid": proc.pid,
                    "cpu": (proc.cpu_percent(None) or 0.0) / ncpu,
                    "rss": proc.memory_info().rss}
    except Exception:
        return {}


# ── llama-server instances ─────────────────────────────────────────────────
# Each instance's configuration is recoverable only from its command line:
# no llama endpoint reports which flags it was started with, and those flags
# are what the dashboard shows per instance.
_FLAGS_INT = {"-c": "ctx_total", "--ctx-size": "ctx_total",
              "-np": "slots", "--parallel": "slots",
              "--reasoning-budget": "reasoning_budget"}
_FLAGS_STR = {"--port": "port", "--alias": "alias", "--spec-type": "spec",
              "-ctk": "kv_type", "--cache-type-k": "kv_type"}
_FLAGS_PATH = {"-md": "draft_model", "--model-draft": "draft_model",
               "--spec-draft-model": "draft_model", "--mmproj": "mmproj",
               "-m": "model_path", "--model": "model_path"}

# Per-slot token counts. The update_slots line only appears at higher
# verbosity; the release line is logged at the default level and gives what
# the slot keeps cached after a request.
RE_SLOT_KV = re.compile(
    r"slot update_slots: id\s+(\d+) \|.*?slot\.prompt\.tokens\.size\(\) = (\d+)"
    r"|slot\s+release: id\s+(\d+) \|.*?stop processing: n_tokens = (\d+)")
# Where one run of a server begins in a log that may hold several.
RUN_START = "load_model: loading model"


def instance_kv_tokens(alias):
    """Tokens currently held in KV, summed across slots — None if unknown.

    llama-server exposes no KV gauge on /metrics or /slots, but its log
    lines carry the cached token count per slot; the last value seen for
    each slot, in the current run, is that slot's occupancy.
    """
    if not alias:
        return None
    path = config.LOG_DIR / f"{alias}.log"
    if not path.exists():
        return None
    text = tail_text(path, 256 * 1024)
    idx = text.rfind(RUN_START)
    if idx >= 0:
        text = text[idx:]
    per_slot = {}
    for m in RE_SLOT_KV.finditer(text):
        slot = m.group(1) if m.group(1) is not None else m.group(3)
        per_slot[slot] = int(m.group(2) if m.group(2) is not None else m.group(4))
    return sum(per_slot.values()) if per_slot else None


def _norm(path) -> str:
    try:
        return os.path.normcase(os.path.realpath(str(path)))
    except Exception:
        return str(path)


# Resolving a path is a syscall, so the map is cached — and rebuilt whenever
# the backend list changes under it (the settings page can do that live).
_backend_map: dict = {"key": None, "map": {}}


def _backend_by_exe() -> dict:
    key = tuple(sorted(config.BACKENDS.items()))
    if _backend_map["key"] != key:
        _backend_map["key"] = key
        _backend_map["map"] = {_norm(b): label for label, b in key if b}
    return _backend_map["map"]


def backend_of(exe, cmd: str) -> str:
    """Which build this instance came from.

    A configured backend is named by its label. Anything else — an instance
    someone started by hand, or a fork being A/B'd — is named after the
    checkout its binary lives in, which is more use than "unknown".
    """
    label = _backend_by_exe().get(_norm(exe)) if exe else None
    if label:
        return label
    parts = [q for q in Path(exe or cmd.split(" ")[0]).parts
             if q not in ("build", "bin", "Release", "Debug", "/", "\\")]
    # <checkout>/build/bin/llama-server -> "<checkout>"
    return parts[-2] if len(parts) >= 2 else "llama-server"


def llama_instances() -> list:
    """Every running llama-server, with what it was launched with."""
    out, seen = [], set()
    for p in psutil.process_iter(["name", "cmdline", "status"]):
        try:
            if (p.info["name"] or "") not in LLAMA_NAMES:
                continue
            # An exited server nobody has reaped yet: no port, no command
            # line, and nothing to stop.
            if p.info["status"] == psutil.STATUS_ZOMBIE:
                continue
            cmd = p.info["cmdline"] or []
            inst = {"port": None, "alias": None, "ctx_total": None, "slots": 1,
                    "spec": None, "draft_model": None, "mmproj": None,
                    "kv_type": None, "reasoning_budget": None,
                    "embeddings": False, "model_path": None}
            for i, arg in enumerate(cmd):
                nxt = cmd[i + 1] if i + 1 < len(cmd) else None
                if arg == "--embeddings":
                    inst["embeddings"] = True
                elif nxt is None:
                    continue
                elif arg in _FLAGS_INT:
                    try:
                        inst[_FLAGS_INT[arg]] = int(nxt)
                    except ValueError:
                        pass
                elif arg in _FLAGS_STR:
                    inst[_FLAGS_STR[arg]] = nxt
                elif arg in _FLAGS_PATH:
                    inst[_FLAGS_PATH[arg]] = os.path.basename(nxt)
            try:
                inst["port"] = int(inst["port"]) if inst["port"] else None
            except ValueError:
                inst["port"] = None
            inst["slots"] = max(1, int(inst.get("slots") or 1))
            seen.add(p.pid)
            proc = _llama_cache.get(p.pid)
            if proc is None:
                p.cpu_percent(None)
                _llama_cache[p.pid] = proc = p
            kv = instance_kv_tokens(inst["alias"])
            ctx = inst["ctx_total"]
            out.append({
                **(metrics(proc) or {"pid": p.pid}),
                **inst,
                # speculation as CONFIGURED: the draft counters only appear
                # once a request has actually drafted, so the launch flags
                # are the reliable record of whether it is on
                "spec": inst["spec"] if inst["spec"] not in (None, "none") else None,
                # llama.cpp splits -c across slots; this is the limit a
                # single request actually hits
                "ctx_per_req": ctx // inst["slots"] if ctx else None,
                "kv_tokens": kv,
                "kv_pct": round(kv / ctx * 100, 1) if kv is not None and ctx else None,
                "backend": backend_of(_exe_of(p), " ".join(cmd)),
            })
        except Exception:
            continue
    for pid in [k for k in _llama_cache if k not in seen]:
        del _llama_cache[pid]
    return sorted(out, key=lambda i: i["port"] or 0)


# ── ports ──────────────────────────────────────────────────────────────────

def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def llama_probe_host() -> str:
    """Where to reach a llama-server we launched: the configured host,
    unless that is a wildcard bind, which loopback reaches."""
    host = (config.LLAMA_HOST or "").strip()
    return "127.0.0.1" if host in ("", "0.0.0.0", "::", "[::]", "*") else host


def llama_port_busy(port: int) -> bool:
    """Taken on loopback or on the host llama-server binds to — either one
    makes the bind fail."""
    if port_in_use(port):
        return True
    host = llama_probe_host()
    try:
        return host != "127.0.0.1" and port_in_use(port, host)
    except OSError:
        return False


def next_free_llama_port(instances: list = None):
    used = {i["port"] for i in (llama_instances() if instances is None
                                else instances)}
    for port in config.LLAMA_PORTS:
        if port not in used and not llama_port_busy(port):
            return port
    return None


def wait_port_free(port: int, timeout: float = 30.0) -> bool:
    """A loaded llama-server can hold its port for seconds while it frees
    VRAM; a relaunch that does not wait fails to bind."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not llama_port_busy(port):
            return True
        time.sleep(0.5)
    return not llama_port_busy(port)


def _gone(p) -> bool:
    try:
        return not p.is_running() or p.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True
    except Exception:
        return False


def stop_pid(pid: int, timeout: float = 10.0) -> bool:
    """Terminate one pid (and its children), escalating to kill.

    True only if the pid is actually gone afterwards — a process we may not
    signal (AccessDenied) is still running, and saying otherwise would have
    a relaunch wait on a port that never frees.
    """
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    kids = []
    try:
        kids = proc.children(recursive=True)
    except Exception:
        pass
    for p in [proc] + kids:
        try:
            p.terminate()
        except Exception:
            pass
    gone, alive = psutil.wait_procs([proc] + kids, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=3.0)
    reap_children()
    return _gone(proc)


def stop_all_llama() -> int:
    """Stop the instances on the deck's own ports. A llama-server someone
    runs elsewhere on the machine is not ours to stop."""
    stopped = 0
    for inst in llama_instances():
        if inst.get("port") not in config.LLAMA_PORTS:
            continue
        if inst.get("pid") and stop_pid(inst["pid"]):
            stopped += 1
    return stopped


def reap_children() -> int:
    """Collect exited children of this process (POSIX).

    llama-servers and services are started detached but remain our
    children, and one that exits — stopped, or crashed — stays a zombie
    until reaped. Only pids already seen as zombie children are waited
    on, so a live child's own Popen handle is never robbed of its status.
    """
    if config.IS_WINDOWS:
        return 0
    n = 0
    try:
        kids = psutil.Process().children()
    except Exception:
        return 0
    for p in kids:
        try:
            if p.status() != psutil.STATUS_ZOMBIE:
                continue
            os.waitpid(p.pid, os.WNOHANG)
            n += 1
        except (psutil.Error, ChildProcessError, OSError):
            continue
    return n


# ── log tails ──────────────────────────────────────────────────────────────

def tail_text(path, n_bytes: int = 262144) -> str:
    """The last n_bytes of a file, decoded leniently. "" if unreadable."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def newest_llama_log():
    """The most recently written instance log — the dashboard's headline
    inference numbers come from whichever instance is busiest."""
    logs = sorted(config.LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime
                  if p.exists() else 0, reverse=True)
    skip = {config.PROXY_LOG.name, config.WHISPER_LOG.name,
            config.COMFY_LOG.name, config.TTS_LOG.name, "deck.log"}
    for p in logs:
        if p.name not in skip:
            return str(p)
    return None
