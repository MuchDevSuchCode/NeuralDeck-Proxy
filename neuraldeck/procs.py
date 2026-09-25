"""Process and port helpers.

Everything here goes through psutil rather than shelling out to pkill or
lsof: those are absent on Windows, and a pattern-matching kill is a hazard
anyway — `pkill -f llama` cheerfully matches the dashboard that ran it.
Stopping something means signalling a pid we identified, never a pattern.
"""

import json
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
    """Every running model server — llama-server and vLLM — with what it
    was launched with. Each entry's "kind" says which."""
    out, seen = [], set()
    vllm_procs = []
    for p in psutil.process_iter(["name", "cmdline", "status", "ppid"]):
        try:
            if (p.info["name"] or "") not in LLAMA_NAMES:
                if p.info["status"] != psutil.STATUS_ZOMBIE \
                        and (p.info["name"] or "") not in _WRAPPERS \
                        and _vllm_script(p.info["cmdline"] or []) is not None:
                    vllm_procs.append(p)
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
                "kind": "llama.cpp",
            })
        except Exception:
            continue
    # vLLM forks an engine process (and, with some settings, API server
    # workers) that can carry the same command line: only the top-level
    # serve process is the instance.
    vpids = {p.pid for p in vllm_procs}
    for p in vllm_procs:
        try:
            if p.info.get("ppid") in vpids or _has_ancestor(p, vpids):
                continue
            inst = _vllm_instance(p)
            if inst is None:
                continue
            seen.add(p.pid)
            proc = _llama_cache.get(p.pid)
            if proc is None:
                p.cpu_percent(None)
                _llama_cache[p.pid] = proc = p
            out.append({**(metrics(proc) or {"pid": p.pid}), **inst})
        except Exception:
            continue
    for pid in [k for k in _llama_cache if k not in seen]:
        del _llama_cache[pid]
    for port in [k for k in _vllm_scrape if k not in
                 {i["port"] for i in out if i.get("kind") == "vllm"}]:
        _vllm_scrape.pop(port, None)
    return sorted(out, key=lambda i: i["port"] or 0)


# ── vLLM instances ─────────────────────────────────────────────────────────
# `vllm serve` is a Python script, so the process is python running it: the
# instance is recognised by its command line, never by process name.

_VLLM_MODULES = {"vllm.entrypoints.openai.api_server"}


def _vllm_script(cmd: list):
    """Index of the token that makes this a vLLM server's command line
    (the `vllm` script before `serve`, or the api_server module), or None.

    The token must be the program itself or what a Python interpreter runs
    — a shell, `tail -f vllm.log` or an editor mentioning vllm is not one.
    """
    if not cmd:
        return None
    cands = [0]
    if os.path.basename(cmd[0]).lower().startswith("python"):
        j = 1
        while j < len(cmd) and cmd[j].startswith("-") and cmd[j] != "-m":
            j += 1                       # interpreter switches (-u, -O …)
        if j < len(cmd) and cmd[j] == "-m":
            j += 1
        cands.append(j)
    for i in cands:
        if i >= len(cmd):
            continue
        tok = cmd[i]
        if os.path.basename(tok).lower() in config.VLLM_NAMES \
                or tok == "vllm.entrypoints.cli.main":
            if i + 1 < len(cmd) and cmd[i + 1] == "serve":
                return i
        elif tok in _VLLM_MODULES:
            return i
    return None


def _has_ancestor(p, pids: set) -> bool:
    try:
        return any(a.pid in pids for a in p.parents())
    except Exception:
        return False


def _vllm_args(cmd: list) -> dict:
    """{flag: value} from a vLLM command line — both `--flag value` and
    `--flag=value` spellings — plus the positional model as "model"."""
    i = _vllm_script(cmd)
    rest = cmd[i + 1:] if i is not None else cmd
    if rest and rest[0] == "serve":
        rest = rest[1:]
    out = {}
    if rest and not rest[0].startswith("-"):
        out["model"] = rest[0]
        rest = rest[1:]
    k = 0
    while k < len(rest):
        tok = rest[k]
        if tok.startswith("--") and "=" in tok:
            flag, _, val = tok.partition("=")
            out[flag] = val
        elif tok.startswith("-"):
            nxt = rest[k + 1] if k + 1 < len(rest) else None
            if nxt is not None and (not nxt.startswith("-")
                                    or re.fullmatch(r"-[\d.]+", nxt)):
                out[tok] = nxt
                k += 1
            else:
                out[tok] = True          # a bare switch
        k += 1
    return out


def parse_len(v):
    """vLLM's --max-model-len: an int, or 32k (x1000) / 32K (x1024) style."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([kmgKMG]?)\s*", str(v or ""))
    if not m:
        return None
    mult = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9,
            "K": 2**10, "M": 2**20, "G": 2**30}[m.group(2)]
    try:
        return int(float(m.group(1)) * mult)
    except ValueError:
        return None


def _vllm_instance(p):
    cmd = p.info["cmdline"] or []
    a = _vllm_args(cmd)
    model = a.get("--model") if isinstance(a.get("--model"), str) else a.get("model")
    served = a.get("--served-model-name")
    alias = served if isinstance(served, str) else None
    if not alias and isinstance(model, str):
        alias = os.path.basename(model.rstrip("/\\")) or model
    try:
        port = int(a.get("--port", 8000))
    except (TypeError, ValueError):
        port = None
    ctx = parse_len(a.get("--max-model-len"))
    try:
        slots = max(1, int(a.get("--max-num-seqs")))
    except (TypeError, ValueError):
        slots = None                   # vLLM's own default, not known here
    spec = None
    raw_spec = a.get("--speculative-config") or a.get("-sc")
    if isinstance(raw_spec, str):
        try:
            spec = (json.loads(raw_spec) or {}).get("method") or "on"
        except (ValueError, AttributeError):
            spec = "on"
    host = a.get("--host") if isinstance(a.get("--host"), str) else ""
    script = cmd[_vllm_script(cmd)]
    m = vllm_metrics(port, host) if port else {}
    kv = m.get("kv_pct")
    return {
        "port": port, "alias": alias, "ctx_total": ctx, "ctx": ctx,
        "slots": slots or 1,
        "spec": spec, "draft_model": None, "mmproj": None,
        "kv_type": a.get("--kv-cache-dtype") if isinstance(
            a.get("--kv-cache-dtype"), str) else None,
        "reasoning_budget": None, "embeddings": False,
        "model_path": os.path.basename(str(model).rstrip("/\\")) if model else None,
        # --max-model-len is per request in vLLM; nothing is split by slot
        "ctx_per_req": ctx,
        "kv_tokens": m.get("kv_tokens"),
        "kv_pct": kv,
        "running": m.get("running"),
        "waiting": m.get("waiting"),
        "prefill_tps": m.get("prefill_tps"),
        "decode_tps": m.get("decode_tps"),
        "backend": backend_of(script if os.path.isabs(script) else _exe_of(p),
                              " ".join(cmd)),
        "kind": "vllm",
    }


# port -> {"at", "val", "prev": (t, prompt_total, gen_total)}
_vllm_scrape: dict = {}
_METRIC_RE = re.compile(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+(\S+)")
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict:
    """name -> list of (labels, value) from Prometheus exposition text."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        try:
            val = float(m.group(3))
        except ValueError:
            continue
        labels = dict(_LABEL_RE.findall(m.group(2) or ""))
        out.setdefault(m.group(1), []).append((labels, val))
    return out


def _metric_sum(parsed: dict, *names):
    for n in names:
        if n in parsed:
            return sum(v for _, v in parsed[n])
    return None


def vllm_metrics(port: int, host: str = "", ttl: float = 1.0) -> dict:
    """KV occupancy, queue and token rates from a vLLM server's /metrics.

    Short timeout and cached for `ttl`: called from the sampler thread every
    second, and from API handlers (always via a thread). Token rates are
    the counters' growth between two scrapes; a counter that went down
    means the server restarted, which is a new baseline, not a rate.
    """
    import time
    now = time.monotonic()
    st = _vllm_scrape.setdefault(port, {"at": 0.0, "val": {}, "prev": None})
    if now - st["at"] < ttl:
        return st["val"]
    st["at"] = now
    h = (host or "").strip()
    h = "127.0.0.1" if h in ("", "0.0.0.0", "::", "[::]", "*") else h
    url = f"http://{'[%s]' % h if ':' in h else h}:{port}/metrics"
    try:
        import httpx
        r = httpx.get(url, timeout=0.5)
        if r.status_code != 200:
            raise ValueError(r.status_code)
        parsed = parse_prometheus(r.text)
    except Exception:
        st["val"], st["prev"] = {}, None
        return {}
    val = {}
    usage = parsed.get("vllm:kv_cache_usage_perc") \
        or parsed.get("vllm:gpu_cache_usage_perc")
    if usage:
        frac = max(v for _, v in usage)          # the fullest engine
        val["kv_pct"] = round(frac * 100, 1)
        blocks = None
        for labels, _ in parsed.get("vllm:cache_config_info") or []:
            # A hybrid (Mamba-style) model allocates in blocks of hundreds of
            # tokens and parks each sequence's state in whole blocks, so
            # usage x capacity reads as thousands of tokens for a short chat.
            # The percentage is still vLLM's own truth; only the token count
            # is dropped there. Attention-only models use 16-token blocks.
            if labels.get("mamba_block_size") not in (None, "", "None"):
                break
            try:
                blocks = int(labels["num_gpu_blocks"]) * int(labels["block_size"])
                break
            except (KeyError, ValueError):
                continue
        val["kv_tokens"] = int(round(frac * blocks)) if blocks else None
    running = _metric_sum(parsed, "vllm:num_requests_running")
    waiting = _metric_sum(parsed, "vllm:num_requests_waiting")
    val["running"] = int(running) if running is not None else None
    val["waiting"] = int(waiting) if waiting is not None else None
    pt = _metric_sum(parsed, "vllm:prompt_tokens_total", "vllm:prompt_tokens")
    gt = _metric_sum(parsed, "vllm:generation_tokens_total",
                     "vllm:generation_tokens")
    prev = st["prev"]
    val["prefill_tps"] = val["decode_tps"] = None
    if prev and pt is not None and gt is not None:
        dt = now - prev[0]
        if dt > 0 and pt >= prev[1] and gt >= prev[2]:
            val["prefill_tps"] = round((pt - prev[1]) / dt, 1)
            val["decode_tps"] = round((gt - prev[2]) / dt, 1)
    st["prev"] = (now, pt, gt) if pt is not None and gt is not None else None
    st["val"] = val
    return val


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


# The launcher opens every vLLM log with this line; vLLM's own startup
# banner identifies logs of instances started some other way.
VLLM_LOG_MARK = "[neuraldeck] backend=vllm"
_VLLM_LOG_SIGNS = (VLLM_LOG_MARK, "vLLM API server version", "(APIServer pid=")
_log_kind: dict = {}     # (path, inode) -> is vLLM


def is_vllm_log(path) -> bool:
    """Whether an instance log was written by vLLM. Only the head is read,
    and the answer is kept per file (a rotated log is a new inode)."""
    try:
        key = (str(path), os.stat(path).st_ino)
    except OSError:
        return False
    if key not in _log_kind:
        try:
            with open(path, "rb") as f:
                head = f.read(65536).decode("utf-8", errors="replace")
        except OSError:
            return False
        found = any(s in head for s in _VLLM_LOG_SIGNS)
        if not found and len(head) < 16384:
            return False        # vLLM may not have printed its banner yet
        if len(_log_kind) > 256:
            _log_kind.clear()
        _log_kind[key] = found
    return _log_kind[key]


def newest_llama_log(skip_vllm: bool = False):
    """The most recently written instance log — the dashboard's headline
    inference numbers come from whichever instance is busiest. Those come
    from llama.cpp's timing lines, so that caller passes skip_vllm."""
    def mtime(p):
        try:
            return p.stat().st_mtime
        except OSError:
            return 0
    logs = sorted(config.LOG_DIR.glob("*.log"), key=mtime, reverse=True)
    skip = {config.PROXY_LOG.name, config.WHISPER_LOG.name,
            config.COMFY_LOG.name, config.TTS_LOG.name, "deck.log"}
    for p in logs:
        if p.name in skip:
            continue
        if skip_vllm and is_vllm_log(p):
            continue
        return str(p)
    return None
