"""The editable settings surface behind the Settings page.

One schema, declared here, drives everything: what the page renders, how a
submitted value is validated, and what gets written to config.json. Adding
a setting means adding a row here and nothing else.

Three things the page has to be honest about, and the schema carries all
three:

  * a setting fixed in the environment cannot be changed by writing the
    file, so it is reported as locked rather than silently ignored;
  * some settings take effect the moment the file is re-read, others were
    captured when a process started (a bound port, a deque's length), so
    each row says which: `restart` is False (applies at once), True (the
    dashboard needs a restart), "proxy" (read by the proxy process, which
    is restarted for you) or "both";
  * a default is not a stored value — clearing a field means "go back to
    the default", not "set it to empty".
"""

import ipaddress
import json
import math
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

from . import config

SPEC_CHOICES = ["auto", "ngram", "off"]
THINK_CHOICES = ["off", "low", "medium", "high"]
VLLM_KV_CHOICES = ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]


# restart: False  — the dashboard re-reads it on save, applies at once
#          True   — captured when the dashboard started; restart it
#          "proxy"— read by the proxy process at startup; saving restarts it
#          "both" — both of the above
def _f(key, label, type_, group, help_="", restart=False, choices=None,
       min_=None, max_=None, placeholder=""):
    return {"key": key, "label": label, "type": type_, "group": group,
            "help": help_, "restart": restart, "choices": choices,
            "min": min_, "max": max_, "placeholder": placeholder}


# type: str | int | float | bool | choice | dir | file | dirlist | strlist | map
SCHEMA = [
    # ── Models ────────────────────────────────────────────────────────────
    _f("model_dirs", "Model directories", "dirlist", "Models",
       "Folders scanned for models. A model is a subfolder holding a "
       ".gguf (plus any mmproj or draft head beside it), or a Hugging Face "
       "safetensors folder (config.json + *.safetensors) for vLLM."),
    _f("download_dir", "Download directory", "dir", "Models",
       "Where the Hugging Face downloader puts new models. Usually the "
       "first model directory."),

    # ── Backends ──────────────────────────────────────────────────────────
    _f("backends", "llama-server builds", "map", "Backends",
       "Label → path to a llama-server binary. Add several to A/B a fork "
       "against upstream from the launch form. A vLLM executable (the "
       "`vllm` script in its environment's bin folder) works too, and "
       "serves safetensors models."),
    _f("default_backend", "Default build", "str", "Backends",
       "Which label the launch form starts on."),
    _f("llama_port_range", "llama port range", "str", "Backends",
       "Ports the deck launches into and the proxy discovers, e.g. 8081-8089 "
       f"(at most {config.MAX_LLAMA_PORTS}). Saving restarts the proxy; "
       "running instances keep their ports.",
       restart="proxy", placeholder="8081-8089"),
    _f("llama_host", "llama bind address", "str", "Backends",
       "What launched instances bind to. 127.0.0.1 keeps them off the "
       "network. Applies to the next launch.",
       placeholder="0.0.0.0"),

    # ── Launch defaults ───────────────────────────────────────────────────
    _f("ctx", "Context size", "int", "Launch defaults",
       "Total KV budget for a launch. llama.cpp splits this across slots.",
       min_=1024, max_=1048576),
    _f("slots", "Slots", "int", "Launch defaults",
       "Parallel requests. Each slot gets ctx ÷ slots tokens.", min_=1, max_=64),
    _f("spec", "Speculation", "choice", "Launch defaults",
       "auto uses a model's MTP head when it has one; ngram drafts from "
       "context and needs no draft model.", choices=SPEC_CHOICES),
    _f("thinking", "Thinking", "choice", "Launch defaults",
       "Reasoning budget for models whose template supports it.",
       choices=THINK_CHOICES),
    _f("kv_cache_type", "KV cache type", "str", "Launch defaults",
       "-ctk/-ctv value, e.g. f16, q8_0, q4_0. Smaller trades quality for "
       "context length.", placeholder="q8_0"),
    _f("flash_attn", "Flash attention", "choice", "Launch defaults",
       "Passed as -fa on builds that take a value.",
       choices=["on", "off", "auto"]),
    _f("n_gpu_layers", "GPU layers", "int", "Launch defaults",
       "-ngl. 999 offloads everything that fits.", min_=0, max_=999),
    _f("threads", "CPU threads", "int", "Launch defaults",
       "0 uses the physical core count.", min_=0, max_=512),
    _f("extra_llama_args", "Extra llama-server args", "strlist",
       "Launch defaults",
       "Appended to every launch, one argument per entry."),

    # ── vLLM ──────────────────────────────────────────────────────────────
    _f("vllm_kv_cache_dtype", "vLLM KV cache dtype", "choice", "vLLM",
       "--kv-cache-dtype for vLLM launches. auto keeps the model's dtype; "
       "fp8 halves KV memory at a small quality cost.",
       choices=VLLM_KV_CHOICES),
    _f("vllm_gpu_frac_max", "vLLM max GPU memory fraction", "float", "vLLM",
       "Ceiling for --gpu-memory-utilization. Each launch is sized to the "
       "VRAM actually free (so vLLM can run beside llama-servers), never "
       "above this.", min_=0.5, max_=0.98),
    _f("vllm_extra_args", "Extra vLLM args", "strlist", "vLLM",
       "Appended to every `vllm serve`, one argument per entry, e.g. "
       "--max-num-batched-tokens then 2048."),

    # ── Sampling ──────────────────────────────────────────────────────────
    _f("temp", "Temperature", "float", "Sampling",
       "Launch-time default; a request can still override it.", min_=0, max_=5),
    _f("top_p", "Top-p", "float", "Sampling", "", min_=0, max_=1),
    _f("min_p", "Min-p", "float", "Sampling", "", min_=0, max_=1),
    _f("repeat_penalty", "Repeat penalty", "float", "Sampling", "", min_=0, max_=5),

    # ── Ports ─────────────────────────────────────────────────────────────
    _f("deck_port", "Dashboard port", "int", "Ports",
       "This page. Changing it needs a restart.", restart=True,
       min_=1, max_=65535),
    _f("deck_host", "Dashboard bind address", "str", "Ports",
       "0.0.0.0 serves the whole network with no authentication; "
       "127.0.0.1 keeps it on this machine. Changing it needs a restart.",
       restart=True),
    _f("proxy_port", "Proxy port", "int", "Ports",
       "The client-facing endpoint your apps point at. Saving restarts the "
       "proxy on the new port.", restart="proxy", min_=1, max_=65535),
    _f("proxy_host", "Proxy bind address", "str", "Ports",
       "Saving restarts the proxy.", restart="proxy"),

    # ── Speech, vision and video ──────────────────────────────────────────
    _f("whisper_bin", "whisper-server binary", "file", "Speech & video",
       "whisper.cpp's server, for speech input. Optional. Used the next "
       "time whisper-server is started."),
    _f("whisper_model", "Whisper model", "file", "Speech & video",
       "A ggml-*.bin model file. Used the next time whisper-server is "
       "started."),
    _f("whisper_port", "Whisper port", "int", "Speech & video",
       "Saving restarts the proxy; restart whisper-server from the "
       "dashboard to move it.", restart="proxy", min_=1, max_=65535),
    _f("ffmpeg", "ffmpeg", "file", "Speech & video",
       "Needed only for video input. Without it, video requests fail and "
       "everything else works. Read by the proxy.", restart="proxy"),
    _f("video_fps", "Video frame rate", "float", "Speech & video",
       "Frames per second extracted from video for vision models.",
       restart="proxy", min_=0.1, max_=30),
    _f("max_frames", "Max frames per video", "int", "Speech & video", "",
       restart="proxy", min_=1, max_=500),
    _f("tts_endpoint", "TTS endpoint", "str", "Speech & video",
       "An OpenAI-compatible /v1/audio/speech backend.", restart="proxy"),
    _f("tts_default_voice", "Default voice", "str", "Speech & video",
       "Filled in when a client does not name one.", restart="proxy"),

    # ── Proxy behaviour ───────────────────────────────────────────────────
    _f("strict_model_routing", "Strict model routing", "bool", "Proxy",
       "On: a request naming a model nothing is serving is an error. Off: "
       "it is answered by the default instance, which misattributes "
       "benchmarks.", restart="proxy"),
    _f("llama_timeout", "LLM timeout (s)", "float", "Proxy", "",
       restart="proxy", min_=1, max_=7200),
    _f("whisper_timeout", "Whisper timeout (s)", "float", "Proxy", "",
       restart="proxy", min_=1, max_=7200),

    # ── Dashboard ─────────────────────────────────────────────────────────
    _f("peak_bw_gbs", "Peak memory bandwidth (GB/s)", "float", "Dashboard",
       "Used by the roofline panel to estimate a decode ceiling. The "
       "default (89.6) is a dual-channel DDR5 figure: set it to your GPU's "
       "or APU's real memory bandwidth or the panel means nothing.",
       min_=1, max_=100000),
    _f("vram_total_gb", "VRAM total override (GiB)", "float", "Dashboard",
       "Only used when no driver reports a total. 0 leaves it unknown.",
       min_=0, max_=100000),
    _f("history_len", "History length (samples)", "int", "Dashboard",
       "Points kept in the charts, one per sample interval. Changing it "
       "needs a restart.", restart=True,
       min_=60, max_=86400),
    _f("sample_interval", "Sample interval (s)", "float", "Dashboard",
       "How often telemetry is collected. Takes effect immediately.",
       min_=0.2, max_=60),

    # ── Optional services ─────────────────────────────────────────────────
    _f("tts_cmd", "TTS start command", "strlist", "Optional services",
       "Command to start a TTS server, one argument per entry. Leave empty "
       "if you start it yourself. Used the next time it is started."),
    _f("tts_port", "TTS port", "int", "Optional services",
       "Where the dashboard looks for the TTS server, and the proxy's TTS "
       "endpoint unless one is set. Saving restarts the proxy.",
       restart="proxy", min_=1, max_=65535),
    _f("comfy_cmd", "ComfyUI start command", "strlist", "Optional services",
       "One argument per entry. Used the next time it is started."),
    _f("comfy_port", "ComfyUI port", "int", "Optional services", "",
       min_=1, max_=65535),
]

BY_KEY = {f["key"]: f for f in SCHEMA}
GROUPS = list(dict.fromkeys(f["group"] for f in SCHEMA))

# config.py's attribute name for each key, so the schema can show what is
# actually in force right now.
ATTR = {
    "model_dirs": "MODEL_DIRS", "download_dir": "MODELS_DOWNLOAD_DIR",
    "backends": "BACKENDS", "default_backend": "DEFAULT_BACKEND",
    "llama_port_range": "LLAMA_PORT_RANGE", "llama_host": "LLAMA_HOST",
    "ctx": "CTX_DEFAULT", "slots": "SLOTS_DEFAULT", "spec": "SPEC_DEFAULT",
    "thinking": "THINKING_DEFAULT", "kv_cache_type": "KV_CACHE_TYPE",
    "flash_attn": "FLASH_ATTN", "n_gpu_layers": "N_GPU_LAYERS",
    "threads": "THREADS", "extra_llama_args": "EXTRA_LLAMA_ARGS",
    "vllm_kv_cache_dtype": "VLLM_KV_CACHE_DTYPE",
    "vllm_gpu_frac_max": "VLLM_GPU_FRAC_MAX", "vllm_extra_args": "VLLM_EXTRA_ARGS",
    "temp": "TEMP", "top_p": "TOP_P", "min_p": "MIN_P",
    "repeat_penalty": "REPEAT_PENALTY",
    "deck_port": "DECK_PORT", "deck_host": "DECK_HOST",
    "proxy_port": "PROXY_PORT", "proxy_host": "PROXY_HOST",
    "whisper_bin": "WHISPER_BIN", "whisper_model": "WHISPER_MODEL",
    "whisper_port": "WHISPER_PORT", "ffmpeg": "FFMPEG",
    "video_fps": "VIDEO_FPS", "max_frames": "MAX_FRAMES",
    "tts_endpoint": "TTS_ENDPOINT", "tts_default_voice": "TTS_DEFAULT_VOICE",
    "strict_model_routing": "STRICT_MODEL_ROUTING",
    "llama_timeout": "LLAMA_TIMEOUT", "whisper_timeout": "WHISPER_TIMEOUT",
    "peak_bw_gbs": "PEAK_BW_GBS", "vram_total_gb": "VRAM_TOTAL_GB_FALLBACK",
    "history_len": "HISTORY_LEN", "sample_interval": "SAMPLE_INTERVAL",
    "tts_cmd": "TTS_CMD", "tts_port": "TTS_PORT",
    "comfy_cmd": "COMFY_CMD", "comfy_port": "COMFY_PORT",
}


# ── reading ────────────────────────────────────────────────────────────────

def describe() -> dict:
    """The whole settings surface: schema, stored values, effective values."""
    fields = []
    for f in SCHEMA:
        key = f["key"]
        env = config.env_override(key)
        fields.append({
            **f,
            # what the file holds (None = not set, i.e. using the default)
            "value": config.from_file(key),
            # what is actually in force, whatever the source
            "effective": _effective(key),
            "env": env,
            "locked": env is not None,
            "exists": _exists(f, _effective(key)),
        })
    return {
        "groups": GROUPS,
        "fields": fields,
        "config_file": str(config.CONFIG_FILE),
        "data_dir": str(config.DATA_DIR),
        "log_dir": str(config.LOG_DIR),
        "bench_file": str(config.BENCH_FILE),
        "platform": "windows" if config.IS_WINDOWS else "posix",
        "path_sep": os.sep,
        # why config.json is not in use (unparseable, wrong shape), or None;
        # saving is refused until it is fixed, so the page must show this
        "config_error": config.FILE_ERROR,
    }


def _effective(key: str):
    attr = ATTR.get(key)
    val = getattr(config, attr, None) if attr else None
    return list(val) if isinstance(val, (list, tuple)) else \
        dict(val) if isinstance(val, dict) else val


def _exists(field: dict, value) -> dict:
    """Whether the paths a setting names are actually there.

    Reported rather than enforced: a model directory on a drive that is not
    mounted right now is still the setting you meant.
    """
    t = field["type"]
    if t == "dirlist":
        return {p: os.path.isdir(p) for p in (value or [])}
    if t == "dir":
        return {value: os.path.isdir(value)} if value else {}
    if t == "file":
        return {value: bool(value and (os.path.isfile(value)
                                       or shutil.which(str(value))))} if value else {}
    if t == "map":
        return {p: bool(p and (os.path.isfile(p) or shutil.which(str(p))))
                for p in (value or {}).values()}
    return {}


# ── writing ────────────────────────────────────────────────────────────────

class Invalid(ValueError):
    """A submitted value the schema will not accept."""


HOST_KEYS = ("deck_host", "proxy_host", "llama_host")
PORT_KEYS = ("deck_port", "proxy_port", "whisper_port", "tts_port", "comfy_port")
_LABEL = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")


def _valid_host(h: str) -> bool:
    """An IP literal (0.0.0.0, ::, 127.0.0.1 …) or a bare hostname — no
    scheme, port, brackets or spaces, since it goes straight to bind()."""
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        pass
    if len(h) > 253:
        return False
    labels = h.rstrip(".").split(".")
    if all(x.isdigit() for x in labels):     # 999.1.1.1 is not a hostname
        return False
    return all(_LABEL.match(x) for x in labels)


def _coerce(field: dict, raw):
    key, t = field["key"], field["type"]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None                      # cleared: fall back to the default
    try:
        if t == "int":
            if isinstance(raw, bool):
                raise ValueError
            num = raw if isinstance(raw, int) else float(str(raw).strip())
            if isinstance(num, float):
                if not math.isfinite(num) or not num.is_integer():
                    raise Invalid(f"{key}: must be a whole number")
            v = int(num)
        elif t == "float":
            if isinstance(raw, bool):
                raise ValueError
            v = float(str(raw).strip()) if isinstance(raw, str) else float(raw)
            if not math.isfinite(v):
                raise Invalid(f"{key}: must be a finite number")
        elif t == "bool":
            if isinstance(raw, bool):
                v = raw
            else:
                word = str(raw).strip().lower()
                if word in ("1", "true", "yes", "on"):
                    v = True
                elif word in ("0", "false", "no", "off"):
                    v = False
                else:
                    raise ValueError
        elif t == "choice":
            v = str(raw).strip()
            if field["choices"] and v not in field["choices"]:
                raise Invalid(f"{key}: must be one of "
                              f"{', '.join(field['choices'])}")
        elif t in ("dirlist", "strlist"):
            if isinstance(raw, str):
                raw = raw.splitlines()
            if not isinstance(raw, list):
                raise Invalid(f"{key}: expected a list")
            v = [str(x).strip() for x in raw if str(x).strip()]
            if t == "dirlist":
                v = [os.path.expanduser(x) for x in v]
        elif t == "map":
            if not isinstance(raw, dict):
                raise Invalid(f"{key}: expected an object of label → path")
            v = {str(k).strip(): os.path.expanduser(str(x).strip())
                 for k, x in raw.items() if str(k).strip() and str(x).strip()}
        elif t in ("dir", "file"):
            v = os.path.expanduser(str(raw).strip())
        else:
            v = str(raw).strip()
    except Invalid:
        raise
    except (TypeError, ValueError, OverflowError):
        raise Invalid(f"{key}: '{raw}' is not a valid {t}")
    # whitespace-only entries strip to nothing: that is a clear, not a value
    # (an empty model_dirs would leave the deck with nowhere to look)
    if v == [] or v == {} or v == "":
        return None
    if t in ("int", "float"):
        if field["min"] is not None and v < field["min"]:
            raise Invalid(f"{key}: must be at least {field['min']}")
        if field["max"] is not None and v > field["max"]:
            raise Invalid(f"{key}: must be at most {field['max']}")
    if key in HOST_KEYS and not _valid_host(v):
        raise Invalid(f"{key}: '{v}' is not an IP address or hostname "
                      "(no scheme, port or spaces — e.g. 0.0.0.0 or 127.0.0.1)")
    if key == "llama_port_range":
        bounds = config.parse_port_range(v)
        if bounds is None:
            raise Invalid("llama_port_range: expected something like 8081-8089, "
                          f"ports 1-65535, at most {config.MAX_LLAMA_PORTS} of them")
        v = f"{bounds[0]}-{bounds[1]}"   # stored the right way round
    return v


def _cross_check(pending: dict, cleared: list, merged: dict,
                 computed: dict) -> list:
    """Checks that only make sense once the whole submission is known.

    `computed` is config as it would be after the save, so a cleared key is
    checked at its default and an env-fixed key at its env value — not at
    whatever happens to be in force right now. Conflicts that involve no
    submitted key are warned about rather than refused, since the page may
    not be able to fix them (an environment variable, say).
    """
    warnings = []
    touched = set(pending) | set(cleared)

    backends = computed["BACKENDS"]
    default = config._get("default_backend", None, merged)
    if default is not None and str(default) not in backends:
        msg = (f"default_backend: '{default}' is not one of the configured "
               f"builds ({', '.join(backends)})")
        if touched & {"backends", "default_backend"}:
            raise Invalid(msg)
        warnings.append(msg + f" — using '{computed['DEFAULT_BACKEND']}'")

    ports = {"deck_port": computed["DECK_PORT"],
             "proxy_port": computed["PROXY_PORT"],
             "whisper_port": computed["WHISPER_PORT"],
             "tts_port": computed["TTS_PORT"],
             "comfy_port": computed["COMFY_PORT"]}
    llama = computed["LLAMA_PORTS"]
    problems = []
    keys = list(ports)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if ports[a] == ports[b]:
                problems.append(({a, b}, f"{a} and {b} are both {ports[a]}"))
        if ports[a] in llama:
            problems.append(({a, "llama_port_range"},
                             f"{a} {ports[a]} is inside llama_port_range "
                             f"{computed['LLAMA_PORT_RANGE']}"))
    for involved, msg in problems:
        if involved & touched:
            raise Invalid(msg)
        warnings.append(msg)

    for key in ("model_dirs",):
        for path in pending.get(key) or []:
            if not os.path.isdir(path):
                warnings.append(f"{path} does not exist right now — kept anyway")
    for label, binary in (pending.get("backends") or {}).items():
        if not (os.path.isfile(binary) or shutil.which(binary)):
            warnings.append(f"backend '{label}': {binary} not found right now")
    return warnings


# One save at a time: each one reads, merges and replaces the whole file.
_save_lock = threading.Lock()


def _write_atomic(data: dict) -> None:
    """Replace config.json in one step via a uniquely named temp file beside
    it — never a half-written config, and two writers never share a temp."""
    target = config.CONFIG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".config.",
                               suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        # mkstemp creates 0600; keep the mode the file already had
        mode = os.stat(target).st_mode & 0o777 if target.exists() else 0o644
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save(submitted: dict) -> dict:
    """Validate, merge into config.json, and re-read it.

    Only keys present in the submission are touched, so a page that renders
    one group can save just that group. A key submitted as empty is removed
    from the file, which restores its default.
    """
    if not isinstance(submitted, dict):
        raise Invalid("expected an object of settings")
    unknown = [k for k in submitted if k not in BY_KEY]
    if unknown:
        raise Invalid(f"unknown setting(s): {', '.join(sorted(unknown))}")

    pending, cleared, locked = {}, [], []
    for key, raw in submitted.items():
        field = BY_KEY[key]
        if config.env_override(key) is not None:
            locked.append(key)
            continue
        value = _coerce(field, raw)
        if value is None:
            cleared.append(key)
        else:
            pending[key] = value

    with _save_lock:
        stored, error = config._read_file()
        if error:
            # never overwrite a file we could not read: it may hold settings
            # the user wrote by hand, and saving would silently drop them
            raise Invalid(f"{error}. Fix or remove that file, then save "
                          "again — nothing was written.")
        before = {k: stored.get(k) for k in list(pending) + cleared}
        merged = dict(stored)
        merged.update(pending)
        for key in cleared:
            merged.pop(key, None)

        # prove the result still loads before it replaces the file
        try:
            computed = config._compute(merged, make_dirs=False)
        except Exception as e:
            raise Invalid(f"these settings would not load ({e}); "
                          "nothing was written")
        warnings = _cross_check(pending, cleared, merged, computed)

        _write_atomic(merged)
        config.reload()

    changed = [k for k in list(pending) + cleared
               if before.get(k) != merged.get(k)]
    needs_restart = sorted(k for k in changed
                           if BY_KEY[k]["restart"] in (True, "both"))
    proxy_restart = sorted(k for k in changed
                           if BY_KEY[k]["restart"] in ("proxy", "both"))
    if locked:
        warnings += [f"{k} is fixed by NEURALDECK_{k.upper()} in the "
                     "environment and was not changed" for k in locked]
    return {"saved": sorted(changed), "cleared": sorted(cleared),
            "needs_restart": needs_restart, "proxy_restart": proxy_restart,
            "warnings": warnings, "locked": sorted(locked),
            "config_file": str(config.CONFIG_FILE)}


# ── directory browsing (there is no native file picker in a browser) ───────

def browse(path: str = None, mode: str = "dir", ext: str = None) -> dict:
    """Subdirectories of `path`, with a GGUF count so model folders stand out.

    mode="file" also lists the regular files there — for picking a binary
    or a model file — optionally only those whose extension is in `ext`
    (comma-separated, e.g. ".bin,.gguf"; case does not matter).
    """
    if mode not in ("dir", "file"):
        raise Invalid("mode must be 'dir' or 'file'")
    if not path:
        path = config.MODEL_DIRS[0] if config.MODEL_DIRS else str(Path.home())
    target = Path(os.path.expanduser(path))
    if target.is_file():                 # a file path: open its folder
        target = target.parent
    if not target.is_dir():
        target = Path.home()
    dirs = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            try:
                ggufs = sum(1 for f in child.iterdir()
                            if f.is_file() and f.suffix.lower() == ".gguf")
            except OSError:
                ggufs = 0
            dirs.append({"name": child.name, "path": str(child), "ggufs": ggufs})
    except OSError as e:
        raise Invalid(f"cannot read {target}: {e}")
    here = 0
    try:
        here = sum(1 for f in target.iterdir()
                   if f.is_file() and f.suffix.lower() == ".gguf")
    except OSError:
        pass
    out = {"path": str(target),
           "parent": str(target.parent) if target.parent != target else None,
           "dirs": dirs, "ggufs_here": here, "roots": _roots(),
           "sep": os.sep}
    if mode == "file":
        exts = {("." + e.strip().lstrip(".")).lower()
                for e in (ext or "").split(",") if e.strip().lstrip(".")}
        files = []
        try:
            for child in target.iterdir():
                if child.name.startswith("."):
                    continue
                try:
                    if not child.is_file():
                        continue
                    if exts and child.suffix.lower() not in exts:
                        continue
                    size = child.stat().st_size
                except OSError:
                    continue
                files.append({"name": child.name, "path": str(child),
                              "size": size})
        except OSError:
            pass
        files.sort(key=lambda f: f["name"].lower())
        out["files"] = files
    return out


def _roots() -> list:
    """Somewhere to start from: drives on Windows, useful anchors elsewhere."""
    if config.IS_WINDOWS:
        import string
        return [f"{d}:{os.sep}" for d in string.ascii_uppercase
                if os.path.exists(f"{d}:{os.sep}")]
    return [p for p in (str(Path.home()), "/mnt", "/media", "/opt", "/")
            if os.path.isdir(p)]
