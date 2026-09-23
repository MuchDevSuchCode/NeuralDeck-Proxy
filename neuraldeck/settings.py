"""The editable settings surface behind the Settings page.

One schema, declared here, drives everything: what the page renders, how a
submitted value is validated, and what gets written to config.json. Adding
a setting means adding a row here and nothing else.

Three things the page has to be honest about, and the schema carries all
three:

  * a setting fixed in the environment cannot be changed by writing the
    file, so it is reported as locked rather than silently ignored;
  * some settings take effect the moment the file is re-read, others were
    captured when the process started (a bound port, a deque's length), so
    each row says which it is;
  * a default is not a stored value — clearing a field means "go back to
    the default", not "set it to empty".
"""

import json
import os
import shutil
from pathlib import Path

from . import config

SPEC_CHOICES = ["auto", "ngram", "off"]
THINK_CHOICES = ["off", "low", "medium", "high"]


def _f(key, label, type_, group, help_="", restart=False, choices=None,
       min_=None, max_=None, placeholder=""):
    return {"key": key, "label": label, "type": type_, "group": group,
            "help": help_, "restart": restart, "choices": choices,
            "min": min_, "max": max_, "placeholder": placeholder}


# type: str | int | float | bool | choice | dir | file | dirlist | strlist | map
SCHEMA = [
    # ── Models ────────────────────────────────────────────────────────────
    _f("model_dirs", "Model directories", "dirlist", "Models",
       "Folders scanned for GGUF models. A model is a subfolder holding a "
       ".gguf, plus any mmproj or draft head beside it."),
    _f("download_dir", "Download directory", "dir", "Models",
       "Where the Hugging Face downloader puts new models. Usually the "
       "first model directory."),

    # ── Backends ──────────────────────────────────────────────────────────
    _f("backends", "llama-server builds", "map", "Backends",
       "Label → path to a llama-server binary. Add several to A/B a fork "
       "against upstream from the launch form."),
    _f("default_backend", "Default build", "str", "Backends",
       "Which label the launch form starts on."),
    _f("llama_port_range", "llama port range", "str", "Backends",
       "Ports the deck launches into and the proxy discovers, e.g. 8081-8089.",
       restart=True, placeholder="8081-8089"),
    _f("llama_host", "llama bind address", "str", "Backends",
       "What launched instances bind to. 127.0.0.1 keeps them off the network.",
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
       "127.0.0.1 keeps it on this machine.", restart=True),
    _f("proxy_port", "Proxy port", "int", "Ports",
       "The client-facing endpoint your apps point at.", restart=True,
       min_=1, max_=65535),
    _f("proxy_host", "Proxy bind address", "str", "Ports", "", restart=True),

    # ── Speech, vision and video ──────────────────────────────────────────
    _f("whisper_bin", "whisper-server binary", "file", "Speech & video",
       "whisper.cpp's server, for speech input. Optional."),
    _f("whisper_model", "Whisper model", "file", "Speech & video",
       "A ggml-*.bin model file."),
    _f("whisper_port", "Whisper port", "int", "Speech & video", "",
       restart=True, min_=1, max_=65535),
    _f("ffmpeg", "ffmpeg", "file", "Speech & video",
       "Needed only for video input. Without it, video requests fail and "
       "everything else works."),
    _f("video_fps", "Video frame rate", "float", "Speech & video",
       "Frames per second extracted from video for vision models.",
       min_=0.1, max_=30),
    _f("max_frames", "Max frames per video", "int", "Speech & video", "",
       min_=1, max_=500),
    _f("tts_endpoint", "TTS endpoint", "str", "Speech & video",
       "An OpenAI-compatible /v1/audio/speech backend.", restart=True),
    _f("tts_default_voice", "Default voice", "str", "Speech & video",
       "Filled in when a client does not name one."),

    # ── Proxy behaviour ───────────────────────────────────────────────────
    _f("strict_model_routing", "Strict model routing", "bool", "Proxy",
       "On: a request naming a model nothing is serving is an error. Off: "
       "it is answered by the default instance, which misattributes "
       "benchmarks.", restart=True),
    _f("llama_timeout", "LLM timeout (s)", "float", "Proxy", "",
       restart=True, min_=1, max_=7200),
    _f("whisper_timeout", "Whisper timeout (s)", "float", "Proxy", "",
       restart=True, min_=1, max_=7200),

    # ── Dashboard ─────────────────────────────────────────────────────────
    _f("peak_bw_gbs", "Peak memory bandwidth (GB/s)", "float", "Dashboard",
       "Used by the roofline panel to estimate a decode ceiling. Set it to "
       "your machine's real figure or the panel means nothing.",
       min_=1, max_=100000),
    _f("vram_total_gb", "VRAM total override (GiB)", "float", "Dashboard",
       "Only used when no driver reports a total. 0 leaves it unknown.",
       min_=0, max_=100000),
    _f("history_len", "History length (samples)", "int", "Dashboard",
       "Points kept in the charts, at one per second.", restart=True,
       min_=60, max_=86400),
    _f("sample_interval", "Sample interval (s)", "float", "Dashboard",
       "How often telemetry is collected. Takes effect immediately.",
       min_=0.2, max_=60),

    # ── Optional services ─────────────────────────────────────────────────
    _f("tts_cmd", "TTS start command", "strlist", "Optional services",
       "Command to start a TTS server, one argument per entry. Leave empty "
       "if you start it yourself.", restart=True),
    _f("tts_port", "TTS port", "int", "Optional services", "", restart=True,
       min_=1, max_=65535),
    _f("comfy_cmd", "ComfyUI start command", "strlist", "Optional services",
       "", restart=True),
    _f("comfy_port", "ComfyUI port", "int", "Optional services", "",
       restart=True, min_=1, max_=65535),
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


def _coerce(field: dict, raw):
    key, t = field["key"], field["type"]
    if raw is None or raw == "" or raw == [] or raw == {}:
        return None                      # cleared: fall back to the default
    try:
        if t == "int":
            v = int(raw)
        elif t == "float":
            v = float(raw)
        elif t == "bool":
            v = raw if isinstance(raw, bool) else \
                str(raw).strip().lower() in ("1", "true", "yes", "on")
        elif t == "choice":
            v = str(raw)
            if field["choices"] and v not in field["choices"]:
                raise Invalid(f"{key}: must be one of "
                              f"{', '.join(field['choices'])}")
        elif t in ("dirlist", "strlist"):
            if isinstance(raw, str):
                raw = [x for x in raw.splitlines() if x.strip()]
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
    except (TypeError, ValueError):
        raise Invalid(f"{key}: '{raw}' is not a valid {t}")
    if t in ("int", "float"):
        if field["min"] is not None and v < field["min"]:
            raise Invalid(f"{key}: must be at least {field['min']}")
        if field["max"] is not None and v > field["max"]:
            raise Invalid(f"{key}: must be at most {field['max']}")
    return v


def _cross_check(pending: dict) -> list:
    """Checks that only make sense once the whole submission is known."""
    warnings = []
    backends = pending.get("backends") or _effective("backends") or {}
    default = pending.get("default_backend") or _effective("default_backend")
    if backends and default and default not in backends:
        raise Invalid(f"default_backend: '{default}' is not one of the "
                      f"configured builds ({', '.join(backends)})")
    rng = pending.get("llama_port_range") or _effective("llama_port_range")
    if rng:
        lo, _, hi = str(rng).partition("-")
        try:
            lo_i, hi_i = int(lo), int(hi or lo)
            if hi_i < lo_i:
                raise ValueError
        except ValueError:
            raise Invalid("llama_port_range: expected something like 8081-8089")
    ports = {"deck_port": pending.get("deck_port") or _effective("deck_port"),
             "proxy_port": pending.get("proxy_port") or _effective("proxy_port")}
    if ports["deck_port"] == ports["proxy_port"]:
        raise Invalid("deck_port and proxy_port cannot be the same")
    for key in ("model_dirs",):
        for path in pending.get(key) or []:
            if not os.path.isdir(path):
                warnings.append(f"{path} does not exist right now — kept anyway")
    for label, binary in (pending.get("backends") or {}).items():
        if not (os.path.isfile(binary) or shutil.which(binary)):
            warnings.append(f"backend '{label}': {binary} not found right now")
    return warnings


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

    warnings = _cross_check(pending)

    stored = config._load_file()
    before = {k: stored.get(k) for k in list(pending) + cleared}
    stored.update(pending)
    for key in cleared:
        stored.pop(key, None)

    config.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.CONFIG_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
    tmp.replace(config.CONFIG_FILE)      # atomic: never a half-written config
    config.reload()

    changed = [k for k in list(pending) + cleared
               if before.get(k) != stored.get(k)]
    needs_restart = sorted({k for k in changed if BY_KEY[k]["restart"]})
    if locked:
        warnings += [f"{k} is fixed by NEURALDECK_{k.upper()} in the "
                     "environment and was not changed" for k in locked]
    return {"saved": sorted(changed), "cleared": sorted(cleared),
            "needs_restart": needs_restart, "warnings": warnings,
            "locked": sorted(locked), "config_file": str(config.CONFIG_FILE)}


# ── directory browsing (there is no native file picker in a browser) ───────

def browse(path: str = None) -> dict:
    """Subdirectories of `path`, with a GGUF count so model folders stand out."""
    if not path:
        path = config.MODEL_DIRS[0] if config.MODEL_DIRS else str(Path.home())
    target = Path(os.path.expanduser(path))
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
    return {"path": str(target),
            "parent": str(target.parent) if target.parent != target else None,
            "dirs": dirs, "ggufs_here": here, "roots": _roots(),
            "sep": os.sep}


def _roots() -> list:
    """Somewhere to start from: drives on Windows, useful anchors elsewhere."""
    if config.IS_WINDOWS:
        import string
        return [f"{d}:{os.sep}" for d in string.ascii_uppercase
                if os.path.exists(f"{d}:{os.sep}")]
    return [p for p in (str(Path.home()), "/mnt", "/media", "/opt", "/")
            if os.path.isdir(p)]
