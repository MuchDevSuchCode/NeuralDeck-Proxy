"""Resolved configuration for this machine.

Every setting has three sources, in falling precedence:

  1. environment, as ``NEURALDECK_<KEY>``
  2. ``config.json`` in the data directory (keys lowercase, no prefix)
  3. the defaults below

The data directory holds everything the deck writes — instance logs, the
benchmark history, the pid files — and defaults to the platform's own
per-user location so a Windows install never writes into Program Files.
"""

import json
import math
import os
import shlex
import shutil
import sys
from pathlib import Path

IS_WINDOWS = os.name == "nt"
EXE = ".exe" if IS_WINDOWS else ""


def _env(key: str):
    """NEURALDECK_<KEY>, or None. An empty variable counts as unset, so
    `export NEURALDECK_MODEL_DIRS=` does not mean "no model directories"."""
    raw = os.environ.get("NEURALDECK_" + key.upper())
    return raw if raw is not None and raw.strip() != "" else None


def _default_data_dir() -> Path:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "NeuralDeck"
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "neuraldeck"


DATA_DIR = Path(os.path.expanduser(_env("home") or str(_default_data_dir())))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = DATA_DIR / "config.json"

# Why config.json is not being used, or None when it is fine (or absent).
# A broken file is reported, never silently replaced: the settings page
# shows this and refuses to save over it.
FILE_ERROR = None


def _read_file():
    """(settings, error) from config.json. A missing or blank file is {}."""
    try:
        # utf-8-sig: Notepad's "UTF-8" writes a BOM that json.loads rejects
        text = CONFIG_FILE.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}, None
    except (OSError, UnicodeDecodeError) as e:
        return {}, f"cannot read {CONFIG_FILE}: {e}"
    if not text.strip():
        return {}, None
    try:
        data = json.loads(text)
    except ValueError as e:
        return {}, f"{CONFIG_FILE} is not valid JSON ({e})"
    if not isinstance(data, dict):
        return {}, f"{CONFIG_FILE} must hold a JSON object ({{...}})"
    return data, None


def _load_file() -> dict:
    global FILE_ERROR
    data, FILE_ERROR = _read_file()
    return data


_FILE: dict = _load_file()


def from_file(key: str):
    """The stored value for a key, or None if the file does not set it."""
    return _FILE.get(key.lower())


def env_override(key: str):
    """The environment's value for a key, if it sets one.

    A setting fixed in the environment cannot be changed by writing the
    config file, and the settings page has to say so rather than pretend
    the save took effect.
    """
    return _env(key)


def _get(key: str, default, src: dict):
    raw = _env(key)
    if raw is not None:
        return raw
    if key.lower() in src:
        return src[key.lower()]
    return default


def get(key: str, default=None):
    """One setting, env first, then config.json, then the given default."""
    return _get(key, default, _FILE)


# Coercions from a raw setting (env string or JSON value) to its type; each
# falls back to the default rather than raising, so a bad value cannot stop
# the process from starting.

def _as_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v, default):
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _as_bool(v, default):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_path(v) -> Path:
    if not isinstance(v, (str, os.PathLike)):   # {} or 5 would become a dir
        raise TypeError(f"expected a path, got {type(v).__name__}")
    return Path(os.path.expanduser(str(v)))


def _as_list(v, default):
    """A list of paths: JSON array in config.json, os.pathsep-joined in env."""
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return [p for p in str(v).split(os.pathsep) if p.strip()]


def _as_cmd(v, default):
    """A command line: JSON array, or a string split the way a shell would
    (so "python -m tts --port 8004" is four arguments, not one)."""
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    try:
        parts = shlex.split(str(v), posix=not IS_WINDOWS)
    except ValueError:                  # unbalanced quote
        parts = str(v).split()
    if IS_WINDOWS:                      # non-posix mode keeps the quotes
        parts = [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in "\"'" else p
                 for p in parts]
    return [p for p in parts if p]


def _as_dict(v, default):
    """A mapping setting: JSON object, or "k=v,k=v" in the environment."""
    if v is None:
        return dict(default)
    if isinstance(v, dict):
        return {str(k): str(x) for k, x in v.items()}
    out = {}
    for part in str(v).split(","):
        k, _, val = part.partition("=")
        if k.strip() and val.strip():
            out[k.strip()] = val.strip()
    return out


MAX_LLAMA_PORTS = 64


def parse_port_range(spec):
    """(lo, hi) from "8081-8089" or "8081", reversed bounds swapped; None if
    it is not a range of 1..MAX_LLAMA_PORTS valid ports."""
    lo, _, hi = str(spec).strip().partition("-")
    try:
        lo_i, hi_i = int(lo), int(hi or lo)
    except ValueError:
        return None
    lo_i, hi_i = min(lo_i, hi_i), max(lo_i, hi_i)
    if lo_i < 1 or hi_i > 65535 or hi_i - lo_i + 1 > MAX_LLAMA_PORTS:
        return None
    return lo_i, hi_i


# ── derived settings ───────────────────────────────────────────────────────
# Computed in one function so the settings page can write config.json and
# have the running process pick the change up without a restart. Anything
# captured at import time by another module (a port already bound, the
# history deque's length) still needs one — settings.py marks those.

def _compute(src: dict = None, make_dirs: bool = True) -> dict:
    """Every derived setting from `src` (default: the loaded config.json).

    settings.py calls this on a merged-but-unsaved dict to prove a save
    would still load before it replaces the file.
    """
    src = _FILE if src is None else src

    def get(key, default=None):
        return _get(key, default, src)

    def _int(key, default):
        return _as_int(get(key, default), default)

    def _float(key, default):
        return _as_float(get(key, default), default)

    def _bool(key, default):
        return _as_bool(get(key, default), default)

    def _list(key, default):
        return _as_list(get(key), default)

    def _cmd(key, default):
        return _as_cmd(get(key), default)

    def _dict(key, default):
        return _as_dict(get(key), default)

    # ── where things live ──────────────────────────────────────────────────────
    LOG_DIR = _as_path(get("log_dir", DATA_DIR / "logs"))
    RUN_DIR = DATA_DIR / "run"
    if make_dirs:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_FILE = _as_path(get("bench_file", DATA_DIR / "bench.jsonl"))
    LAST_MODEL_FILE = DATA_DIR / "last-model.txt"

    PROXY_LOG = _as_path(get("proxy_log", LOG_DIR / "proxy.log"))
    WHISPER_LOG = _as_path(get("whisper_log", LOG_DIR / "whisper.log"))
    COMFY_LOG = _as_path(get("comfy_log", LOG_DIR / "comfy.log"))
    TTS_LOG = _as_path(get("tts_log", LOG_DIR / "tts.log"))


    def _default_model_dirs() -> list:
        dirs = [str(Path.home() / "models")]
        if not IS_WINDOWS:
            dirs.append("/mnt/models")
        return dirs


    MODEL_DIRS = [os.path.expanduser(d) for d in _list("model_dirs", _default_model_dirs())]

    # ── ports ──────────────────────────────────────────────────────────────────
    # The proxy is the client-facing endpoint; llama-server instances sit behind
    # it. The deck (this dashboard) is a third, separate port.
    DECK_HOST = str(get("deck_host", "0.0.0.0"))
    DECK_PORT = _int("deck_port", 8770)
    PROXY_HOST = str(get("proxy_host", "0.0.0.0"))
    PROXY_PORT = _int("proxy_port", 8080)
    LLAMA_HOST = str(get("llama_host", "0.0.0.0"))
    LLAMA_PORT_RANGE = str(get("llama_port_range", "8081-8089")).strip()
    WHISPER_PORT = _int("whisper_port", 8090)
    TTS_PORT = _int("tts_port", 8004)
    COMFY_PORT = _int("comfy_port", 8188)


    # A reversed range is swapped; one that is not a range at all falls back
    # to the default rather than leaving the deck with no port to launch on.
    bounds = parse_port_range(LLAMA_PORT_RANGE) or (8081, 8089)
    LLAMA_PORTS = list(range(bounds[0], bounds[1] + 1))
    LLAMA_PORT_RANGE = f"{bounds[0]}-{bounds[1]}"
    # The dashboard's KV-cache chart has four series; give it the first four.
    KV_CHART_PORTS = LLAMA_PORTS[:4]

    # ── binaries ───────────────────────────────────────────────────────────────
    def _find_llama() -> str:
        """llama-server from config, PATH, or the usual build locations."""
        explicit = get("llama_bin", None)
        if explicit:
            return os.path.expanduser(str(explicit))
        found = shutil.which("llama-server")
        if found:
            return found
        home = Path.home()
        for cand in (
            home / "llama.cpp" / "build" / "bin" / f"llama-server{EXE}",
            home / "llama.cpp" / "build" / "Release" / f"llama-server{EXE}",
            Path("C:/llama.cpp/build/bin/Release/llama-server.exe"),
            Path("/usr/local/bin/llama-server"),
        ):
            if cand.exists():
                return str(cand)
        return f"llama-server{EXE}"


    # label -> llama-server binary. One entry is the common case; add more to
    # A/B a fork against upstream from the dashboard's backend picker.
    BACKENDS = _dict("backends", {"llama.cpp": _find_llama()})
    if not BACKENDS:                        # an empty override would break launch
        BACKENDS = {"llama.cpp": _find_llama()}
    DEFAULT_BACKEND = str(get("default_backend", next(iter(BACKENDS))))
    if DEFAULT_BACKEND not in BACKENDS:     # a label that was since removed
        DEFAULT_BACKEND = next(iter(BACKENDS))

    WHISPER_BIN = os.path.expanduser(str(get(
        "whisper_bin", shutil.which("whisper-server")
        or Path.home() / "whisper.cpp" / "build" / "bin" / f"whisper-server{EXE}")))
    WHISPER_MODEL = os.path.expanduser(str(get(
        "whisper_model",
        Path.home() / "whisper.cpp" / "models" / "ggml-large-v3-turbo.bin")))
    FFMPEG = os.path.expanduser(str(get("ffmpeg", shutil.which("ffmpeg")
                                        or f"ffmpeg{EXE}")))

    # Optional extras: a start command each, as a list or a shell-style
    # string. Empty means "not installed here" and the service is hidden.
    COMFY_CMD = _cmd("comfy_cmd", [])
    TTS_CMD = _cmd("tts_cmd", [])

    # ── launch defaults (the dashboard's launch form starts from these) ─────────
    CTX_DEFAULT = _int("ctx", 32768)
    SLOTS_DEFAULT = _int("slots", 1)
    SPEC_DEFAULT = str(get("spec", "auto"))
    THINKING_DEFAULT = str(get("thinking", "off"))
    KV_CACHE_TYPE = str(get("kv_cache_type", "q8_0"))
    TEMP = _float("temp", 0.7)
    TOP_P = _float("top_p", 0.95)
    MIN_P = _float("min_p", 0.05)
    REPEAT_PENALTY = _float("repeat_penalty", 1.05)
    N_GPU_LAYERS = _int("n_gpu_layers", 999)
    FLASH_ATTN = str(get("flash_attn", "on"))
    THREADS = _int("threads", 0)  # 0 = physical core count
    EXTRA_LLAMA_ARGS = _cmd("extra_llama_args", [])

    # ── dashboard tuning ───────────────────────────────────────────────────────
    # Memory bandwidth for the roofline panel. The default is a dual-channel
    # DDR5 figure; set it to your GPU's (or APU's) real peak for the ceiling
    # to mean anything.
    PEAK_BW_GBS = _float("peak_bw_gbs", 89.6)
    # Used only when no driver reports a VRAM total (some Windows setups).
    VRAM_TOTAL_GB_FALLBACK = _float("vram_total_gb", 0.0)
    HISTORY_LEN = _int("history_len", 600)
    SAMPLE_INTERVAL = _float("sample_interval", 1.0)

    # ── proxy backends ─────────────────────────────────────────────────────────
    LLAMA_ENDPOINT = str(get("llama_endpoint",
                             f"http://127.0.0.1:{LLAMA_PORTS[0]}/v1/chat/completions"))
    WHISPER_ENDPOINT = str(get("whisper_endpoint",
                               f"http://127.0.0.1:{WHISPER_PORT}/v1/audio/transcriptions"))
    TTS_ENDPOINT = str(get("tts_endpoint", f"http://127.0.0.1:{TTS_PORT}"))
    COMFY_URL = str(get("comfy_url", f"http://127.0.0.1:{COMFY_PORT}"))
    PROXY_CHAT = str(get("proxy_chat",
                         f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions"))
    STRICT_MODEL_ROUTING = _bool("strict_model_routing", True)
    VIDEO_FPS = _float("video_fps", 1.0)
    MAX_FRAMES = _int("max_frames", 30)
    TTS_DEFAULT_VOICE = str(get("tts_default_voice", ""))
    TTS_DEFAULT_MODEL = str(get("tts_default_model", "tts-1"))
    BACKEND_CONNECT_TIMEOUT = _float("backend_connect_timeout", 5.0)
    WHISPER_TIMEOUT = _float("whisper_timeout", 300.0)
    LLAMA_TIMEOUT = _float("llama_timeout", 600.0)

    # ── misc ───────────────────────────────────────────────────────────────────
    MODELS_DOWNLOAD_DIR = os.path.expanduser(str(get(
        "download_dir", MODEL_DIRS[0] if MODEL_DIRS else str(Path.home()))))

    return {
        'LOG_DIR': LOG_DIR,
        'RUN_DIR': RUN_DIR,
        'BENCH_FILE': BENCH_FILE,
        'LAST_MODEL_FILE': LAST_MODEL_FILE,
        'PROXY_LOG': PROXY_LOG,
        'WHISPER_LOG': WHISPER_LOG,
        'COMFY_LOG': COMFY_LOG,
        'TTS_LOG': TTS_LOG,
        'MODEL_DIRS': MODEL_DIRS,
        'DECK_HOST': DECK_HOST,
        'DECK_PORT': DECK_PORT,
        'PROXY_HOST': PROXY_HOST,
        'PROXY_PORT': PROXY_PORT,
        'LLAMA_HOST': LLAMA_HOST,
        'LLAMA_PORT_RANGE': LLAMA_PORT_RANGE,
        'WHISPER_PORT': WHISPER_PORT,
        'TTS_PORT': TTS_PORT,
        'COMFY_PORT': COMFY_PORT,
        'LLAMA_PORTS': LLAMA_PORTS,
        'KV_CHART_PORTS': KV_CHART_PORTS,
        'BACKENDS': BACKENDS,
        'DEFAULT_BACKEND': DEFAULT_BACKEND,
        'WHISPER_BIN': WHISPER_BIN,
        'WHISPER_MODEL': WHISPER_MODEL,
        'FFMPEG': FFMPEG,
        'COMFY_CMD': COMFY_CMD,
        'TTS_CMD': TTS_CMD,
        'CTX_DEFAULT': CTX_DEFAULT,
        'SLOTS_DEFAULT': SLOTS_DEFAULT,
        'SPEC_DEFAULT': SPEC_DEFAULT,
        'THINKING_DEFAULT': THINKING_DEFAULT,
        'KV_CACHE_TYPE': KV_CACHE_TYPE,
        'TEMP': TEMP,
        'TOP_P': TOP_P,
        'MIN_P': MIN_P,
        'REPEAT_PENALTY': REPEAT_PENALTY,
        'N_GPU_LAYERS': N_GPU_LAYERS,
        'FLASH_ATTN': FLASH_ATTN,
        'THREADS': THREADS,
        'EXTRA_LLAMA_ARGS': EXTRA_LLAMA_ARGS,
        'PEAK_BW_GBS': PEAK_BW_GBS,
        'VRAM_TOTAL_GB_FALLBACK': VRAM_TOTAL_GB_FALLBACK,
        'HISTORY_LEN': HISTORY_LEN,
        'SAMPLE_INTERVAL': SAMPLE_INTERVAL,
        'LLAMA_ENDPOINT': LLAMA_ENDPOINT,
        'WHISPER_ENDPOINT': WHISPER_ENDPOINT,
        'TTS_ENDPOINT': TTS_ENDPOINT,
        'COMFY_URL': COMFY_URL,
        'PROXY_CHAT': PROXY_CHAT,
        'STRICT_MODEL_ROUTING': STRICT_MODEL_ROUTING,
        'VIDEO_FPS': VIDEO_FPS,
        'MAX_FRAMES': MAX_FRAMES,
        'TTS_DEFAULT_VOICE': TTS_DEFAULT_VOICE,
        'TTS_DEFAULT_MODEL': TTS_DEFAULT_MODEL,
        'BACKEND_CONNECT_TIMEOUT': BACKEND_CONNECT_TIMEOUT,
        'WHISPER_TIMEOUT': WHISPER_TIMEOUT,
        'LLAMA_TIMEOUT': LLAMA_TIMEOUT,
        'MODELS_DOWNLOAD_DIR': MODELS_DOWNLOAD_DIR,
    }


def _apply() -> dict:
    """Compute from the loaded file; if its contents cannot be applied
    (a value of the wrong JSON type, say), report that and run on the
    defaults instead of refusing to start."""
    global _FILE, FILE_ERROR
    try:
        values = _compute()
    except Exception as e:
        FILE_ERROR = (f"settings in {CONFIG_FILE} could not be applied ({e}); "
                      "running on the defaults")
        _FILE = {}
        values = _compute()
    globals().update(values)
    return values


_apply()


def reload() -> dict:
    """Re-read config.json and rebind every derived setting."""
    global _FILE
    _FILE = _load_file()
    values = _apply()
    for hook in _on_reload:
        try:
            hook()
        except Exception:
            pass
    return values


# Modules that cache something derived from a setting register here.
_on_reload: list = []


def on_reload(fn):
    _on_reload.append(fn)
    return fn


def summary() -> dict:
    """Everything worth printing in `neuraldeck doctor`."""
    return {
        "platform": f"{sys.platform} ({'windows' if IS_WINDOWS else 'posix'})",
        "python": sys.version.split()[0],
        "data_dir": str(DATA_DIR),
        "config_file": f"{CONFIG_FILE} ({'present' if CONFIG_FILE.exists() else 'not created'})",
        "config_error": FILE_ERROR,
        "log_dir": str(LOG_DIR),
        "bench_file": str(BENCH_FILE),
        "deck": f"http://{DECK_HOST}:{DECK_PORT}",
        "proxy": f"http://{PROXY_HOST}:{PROXY_PORT}",
        "llama_ports": LLAMA_PORT_RANGE,
        "model_dirs": MODEL_DIRS,
        "backends": BACKENDS,
        "whisper_bin": WHISPER_BIN,
        "whisper_model": WHISPER_MODEL,
        "ffmpeg": FFMPEG,
        "peak_bw_gbs": PEAK_BW_GBS,
    }
