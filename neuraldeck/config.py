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
import os
import shutil
import sys
from pathlib import Path

IS_WINDOWS = os.name == "nt"
EXE = ".exe" if IS_WINDOWS else ""


def _default_data_dir() -> Path:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "NeuralDeck"
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "neuraldeck"


DATA_DIR = Path(os.environ.get("NEURALDECK_HOME") or _default_data_dir())
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = DATA_DIR / "config.json"

try:
    _FILE: dict = json.loads(CONFIG_FILE.read_text())
    if not isinstance(_FILE, dict):
        _FILE = {}
except Exception:
    _FILE = {}


def get(key: str, default=None):
    """One setting, env first, then config.json, then the given default."""
    raw = os.environ.get("NEURALDECK_" + key.upper())
    if raw is not None:
        return raw
    if key.lower() in _FILE:
        return _FILE[key.lower()]
    return default


def _int(key, default):
    try:
        return int(get(key, default))
    except (TypeError, ValueError):
        return default


def _float(key, default):
    try:
        return float(get(key, default))
    except (TypeError, ValueError):
        return default


def _bool(key, default):
    v = get(key, default)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _list(key, default):
    """A list setting: JSON array in config.json, os.pathsep-joined in env."""
    v = get(key, None)
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [p for p in str(v).split(os.pathsep) if p.strip()]


def _dict(key, default):
    """A mapping setting: JSON object, or "k=v,k=v" in the environment."""
    v = get(key, None)
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


# ── where things live ──────────────────────────────────────────────────────
LOG_DIR = Path(get("log_dir", DATA_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = DATA_DIR / "run"
RUN_DIR.mkdir(parents=True, exist_ok=True)
BENCH_FILE = Path(get("bench_file", DATA_DIR / "bench.jsonl"))
LAST_MODEL_FILE = DATA_DIR / "last-model.txt"

PROXY_LOG = Path(get("proxy_log", LOG_DIR / "proxy.log"))
WHISPER_LOG = Path(get("whisper_log", LOG_DIR / "whisper.log"))
COMFY_LOG = Path(get("comfy_log", LOG_DIR / "comfy.log"))
TTS_LOG = Path(get("tts_log", LOG_DIR / "tts.log"))


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
LLAMA_PORT_RANGE = str(get("llama_port_range", "8081-8089"))
WHISPER_PORT = _int("whisper_port", 8090)
TTS_PORT = _int("tts_port", 8004)
COMFY_PORT = _int("comfy_port", 8188)


def _ports(spec: str) -> list:
    lo, _, hi = spec.partition("-")
    try:
        return list(range(int(lo), int(hi or lo) + 1))
    except ValueError:
        return [8081]


LLAMA_PORTS = _ports(LLAMA_PORT_RANGE)
# The dashboard's KV-cache chart has four series; give it the first four.
KV_CHART_PORTS = LLAMA_PORTS[:4]

# ── binaries ───────────────────────────────────────────────────────────────
def _find_llama() -> str:
    """llama-server from config, PATH, or the usual build locations."""
    explicit = get("llama_bin", None)
    if explicit:
        return str(explicit)
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

WHISPER_BIN = str(get("whisper_bin", shutil.which("whisper-server")
                      or Path.home() / "whisper.cpp" / "build" / "bin"
                      / f"whisper-server{EXE}"))
WHISPER_MODEL = str(get("whisper_model", Path.home() / "whisper.cpp" / "models"
                        / "ggml-large-v3-turbo.bin"))
FFMPEG = str(get("ffmpeg", shutil.which("ffmpeg") or f"ffmpeg{EXE}"))

# Optional extras: a start command each, as a list or a shell-ish string.
# Empty means "not installed here" and the service is hidden from the deck.
COMFY_CMD = _list("comfy_cmd", [])
TTS_CMD = _list("tts_cmd", [])

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
EXTRA_LLAMA_ARGS = _list("extra_llama_args", [])

# ── dashboard tuning ───────────────────────────────────────────────────────
# Memory bandwidth for the roofline panel. The default is a modest DDR5
# figure; set it to your box's real peak for the ceiling to mean anything.
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
MODELS_DOWNLOAD_DIR = str(get("download_dir", MODEL_DIRS[0]))


def summary() -> dict:
    """Everything worth printing in `neuraldeck doctor`."""
    return {
        "platform": f"{sys.platform} ({'windows' if IS_WINDOWS else 'posix'})",
        "python": sys.version.split()[0],
        "data_dir": str(DATA_DIR),
        "config_file": f"{CONFIG_FILE} ({'present' if _FILE else 'not created'})",
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
