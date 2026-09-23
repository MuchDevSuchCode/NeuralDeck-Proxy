"""Cross-platform hardware probes.

The dashboard asks for one shape of data whatever the box is; this module
answers it from whatever that box actually exposes, and returns None for
anything it cannot know rather than inventing a number.

Probe order for the GPU: NVIDIA's nvidia-smi, AMD's amd-smi, AMD's
rocm-smi, the amdgpu sysfs nodes (Linux), then a name-only lookup from
Windows CIM. The first one that answers wins; a second GPU vendor in the
same box is not something this dashboard tries to chart.
"""

import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

from . import config

IS_WINDOWS = config.IS_WINDOWS
IS_LINUX = sys.platform.startswith("linux")

# A probe that needs a subprocess is only worth trying if the tool exists;
# the answer to "does it exist" is cached because PATH does not change under
# a running dashboard.
_which_cache: dict = {}


def _which(name: str):
    if name not in _which_cache:
        _which_cache[name] = shutil.which(name)
    return _which_cache[name]


def _run(cmd, timeout=2.0) -> str:
    """Run a probe command, returning stdout or "" — probes never raise."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=(subprocess.CREATE_NO_WINDOW
                                          if IS_WINDOWS else 0))
        return p.stdout if p.returncode == 0 else ""
    except Exception:
        return ""


def _f(v):
    try:
        f = float(str(v).strip())
        return f
    except (TypeError, ValueError):
        return None


# ── CPU ────────────────────────────────────────────────────────────────────

def cpu_name() -> str:
    if IS_LINUX:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    if IS_WINDOWS:
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor).Name"], timeout=6.0)
        if out.strip():
            return out.strip().splitlines()[0].strip()
        ident = os.environ.get("PROCESSOR_IDENTIFIER")
        if ident:
            return ident
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out.strip():
            return out.strip()
    return platform.processor() or platform.machine() or "Unknown CPU"


def cpu_metrics():
    """(temperature_c, core_volts) — both None where unavailable."""
    temp = volts = None
    sensors = getattr(psutil, "sensors_temperatures", None)
    if sensors:
        try:
            for name, entries in (sensors() or {}).items():
                if name not in ("k10temp", "zenpower", "amdgpu", "coretemp",
                                "acpitz", "cpu_thermal"):
                    continue
                for e in entries:
                    if e.label in ("Tdie", "Tctl", "edge", "Package id 0", "") \
                            and e.current and e.current > 0:
                        temp = e.current
                        break
                if temp:
                    break
        except Exception:
            pass
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = Path(hwmon, "name").read_text().strip()
        except Exception:
            continue
        if name not in ("k10temp", "zenpower"):
            continue
        try:
            if os.path.exists(f"{hwmon}/in0_input"):
                volts = float(Path(f"{hwmon}/in0_input").read_text()) / 1000.0
            if temp is None and os.path.exists(f"{hwmon}/temp1_input"):
                temp = float(Path(f"{hwmon}/temp1_input").read_text()) / 1000.0
        except Exception:
            continue
    return temp, volts


# ── NPU (AMD XDNA) ─────────────────────────────────────────────────────────

def npu_metrics():
    """(utilisation, power) as display strings, "N/A" when there is no NPU."""
    util = power = "N/A"
    if _which("xrt-smi"):
        out = _run(["xrt-smi", "examine", "-r", "utilization"], timeout=1.0)
        m = re.search(r"Utilization.*?:\s*(\d+(?:\.\d+)?)", out, re.IGNORECASE)
        if m:
            util = f"{m.group(1)}%"
    if util == "N/A":
        for accel in glob.glob("/sys/class/accel/accel*"):
            try:
                busy = Path(accel, "device/npu_busy_percent")
                if busy.exists():
                    util = f"{busy.read_text().strip()}%"
                pw = Path(accel, "device/power_now")
                if pw.exists():
                    power = f"{float(pw.read_text().strip()) / 1e6:.1f}W"
            except Exception:
                continue
    return util, power


# ── GPU ────────────────────────────────────────────────────────────────────

def _gpu_nvidia():
    if not _which("nvidia-smi"):
        return None
    fields = ("name,utilization.gpu,temperature.gpu,power.draw,"
              "memory.used,memory.total,clocks.current.graphics,"
              "clocks.current.memory,fan.speed")
    out = _run(["nvidia-smi", f"--query-gpu={fields}",
                "--format=csv,noheader,nounits"])
    line = next((l for l in out.splitlines() if l.strip()), None)
    if not line:
        return None
    p = [x.strip() for x in line.split(",")]
    while len(p) < 9:
        p.append("")
    mu, mt = _f(p[4]), _f(p[5])
    sclk, mclk = _f(p[6]), _f(p[7])
    return {
        "name": p[0] or "NVIDIA GPU",
        "vendor": "nvidia",
        "utilization": _f(p[1]),
        "temperature": _f(p[2]),
        "power": _f(p[3]),
        "vram_used_bytes": int(mu * 1024**2) if mu is not None else None,
        "vram_total_bytes": int(mt * 1024**2) if mt is not None else None,
        "sclk": f"{sclk:.0f}Mhz" if sclk else None,
        "mclk": f"{mclk:.0f}Mhz" if mclk else None,
        "fclk": None,
        # nvidia-smi reports fan as a percentage, not RPM
        "fan_pct": _f(p[8]),
        "fan_rpm": None,
    }


def _amd_smi_metrics():
    """amd-smi ships on Windows and current ROCm, and on an APU it is often
    the only source for VRAM size and edge temperature — while reporting
    "N/A" for utilisation, which sysfs does have. Hence the merge in gpu():
    every probe contributes only the fields it actually knows."""
    if not _which("amd-smi"):
        return None
    try:
        data = json.loads(_run(["amd-smi", "metric", "--json"], timeout=4.0))
    except Exception:
        return None
    d = _first_gpu(data)
    if d is None:
        return None

    def dig(*path):
        """Walk amd-smi's nested {"value": n, "unit": "MB"} shape. Any "N/A"
        along the way means the field is unsupported on this part."""
        cur = d
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        if isinstance(cur, dict):
            cur = cur.get("value")
        return _f(cur)

    used_mb, total_mb = dig("mem_usage", "used_vram"), dig("mem_usage", "total_vram")
    sclk, mclk = dig("clock", "gfx_0", "clk"), dig("clock", "mem_0", "clk")
    return {
        "name": _amd_smi_name(),
        "vendor": "amd",
        "utilization": dig("usage", "gfx_activity"),
        "temperature": dig("temperature", "edge") or dig("temperature", "hotspot"),
        "power": dig("power", "socket_power"),
        "vram_used_bytes": int(used_mb * 1024**2) if used_mb else None,
        "vram_total_bytes": int(total_mb * 1024**2) if total_mb else None,
        "sclk": f"{sclk:.0f}Mhz" if sclk else None,
        "mclk": f"{mclk:.0f}Mhz" if mclk else None,
        "fclk": None,
        "fan_pct": dig("fan", "speed"),
        "fan_rpm": dig("fan", "rpm"),
    }


def _first_gpu(data):
    """amd-smi wraps its output differently across versions: a bare list, or
    a {"gpu_data": [...]} envelope."""
    if isinstance(data, dict):
        data = data.get("gpu_data", [data])
    if isinstance(data, list) and data:
        return data[0] if isinstance(data[0], dict) else None
    return None


_amd_name_cache = []


def _amd_smi_name():
    """Marketing name from `amd-smi static`; read once, it cannot change."""
    if _amd_name_cache:
        return _amd_name_cache[0]
    name = None
    try:
        d = _first_gpu(json.loads(_run(["amd-smi", "static", "--json"], timeout=4.0)))
        asic = (d or {}).get("asic") or {}
        name = asic.get("market_name") or asic.get("product_name")
        gfx = asic.get("target_graphics_version")
        if name and gfx and gfx not in str(name):
            name = f"{name} ({gfx})"
    except Exception:
        pass
    _amd_name_cache.append(name)
    return name


def _gpu_rocm_smi():
    if not _which("rocm-smi"):
        return None
    out = _run(["rocm-smi", "--all", "--json"], timeout=4.0)
    try:
        data = json.loads(out)
    except Exception:
        return None
    for key, m in (data or {}).items():
        if not isinstance(m, dict):
            continue
        used = _f(m.get("VRAM Total Used (B)"))
        total = _f(m.get("VRAM Total Memory (B)"))
        if used is None:
            mib = _f(m.get("VRAM Total Used (MiB)"))
            used = mib * 1024**2 if mib else None
        if total is None:
            mib = _f(m.get("VRAM Total Memory (MiB)"))
            total = mib * 1024**2 if mib else None
        idx = int(re.search(r"\d+", key).group()) if re.search(r"\d+", key) else 0
        return {
            "name": m.get("Card series") or m.get("Card model") or "AMD GPU",
            "vendor": "amd",
            "utilization": _f(m.get("GPU use (%)")),
            "temperature": _f(m.get("Temperature (Sensor edge) (C)")),
            "power": _f(m.get("Average Graphics Package Power (W)")),
            "vram_used_bytes": int(used) if used else None,
            "vram_total_bytes": int(total) if total else None,
            "sclk": m.get("sclk clock speed:"),
            "mclk": m.get("mclk clock speed:"),
            "fclk": None,
            "fan_pct": _f(m.get("Fan speed (%)")),
            "fan_rpm": _fan_rpm(idx),
        }
    return None


def _fan_rpm(card_index=0):
    for hwmon in glob.glob(f"/sys/class/drm/card{card_index}/device/hwmon/hwmon*"):
        try:
            return int(Path(hwmon, "fan1_input").read_text().strip())
        except Exception:
            continue
    return None


def _sysfs_card_dirs():
    return sorted(glob.glob("/sys/class/drm/card*/device"))


def _read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def _gpu_amdgpu_sysfs():
    """The amdgpu driver's own nodes: always right when they are there, and
    the only source that separates VRAM from GTT on a unified-memory APU."""
    for d in _sysfs_card_dirs():
        busy = _read_int(f"{d}/gpu_busy_percent")
        used = _read_int(f"{d}/mem_info_vram_used")
        total = _read_int(f"{d}/mem_info_vram_total")
        if busy is None and used is None:
            continue
        idx_m = re.search(r"card(\d+)", d)
        idx = int(idx_m.group(1)) if idx_m else 0
        return {
            # deliberately nameless: amd-smi/rocm-smi know the marketing
            # name, and "AMD GPU (amdgpu)" would win the merge over it
            "name": None,
            "vendor": "amd",
            "utilization": float(busy) if busy is not None else None,
            "temperature": _hwmon_temp(d),
            "power": _hwmon_power(d),
            "vram_used_bytes": used,
            "vram_total_bytes": total,
            "sclk": dpm_clock("pp_dpm_sclk"),
            "mclk": dpm_clock("pp_dpm_mclk"),
            "fclk": dpm_clock("pp_dpm_fclk"),
            "fan_pct": None,
            "fan_rpm": _fan_rpm(idx),
        }
    return None


def _hwmon_temp(dev_dir):
    for hwmon in glob.glob(f"{dev_dir}/hwmon/hwmon*"):
        v = _read_int(f"{hwmon}/temp1_input")
        if v:
            return v / 1000.0
    return None


def _hwmon_power(dev_dir):
    for hwmon in glob.glob(f"{dev_dir}/hwmon/hwmon*"):
        for node in ("power1_average", "power1_input"):
            v = _read_int(f"{hwmon}/{node}")
            if v:
                return v / 1e6
    return None


def _gpu_windows_name():
    """Last resort on Windows: the adapter's name and memory from CIM, with
    no utilisation — better than an empty panel."""
    if not IS_WINDOWS:
        return None
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -First 1 Name,AdapterRAM | ConvertTo-Json"],
               timeout=8.0)
    try:
        d = json.loads(out)
    except Exception:
        return None
    if isinstance(d, list):
        d = d[0] if d else {}
    ram = d.get("AdapterRAM")
    return {
        "name": d.get("Name") or "GPU",
        "vendor": "unknown",
        "utilization": None, "temperature": None, "power": None,
        "vram_used_bytes": None,
        "vram_total_bytes": int(ram) if isinstance(ram, int) and ram > 0 else None,
        "sclk": None, "mclk": None, "fclk": None,
        "fan_pct": None, "fan_rpm": None,
    }


# Probe order is merge order: the first probe to supply a field owns it.
# sysfs comes before amd-smi because its utilisation counter is live and
# free, while amd-smi fills in the name, VRAM size and edge temperature an
# APU does not expose through sysfs.
_PROBES = (
    ("nvidia-smi", _gpu_nvidia, 1.5),
    ("amdgpu-sysfs", _gpu_amdgpu_sysfs, 0.0),
    ("amd-smi", _amd_smi_metrics, 2.0),
    ("rocm-smi", _gpu_rocm_smi, 2.0),
    ("windows-cim", _gpu_windows_name, 60.0),
)
_GPU_FIELDS = ("name", "vendor", "utilization", "temperature", "power",
               "vram_used_bytes", "vram_total_bytes", "sclk", "mclk", "fclk",
               "fan_pct", "fan_rpm")
# Per-probe memo: {"at": monotonic, "val": result}. A probe that costs a
# subprocess refreshes on its own interval and is reused in between; one
# that produced nothing at all is dropped for the rest of the session.
_gpu_memo: dict = {}
_gpu_dead: set = set()


def _probe_value(name, fn, interval):
    import time
    memo = _gpu_memo.get(name)
    if memo and (interval == 0.0 or time.monotonic() - memo["at"] < interval):
        if interval != 0.0:
            return memo["val"]
    try:
        val = fn()
    except Exception:
        val = None
    _gpu_memo[name] = {"at": time.monotonic(), "val": val}
    return val


def gpu() -> dict:
    """One GPU snapshot, merged from every probe that has something to say.

    Returns the full field set with None for anything unknown, so callers
    never have to care which tool answered.
    """
    out = {k: None for k in _GPU_FIELDS}
    out["sources"] = []
    for name, fn, interval in _PROBES:
        if name in _gpu_dead:
            continue
        val = _probe_value(name, fn, interval)
        if not val:
            # nothing on the first attempt means the tool is absent or this
            # box has no part it understands — stop paying for it
            if name in _gpu_memo and _gpu_memo[name]["val"] is None:
                _gpu_dead.add(name)
            continue
        used = False
        for k in _GPU_FIELDS:
            if out[k] is None and val.get(k) is not None:
                out[k] = val[k]
                used = used or k not in ("vendor",)
        if used:
            out["sources"].append(name)
    if out["name"] is None and out["vendor"]:
        out["name"] = f"{out['vendor'].upper()} GPU"
    return out


def dpm_clock(name: str):
    """Current level from an amdgpu pp_dpm_* node (the line marked '*')."""
    for f in glob.glob(f"/sys/class/drm/card*/device/{name}"):
        try:
            for line in Path(f).read_text().splitlines():
                if "*" in line:
                    return line.split(":", 1)[1].replace("*", "").strip()
        except Exception:
            continue
    return None


def gtt() -> tuple:
    """(used_gb, total_gb) of the shared GTT pool, AMD/Linux only."""
    for d in _sysfs_card_dirs():
        used = _read_int(f"{d}/mem_info_gtt_used")
        total = _read_int(f"{d}/mem_info_gtt_total")
        if used is not None and total:
            return round(used / 1024**3, 2), round(total / 1024**3, 1)
    return None, None


# ── storage ────────────────────────────────────────────────────────────────

def disk_mounts() -> list:
    """The volumes that matter: where models live, plus the system volume."""
    seen, out = set(), []
    paths = []
    for d in config.MODEL_DIRS:
        if os.path.isdir(d):
            paths.append((d, d))
    system = "C:\\" if IS_WINDOWS else "/"
    paths.append((system, system))
    for label, path in paths:
        try:
            u = psutil.disk_usage(path)
        except OSError:
            continue
        key = (u.total, u.used // (1 << 30))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "label": _short(label),
            "pct": u.percent,
            "total_gb": round(u.total / 1024**3, 1),
            "used_gb": round(u.used / 1024**3, 1),
            "free_gb": round(u.free / 1024**3, 1),
        })
    return out


def _short(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home) else path
