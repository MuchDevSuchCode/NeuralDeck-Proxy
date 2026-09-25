#!/usr/bin/env python3
"""NeuralDeck — the dashboard server.

Serves the single-page deck (Dashboard, Prompt Lab, Benchmarks) and the JSON
API behind it:

  GET  /                        the deck
  GET  /api/config              ports, backends and defaults for the UI
  GET  /api/state               latest snapshot + metric history
  GET  /api/stream              SSE: one snapshot per second
  GET  /api/models              discovered models
  POST /api/models/delete       remove a model's files
  GET  /api/logs/{service}      tail of a log
  POST /api/service/{name}/start|stop
  POST /api/llama/launch        start a model
  POST /api/llama/{port}/stop   stop one instance
  GET  /api/launch/status       progress of an in-flight launch
  POST /api/chat                same-origin relay to the proxy (Prompt Lab)
  GET  /api/bench               benchmark history
  GET  /api/hf/*                Hugging Face browse + download

There is no authentication: run it on a network you trust, the same as
every other service in this stack.
"""

import asyncio
import json
import os
import re
import shutil
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from . import (bench, config, launcher, llama_log, models, procs, services,
               settings, sysinfo)

STATIC_DIR = Path(__file__).resolve().parent / "static"

HISTORY_KEYS = ("ts", "cpu", "gpu", "vram", "ram", "gpu_temp", "gpu_power",
                "prefill", "decode", "kv1", "kv2", "kv3", "kv4")
history: dict = {k: deque(maxlen=config.HISTORY_LEN) for k in HISTORY_KEYS}
latest_snapshot: dict = {}
_prev = {"io": None, "net": None, "t": None}
_launcher = launcher.Launcher()
CPU_NAME = sysinfo.cpu_name()        # static: read once, not every sample


# ---------------------------------------------------------------------------
# Log-derived stats for the multimodal stack
# ---------------------------------------------------------------------------

def whisper_stats() -> dict:
    """Per-request lines from whisper-server's log, current run only."""
    stats = {"model": None, "reqs": 0, "audio_sec": 0.0,
             "last_file": None, "last_sec": None}
    text = procs.tail_text(config.WHISPER_LOG)
    if not text:
        return stats
    found = re.findall(r"loading model from '([^']+)'", text)
    if found:
        stats["model"] = os.path.basename(found[-1]).replace("ggml-", "").replace(".bin", "")
    # The log appends across restarts; count only after the last startup.
    idx = text.rfind("whisper server listening")
    seg = text[idx:] if idx >= 0 else text
    for m in re.finditer(r"processing '([^']+)' \((\d+) samples, ([\d.]+) sec\)", seg):
        stats["reqs"] += 1
        stats["audio_sec"] += float(m.group(3))
        stats["last_file"], stats["last_sec"] = m.group(1), float(m.group(3))
    return stats


def proxy_stats() -> dict:
    """Access lines from the proxy's log, current run only."""
    stats = {"reqs": 0, "errors": 0, "health": 0, "last": None, "port": None}
    text = procs.tail_text(config.PROXY_LOG)
    if not text:
        return stats
    idx = text.rfind("Multimodal Orchestrator")
    seg = text[idx:] if idx >= 0 else text
    m_port = re.search(r"Uvicorn running on https?://[^:]+:(\d+)", seg)
    if m_port:
        stats["port"] = m_port.group(1)
    for m in re.finditer(r'"(GET|POST|PUT|DELETE|PATCH) (/[^ ]*) HTTP/[\d.]+" (\d+)', seg):
        method, path, code = m.groups()
        if path.startswith("/health"):
            stats["health"] += 1
            continue
        stats["reqs"] += 1
        if not code.startswith("2"):
            stats["errors"] += 1
        stats["last"] = f"{method} {path.split('?')[0][:18]} {code}"
    return stats


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def collect_snapshot() -> dict:
    """One full metrics sample. Runs in a thread — everything here is sync."""
    now = time.time()
    g = sysinfo.gpu()
    gtt_used, gtt_total = sysinfo.gtt()
    mem, swap = psutil.virtual_memory(), psutil.swap_memory()
    cores = psutil.cpu_percent(interval=None, percpu=True)
    cpu_util = sum(cores) / len(cores) if cores else 0.0
    try:
        freq = psutil.cpu_freq()          # unimplemented on some platforms
    except Exception:
        freq = None
    cpu_temp, cpu_volts = sysinfo.cpu_metrics()
    npu_util, npu_pwr = sysinfo.npu_metrics()

    try:
        io, net = psutil.disk_io_counters(), psutil.net_io_counters()
    except Exception:                      # no counters in some containers
        io = net = None
    dt = max(0.1, now - _prev["t"]) if _prev["t"] else 1.0
    rd = wr = up = dn = 0.0
    if _prev["io"] and io:
        rd = (io.read_bytes - _prev["io"].read_bytes) / dt / 1024**2
        wr = (io.write_bytes - _prev["io"].write_bytes) / dt / 1024**2
    if _prev["net"] and net:
        up = (net.bytes_sent - _prev["net"].bytes_sent) / dt / 1024**2
        dn = (net.bytes_recv - _prev["net"].bytes_recv) / dt / 1024**2
    _prev.update(io=io, net=net, t=now)

    ncpu = len(cores) or 1
    procs_seen, total_threads = [], 0
    for p in psutil.process_iter(["name", "cpu_percent", "memory_percent",
                                  "num_threads"]):
        try:
            total_threads += p.info["num_threads"] or 0
            procs_seen.append(p.info)
        except Exception:
            pass
    top = sorted(procs_seen, key=lambda p: p["cpu_percent"] or 0, reverse=True)[:6]
    top = [{"name": (p["name"] or "?")[:20],
            "cpu": round((p["cpu_percent"] or 0) / ncpu, 1),
            "mem": round(p["memory_percent"] or 0, 1)} for p in top]

    llama = None
    log = procs.newest_llama_log()
    if log:
        try:
            llama = llama_log.quick_stats(log)
        except Exception:
            llama = None

    instances = procs.llama_instances()
    svcs = services.snapshot()
    primary = instances[0] if instances else {}
    svcs["llama"] = {**primary, "up": bool(instances),
                     "port": primary.get("port") or config.LLAMA_PORTS[0],
                     "label": "llama-server", "can_start": False}

    vram_used = g.get("vram_used_bytes")
    vram_total = g.get("vram_total_bytes")
    total_gb = (round(vram_total / 1024**3, 1) if vram_total
                else (config.VRAM_TOTAL_GB_FALLBACK or None))

    return {
        "ts": round(now, 1),
        "cpu": {"util": round(cpu_util, 1),
                "cores": [round(c, 1) for c in cores],
                "freq": round(freq.current, 0) if freq else None,
                "temp": cpu_temp, "volts": cpu_volts, "name": CPU_NAME},
        "gpu": {"util": g.get("utilization"), "temp": g.get("temperature"),
                "power": g.get("power"), "fan": g.get("fan_rpm"),
                "fan_pct": g.get("fan_pct"),
                "vram_used_gb": round(vram_used / 1024**3, 2) if vram_used else None,
                "vram_total_gb": total_gb,
                "name": g.get("name"), "sclk": g.get("sclk"),
                "mclk": g.get("mclk"), "fclk": g.get("fclk"),
                "gtt_used_gb": gtt_used, "gtt_total_gb": gtt_total,
                "peak_bw_gbs": config.PEAK_BW_GBS,
                "sources": g.get("sources") or []},
        "npu": {"util": npu_util, "power": npu_pwr},
        "mem": {"used_gb": round(mem.used / 1024**3, 1),
                "total_gb": round(mem.total / 1024**3, 1),
                "pct": mem.percent, "swap_pct": swap.percent},
        "disk": {"mounts": sysinfo.disk_mounts(),
                 "read_mbs": round(rd, 2), "write_mbs": round(wr, 2)},
        "net": {"up_mbs": round(up, 2), "down_mbs": round(dn, 2)},
        "system": {"uptime_s": int(now - psutil.boot_time()),
                   "threads": total_threads, "top": top},
        "llama": llama,
        "llama_instances": instances,
        "whisper_stats": whisper_stats(),
        "proxy_stats": proxy_stats(),
        "services": svcs,
        "launch": {"running": _launcher.running},
    }


async def sampler():
    global latest_snapshot
    psutil.cpu_percent(interval=None, percpu=True)  # prime the deltas
    while True:
        try:
            snap = await asyncio.to_thread(collect_snapshot)
            latest_snapshot = snap
            history["ts"].append(snap["ts"])
            history["cpu"].append(snap["cpu"]["util"])
            history["gpu"].append(snap["gpu"]["util"] or 0)
            history["vram"].append(snap["gpu"]["vram_used_gb"] or 0)
            history["ram"].append(snap["mem"]["pct"])
            history["gpu_temp"].append(snap["gpu"]["temp"] or 0)
            history["gpu_power"].append(snap["gpu"]["power"] or 0)
            ll = snap.get("llama") or {}
            history["prefill"].append(round(ll.get("prefill_tps_weighted") or 0, 1))
            history["decode"].append(round(ll.get("decode_tps_weighted") or 0, 1))
            by_port = {i["port"]: i for i in snap.get("llama_instances", [])}
            for n, kp in enumerate(config.KV_CHART_PORTS, start=1):
                history[f"kv{n}"].append((by_port.get(kp) or {}).get("kv_pct") or 0)
            await asyncio.to_thread(bench.collect_traffic,
                                    snap.get("llama_instances", []))
        except Exception as e:
            print(f"sampler error: {e}", flush=True)
        await asyncio.sleep(config.SAMPLE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(sampler())
    print(f"NeuralDeck on http://{config.DECK_HOST}:{config.DECK_PORT}", flush=True)
    yield
    task.cancel()


app = FastAPI(title="NeuralDeck", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Static + config
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/api/config")
async def ui_config():
    """What the page needs in order to stop hard-coding this machine."""
    return JSONResponse({
        "backends": list(config.BACKENDS.keys()),
        "default_backend": config.DEFAULT_BACKEND,
        "llama_ports": config.LLAMA_PORTS,
        "kv_chart_ports": config.KV_CHART_PORTS,
        "proxy_port": config.PROXY_PORT,
        "deck_port": config.DECK_PORT,
        "logs": services.log_names(),
        "links": _links(),
        "defaults": {"ctx": config.CTX_DEFAULT, "slots": config.SLOTS_DEFAULT,
                     "spec": config.SPEC_DEFAULT, "thinking": config.THINKING_DEFAULT},
        "model_dirs": config.MODEL_DIRS,
        "download_dir": config.MODELS_DOWNLOAD_DIR,
        "hf_available": _hf_available(),
        "data_dir": str(config.DATA_DIR),
        "config_file": str(config.CONFIG_FILE),
        "platform": "windows" if config.IS_WINDOWS else sys.platform,
    })


def _links() -> list:
    out = [{"label": "llama webui", "port": config.PROXY_PORT}]
    if config.COMFY_CMD or procs.port_in_use(config.COMFY_PORT):
        out.append({"label": "ComfyUI", "port": config.COMFY_PORT})
    if config.TTS_CMD or procs.port_in_use(config.TTS_PORT):
        out.append({"label": "TTS UI", "port": config.TTS_PORT})
    return out


def _hf_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        return False


@app.get("/api/state")
async def state():
    return JSONResponse({"snapshot": latest_snapshot,
                         "history": {k: list(v) for k, v in history.items()}})


@app.get("/api/stream")
async def stream():
    async def gen():
        last_ts = None
        while True:
            if latest_snapshot and latest_snapshot.get("ts") != last_ts:
                last_ts = latest_snapshot.get("ts")
                yield f"data: {json.dumps(latest_snapshot)}\n\n"
            await asyncio.sleep(config.SAMPLE_INTERVAL)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@app.get("/api/settings")
async def settings_get():
    return JSONResponse(settings.describe())


@app.post("/api/settings")
async def settings_post(body: dict):
    """Write the submitted settings and re-read the file.

    Only the keys in the body are touched, and a key sent empty is removed
    so it falls back to its default. Most settings apply at once; the
    response names the ones that were captured at startup and so need a
    restart.
    """
    try:
        result = await asyncio.to_thread(settings.save, body)
    except settings.Invalid as e:
        raise HTTPException(400, str(e))
    except OSError as e:
        raise HTTPException(500, f"could not write {config.CONFIG_FILE}: {e}")
    return JSONResponse(result)


@app.get("/api/browse")
async def browse(path: str = None):
    """Directory listing for the settings page's folder picker — a browser
    cannot open a native one, and typing paths by hand is worse."""
    try:
        return JSONResponse(await asyncio.to_thread(settings.browse, path))
    except settings.Invalid as e:
        raise HTTPException(400, str(e))


@app.post("/api/restart")
async def restart():
    """Re-exec this process, for settings that were read at startup.

    The response goes out first and the exec happens a moment later, so the
    page knows the restart was accepted (and where to reconnect) instead of
    just seeing the connection drop.
    """
    async def go():
        await asyncio.sleep(0.4)
        print("restarting on request from the settings page", flush=True)
        # -m neuraldeck works whether this was started as a module or via
        # the installed console script; the listening socket is not
        # inherited across exec, so the new process can bind it.
        os.execv(sys.executable,
                 [sys.executable, "-m", "neuraldeck", *sys.argv[1:]])

    asyncio.create_task(go())
    return JSONResponse({"restarting": True, "port": config.DECK_PORT,
                         "url": f"http://HOST:{config.DECK_PORT}/"})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    found = await asyncio.to_thread(models.discover)
    last = None
    try:
        last = config.LAST_MODEL_FILE.read_text().strip()
    except Exception:
        pass
    return JSONResponse({
        "models": [{k: v for k, v in m.items()
                    if k not in ("path", "mmproj_path", "draft_path")}
                   for m in found],
        "last": last})


@app.post("/api/models/delete")
async def delete_model(body: dict):
    """Delete a model's files.

    The path comes from discovery, never from the client; a serving instance
    blocks the delete; and a model living loose in a root loses only its own
    .gguf, never the shared directory.
    """
    name = body.get("name")
    model = await asyncio.to_thread(models.by_name, name)
    if model is None:
        raise HTTPException(404, f"unknown model '{name}'")
    for inst in procs.llama_instances():
        if models.fuzzy_eq(inst.get("alias") or "", name):
            raise HTTPException(409, f"'{name}' is serving on :{inst['port']} "
                                     "— stop it first")
    folder = os.path.realpath(os.path.dirname(model["path"]))
    roots = {os.path.realpath(r) for r in config.MODEL_DIRS}
    if folder in roots:
        os.remove(model["path"])
        removed = model["path"]
    else:
        shutil.rmtree(folder)
        removed = folder
    return JSONResponse({"deleted": name, "removed": removed})


# ---------------------------------------------------------------------------
# Logs and services
# ---------------------------------------------------------------------------

@app.get("/api/logs/{service}")
async def logs(service: str, lines: int = 120):
    if service == "launch":
        return JSONResponse({"lines": _launcher.status()["lines"][-lines:]})
    path = services.log_path(service)
    if service not in services.log_names():
        raise HTTPException(404, f"unknown log '{service}'")
    if not path or not os.path.exists(path):
        # normal before the first launch; the dashboard polls this
        return JSONResponse({"path": None, "lines": []})
    text = procs.tail_text(path, 256 * 1024)
    return JSONResponse({"path": str(path), "lines": text.splitlines()[-lines:]})


@app.post("/api/service/{name}/start")
async def service_start(name: str):
    try:
        return JSONResponse(await asyncio.to_thread(services.start, name))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(500, f"could not start {name}: {e}")


@app.post("/api/service/{name}/stop")
async def service_stop(name: str):
    try:
        return JSONResponse(await asyncio.to_thread(services.stop, name))
    except LookupError as e:
        raise HTTPException(404, str(e))


# ---------------------------------------------------------------------------
# llama-server control
# ---------------------------------------------------------------------------

@app.post("/api/llama/launch")
async def llama_launch(body: dict):
    model = body.get("model")
    backend = body.get("backend") or config.DEFAULT_BACKEND
    thinking = body.get("thinking", "off")
    spec = body.get("spec", "auto")
    try:
        slots = max(1, int(body.get("slots") or 1))
        ctx = int(body.get("ctx") or config.CTX_DEFAULT)
    except (TypeError, ValueError):
        raise HTTPException(400, "slots and ctx must be numbers")
    if thinking not in ("off", "low", "medium", "high"):
        raise HTTPException(400, "thinking must be off|low|medium|high")
    if spec not in ("auto", "off", "ngram"):
        raise HTTPException(400, "spec must be auto, ngram or off")
    if not 1024 <= ctx <= 1048576:
        raise HTTPException(400, "ctx must be between 1024 and 1048576 tokens")
    if model:
        try:
            config.LAST_MODEL_FILE.write_text(model)
        except OSError:
            pass
    try:
        return JSONResponse(await _launcher.start(
            model_name=model, backend=backend, ctx=ctx, slots=slots,
            thinking=thinking, spec=spec, replace=bool(body.get("replace")),
            force=bool(body.get("force"))))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except MemoryError as e:
        raise HTTPException(409, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/api/launch/status")
async def launch_status():
    return JSONResponse(_launcher.status())


@app.post("/api/llama/{port}/stop")
async def llama_stop(port: int):
    for inst in procs.llama_instances():
        if inst["port"] == port:
            await asyncio.to_thread(procs.stop_pid, inst["pid"])
            return JSONResponse({"stopped": inst["pid"], "port": port})
    raise HTTPException(404, f"no llama instance on port {port}")


# ---------------------------------------------------------------------------
# Prompt Lab: same-origin streaming relay to the proxy
# (the proxy sends no CORS headers, so the browser cannot call it directly)
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_relay(request: Request):
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))
    upstream = client.build_request(
        "POST", config.PROXY_CHAT, content=await request.body(),
        headers={"Content-Type": "application/json"})
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        raise HTTPException(502, f"proxy unreachable on :{config.PROXY_PORT} "
                                 "— start it from the Servers panel")
    except httpx.TimeoutException:
        await client.aclose()
        raise HTTPException(504, "proxy timed out")

    async def cleanup():
        await resp.aclose()
        await client.aclose()

    return StreamingResponse(
        resp.aiter_raw(), status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/event-stream"),
        headers={"Cache-Control": "no-cache"},
        background=BackgroundTask(cleanup))


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@app.post("/api/bench")
async def bench_add(body: dict):
    if not body.get("model"):
        raise HTTPException(400, "model required")
    rec = bench.add_client_run(body, procs.llama_instances())
    return JSONResponse({"saved": rec["id"]})


@app.get("/api/bench")
async def bench_list(limit: int = 1000):
    return JSONResponse({"runs": bench.read(limit)})


@app.post("/api/bench/delete")
async def bench_delete(body: dict):
    if not bench.delete(body.get("id")):
        raise HTTPException(404, f"no benchmark record '{body.get('id')}'")
    return JSONResponse({"deleted": body.get("id")})


@app.post("/api/bench/clear")
async def bench_clear():
    bench.clear()
    return JSONResponse({"cleared": True})


# ---------------------------------------------------------------------------
# Hugging Face downloader
# ---------------------------------------------------------------------------

download_queue: deque = deque()
download_recent: deque = deque(maxlen=12)
download_state = {"proc": None, "buffer": deque(maxlen=200), "repo": None,
                  "target": None, "total_bytes": 0, "started": None,
                  "id": None, "files": 0, "cancelled": False}
_dl_task = None


def _require_hf():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except Exception:
        raise HTTPException(
            501, "huggingface_hub is not installed — `pip install huggingface_hub` "
                 "to browse and download models from here")


def _hf_caps(tags, pipeline, repo_id, file_names=None, chat_template=None):
    """Capabilities from hub metadata: weaker than reading the GGUF, but
    good enough for browse-time icons."""
    t = {str(x).lower() for x in (tags or [])}
    p = (pipeline or "").lower()
    rid = (repo_id or "").lower()
    return {
        "vision": p in ("image-text-to-text", "visual-question-answering",
                        "video-text-to-text") or "vision" in t
                  or bool(file_names and any("mmproj" in f.lower() for f in file_names)),
        "embed": p in ("feature-extraction", "sentence-similarity")
                 or "embeddings" in t or "sentence-transformers" in t,
        "audio": p in ("audio-text-to-text", "automatic-speech-recognition",
                       "any-to-any") or "audio" in t,
        "thinking": "reasoning" in t or "thinking" in t
                    or bool(chat_template and "think" in str(chat_template).lower()),
        "mtp": "mtp" in rid,
    }


@app.get("/api/hf/search")
async def hf_search(q: str = "", limit: int = 25, sort: str = "trending"):
    _require_hf()
    sort_key = {"trending": "trending_score", "downloads": "downloads",
                "likes": "likes"}.get(sort, "trending_score")

    def _search(key):
        from huggingface_hub import HfApi
        kwargs = dict(filter="gguf", sort=key, limit=limit)
        if q:
            kwargs["search"] = q
        return [{"id": m.id, "downloads": m.downloads, "likes": m.likes,
                 "updated": str(m.last_modified or "")[:10],
                 "caps": _hf_caps(m.tags, m.pipeline_tag, m.id)}
                for m in HfApi().list_models(**kwargs)]

    try:
        try:
            results = await asyncio.to_thread(_search, sort_key)
        except Exception:
            # older hub versions have no trending_score
            results = await asyncio.to_thread(_search, "downloads")
        return JSONResponse({"results": results})
    except Exception as e:
        raise HTTPException(502, f"HF search failed: {e}")


@app.get("/api/hf/files")
async def hf_files(repo: str):
    _require_hf()

    def _files():
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        files = [{"name": s.rfilename, "size": s.size or 0}
                 for s in info.siblings if s.rfilename.lower().endswith(".gguf")]
        gguf_meta = getattr(info, "gguf", None) or {}
        caps = _hf_caps(info.tags, info.pipeline_tag, repo,
                        file_names=[f["name"] for f in files],
                        chat_template=gguf_meta.get("chat_template")
                        if isinstance(gguf_meta, dict) else None)
        return files, caps

    try:
        files, caps = await asyncio.to_thread(_files)
        return JSONResponse({"repo": repo, "files": files, "caps": caps})
    except Exception as e:
        raise HTTPException(502, f"HF file listing failed: {e}")


def _download_folder_name(files: list) -> str:
    """Folder = the main gguf's name minus any shard suffix, so an mmproj
    lands beside its model and discovery pairs them."""
    main = next((f for f in files if "mmproj" not in f.lower()), files[0])
    base = re.sub(r"-\d{5}-of-\d{5}", "", os.path.basename(main))
    return re.sub(r"\.gguf$", "", base, flags=re.I)


def _download_job(job: dict) -> int:
    """Fetch one job's files with huggingface_hub, which works the same on
    every platform (the `hf` CLI is not always on PATH on Windows)."""
    from huggingface_hub import hf_hub_download
    os.makedirs(job["target"], exist_ok=True)
    for name in job["files"]:
        if download_state["cancelled"]:
            download_state["buffer"].append("[cancelled]")
            return 1
        download_state["buffer"].append(f"downloading {name}")
        hf_hub_download(repo_id=job["repo"], filename=name,
                        local_dir=job["target"])
        download_state["buffer"].append(f"done {name}")
    return 0


async def _dl_run_queue():
    """One job at a time: each saturates the link, and the progress figure
    walks a single target directory."""
    while download_queue:
        job = download_queue.popleft()
        download_state.update(id=job["id"], repo=job["repo"], target=job["target"],
                              total_bytes=job["total_bytes"], files=len(job["files"]),
                              started=time.time(), cancelled=False, proc="running")
        download_state["buffer"].clear()
        download_state["buffer"].append(
            f"[{len(job['files'])} file(s) from {job['repo']} -> {job['target']}]")
        rc = 0
        try:
            rc = await asyncio.to_thread(_download_job, job)
        except Exception as e:
            download_state["buffer"].append(f"[failed: {e}]")
            rc = 1
        download_state["proc"] = None
        download_state["returncode"] = rc
        download_state["buffer"].append(f"[finished, exit {rc}]")
        download_recent.append({"id": job["id"], "repo": job["repo"],
                                "target": job["target"], "rc": rc,
                                "ts": time.time()})


@app.post("/api/hf/download")
async def hf_download(body: dict):
    global _dl_task
    _require_hf()
    repo, files = body.get("repo"), body.get("files") or []
    if not repo or not files:
        raise HTTPException(400, "repo and files are required")
    job = {"id": f"{int(time.time() * 1000)}-{os.urandom(3).hex()}",
           "repo": repo, "files": files,
           "total_bytes": int(body.get("total_bytes") or 0),
           "target": os.path.join(config.MODELS_DOWNLOAD_DIR,
                                  _download_folder_name(files))}
    download_queue.append(job)
    if _dl_task is None or _dl_task.done():
        _dl_task = asyncio.create_task(_dl_run_queue())
    return JSONResponse({"queued": job["id"], "repo": repo,
                         "target": job["target"], "position": len(download_queue)})


@app.get("/api/hf/status")
async def hf_status():
    done = 0
    if download_state["target"]:
        for root, _, names in os.walk(download_state["target"]):
            for n in names:
                try:
                    done += os.path.getsize(os.path.join(root, n))
                except OSError:
                    pass
    return JSONResponse({
        "active": download_state["proc"] is not None,
        "returncode": download_state.get("returncode"),
        "repo": download_state["repo"], "target": download_state["target"],
        "done_bytes": done, "total_bytes": download_state["total_bytes"],
        "elapsed": round(time.time() - download_state["started"], 1)
        if download_state["started"] else None,
        "lines": list(download_state["buffer"])[-12:],
        "id": download_state["id"],
        "queue": [{"id": j["id"], "repo": j["repo"], "files": len(j["files"]),
                   "total_bytes": j["total_bytes"],
                   "name": os.path.basename(j["target"])} for j in download_queue],
        "recent": [{**r, "name": os.path.basename(r["target"])}
                   for r in list(download_recent)[-6:]],
    })


@app.post("/api/hf/cancel")
async def hf_cancel(body: dict = None):
    """Cancel: a queued job is dropped outright, the running one stops after
    the file it is on (hf_hub_download has no mid-file abort)."""
    job_id = (body or {}).get("id")
    if job_id:
        for j in list(download_queue):
            if j["id"] == job_id:
                download_queue.remove(j)
                return JSONResponse({"dequeued": job_id})
    if download_state["proc"] is None:
        raise HTTPException(404, "no download in progress")
    download_state["cancelled"] = True
    return JSONResponse({"cancelling": True})


def main():
    import uvicorn
    # Cap graceful shutdown: an open SSE stream (a dashboard tab left open)
    # must not hold the process half-dead through a restart.
    uvicorn.run(app, host=config.DECK_HOST, port=config.DECK_PORT,
                log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
