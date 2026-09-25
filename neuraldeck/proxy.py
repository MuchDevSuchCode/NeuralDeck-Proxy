#!/usr/bin/env python3
"""Multimodal proxy — the client-facing endpoint.

Apps point at this server as if it were llama-server itself. It owns the
client port; llama-server instances sit behind it. What it adds:

  * multi-instance routing — several llama-servers are discovered by
    probing /props (vLLM servers, which have no /props, by /v1/models), and
    a request's "model" field picks one
  * speech input — an `input_audio` content part is transcribed by whisper
    and rewritten to text before llama ever sees it (llama has no audio
    projector and would fail with a misleading mmproj error)
  * video input — passed through when the routed model decodes video
    natively, otherwise ffmpeg extracts frames and they are rewritten as
    images for the vision projector
  * speech output — /v1/audio/speech is relayed to a TTS backend
  * Anthropic-format endpoints routed by model, like the OpenAI ones
  * everything else relayed to llama untouched

Run it with `python -m neuraldeck.proxy` or `neuraldeck proxy`.
"""

import asyncio
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from . import config

LLAMA_ENDPOINT = config.LLAMA_ENDPOINT
WHISPER_ENDPOINT = config.WHISPER_ENDPOINT
TTS_ENDPOINT = config.TTS_ENDPOINT
HOST, PORT = config.PROXY_HOST, config.PROXY_PORT
VIDEO_FPS, MAX_FRAMES = config.VIDEO_FPS, config.MAX_FRAMES
CONNECT_TIMEOUT = config.BACKEND_CONNECT_TIMEOUT
WHISPER_TIMEOUT = config.WHISPER_TIMEOUT
LLAMA_TIMEOUT = config.LLAMA_TIMEOUT
_LLAMA_T = httpx.Timeout(LLAMA_TIMEOUT, connect=CONNECT_TIMEOUT)
_WHISPER_T = httpx.Timeout(WHISPER_TIMEOUT, connect=CONNECT_TIMEOUT)
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if config.IS_WINDOWS else 0


def _max_body_bytes() -> int:
    """Largest request body accepted. Read per request so a config change
    applies without a restart; config.py need not define the setting."""
    raw = getattr(config, "MAX_UPLOAD_MB", None) or config.get("max_upload_mb", 512)
    try:
        return int(float(raw) * 1024 * 1024)
    except (TypeError, ValueError):
        return 512 * 1024 * 1024


def _base_url(endpoint: str) -> str:
    p = urlparse(endpoint)
    return f"{p.scheme}://{p.netloc}"


LLAMA_BASE = _base_url(LLAMA_ENDPOINT)


def ffmpeg_bin() -> Optional[str]:
    """ffmpeg, if this machine has it. Only video needs it, so a missing
    binary is a per-request error, not a refusal to start — plenty of boxes
    run this stack for text, images and audio alone."""
    p = config.FFMPEG
    return p if (os.path.isfile(p) or shutil.which(p)) else None


# ---------------------------------------------------------------------------
# Shared HTTP client: one pool for every backend call instead of a client
# (and fresh connections) per request. Timeouts are passed per call. It is
# tied to the event loop that made it, so a new loop gets a new one.
# ---------------------------------------------------------------------------

_shared = {"loop": None, "client": None, "lock": None}


def _http() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _shared["client"]
    if _shared["loop"] is not loop or client is None or client.is_closed:
        _shared.update(loop=loop, lock=asyncio.Lock(), client=httpx.AsyncClient(
            timeout=_LLAMA_T,
            # Idle connections are dropped well before llama-server's (and
            # uvicorn's) 5s keep-alive, so a reused one is never mid-close.
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=32,
                                keepalive_expiry=2.0),
            # A shared jar would carry one caller's Set-Cookie into every
            # other caller's requests; refuse all cookies. (A bare jar: httpx
            # would copy an httpx.Cookies into a fresh, accepting one.)
            cookies=CookieJar(DefaultCookiePolicy(allowed_domains=[]))))
    return _shared["client"]


def _discovery_lock() -> asyncio.Lock:
    _http()
    return _shared["lock"]


_HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "upgrade",
                "host", "content-length", "proxy-authenticate",
                "proxy-authorization", "te", "trailer", "date", "server",
                "expect"}
# Also dropped when the proxy re-serialises the body itself: the type is
# httpx's to set, and a client's accept-encoding could ask for a compression
# httpx cannot decode.
_REBODY_HEADERS = _HOP_HEADERS | {"content-type", "accept-encoding"}


def _client_headers(request: Request, drop=_HOP_HEADERS) -> dict:
    """The caller's headers minus hop-by-hop ones — Authorization included,
    so an instance started with --api-key still accepts the request."""
    return {k: v for k, v in request.headers.items() if k.lower() not in drop}


def _transport_error(e: httpx.HTTPError, backend: str, where: str) -> HTTPException:
    """502/504 for a backend that could not be talked to properly — distinct
    from an error the backend answered with, which is relayed as-is."""
    if isinstance(e, httpx.TimeoutException):
        return HTTPException(504, f"{backend} backend timed out at {where}")
    # The instance may have just stopped; look again on the next request
    # rather than trust the cache for another few seconds.
    _instances_cache["ts"] = float("-inf")
    if isinstance(e, httpx.ConnectError):
        return HTTPException(502, f"{backend} backend unreachable at {where}")
    return HTTPException(502, f"{backend} backend at {where} failed: "
                              f"{type(e).__name__}: {e}")


def _upstream_error(status: int, body: bytes, backend: str) -> JSONResponse:
    """Relay a backend's error response with its own status. Turning a 400
    (context overflow, bad parameter) into 502 hides the reason and makes
    OpenAI SDKs retry a request that can never succeed."""
    try:
        content = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        text = body.decode(errors="replace").strip()[:2000]
        content = {"error": {"message": text or f"{backend} backend returned {status}",
                             "type": "backend_error", "code": status}}
    return JSONResponse(status_code=status, content=content)


# ---------------------------------------------------------------------------
# Instance registry: discovered by probing, never from a state file, so a
# stopped instance simply vanishes on the next pass.
# ---------------------------------------------------------------------------

_instances_cache = {"ts": float("-inf"), "list": []}
_LOCAL_HOSTS = ("", "0.0.0.0", "::", "127.0.0.1", "localhost", "::1")


def _probe_ports() -> list:
    """The llama port range minus this proxy's own port: probing ourselves
    would recurse /props -> discovery -> /props, a request storm."""
    return [p for p in config.LLAMA_PORTS
            if not (p == PORT and HOST in _LOCAL_HOSTS)]


def _norm_model(s: str) -> str:
    s = (s or "").split("/")[-1]          # strip hf-repo/path prefixes
    return re.sub(r"[._]", "-", re.sub(r"(\.gguf|-gguf)$", "", s.lower()))


def _route_match(requested: str, instances: list) -> list:
    """Instances a requested name could mean.

    An exact normalised match wins outright. Failing that, instances whose
    name extends the request at a '-' boundary (an HF repo id against a
    quant-suffixed filename) are candidates, and the caller routes only when
    there is exactly one: choosing between Bonsai-27B-PTQ1_0 and -PQ2_0 by
    length would answer from a model nobody named. Substrings never match —
    "8b" is not a model name.
    """
    req = _norm_model(requested)
    if not req:
        return []
    prefixed = []
    for inst in instances:
        alias = _norm_model(inst.get("alias"))
        if not alias:
            continue
        if alias == req:
            return [inst]
        if alias.startswith(req + "-"):
            prefixed.append(inst)
    return prefixed


def _inst_label(inst: dict) -> str:
    return inst.get("alias") or f":{inst.get('port')}"


async def discover_instances(ttl: float = 10.0) -> list:
    asked = time.monotonic()
    if asked - _instances_cache["ts"] < ttl:
        return _instances_cache["list"]

    async def probe(port: int):
        base = f"http://127.0.0.1:{port}"
        try:
            resp = await _http().get(f"{base}/props", timeout=2.0)
            if resp.status_code == 404:
                return await probe_vllm(port, base)
            if resp.status_code != 200:
                return None             # e.g. llama-server still loading (503)
            try:
                props = resp.json()
            except ValueError:
                props = None
            if not isinstance(props, dict):
                return await probe_vllm(port, base)
            alias = (props.get("model_alias")
                     or os.path.basename(props.get("model_path") or ""))
            return [{"port": port, "base": base, "alias": alias,
                     "modalities": props.get("modalities") or {},
                     "kind": "llama.cpp"}]
        except Exception:
            return None

    async def probe_vllm(port: int, base: str):
        """A server with no /props: vLLM, known by what /v1/models serves.
        One instance per served id, so each name routes."""
        resp = await _http().get(f"{base}/v1/models", timeout=2.0)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data") or []
        return [{"port": port, "base": base, "alias": m["id"],
                 "modalities": await asyncio.to_thread(_vllm_modalities,
                                                       m.get("root")),
                 "kind": "vllm", "max_model_len": m.get("max_model_len")}
                for m in data if isinstance(m, dict) and m.get("id")] or None

    # One probe pass at a time: when the cache expires under load, waiters
    # reuse the pass that finished while they queued instead of each
    # sweeping the port range again.
    async with _discovery_lock():
        ts = _instances_cache["ts"]
        if ts >= asked or time.monotonic() - ts < ttl:
            return _instances_cache["list"]
        found = await asyncio.gather(*[probe(p) for p in _probe_ports()])
        instances = [i for f in found if f for i in f]
        _instances_cache.update(ts=time.monotonic(), list=instances)
        return instances


async def pick_backend(model: Optional[str]) -> dict:
    """Choose the instance for a request.

    A model that was explicitly asked for but is not serving raises instead
    of quietly falling back: answering from a different model than the one
    named misattributes benchmarks and hands back output the caller believes
    came from elsewhere. An instance still loading fails /props, so it is
    not a match either. A name matching several instances is refused the
    same way. A miss is re-checked against a fresh probe before failing, so
    an instance that finished loading a moment ago is not a 404.
    """
    instances = await discover_instances()
    candidates = _route_match(model, instances) if model else []
    if (model and not candidates) or not instances:
        instances = await discover_instances(ttl=0)
        candidates = _route_match(model, instances) if model else []
    if len(candidates) == 1:
        return candidates[0]
    default = next((i for i in instances if i["base"] == LLAMA_BASE), None)
    if candidates:
        if config.STRICT_MODEL_ROUTING:
            raise HTTPException(
                404, f"model '{model}' is ambiguous: it could mean any of "
                     f"{', '.join(_inst_label(i) for i in candidates)}. "
                     f"Send the exact id (see /v1/models).")
        # Lenient mode still stays among the models the name could mean.
        return default if default in candidates else candidates[0]
    if model and instances and config.STRICT_MODEL_ROUTING:
        serving = ", ".join(_inst_label(i) for i in instances)
        raise HTTPException(
            404, f"model '{model}' is not being served (loading, stopped, "
                 f"or never launched). Currently serving: {serving or 'nothing'}")
    if default:
        return default
    if instances:
        return instances[0]
    return {"port": None, "base": LLAMA_BASE, "alias": None, "modalities": None,
            "kind": None}


def _vllm_modalities(root) -> Optional[dict]:
    """What a vLLM model can take, read from its folder (vLLM reports the
    path it loaded as each model's "root"). None when that is not a local
    folder: unknown, so video goes as frames and is never refused."""
    if not isinstance(root, str) or not os.path.isfile(
            os.path.join(root, "config.json")):
        return None
    from . import models
    try:
        vision = models._hf_info(Path(root))["vision"]
    except Exception:
        return None
    # video_url is vLLM's own wire format, but frames work with every
    # vision model, so video is not claimed as native
    return {"vision": bool(vision), "video": False}


def _is_vllm(target: dict) -> bool:
    return target.get("kind") == "vllm"


def _vllm_model(payload: dict, target: dict) -> bool:
    """vLLM, unlike llama-server, rejects a model field that is not exactly
    a served id — and routing accepts prefixes and case differences. Name
    the served id before forwarding. True if the payload changed."""
    if _is_vllm(target) and target.get("alias") \
            and payload.get("model") != target["alias"]:
        payload["model"] = target["alias"]
        return True
    return False


async def probe_backend(endpoint: str) -> bool:
    base = _base_url(endpoint)
    for url in (f"{base}/health", base):
        try:
            await _http().get(url, timeout=CONNECT_TIMEOUT)
            return True
        except httpx.HTTPError:
            continue
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A llama endpoint pointing at this proxy's own port would make the
    # catch-all passthrough recurse into itself forever.
    lp = urlparse(LLAMA_BASE)
    if lp.port == PORT and lp.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
        print(f"FATAL: llama_endpoint ({LLAMA_ENDPOINT}) points at this proxy's "
              f"own port {PORT}. Point it at a llama-server port "
              f"(default {config.LLAMA_PORTS[0]}).", file=sys.stderr)
        sys.exit(1)
    if PORT in config.LLAMA_PORTS:
        print(f"WARNING: proxy port {PORT} is inside the llama port range "
              f"({config.LLAMA_PORT_RANGE}); discovery skips it, so no "
              f"llama-server can be found there.", file=sys.stderr, flush=True)
    _http()
    llama_ok, whisper_ok, tts_ok = await asyncio.gather(
        probe_backend(LLAMA_ENDPOINT), probe_backend(WHISPER_ENDPOINT),
        probe_backend(TTS_ENDPOINT))
    state = lambda ok: "reachable" if ok else "UNREACHABLE"
    ff = ffmpeg_bin()
    print(f"""
=========================================================
  Multimodal Orchestrator  (NeuralDeck proxy)
=========================================================
  Listening on          : http://{HOST}:{PORT}
  LLM endpoint          : {LLAMA_ENDPOINT}  [{state(llama_ok)}]
  llama port range      : {config.LLAMA_PORT_RANGE}
  Whisper endpoint      : {WHISPER_ENDPOINT}  [{state(whisper_ok)}]
  TTS endpoint          : {TTS_ENDPOINT}  [{state(tts_ok)}]
  Video frame rate      : {VIDEO_FPS} fps (max {MAX_FRAMES} frames/video)
  ffmpeg                : {ff or 'NOT FOUND — video input disabled'}
=========================================================
""", flush=True)
    try:
        yield
    finally:
        if _shared["client"] is not None:
            await _shared["client"].aclose()


app = FastAPI(title="Multimodal Orchestrator", lifespan=lifespan)


# Anthropic clients parse errors in their own shape, not FastAPI's
# {"detail": ...}.
_ANTHROPIC_ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error",
                          403: "permission_error", 404: "not_found_error",
                          413: "request_too_large", 429: "rate_limit_error",
                          504: "timeout_error", 529: "overloaded_error"}


def _anthropic_error(status: int, message: str) -> JSONResponse:
    kind = (_ANTHROPIC_ERROR_TYPES.get(status)
            or ("invalid_request_error" if status < 500 else "api_error"))
    return JSONResponse(status_code=status, content={
        "type": "error", "error": {"type": kind, "message": message}})


class BodyLimit:
    """Refuse oversized request bodies — by Content-Length before any of it
    is read, and by counting for chunked uploads that declare none. Raised
    from receive() as an HTTPException, which FastAPI's body parsing passes
    through as-is."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = _max_body_bytes()
        msg = (f"request body exceeds the proxy's {limit / 2**20:g} MB limit "
               f"(setting max_upload_mb)")
        declared = dict(scope.get("headers") or []).get(b"content-length")
        try:
            too_big = declared is not None and int(declared) > limit
        except ValueError:
            too_big = False
        if too_big:
            resp = (_anthropic_error(413, msg)
                    if scope.get("path", "").startswith("/v1/messages")
                    else JSONResponse(status_code=413, content={"detail": msg}))
            return await resp(scope, receive, send)
        seen = 0

        async def counted():
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body") or b"")
                if seen > limit:
                    raise HTTPException(413, msg)
            return message

        await self.app(scope, counted, send)


app.add_middleware(BodyLimit)


# ---------------------------------------------------------------------------
# Modality handlers
# ---------------------------------------------------------------------------

class _LRU:
    """Bounded by entry count and by total size. Clients resend the whole
    conversation every turn, so without it every earlier audio clip would be
    re-transcribed, and every earlier video re-extracted, on every turn."""

    def __init__(self, max_entries: int, max_bytes: int):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0

    def get(self, key):
        hit = self._items.get(key)
        if hit is None:
            return None
        self._items.move_to_end(key)
        return hit[0]

    def put(self, key, value, size: int):
        if size > self.max_bytes:
            return
        old = self._items.pop(key, None)
        if old is not None:
            self._bytes -= old[1]
        self._items[key] = (value, size)
        self._bytes += size
        while len(self._items) > self.max_entries or self._bytes > self.max_bytes:
            _, (_, s) = self._items.popitem(last=False)
            self._bytes -= s


# Keyed by sha256 of the raw media bytes.
_transcript_cache = _LRU(256, 4 * 2**20)
_frame_cache = _LRU(32, 256 * 2**20)      # (caption, image blocks) per video


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def transcribe_audio(filename: str, data: bytes, content_type: str) -> str:
    try:
        resp = await _http().post(
            WHISPER_ENDPOINT, files={"file": (filename, data, content_type)},
            data={"response_format": "json"}, timeout=_WHISPER_T)
    except httpx.TimeoutException:
        raise HTTPException(504, f"Whisper backend timed out transcribing '{filename}'")
    except httpx.HTTPError as e:
        raise _transport_error(e, "Whisper", WHISPER_ENDPOINT)
    if resp.status_code != 200:
        # A 4xx means whisper rejected the audio itself — the caller's
        # problem, and retrying won't help; anything else is the backend's.
        status = resp.status_code if 400 <= resp.status_code < 500 else 502
        raise HTTPException(status, f"Whisper backend returned {resp.status_code} "
                                    f"for '{filename}': {resp.text[:500]}")
    try:
        return str(resp.json().get("text", "")).strip()
    except (ValueError, AttributeError):
        raise HTTPException(502, f"Whisper backend returned an unreadable "
                                 f"response for '{filename}'")


async def transcribe_cached(digest: str, filename: str, data: bytes,
                            content_type: str) -> str:
    text = _transcript_cache.get(digest)
    if text is None:
        text = await transcribe_audio(filename, data, content_type)
        _transcript_cache.put(digest, text, len(text) + 64)
    return text


# Input guard for ffmpeg/ffprobe: local files only, and only real video
# containers — never playlist or concat demuxers (HLS, ffconcat), which would
# let an uploaded "video" make ffmpeg read other local files.
_VIDEO_DEMUXERS = "mov,matroska,avi,mpeg,mpegts,ogg,flv,asf,gif,m4v,h264,hevc"
_INPUT_GUARD = ["-protocol_whitelist", "file", "-format_whitelist", _VIDEO_DEMUXERS]


def _ffprobe_bin(ff: str) -> Optional[str]:
    path = ff if os.path.isfile(ff) else shutil.which(ff)
    if path:
        cand = os.path.join(os.path.dirname(path), f"ffprobe{config.EXE}")
        if os.path.isfile(cand):
            return cand
    return shutil.which("ffprobe")


def _video_duration(ff: str, video_path: Path) -> Optional[float]:
    """Clip length in seconds, or None when ffprobe is missing or can't tell."""
    probe = _ffprobe_bin(ff)
    if not probe:
        return None
    try:
        proc = subprocess.run(
            [probe, "-v", "error", *_INPUT_GUARD, "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW)
        d = float(proc.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return d if d > 0 and math.isfinite(d) else None


def _extract_frames_sync(video_path: Path) -> tuple:
    """(caption, image blocks) for a video on disk. Runs in a worker thread:
    ffmpeg, the frame reads and the base64 encoding are all blocking.

    The sampling rate is lowered for long clips so MAX_FRAMES spans the
    whole video instead of only its first MAX_FRAMES/VIDEO_FPS seconds.
    """
    ff = ffmpeg_bin()
    if ff is None:
        raise HTTPException(
            501, "video input needs ffmpeg, which is not installed (or set "
                 "NEURALDECK_FFMPEG to its path)")
    duration = _video_duration(ff, video_path)
    fps = min(VIDEO_FPS, MAX_FRAMES / duration) if duration else VIDEO_FPS
    out_dir = video_path.parent / "frames"
    out_dir.mkdir(exist_ok=True)
    cmd = [ff, "-hide_banner", "-loglevel", "error", *_INPUT_GUARD,
           "-i", str(video_path), "-vf", f"fps={fps:.6g}",
           "-frames:v", str(MAX_FRAMES), "-q:v", "2", str(out_dir / "frame_%04d.jpg")]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                              creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"ffmpeg timed out extracting frames from "
                                 f"'{video_path.name}'")
    if proc.returncode != 0:
        raise HTTPException(422, f"ffmpeg failed to extract frames from "
                                 f"'{video_path.name}': {proc.stderr.strip()[:500]}")
    frames = [image_to_content_block(p.read_bytes(), "image/jpeg")
              for p in sorted(out_dir.glob("frame_*.jpg"))]
    if not frames:
        raise HTTPException(422, "no frames could be extracted from the video")
    n = len(frames)
    if duration:
        covered = min(duration, n / fps)
        span = (f"covering the whole {duration:.0f}s clip"
                if covered >= duration * 0.95
                else f"covering the first {covered:.0f}s of {duration:.0f}s")
    else:
        span = (f"covering only the first {n / fps:.0f}s"
                if n >= MAX_FRAMES else "covering the clip")
    return f"{n} frames sampled at {fps:.3g} fps, {span}", frames


def _video_bytes_to_blocks_sync(raw: bytes, suffix: str) -> tuple:
    tmpdir = tempfile.mkdtemp(prefix="video_part_")
    try:
        video_path = Path(tmpdir) / f"input{suffix}"
        video_path.write_bytes(raw)
        return _extract_frames_sync(video_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _save_upload_sync(upload: UploadFile, dest: Path) -> tuple:
    """Copy an upload to disk in chunks, hashing on the way: (size, sha256)."""
    h, n = hashlib.sha256(), 0
    upload.file.seek(0)
    with open(dest, "wb") as out:
        while chunk := upload.file.read(1 << 20):
            h.update(chunk)
            out.write(chunk)
            n += len(chunk)
    return n, h.hexdigest()


async def cached_video_frames(digest: str, extract, *args) -> tuple:
    """(caption, image blocks), from the cache or by running `extract(*args)`
    in a worker thread."""
    hit = _frame_cache.get(digest)
    if hit is None:
        hit = await asyncio.to_thread(extract, *args)
        _frame_cache.put(digest, hit, sum(len(b["image_url"]["url"]) for b in hit[1]))
    return hit


def image_to_content_block(data: bytes, mime: str) -> dict:
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def guess_image_mime(upload: UploadFile) -> str:
    if upload.content_type and upload.content_type.startswith("image/"):
        return upload.content_type
    guessed, _ = mimetypes.guess_type(upload.filename or "")
    return guessed if guessed and guessed.startswith("image/") else "image/jpeg"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    instances, whisper_ok, tts_ok = await asyncio.gather(
        discover_instances(ttl=2.0), probe_backend(WHISPER_ENDPOINT),
        probe_backend(TTS_ENDPOINT))
    llama_ok = bool(instances) or await probe_backend(LLAMA_ENDPOINT)
    return JSONResponse(
        status_code=200 if llama_ok else 503,
        content={
            "status": "ok" if llama_ok else "degraded",
            "ffmpeg": ffmpeg_bin(),
            "backends": {
                "llama": {"endpoint": LLAMA_ENDPOINT, "reachable": llama_ok,
                          "instances": [{"port": i["port"], "model": i["alias"],
                                         "kind": i.get("kind")}
                                        for i in instances]},
                "whisper": {"endpoint": WHISPER_ENDPOINT, "reachable": whisper_ok},
                "tts": {"endpoint": TTS_ENDPOINT, "reachable": tts_ok},
            },
        })


@app.post("/v1/multimodal")
async def multimodal(
    request: Request,
    text: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
    videos: List[UploadFile] = File(default=[]),
    audio: List[UploadFile] = File(default=[]),
    stream: bool = Form(False),
    system: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    max_tokens: Optional[int] = Form(None),
):
    """One endpoint for mixed input: text, images, video and audio together."""
    if not text and not images and not videos and not audio:
        raise HTTPException(400, "Provide at least one of: text, images, videos, audio")

    # Route first, so a bad model name fails in milliseconds rather than
    # after transcription and frame extraction.
    target = await pick_backend(model)
    endpoint = f"{target['base']}/v1/chat/completions"

    tmpdir = await asyncio.to_thread(tempfile.mkdtemp, prefix="multimodal_")
    try:
        content: List[dict] = []
        # Images and video frames go before the text: vision-capable chat
        # templates are trained that way round.
        for img in images:
            data = await img.read()
            if data:
                content.append(await asyncio.to_thread(
                    image_to_content_block, data, guess_image_mime(img)))
        for i, vid in enumerate(videos):
            workdir = Path(tmpdir) / f"video_{i}"
            workdir.mkdir()
            suffix = Path(vid.filename or "video.mp4").suffix or ".mp4"
            video_path = workdir / f"input{suffix}"
            size, digest = await asyncio.to_thread(_save_upload_sync, vid, video_path)
            if not size:
                continue
            caption, frames = await cached_video_frames(
                digest, _extract_frames_sync, video_path)
            label = vid.filename or f"video {i + 1}"
            content.append({"type": "text", "text": f"[Video '{label}': {caption}]"})
            content.extend(frames)
        if text:
            content.append({"type": "text", "text": text})

        audio_jobs = [(a, await a.read()) for a in audio]
        audio_jobs = [(a, d) for a, d in audio_jobs if d]
        digests = await asyncio.to_thread(lambda: [_sha256(d) for _, d in audio_jobs])
        transcripts = await asyncio.gather(*[
            transcribe_cached(h, a.filename or "audio", d,
                              a.content_type or "application/octet-stream")
            for (a, d), h in zip(audio_jobs, digests)])
        for (a, _), transcript in zip(audio_jobs, transcripts):
            label = f" ({a.filename})" if a.filename else ""
            content.append({"type": "text",
                            "text": f"[Audio transcript{label}]: {transcript}"})

        if not content:
            raise HTTPException(400, "All uploaded files were empty")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        payload: dict = {"messages": messages, "stream": stream}
        if model:
            payload["model"] = model
        _vllm_model(payload, target)
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = _client_headers(request, _REBODY_HEADERS)
        if stream:
            return await stream_llama(payload, endpoint, headers)
        return await forward_llama(payload, endpoint, headers)
    finally:
        # Frames and transcripts are in memory by now, so the uploads go
        # whatever happens next — stream, plain answer or error.
        await asyncio.to_thread(shutil.rmtree, tmpdir, True)


async def forward_llama(payload: dict, endpoint: str = None,
                        headers: Optional[dict] = None) -> JSONResponse:
    endpoint = endpoint or LLAMA_ENDPOINT
    try:
        resp = await _http().post(endpoint, json=payload, headers=headers,
                                  timeout=_LLAMA_T)
    except httpx.HTTPError as e:
        raise _transport_error(e, "LLM", endpoint)
    if resp.status_code != 200:
        return _upstream_error(resp.status_code, resp.content, "LLM")
    try:
        return JSONResponse(content=resp.json())
    except ValueError:
        raise HTTPException(502, f"LLM backend at {endpoint} returned a non-JSON body")


async def stream_llama(payload: dict, endpoint: str = None,
                       headers: Optional[dict] = None) -> Response:
    """Open the upstream stream before answering, so a backend error (bad
    request, context overflow, model still loading) reaches the client with
    its real status instead of inside an HTTP 200 event stream."""
    endpoint = endpoint or LLAMA_ENDPOINT
    client = _http()
    req = client.build_request("POST", endpoint, json=payload, headers=headers,
                               timeout=_LLAMA_T)
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        raise _transport_error(e, "LLM", endpoint)
    if resp.status_code != 200:
        try:
            body = await resp.aread()
        except httpx.HTTPError as e:
            raise _transport_error(e, "LLM", endpoint)
        finally:
            await resp.aclose()
        return _upstream_error(resp.status_code, body, "LLM")

    async def relay():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        except httpx.HTTPError as e:
            # Headers are long gone, so the failure travels in-band, in the
            # error shape OpenAI clients parse. The leading blank line ends
            # any event the cut left half-written.
            timeout = isinstance(e, httpx.TimeoutException)
            yield "\n\ndata: " + json.dumps({"error": {
                "message": f"LLM backend at {endpoint} failed mid-stream: "
                           f"{type(e).__name__}: {e}",
                "type": "timeout" if timeout else "backend_error",
                "code": 504 if timeout else 502}}) + "\n\n"
        finally:
            await resp.aclose()

    # The background close covers a client that disconnects before the
    # body iterator ever starts.
    return StreamingResponse(relay(), media_type="text/event-stream",
                             background=BackgroundTask(resp.aclose))


# ---------------------------------------------------------------------------
# OpenAI-compatible client-facing surface
# ---------------------------------------------------------------------------

AUDIO_FORMAT_MIME = {"wav": "audio/wav", "mp3": "audio/mpeg",
                     "flac": "audio/flac", "ogg": "audio/ogg", "m4a": "audio/mp4"}
VIDEO_MIME_EXT = {"video/mp4": ".mp4", "video/webm": ".webm",
                  "video/quicktime": ".mov", "video/x-matroska": ".mkv",
                  "video/mpeg": ".mpg", "video/avi": ".avi",
                  "video/x-msvideo": ".avi", "video/ogg": ".ogv"}

_whisper_ok_cache = {"ts": 0.0, "ok": False}


async def whisper_reachable(ttl: float = 10.0) -> bool:
    """Cached, because clients poll /props and each probe is a round trip."""
    now = time.monotonic()
    if now - _whisper_ok_cache["ts"] < ttl:
        return _whisper_ok_cache["ok"]
    ok = await probe_backend(WHISPER_ENDPOINT)
    _whisper_ok_cache.update(ts=now, ok=ok)
    return ok


@app.get("/props")
async def props():
    """The default instance's /props, with modalities corrected for the stack.

    Clients treat `modalities` as authoritative, so audio is only true when
    a speech backend is actually reachable, and video is true when either
    llama decodes it natively or this proxy can extract frames for it.

    vLLM has no /props. When the default instance is a vLLM server the body
    is synthesized from what /v1/models told discovery — the served name
    and max_model_len (as n_ctx) — marked "backend": "vllm".
    """
    target = await pick_backend(None)
    if _is_vllm(target):
        vision = bool((target.get("modalities") or {}).get("vision"))
        body = {"backend": "vllm", "model_alias": target.get("alias"),
                "model_path": target.get("alias"),
                "default_generation_settings": {
                    "n_ctx": target.get("max_model_len")},
                "modalities": {"vision": vision,
                               "audio": await whisper_reachable(),
                               "video": vision and ffmpeg_bin() is not None}}
        body["llama_instances"] = [
            {"port": i["port"], "model": i["alias"],
             "modalities": i["modalities"], "kind": i.get("kind")}
            for i in await discover_instances()]
        return JSONResponse(content=body)
    try:
        resp = await _http().get(f"{target['base']}/props", timeout=CONNECT_TIMEOUT)
        body = resp.json() if resp.status_code == 200 else None
    except httpx.HTTPError:
        raise HTTPException(502, f"LLM backend unreachable at {target['base']}")
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(502, f"LLM backend /props returned {resp.status_code}"
                                 f"{'' if resp.status_code != 200 else ' (unreadable)'}")
    reported = body.get("modalities") or {}
    vision = bool(reported.get("vision"))
    body["modalities"] = {
        "vision": vision,
        "audio": await whisper_reachable(),
        "video": bool(reported.get("video"))
                 or (vision and ffmpeg_bin() is not None),
    }
    body["llama_instances"] = [
        {"port": i["port"], "model": i["alias"], "modalities": i["modalities"],
         "kind": i.get("kind")}
        for i in await discover_instances()]
    return JSONResponse(content=body)


@app.get("/v1/models")
async def models_aggregate():
    """Union of /v1/models across instances, so a client can see and select
    any served model — its choice then routes the chat."""
    instances = await discover_instances() or [{"base": LLAMA_BASE, "port": None}]
    seen, data = set(), []
    for inst in instances:
        try:
            resp = await _http().get(f"{inst['base']}/v1/models",
                                     timeout=CONNECT_TIMEOUT)
            if resp.status_code != 200:
                continue
            for m in resp.json().get("data", []):
                if m.get("id") in seen:
                    continue
                seen.add(m.get("id"))
                m["port"] = inst.get("port")
                data.append(m)
        except (httpx.HTTPError, ValueError, AttributeError):
            continue
    return JSONResponse(content={"object": "list", "data": data})


async def rewrite_audio_parts(messages: list) -> int:
    """Replace every input_audio part with a whisper transcript.

    All messages are scanned, not just the last: clients resend full history,
    so past turns can carry audio too. Errors name the speech backend —
    llama's own audio errors mention mmproj and would point at the wrong
    subsystem.
    """
    jobs = []
    for mi, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for pi, part in enumerate(content):
            if not (isinstance(part, dict) and part.get("type") == "input_audio"):
                continue
            ia = part.get("input_audio") or {}
            fmt = str(ia.get("format") or "wav").lower()
            try:
                raw = await asyncio.to_thread(base64.b64decode, ia.get("data") or "")
            except Exception:
                raise HTTPException(400, f"invalid base64 in input_audio part "
                                         f"(message {mi})")
            if not raw:
                raise HTTPException(400, f"empty input_audio part (message {mi})")
            jobs.append((mi, pi, raw, fmt))
    if not jobs:
        return 0
    digests = await asyncio.to_thread(lambda: [_sha256(j[2]) for j in jobs])
    # The same clip twice in one request is transcribed once.
    unique = {}
    for (_, _, raw, fmt), h in zip(jobs, digests):
        unique.setdefault(h, (raw, fmt))
    texts = await asyncio.gather(*[
        transcribe_cached(h, f"audio_{i}.{fmt}", raw,
                          AUDIO_FORMAT_MIME.get(fmt, "application/octet-stream"))
        for i, (h, (raw, fmt)) in enumerate(unique.items())])
    by_digest = dict(zip(unique, texts))
    for (mi, pi, _, _), h in zip(jobs, digests):
        messages[mi]["content"][pi] = {"type": "text",
                                       "text": f"[Audio transcript]: {by_digest[h]}"}
    return len(jobs)


def _decode_data_uri(url: str):
    if not url.startswith("data:"):
        return None, None
    head, _, b64 = url.partition(",")
    mime = head[5:].split(";")[0] or "application/octet-stream"
    try:
        return base64.b64decode(b64), mime
    except Exception:
        return None, mime


async def rewrite_video_parts(messages: list, mod: Optional[dict] = None) -> int:
    """Handle video_url parts according to what the routed model can do.

    Recent llama.cpp decodes video natively, but its wire format is
    input_video — a video_url part is rejected by every build, so native
    mode translates the client's data URI into input_video. Backends without
    native video get frames instead.
    """
    has_video = any(
        isinstance(p, dict) and p.get("type") == "video_url"
        for m in messages if isinstance(m, dict) and isinstance(m.get("content"), list)
        for p in m["content"])
    if not has_video:
        return 0
    native = bool(mod and mod.get("video"))
    if mod is not None and not native and not mod.get("vision"):
        raise HTTPException(422, "video input requires a backend with video or "
                                 "vision support; the current model reports neither")
    rewritten = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        new_content = []
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "video_url"):
                new_content.append(part)
                continue
            url = (part.get("video_url") or {}).get("url") or ""
            if native:
                if not url.startswith("data:"):
                    raise HTTPException(400, "video_url must be a base64 data: URI "
                                             "(remote URLs are not fetched)")
                new_content.append({"type": "input_video",
                                    "input_video": {"data": url.partition(",")[2]}})
                rewritten += 1
                continue
            raw, mime = await asyncio.to_thread(_decode_data_uri, url)
            if not raw:
                raise HTTPException(400, "video_url must be a base64 data: URI "
                                         "(remote URLs are not fetched)")
            digest = await asyncio.to_thread(_sha256, raw)
            caption, frames = await cached_video_frames(
                digest, _video_bytes_to_blocks_sync, raw,
                VIDEO_MIME_EXT.get(mime, ".mp4"))
            new_content.append({"type": "text", "text": f"[Video: {caption}]"})
            new_content.extend(frames)
            rewritten += 1
        msg["content"] = new_content
    return rewritten


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except HTTPException:
        raise                                   # e.g. 413 from BodyLimit
    except Exception:
        raise HTTPException(400, "invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")
    requested = body.get("model")
    target = await pick_backend(requested if isinstance(requested, str) else None)
    endpoint = f"{target['base']}/v1/chat/completions"
    # Routing shows up in the proxy log (and so in the dashboard's proxy tab),
    # which makes an unroutable model name diagnosable at a glance.
    print(f"[route] requested={requested!r} -> :{target.get('port')} "
          f"({target.get('alias') or 'static default'})", flush=True)

    messages = body.get("messages")
    if isinstance(messages, list):
        # Audio first: it swaps parts 1:1 in place. The video pass may then
        # splice in frame lists, according to what the routed instance
        # supports. Both happen before forwarding, so a whisper failure is a
        # clean HTTP error even for a streaming request.
        await rewrite_audio_parts(messages)
        await rewrite_video_parts(messages, target.get("modalities"))
    # everything else — stream_options included — goes through untouched
    _vllm_model(body, target)

    headers = _client_headers(request, _REBODY_HEADERS)
    if body.get("stream"):
        return await stream_llama(body, endpoint, headers)
    return await forward_llama(body, endpoint, headers)


async def _relay_to(base: str, request: Request, path: str, backend_name: str,
                    content: Optional[bytes] = None) -> StreamingResponse:
    client = _http()
    upstream = client.build_request(
        request.method, f"{base}{path}", params=request.query_params,
        content=content if content is not None else await request.body(),
        headers=_client_headers(request), timeout=_LLAMA_T)
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.HTTPError as e:
        raise _transport_error(e, backend_name, base)

    async def body():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        except httpx.HTTPError as e:
            # Status and headers are already sent; all that's left is to end
            # the body early and say why in the log.
            print(f"[relay] {backend_name} {path}: backend failed mid-response "
                  f"({type(e).__name__}: {e})", flush=True)
        finally:
            await resp.aclose()

    return StreamingResponse(
        body(), status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items()
                 if k.lower() not in _HOP_HEADERS},
        background=BackgroundTask(resp.aclose))


def _join_content(a, b):
    """Concatenate two message contents, each a string or a list of parts."""
    if isinstance(a, str) and isinstance(b, str):
        return "\n\n".join(x for x in (a, b) if x)

    def parts(c):
        if isinstance(c, list):
            return list(c)
        return [{"type": "text", "text": c}] if isinstance(c, str) and c else []
    return parts(a) + parts(b)


def _merge_same_role(msgs: list) -> list:
    """Dropping a system turn can leave two user (or assistant) turns side by
    side, which templates that enforce alternation reject. Join them."""
    out = []
    for m in msgs:
        prev = out[-1] if out else None
        if (isinstance(m, dict) and isinstance(prev, dict) and m.get("role")
                and m.get("role") == prev.get("role")):
            out[-1] = {**prev, "content": _join_content(prev.get("content"),
                                                        m.get("content"))}
        else:
            out.append(m)
    return out


def _fold_system_messages(payload: dict) -> int:
    """Some chat templates abort with "System message must be at the
    beginning" when a system turn appears mid-conversation, which agent
    clients do emit. Fold those turns into the top-level system field
    instead of failing the request."""
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return 0
    folded, kept = [], []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str):
                folded.append(c)
            elif isinstance(c, list):
                folded.extend(b.get("text", "") for b in c
                              if isinstance(b, dict) and b.get("type") == "text")
        else:
            kept.append(m)
    if not folded:
        return 0
    payload["messages"] = _merge_same_role(kept)
    merged = "\n\n".join(x for x in folded if x)
    sys_param = payload.get("system")
    if isinstance(sys_param, str):
        payload["system"] = f"{sys_param}\n\n{merged}" if sys_param else merged
    elif isinstance(sys_param, list):
        sys_param.append({"type": "text", "text": merged})
    else:
        payload["system"] = merged
    return len(folded)


async def _route_anthropic(request: Request, path: str) -> StreamingResponse:
    """Anthropic-format endpoints route by the body's model field, exactly
    like /v1/chat/completions. Without this they fall through to the
    catch-all and always hit the default instance — silently answering from
    the wrong model. Errors come back in Anthropic's shape."""
    try:
        return await _route_anthropic_inner(request, path)
    except HTTPException as e:
        return _anthropic_error(e.status_code, str(e.detail))


async def _route_anthropic_inner(request: Request, path: str) -> StreamingResponse:
    content = await request.body()
    model, shape, payload = None, "", None
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            if isinstance(payload.get("model"), str):
                model = payload["model"]
            roles = [m.get("role") for m in payload.get("messages") or []
                     if isinstance(m, dict)]
            folded = _fold_system_messages(payload)
            if folded:
                content = json.dumps(payload).encode()
            shape = (f" msgs={len(roles)} roles={'/'.join(r or '?' for r in roles[:6])}"
                     f"{'…' if len(roles) > 6 else ''}"
                     f" tools={len(payload.get('tools') or [])}"
                     + (f" FOLDED {folded} system turn(s)" if folded else ""))
    except (ValueError, UnicodeDecodeError):
        pass
    target = await pick_backend(model)
    print(f"[route] anthropic {path} requested={model!r} -> :{target.get('port')} "
          f"({target.get('alias') or 'static default'}){shape}", flush=True)
    # vLLM (0.30 and later) serves the Anthropic Messages API itself
    if _is_vllm(target) and isinstance(payload, dict) \
            and _vllm_model(payload, target):
        content = json.dumps(payload).encode()
    return await _relay_to(target["base"], request, path, "LLM", content=content)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    return await _route_anthropic(request, "/v1/messages")


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    return await _route_anthropic(request, "/v1/messages/count_tokens")


# Registered before the catch-all so speech synthesis reaches the TTS
# backend instead of llama, which would reject it.
@app.post("/v1/audio/speech")
async def tts_speech(request: Request):
    content = await request.body()
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            # OpenAI's spec requires model and voice; fill this stack's
            # defaults so a bare {"input": "..."} works.
            payload.setdefault("model", config.TTS_DEFAULT_MODEL)
            if config.TTS_DEFAULT_VOICE:
                payload.setdefault("voice", config.TTS_DEFAULT_VOICE)
            content = json.dumps(payload).encode()
    except (ValueError, UnicodeDecodeError):
        pass
    return await _relay_to(TTS_ENDPOINT, request, "/v1/audio/speech", "TTS",
                           content=content)


@app.post("/upload_reference")
async def tts_upload_reference(request: Request):
    """Let clients add a voice-cloning reference, relayed to the TTS backend."""
    return await _relay_to(TTS_ENDPOINT, request, "/upload_reference", "TTS")


@app.get("/v1/audio/voices")
async def tts_voices():
    """Predefined voices plus any cloned reference files the backend holds."""
    timeout = httpx.Timeout(10, connect=CONNECT_TIMEOUT)
    try:
        resp = await _http().get(f"{TTS_ENDPOINT}/v1/audio/voices", timeout=timeout)
    except httpx.HTTPError:
        raise HTTPException(502, f"TTS backend unreachable at {TTS_ENDPOINT}")
    if resp.status_code != 200:
        raise HTTPException(502, f"TTS backend /v1/audio/voices returned "
                                 f"{resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(502, "TTS backend /v1/audio/voices returned non-JSON")
    voices = data.get("voices") if isinstance(data, dict) else data
    voices = list(voices) if isinstance(voices, list) else []
    try:
        r = await _http().get(f"{TTS_ENDPOINT}/get_reference_files", timeout=timeout)
        refs = r.json() if r.status_code == 200 else []
    except (httpx.HTTPError, ValueError):
        refs = []
    if not isinstance(refs, list):
        refs = []
    # Backends list voices as plain names or as objects; dedupe by name.
    seen = {v if isinstance(v, str) else str(v.get("name") or v.get("id") or "")
            for v in voices if isinstance(v, (str, dict))}
    cloned = [r for r in refs if isinstance(r, str) and r not in seen]
    return JSONResponse({"status": "ok", "voices": voices + cloned,
                         "cloned_voices": cloned,
                         "default_voice": config.TTS_DEFAULT_VOICE or None})


@app.api_route("/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def passthrough(request: Request, path: str):
    """Transparent relay to llama-server for everything not handled above.

    Registered last so it never shadows this proxy's own routes. It keeps
    llama's web UI, /metrics, /slots and friends working for clients that
    believe they are talking to llama-server directly.

    A JSON body naming a model (/v1/embeddings, /v1/completions,
    /v1/responses, /tokenize, /infill and the rest) routes by it, like chat
    does; anything else goes to the default instance.
    """
    content, model = None, None
    if request.method in ("POST", "PUT", "PATCH"):
        content = await request.body()
        model = _body_model(content)
    target = await pick_backend(model)
    if model:
        print(f"[route] /{path} requested={model!r} -> :{target.get('port')} "
              f"({target.get('alias') or 'static default'})", flush=True)
    if _is_vllm(target):
        first = path.strip("/").split("/")[0]
        if first in _LLAMA_ONLY:
            raise HTTPException(
                404, f"/{path} is a llama-server endpoint; the instance on "
                     f":{target.get('port')} ({target.get('alias')}) is vLLM, "
                     "which has no equivalent")
        content = _vllm_body(path, content, target)
    return await _relay_to(target["base"], request, f"/{path}", "LLM",
                           content=content)


# llama-server's own endpoints with no vLLM counterpart: answered here with
# a clear 404 rather than vLLM's bare {"detail": "Not Found"}.
_LLAMA_ONLY = {"props", "slots", "apply-template", "infill", "completion",
               "embedding", "lora-adapters", "reranking"}


def _vllm_body(path: str, content, target: dict):
    """Adapt a body for vLLM: its /tokenize and /detokenize want "prompt"
    and "model" where llama-server takes "content" (the answers overlap:
    both return "tokens"), and every named model must be the served id."""
    if not content or content.lstrip()[:1] != b"{":
        return content
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return content
    if not isinstance(payload, dict):
        return content
    changed = False
    if path.strip("/") == "tokenize" and "content" in payload \
            and "prompt" not in payload:
        payload["prompt"] = payload.pop("content")
        changed = True
    if path.strip("/") in ("tokenize", "detokenize") or "model" in payload:
        changed = _vllm_model(payload, target) or changed
    return json.dumps(payload).encode() if changed else content


def _body_model(content: bytes) -> Optional[str]:
    """The string `model` of a JSON-object body, else None."""
    if content.lstrip()[:1] != b"{":
        return None
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    return model if isinstance(model, str) and model else None


def main():
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
