#!/usr/bin/env python3
"""Multimodal proxy — the client-facing endpoint.

Apps point at this server as if it were llama-server itself. It owns the
client port; llama-server instances sit behind it. What it adds:

  * multi-instance routing — several llama-servers are discovered by
    probing /props, and a request's "model" field picks one
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
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
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
# Instance registry: discovered by probing, never from a state file, so a
# stopped instance simply vanishes on the next pass.
# ---------------------------------------------------------------------------

_instances_cache = {"ts": 0.0, "list": []}


def _norm_model(s: str) -> str:
    s = (s or "").split("/")[-1]          # strip hf-repo/path prefixes
    return re.sub(r"[._]", "-", re.sub(r"(\.gguf|-gguf)$", "", s.lower()))


def _route_match(requested: str, instances: list) -> Optional[dict]:
    """Best instance for a requested name: exact normalised match, then
    longest prefix overlap, then longest substring — so family names
    (foo-it-qat vs foo-it-qat-heretic) cannot collide when the client sends
    the exact id."""
    req = _norm_model(requested)
    if not req:
        return None
    scored = []
    for inst in instances:
        alias = _norm_model(inst.get("alias"))
        if not alias:
            continue
        if alias == req:
            return inst
        if alias.startswith(req) or req.startswith(alias):
            scored.append((2, len(alias), inst))
        elif alias in req or req in alias:
            scored.append((1, len(alias), inst))
    return max(scored, key=lambda t: t[:2])[2] if scored else None


async def discover_instances(ttl: float = 10.0) -> list:
    now = time.monotonic()
    if now - _instances_cache["ts"] < ttl:
        return _instances_cache["list"]

    async def probe(port: int):
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{base}/props")
            if resp.status_code != 200:
                return None
            props = resp.json()
            alias = (props.get("model_alias")
                     or os.path.basename(props.get("model_path") or ""))
            return {"port": port, "base": base, "alias": alias,
                    "modalities": props.get("modalities") or {}}
        except Exception:
            return None

    found = await asyncio.gather(*[probe(p) for p in config.LLAMA_PORTS])
    instances = [f for f in found if f]
    _instances_cache.update(ts=now, list=instances)
    return instances


async def pick_backend(model: Optional[str]) -> dict:
    """Choose the instance for a request.

    A model that was explicitly asked for but is not serving raises instead
    of quietly falling back: answering from a different model than the one
    named misattributes benchmarks and hands back output the caller believes
    came from elsewhere. An instance still loading fails /props, so it is
    not a match either.
    """
    instances = await discover_instances()
    if model:
        matched = _route_match(model, instances)
        if matched:
            return matched
        if instances and config.STRICT_MODEL_ROUTING:
            serving = ", ".join(i["alias"] or f":{i['port']}" for i in instances)
            raise HTTPException(
                404, f"model '{model}' is not being served (loading, stopped, "
                     f"or never launched). Currently serving: {serving or 'nothing'}")
    for inst in instances:
        if inst["base"] == LLAMA_BASE:
            return inst
    if instances:
        return instances[0]
    return {"port": None, "base": LLAMA_BASE, "alias": None, "modalities": None}


async def probe_backend(endpoint: str) -> bool:
    base = _base_url(endpoint)
    async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT) as client:
        for url in (f"{base}/health", base):
            try:
                await client.get(url)
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
    yield


app = FastAPI(title="Multimodal Orchestrator", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Modality handlers
# ---------------------------------------------------------------------------

async def transcribe_audio(filename: str, data: bytes, content_type: str) -> str:
    try:
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(WHISPER_TIMEOUT, connect=CONNECT_TIMEOUT)) as client:
            resp = await client.post(
                WHISPER_ENDPOINT, files={"file": (filename, data, content_type)},
                data={"response_format": "json"})
    except httpx.ConnectError:
        raise HTTPException(502, f"Whisper backend unreachable at {WHISPER_ENDPOINT}")
    except httpx.TimeoutException:
        raise HTTPException(504, f"Whisper backend timed out transcribing '{filename}'")
    if resp.status_code != 200:
        raise HTTPException(502, f"Whisper backend returned {resp.status_code} "
                                 f"for '{filename}': {resp.text[:500]}")
    return resp.json().get("text", "").strip()


def _extract_frames_sync(video_path: Path, out_dir: Path) -> List[Path]:
    ff = ffmpeg_bin()
    if ff is None:
        raise HTTPException(
            501, "video input needs ffmpeg, which is not installed (or set "
                 "NEURALDECK_FFMPEG to its path)")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", str(video_path),
           "-vf", f"fps={VIDEO_FPS}", "-frames:v", str(MAX_FRAMES), "-q:v", "2",
           str(out_dir / "frame_%04d.jpg")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          creationflags=(subprocess.CREATE_NO_WINDOW
                                         if config.IS_WINDOWS else 0))
    if proc.returncode != 0:
        raise HTTPException(422, f"ffmpeg failed to extract frames from "
                                 f"'{video_path.name}': {proc.stderr.strip()[:500]}")
    return sorted(out_dir.glob("frame_*.jpg"))


async def extract_video_frames(upload: UploadFile, data: bytes,
                               workdir: Path) -> List[bytes]:
    suffix = Path(upload.filename or "video.mp4").suffix or ".mp4"
    video_path = workdir / f"input{suffix}"
    video_path.write_bytes(data)
    frames_dir = workdir / "frames"
    frames_dir.mkdir()
    frame_paths = await asyncio.to_thread(_extract_frames_sync, video_path, frames_dir)
    if not frame_paths:
        raise HTTPException(422, f"No frames could be extracted from "
                                 f"'{upload.filename}'")
    return [p.read_bytes() for p in frame_paths]


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
                          "instances": [{"port": i["port"], "model": i["alias"]}
                                        for i in instances]},
                "whisper": {"endpoint": WHISPER_ENDPOINT, "reachable": whisper_ok},
                "tts": {"endpoint": TTS_ENDPOINT, "reachable": tts_ok},
            },
        })


@app.post("/v1/multimodal")
async def multimodal(
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

    tmpdir = tempfile.mkdtemp(prefix="multimodal_")
    try:
        content: List[dict] = []
        # Images and video frames go before the text: vision-capable chat
        # templates are trained that way round.
        for img in images:
            data = await img.read()
            if data:
                content.append(image_to_content_block(data, guess_image_mime(img)))
        for i, vid in enumerate(videos):
            data = await vid.read()
            if not data:
                continue
            workdir = Path(tmpdir) / f"video_{i}"
            workdir.mkdir()
            frames = await extract_video_frames(vid, data, workdir)
            label = vid.filename or f"video {i + 1}"
            content.append({"type": "text",
                            "text": f"[Video '{label}': {len(frames)} frames "
                                    f"extracted at {VIDEO_FPS} fps]"})
            for frame in frames:
                content.append(image_to_content_block(frame, "image/jpeg"))
        if text:
            content.append({"type": "text", "text": text})

        audio_jobs = [(a, await a.read()) for a in audio]
        audio_jobs = [(a, d) for a, d in audio_jobs if d]
        transcripts = await asyncio.gather(*[
            transcribe_audio(a.filename or "audio", d,
                             a.content_type or "application/octet-stream")
            for a, d in audio_jobs])
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
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        target = await pick_backend(model)
        endpoint = f"{target['base']}/v1/chat/completions"
        if stream:
            return StreamingResponse(stream_llama(payload, tmpdir, endpoint=endpoint),
                                     media_type="text/event-stream")
        return await forward_llama(payload, endpoint)
    finally:
        # A streaming response cleans up the tempdir when the stream ends.
        if not stream:
            shutil.rmtree(tmpdir, ignore_errors=True)


async def forward_llama(payload: dict, endpoint: str = None) -> JSONResponse:
    endpoint = endpoint or LLAMA_ENDPOINT
    try:
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(LLAMA_TIMEOUT, connect=CONNECT_TIMEOUT)) as client:
            resp = await client.post(endpoint, json=payload)
    except httpx.ConnectError:
        raise HTTPException(502, f"LLM backend unreachable at {endpoint}")
    except httpx.TimeoutException:
        raise HTTPException(504, "LLM backend timed out")
    if resp.status_code != 200:
        raise HTTPException(502, f"LLM backend returned {resp.status_code}: "
                                 f"{resp.text[:500]}")
    return JSONResponse(content=resp.json())


async def stream_llama(payload: dict, tmpdir: Optional[str] = None,
                       endpoint: str = None):
    endpoint = endpoint or LLAMA_ENDPOINT
    try:
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(LLAMA_TIMEOUT, connect=CONNECT_TIMEOUT)) as client:
            try:
                async with client.stream("POST", endpoint, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield ("data: " + json.dumps({
                            "error": f"LLM backend returned {resp.status_code}: "
                                     f"{body.decode(errors='replace')[:500]}"}) + "\n\n")
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except httpx.ConnectError:
                yield ("data: " + json.dumps(
                    {"error": f"LLM backend unreachable at {endpoint}"}) + "\n\n")
            except httpx.TimeoutException:
                yield 'data: {"error": "LLM backend timed out"}\n\n'
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


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
    """
    target = await pick_backend(None)
    try:
        async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT) as client:
            resp = await client.get(f"{target['base']}/props")
    except httpx.HTTPError:
        raise HTTPException(502, f"LLM backend unreachable at {target['base']}")
    if resp.status_code != 200:
        raise HTTPException(502, f"LLM backend /props returned {resp.status_code}")
    body = resp.json()
    reported = body.get("modalities") or {}
    vision = bool(reported.get("vision"))
    body["modalities"] = {
        "vision": vision,
        "audio": await whisper_reachable(),
        "video": bool(reported.get("video"))
                 or (vision and ffmpeg_bin() is not None),
    }
    body["llama_instances"] = [
        {"port": i["port"], "model": i["alias"], "modalities": i["modalities"]}
        for i in await discover_instances()]
    return JSONResponse(content=body)


@app.get("/v1/models")
async def models_aggregate():
    """Union of /v1/models across instances, so a client can see and select
    any served model — its choice then routes the chat."""
    instances = await discover_instances() or [{"base": LLAMA_BASE, "port": None}]
    seen, data = set(), []
    async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT) as client:
        for inst in instances:
            try:
                resp = await client.get(f"{inst['base']}/v1/models")
                if resp.status_code != 200:
                    continue
                for m in resp.json().get("data", []):
                    if m.get("id") in seen:
                        continue
                    seen.add(m.get("id"))
                    m["port"] = inst.get("port")
                    data.append(m)
            except httpx.HTTPError:
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
                raw = base64.b64decode(ia.get("data") or "")
            except Exception:
                raise HTTPException(400, f"invalid base64 in input_audio part "
                                         f"(message {mi})")
            if not raw:
                raise HTTPException(400, f"empty input_audio part (message {mi})")
            jobs.append((mi, pi, raw, fmt))
    if not jobs:
        return 0
    transcripts = await asyncio.gather(*[
        transcribe_audio(f"audio_{i}.{fmt}", raw,
                         AUDIO_FORMAT_MIME.get(fmt, "application/octet-stream"))
        for i, (_, _, raw, fmt) in enumerate(jobs)])
    for (mi, pi, _, _), text in zip(jobs, transcripts):
        messages[mi]["content"][pi] = {"type": "text",
                                       "text": f"[Audio transcript]: {text}"}
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
            raw, mime = _decode_data_uri(url)
            if not raw:
                raise HTTPException(400, "video_url must be a base64 data: URI "
                                         "(remote URLs are not fetched)")
            tmpdir = tempfile.mkdtemp(prefix="video_part_")
            try:
                video_path = Path(tmpdir) / f"input{VIDEO_MIME_EXT.get(mime, '.mp4')}"
                video_path.write_bytes(raw)
                frames_dir = Path(tmpdir) / "frames"
                frames_dir.mkdir()
                frame_paths = await asyncio.to_thread(_extract_frames_sync,
                                                      video_path, frames_dir)
                if not frame_paths:
                    raise HTTPException(422, "no frames could be extracted from "
                                             "video_url part")
                new_content.append({"type": "text",
                                    "text": f"[Video: {len(frame_paths)} frames "
                                            f"extracted at {VIDEO_FPS} fps]"})
                for fp in frame_paths:
                    new_content.append(
                        image_to_content_block(fp.read_bytes(), "image/jpeg"))
                rewritten += 1
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        msg["content"] = new_content
    return rewritten


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON body")
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

    if body.get("stream"):
        return StreamingResponse(stream_llama(body, endpoint=endpoint),
                                 media_type="text/event-stream")
    return await forward_llama(body, endpoint)


_HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "upgrade",
                "host", "content-length", "proxy-authenticate",
                "proxy-authorization", "te", "trailer", "date", "server"}


async def _relay_to(base: str, request: Request, path: str, backend_name: str,
                    content: Optional[bytes] = None) -> StreamingResponse:
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(LLAMA_TIMEOUT, connect=CONNECT_TIMEOUT))
    upstream = client.build_request(
        request.method, f"{base}{path}", params=request.query_params,
        content=content if content is not None else await request.body(),
        headers={k: v for k, v in request.headers.items()
                 if k.lower() not in _HOP_HEADERS})
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        raise HTTPException(502, f"{backend_name} backend unreachable at {base}")
    except httpx.TimeoutException:
        await client.aclose()
        raise HTTPException(504, f"{backend_name} backend timed out")

    async def cleanup():
        await resp.aclose()
        await client.aclose()

    return StreamingResponse(
        resp.aiter_raw(), status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items()
                 if k.lower() not in _HOP_HEADERS},
        background=BackgroundTask(cleanup))


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
    payload["messages"] = kept
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
    the wrong model."""
    content = await request.body()
    model, shape = None, ""
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
    async with httpx.AsyncClient(
            timeout=httpx.Timeout(10, connect=CONNECT_TIMEOUT)) as client:
        try:
            resp = await client.get(f"{TTS_ENDPOINT}/v1/audio/voices")
            voices = list(resp.json().get("voices") or [])
        except httpx.HTTPError:
            raise HTTPException(502, f"TTS backend unreachable at {TTS_ENDPOINT}")
        try:
            refs = (await client.get(f"{TTS_ENDPOINT}/get_reference_files")).json()
        except (httpx.HTTPError, ValueError):
            refs = []
    seen = set(voices)
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
    """
    target = await pick_backend(None)
    return await _relay_to(target["base"], request, f"/{path}", "LLM")


def main():
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
