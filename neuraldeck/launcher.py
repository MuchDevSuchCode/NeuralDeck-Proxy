"""Launching llama-server, portably.

This replaces the shell launcher the original deck used. Two things make it
more than a subprocess call:

  * llama.cpp builds differ in which flags they accept — speculative
    decoding types, flash attention taking a value, reasoning budgets. The
    launcher reads `--help` once per binary and passes only flags that
    binary understands, so a stock build and a fork both work.
  * a model can ship an MTP head its runtime cannot consume. The head is an
    optimisation, so a failure there retries without it rather than leaving
    you with nothing.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import httpx
import psutil

from . import config, models, procs

THINK_BUDGET = {"off": 0, "low": 1024, "medium": 4096, "high": -1}

_caps_cache: dict = {}


def caps(binary: str) -> dict:
    """What this llama-server build accepts, read from its own --help."""
    try:
        key = (binary, os.path.getmtime(binary))
    except OSError:
        key = (binary, 0)
    if key in _caps_cache:
        return _caps_cache[key]
    text = ""
    try:
        p = subprocess.run([binary, "--help"], capture_output=True, text=True,
                           timeout=30,
                           creationflags=(subprocess.CREATE_NO_WINDOW
                                          if config.IS_WINDOWS else 0))
        text = (p.stdout or "") + (p.stderr or "")
    except Exception:
        pass
    has = lambda flag: flag in text
    out = {
        "help_read": bool(text),
        "spec_type": has("--spec-type"),
        "spec_draft_n_max": has("--spec-draft-n-max"),
        "draft_gpu_layers": has("-ngld") or has("--n-gpu-layers-draft"),
        "model_draft": has("--model-draft") or has("-md"),
        "chat_template_kwargs": has("--chat-template-kwargs"),
        "reasoning_budget": has("--reasoning-budget"),
        "mmproj": has("--mmproj"),
        "embeddings": has("--embeddings"),
        "metrics": has("--metrics"),
        "jinja": has("--jinja"),
        "no_warmup": has("--no-warmup"),
        "cache_type_k": has("--cache-type-k") or has("-ctk"),
        "flash_attn": has("--flash-attn"),
        "temp": has("--temp"),
        "top_p": has("--top-p"),
        "min_p": has("--min-p"),
        "repeat_penalty": has("--repeat-penalty"),
        # Newer builds take a value: -fa on|off|auto. Older ones are a bare
        # switch and reject "on" as a positional argument.
        "flash_attn_value": bool(re.search(r"--flash-attn\s*\[?on\|off\|auto",
                                           text)),
        "spec_types": _spec_types(text),
    }
    _caps_cache[key] = out
    return out


def _spec_types(text: str) -> set:
    m = re.search(r"--spec-type\s+(\S+)", text)
    return set(m.group(1).split(",")) if m else set()


def _threads() -> int:
    if config.THREADS > 0:
        return config.THREADS
    return psutil.cpu_count(logical=False) or psutil.cpu_count() or 4


def build_argv(model: dict, *, binary: str, port: int, ctx: int, slots: int,
               thinking: str, spec: str, host: str = None) -> tuple:
    """(argv, notes) — the command line, plus what the deck should report."""
    c = caps(binary)
    notes = []
    argv = [binary,
            "-m", model["path"],
            "--alias", model["name"],
            "-c", str(ctx),
            "-np", str(slots),
            "-t", str(_threads()),
            "-tb", str(_threads()),
            "-ngl", str(config.N_GPU_LAYERS),
            "--host", host or config.LLAMA_HOST,
            "--port", str(port)]

    if c["flash_attn"]:
        argv += ["-fa", config.FLASH_ATTN] if c["flash_attn_value"] else ["-fa"]
    if c["jinja"]:
        argv.append("--jinja")
    if c["no_warmup"]:
        argv.append("--no-warmup")
    if c["metrics"]:
        argv.append("--metrics")
    if c["cache_type_k"] and config.KV_CACHE_TYPE:
        argv += ["-ctk", config.KV_CACHE_TYPE, "-ctv", config.KV_CACHE_TYPE]
    for flag, value, supported in (("--temp", config.TEMP, c["temp"]),
                                   ("--top-p", config.TOP_P, c["top_p"]),
                                   ("--min-p", config.MIN_P, c["min_p"]),
                                   ("--repeat-penalty", config.REPEAT_PENALTY,
                                    c["repeat_penalty"])):
        if supported and value is not None:
            argv += [flag, str(value)]

    # ── thinking ──────────────────────────────────────────────────────────
    budget = THINK_BUDGET.get(thinking, 0)
    if c["reasoning_budget"]:
        if budget == 0:
            argv += ["--reasoning-budget", "0"]
            # A budget of 0 alone does not render the template with
            # enable_thinking=false, and some templates then inject an
            # unclosed think tag into the system turn, which visibly
            # degrades output. Say it explicitly.
            if c["chat_template_kwargs"]:
                argv += ["--chat-template-kwargs",
                         json.dumps({"enable_thinking": False})]
        elif budget > 0:
            argv += ["--reasoning-budget", str(budget)]
        notes.append(f"thinking {thinking}"
                     + (f" ({budget} tok)" if budget > 0 else ""))
    elif thinking != "off":
        notes.append(f"thinking {thinking} requested but this build has no "
                     "--reasoning-budget — ignored")

    # ── vision ────────────────────────────────────────────────────────────
    if model.get("mmproj_path") and c["mmproj"]:
        argv += ["--mmproj", model["mmproj_path"]]
        notes.append(f"vision: {os.path.basename(model['mmproj_path'])}")

    # ── speculative decoding ──────────────────────────────────────────────
    if spec == "ngram":
        ngram = next((t for t in ("ngram-mod", "ngram-cache", "ngram-simple")
                      if t in c["spec_types"]), None)
        if ngram:
            argv += ["--spec-type", ngram]
            if c["spec_draft_n_max"]:
                argv += ["--spec-draft-n-max", "8"]
            notes.append(f"spec: {ngram} (drafts from context, no draft model)")
        else:
            notes.append("spec: ngram requested but this build has no ngram "
                         "draft type — plain decoding")
    elif spec == "auto" and model.get("draft_path"):
        if c["model_draft"] and "draft-mtp" in c["spec_types"]:
            argv += ["-md", model["draft_path"], "--spec-type", "draft-mtp"]
            if c["draft_gpu_layers"]:
                argv += ["-ngld", "999"]
            notes.append(f"spec: MTP head ({os.path.basename(model['draft_path'])})")
        elif c["model_draft"] and model["draft_path"] != model["path"]:
            # No unified draft-mtp type: a standalone head can still serve as
            # an ordinary draft model, which is most of the win.
            argv += ["-md", model["draft_path"]]
            if c["draft_gpu_layers"]:
                argv += ["-ngld", "999"]
            notes.append("spec: draft model (this build has no draft-mtp type)")
        else:
            notes.append("spec: model has an MTP head but this build cannot "
                         "use it — plain decoding")
    else:
        notes.append("spec: off")

    if model.get("embed") and c["embeddings"]:
        argv.append("--embeddings")
        notes.append("embedding server (--embeddings)")

    argv += list(config.EXTRA_LLAMA_ARGS)
    if not c["help_read"]:
        notes.append(f"warning: could not read `{os.path.basename(binary)} "
                     "--help`, so flags were not checked against this build")
    return argv, notes


def _spawn(argv: list, log_path: Path):
    """Start llama-server detached, with its output in its own log.

    Detached matters: the server must survive the dashboard restarting, and
    a Ctrl-C in the dashboard's console must not reach it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab", buffering=0)
    kwargs = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if config.IS_WINDOWS:
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                  | getattr(subprocess, "DETACHED_PROCESS", 0))
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(argv, **kwargs)
    finally:
        log.close()


class Launcher:
    """One launch at a time, with its progress readable over HTTP."""

    def __init__(self):
        self.buffer = deque(maxlen=400)
        self.started = None
        self.running = False
        self.returncode = None
        self._task = None

    # ── status ────────────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "running": self.running,
            "returncode": self.returncode,
            "elapsed": round(time.time() - self.started, 1) if self.started else None,
            "lines": list(self.buffer),
        }

    def say(self, line: str) -> None:
        self.buffer.append(line)
        print(f"[launch] {line}", flush=True)

    # ── the launch itself ─────────────────────────────────────────────────
    async def start(self, *, model_name: str, backend: str, ctx: int, slots: int,
                    thinking: str, spec: str, replace: bool, force: bool) -> dict:
        if self.running:
            raise RuntimeError("a launch is already in progress")
        model = models.by_name(model_name)
        if model is None:
            raise LookupError(f"unknown model '{model_name}'")
        binary = config.BACKENDS.get(backend)
        if not binary:
            raise LookupError(f"unknown backend '{backend}'")
        if not (os.path.isfile(binary) or Path(binary).name == binary):
            raise FileNotFoundError(f"llama-server not found: {binary}")

        instances = procs.llama_instances()
        port = config.LLAMA_PORTS[0]
        if replace:
            mode = "replace-all"
        elif instances:
            free = procs.next_free_llama_port()
            if free is None:
                raise RuntimeError(
                    f"no free llama port in {config.LLAMA_PORT_RANGE}")
            port, mode = free, "additive"
            if not force:
                self._check_vram(model)
        else:
            mode = "first"

        self.buffer.clear()
        self.running = True
        self.returncode = None
        self.started = time.time()
        self.say(f"launching {model['name']} · backend={backend} · slots={slots}"
                 f" · ctx={ctx} · thinking={thinking} · spec={spec}"
                 f" · port={port} · mode={mode}")
        self._task = asyncio.create_task(
            self._run(model, binary, backend, port, ctx, slots, thinking, spec,
                      replace))
        return {"launching": True, "model": model["name"], "backend": backend,
                "port": port, "mode": mode}

    def _check_vram(self, model: dict) -> None:
        """Refuse an additive launch that plainly will not fit.

        An allocation that overcommits does not fail cleanly on every
        runtime — it can hang in the driver instead — so this is checked
        before spawning rather than diagnosed afterwards.
        """
        from . import sysinfo
        g = sysinfo.gpu()
        total, used = g.get("vram_total_bytes"), g.get("vram_used_bytes")
        if not total:
            return
        free = total - (used or 0)
        need = model["vram_bytes"] * 1.1     # weights plus room for KV
        if need > free:
            raise MemoryError(
                f"insufficient VRAM: '{model['name']}' needs about "
                f"{model['vram_bytes'] / 1024**3:.1f} GiB (+KV) but only "
                f"{free / 1024**3:.1f} GiB of {total / 1024**3:.0f} GiB is "
                "free — stop an instance first, or force to try anyway")

    async def _run(self, model, binary, backend, port, ctx, slots, thinking,
                   spec, replace):
        try:
            if replace:
                n = await asyncio.to_thread(procs.stop_all_llama)
                self.say(f"stopped {n} running instance(s)")
            if procs.port_in_use(port):
                self.say(f"waiting for port {port} to be released…")
                if not await asyncio.to_thread(procs.wait_port_free, port, 30.0):
                    self.say(f"[error] port {port} is still in use — aborting")
                    self.returncode = 1
                    return

            log_path = config.LOG_DIR / f"{model['name']}.log"
            argv, notes = build_argv(model, binary=binary, port=port, ctx=ctx,
                                     slots=slots, thinking=thinking, spec=spec)
            for n in notes:
                self.say(n)
            self.say(f"ctx {ctx} total / {ctx // max(1, slots)} per slot")
            self.say(f"log: {log_path}")

            ok = await self._spawn_and_wait(argv, log_path, port)
            if not ok and _has_spec(argv):
                # The MTP/draft head is an optimisation; if the runtime could
                # not load it, serve the model plainly rather than nothing.
                plain = _strip_spec(argv)
                self.say("retrying without speculative decoding")
                ok = await self._spawn_and_wait(plain, log_path, port)
            self.returncode = 0 if ok else 1
            self.say("ready" if ok else "[error] launch failed")
        except Exception as e:                       # never leave it "running"
            self.say(f"[error] {e}")
            self.returncode = 1
        finally:
            self.running = False

    async def _spawn_and_wait(self, argv: list, log_path: Path, port: int,
                              max_wait: float = 600.0) -> bool:
        self.say("$ " + _display(argv))
        try:
            proc = await asyncio.to_thread(_spawn, argv, log_path)
        except Exception as e:
            self.say(f"[error] could not start llama-server: {e}")
            return False
        self.say(f"pid {proc.pid} — waiting for the server to become ready")
        deadline = time.monotonic() + max_wait
        url = f"http://127.0.0.1:{port}/health"
        async with httpx.AsyncClient(timeout=3.0) as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    tail = procs.tail_text(log_path, 4096).splitlines()[-8:]
                    self.say(f"[error] llama-server exited "
                             f"(code {proc.returncode}):")
                    for ln in tail:
                        self.say("    " + ln)
                    return False
                try:
                    r = await client.get(url)
                    if r.status_code == 200 and '"ok"' in r.text:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2.0)
                waited = int(max_wait - (deadline - time.monotonic()))
                if waited and waited % 20 == 0:
                    self.say(f"still loading… {waited}s")
        self.say(f"[error] not ready within {int(max_wait)}s")
        return False


_SPEC_FLAGS = {"--spec-type", "-md", "--model-draft", "--spec-draft-model",
               "-ngld", "--n-gpu-layers-draft", "--spec-draft-n-max"}


def _has_spec(argv: list) -> bool:
    return any(a in _SPEC_FLAGS for a in argv)


def _strip_spec(argv: list) -> list:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in _SPEC_FLAGS:
            skip = True          # every speculation flag takes a value
            continue
        out.append(a)
    return out


def _display(argv: list) -> str:
    """The command line as you would type it, so a failed launch can be
    reproduced by hand from the log."""
    if config.IS_WINDOWS:
        return subprocess.list2cmdline(argv)
    import shlex
    return shlex.join(argv)
