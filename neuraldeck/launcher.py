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

A backend whose executable is `vllm` gets the same treatment: its flags are
read from `vllm serve --help=all`, it is sized to the VRAM that is actually
free so it can run beside llama-servers, and it goes through the same
readiness wait, timeout and retry-without-speculation path.
"""

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
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
                           encoding="utf-8", errors="replace", timeout=30,
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
        # -rea/--reasoning on|off|auto replaced enable_thinking passed
        # through --chat-template-kwargs, which newer builds warn about.
        "reasoning": bool(re.search(r"--reasoning\s+\[?on\|off", text)),
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


# ── vLLM ───────────────────────────────────────────────────────────────────

_vllm_caps_cache: dict = {}
_vllm_version_cache: dict = {}
_vllm_env_cache: dict = {}
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if config.IS_WINDOWS else 0


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _vllm_help(binary: str) -> str:
    """`vllm serve --help=all`, kept on disk per binary and mtime.

    Plain --help lists only the config groups, not the flags, and reading
    either imports torch — seconds at best, a minute on a cold disk — so a
    dashboard restart should not pay for it again.
    """
    key = hashlib.sha1(f"{binary}|{_mtime(binary)}".encode()).hexdigest()[:16]
    cache = config.RUN_DIR / f"vllm-help-{key}.txt"
    try:
        text = cache.read_text(encoding="utf-8")
        if text.strip():
            return text
    except OSError:
        pass
    text = ""
    for args in (["serve", "--help=all"], ["serve", "--help"]):
        try:
            p = subprocess.run([binary, *args], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=180,
                               env=vllm_env(binary), creationflags=_NO_WINDOW)
            text = (p.stdout or "") + (p.stderr or "")
        except Exception:
            text = ""
        if "--max-model-len" in text:
            break
    if "--max-model-len" in text:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        except OSError:
            pass
    return text


def _choices(text: str, flag: str) -> set:
    m = re.search(re.escape(flag) + r"\s+\{([^}]*)\}", text)
    return set(m.group(1).split(",")) if m else set()


def vllm_caps(binary: str) -> dict:
    """What this vLLM accepts, read from its own help, cached."""
    key = (binary, _mtime(binary))
    if key in _vllm_caps_cache:
        return _vllm_caps_cache[key]
    text = _vllm_help(binary)
    flags = set(re.findall(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)", text))
    out = {
        "help_read": "--max-model-len" in text,
        "flags": flags,
        "tool_parsers": _choices(text, "--tool-call-parser"),
        "kv_cache_dtypes": _choices(text, "--kv-cache-dtype"),
    }
    if out["help_read"]:
        _vllm_caps_cache[key] = out
    return out


def vllm_version(binary: str):
    """`vllm --version`, or None when it does not run."""
    key = (binary, _mtime(binary))
    if key not in _vllm_version_cache:
        ver = None
        try:
            p = subprocess.run([binary, "--version"], capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               timeout=120, env=vllm_env(binary),
                               creationflags=_NO_WINDOW)
            lines = [ln.strip() for ln in (p.stdout or "").splitlines()
                     if ln.strip()]
            if p.returncode == 0 and lines:
                ver = lines[-1]
        except Exception:
            ver = None
        _vllm_version_cache[key] = ver
    return _vllm_version_cache[key]


# Finds the CUDA toolkit pip ships inside the environment (nvidia-cuda-nvcc
# for cu12, the nvidia/cu13 bundle for cu13): flashinfer JIT-compiles
# kernels and needs an nvcc matching the torch build, which a system CUDA
# often is not.
_NVCC_PROBE = r"""
import os
try:
    import nvidia
except Exception:
    raise SystemExit
for base in list(getattr(nvidia, "__path__", [])):
    for sub in ("cu13", "cu12", "cuda_nvcc"):
        d = os.path.join(base, sub)
        if any(os.path.isfile(os.path.join(d, "bin", n))
               for n in ("nvcc", "nvcc.exe")):
            print(d)
            raise SystemExit
"""


def _env_cuda_home(bindir: str):
    python = next((os.path.join(bindir, n) for n in
                   ("python", "python3", "python.exe")
                   if os.path.isfile(os.path.join(bindir, n))), None)
    if python is None:
        return None
    try:
        p = subprocess.run([python, "-c", _NVCC_PROBE], capture_output=True,
                           text=True, timeout=30, creationflags=_NO_WINDOW)
        found = (p.stdout or "").strip().splitlines()
        return found[0] if found and os.path.isdir(found[0]) else None
    except Exception:
        return None


def vllm_env(binary: str) -> dict:
    """Environment for a vLLM child: its env's bin first on PATH (flashinfer
    JIT runs the env's nvcc and ninja), CUDA_HOME at the env's own toolkit
    when it ships one, and flashinfer's sampler off unless asked for — it
    JIT-compiles on first use and has failed to on consumer cards."""
    bindir = os.path.dirname(os.path.abspath(binary)) if os.path.dirname(binary) \
        else os.path.dirname(shutil.which(binary) or "")
    env = dict(os.environ)
    if bindir:
        env["PATH"] = os.pathsep.join([bindir] + ([env["PATH"]]
                                                  if env.get("PATH") else []))
        if bindir not in _vllm_env_cache:
            _vllm_env_cache[bindir] = _env_cuda_home(bindir)
        if _vllm_env_cache[bindir]:
            env["CUDA_HOME"] = _vllm_env_cache[bindir]
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env


VLLM_MARGIN = 512 * 1024**2          # left free beside vLLM's claim
VLLM_OVERHEAD = int(1.5 * 1024**3)   # activations, CUDA graphs
VLLM_KV_MIN = 1024**3                # the least KV worth starting with


def vllm_budget(model: dict, force: bool = False) -> tuple:
    """(gpu_memory_utilization, note) for a vLLM launch, sized to free VRAM.

    vLLM reads --gpu-memory-utilization as the share of the card's TOTAL
    memory this instance may take, and refuses to start unless that much is
    free — so the fraction is what is free now, less a margin, capped by the
    setting. MemoryError when the model plainly cannot fit (unless forced).
    """
    from . import sysinfo
    g = sysinfo.gpu()
    total, used = g.get("vram_total_bytes"), g.get("vram_used_bytes")
    cap = config.VLLM_GPU_FRAC_MAX
    if not total:
        return cap, f"VRAM unknown — --gpu-memory-utilization {cap}"
    free = total - (used or 0)
    frac = math.floor(min(cap, (free - VLLM_MARGIN) / total) * 100) / 100
    need = (model.get("vram_bytes") or 0) + VLLM_OVERHEAD + VLLM_KV_MIN
    gib = lambda b: b / 1024**3
    note = (f"--gpu-memory-utilization {frac:.2f} ({gib(frac * total):.1f} of "
            f"{gib(total):.0f} GiB; {gib(free):.1f} GiB free, "
            f"model needs ~{gib(need):.1f} GiB)")
    if frac * total < need:
        if not force:
            raise MemoryError(
                f"vLLM needs ~{gib(need):.1f} GiB of VRAM but only "
                f"{gib(max(0, free - VLLM_MARGIN)):.1f} GiB is free — stop a "
                "model or tick replace")
        frac = max(frac, 0.05)
        note += " — forced, may not fit"
    return frac, note


def mtp_method(model: dict) -> str:
    """vLLM's speculative method for a model's own MTP layers."""
    arch = " ".join([str(model.get("arch") or "")]
                    + list(model.get("architectures") or [])).lower()
    if "qwen3_next" in arch or re.search(r"qwen3[._]?\d", arch):
        return "qwen3_next_mtp"
    if "deepseek" in arch:
        return "deepseek_mtp"
    return "mtp"


def _family(model: dict) -> str:
    arch = " ".join([str(model.get("arch") or "")]
                    + list(model.get("architectures") or [])).lower()
    if "qwen3" in arch:
        return "qwen3"
    if "qwen" in arch:
        return "qwen"
    return ""


def build_vllm_argv(model: dict, *, binary: str, port: int, ctx: int,
                    slots: int, thinking: str, spec: str, gpu_frac: float,
                    host: str = None) -> tuple:
    """(argv, notes) for `vllm serve` — only flags this vLLM's help lists."""
    c = vllm_caps(binary)
    has = (lambda f: f in c["flags"]) if c["help_read"] else (lambda f: True)
    notes = []
    argv = [binary, "serve", model["path"],
            "--served-model-name", model["name"],
            "--host", host or config.LLAMA_HOST,
            "--port", str(port),
            "--max-model-len", str(ctx)]
    if has("--max-num-seqs"):
        argv += ["--max-num-seqs", str(slots)]
    if has("--gpu-memory-utilization"):
        argv += ["--gpu-memory-utilization", f"{gpu_frac:.2f}"]
    if has("--enable-prefix-caching"):
        argv.append("--enable-prefix-caching")
    kv = config.VLLM_KV_CACHE_DTYPE
    if kv and kv != "auto":
        if has("--kv-cache-dtype") and (not c["kv_cache_dtypes"]
                                        or kv in c["kv_cache_dtypes"]):
            argv += ["--kv-cache-dtype", kv]
            notes.append(f"kv cache: {kv}")
        else:
            notes.append(f"kv cache: {kv} not accepted by this vLLM — auto")

    # ── reasoning and tool calls ──────────────────────────────────────────
    family = _family(model)
    if family == "qwen3" and model.get("thinking") and has("--reasoning-parser"):
        argv += ["--reasoning-parser", "qwen3"]
        notes.append("reasoning parser: qwen3")
    if family:
        parser = "qwen3_xml" if model.get("tool_xml") else "hermes"
        if (has("--tool-call-parser") and has("--enable-auto-tool-choice")
                and (not c["tool_parsers"] or parser in c["tool_parsers"])):
            argv += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
            notes.append(f"tool calls: {parser}")
    else:
        notes.append("tool/reasoning parsers: unknown model family — none set")

    # ── thinking ──────────────────────────────────────────────────────────
    if model.get("thinking"):
        if thinking == "off":
            if has("--default-chat-template-kwargs"):
                argv += ["--default-chat-template-kwargs",
                         json.dumps({"enable_thinking": False})]
                notes.append("thinking off (enable_thinking=false by default; "
                             "a request can still turn it on)")
            else:
                notes.append("thinking off requested but this vLLM has no "
                             "--default-chat-template-kwargs — the template "
                             "decides")
        else:
            notes.append(f"thinking {thinking} (vLLM has no reasoning "
                         "budget: unbudgeted)")

    # ── speculative decoding ──────────────────────────────────────────────
    if spec in ("auto", "mtp") and model.get("mtp"):
        if has("--speculative-config"):
            method = mtp_method(model)
            argv += ["--speculative-config", json.dumps(
                {"method": method, "num_speculative_tokens": 2})]
            notes.append(f"spec: {method} (the model's own MTP layers, 2 tokens)")
        else:
            notes.append("spec: model has MTP layers but this vLLM has no "
                         "--speculative-config — plain decoding")
    elif spec == "ngram" and has("--speculative-config"):
        argv += ["--speculative-config", json.dumps(
            {"method": "ngram", "num_speculative_tokens": 4,
             "prompt_lookup_max": 4})]
        notes.append("spec: ngram (drafts from context, no draft model)")
    else:
        notes.append("spec: off")

    argv += list(config.VLLM_EXTRA_ARGS)
    if not c["help_read"]:
        notes.append(f"warning: could not read `{os.path.basename(binary)} "
                     "serve --help=all`, so flags were not checked")
    return argv, notes


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
    if c["reasoning"]:
        argv += ["--reasoning", "off" if budget == 0 else "on"]
    if c["reasoning_budget"]:
        if budget == 0:
            argv += ["--reasoning-budget", "0"]
            # A budget of 0 alone does not render the template with
            # enable_thinking=false, and some templates then inject an
            # unclosed think tag into the system turn, which visibly
            # degrades output. Say it explicitly — with --reasoning off
            # where the build has it, the older kwargs route otherwise.
            if not c["reasoning"] and c["chat_template_kwargs"]:
                argv += ["--chat-template-kwargs",
                         json.dumps({"enable_thinking": False})]
        elif budget > 0:
            argv += ["--reasoning-budget", str(budget)]
        notes.append(f"thinking {thinking}"
                     + (f" ({budget} tok)" if budget > 0 else ""))
    elif c["reasoning"]:
        notes.append(f"thinking {thinking}"
                     + (" (no --reasoning-budget in this build: unbudgeted)"
                        if budget > 0 else ""))
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
        if model["draft_path"] == model["path"] and "draft-mtp" in c["spec_types"]:
            # Combined build: the head is inside the weights already, and
            # naming the same file as -md would load every weight twice.
            argv += ["--spec-type", "draft-mtp"]
            notes.append("spec: MTP head (inside the model file)")
        elif (c["model_draft"] and "draft-mtp" in c["spec_types"]
              and model["draft_path"] != model["path"]):
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


def _spawn(argv: list, log_path: Path, env: dict = None, header: str = None):
    """Start a model server detached, with its output in its own log.

    Detached matters: the server must survive the dashboard restarting, and
    a Ctrl-C in the dashboard's console must not reach it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab", buffering=0)
    if header:
        log.write((header + "\n").encode("utf-8"))
    kwargs = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if env is not None:
        kwargs["env"] = env
    if config.IS_WINDOWS:
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                  | getattr(subprocess, "DETACHED_PROCESS", 0))
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(argv, **kwargs)
    finally:
        log.close()


def _rotate(log_path: Path) -> None:
    """Start each launch on a fresh log, keeping the previous run as .1.

    Everything that reads an instance log (headline numbers, KV, traffic)
    wants the current run only, and an append-forever log also grows
    without bound.
    """
    try:
        if log_path.exists() and log_path.stat().st_size > 0:
            os.replace(log_path, log_path.with_name(log_path.name + ".1"))
    except OSError:
        pass            # held open elsewhere (Windows): append instead


class Launcher:
    """One launch at a time, with its progress readable over HTTP."""

    def __init__(self):
        self.buffer = deque(maxlen=400)
        self.started = None
        self.running = False
        self.returncode = None
        self._task = None
        self._claimed = False

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
                    thinking: str, spec: str, replace: bool, force: bool,
                    relaunch: bool = False) -> dict:
        # The checks read GGUF headers, walk the process table and probe the
        # GPU, so they run in a thread. The claim is taken before the first
        # await, so two clicks cannot both get past it.
        if self.running or self._claimed:
            raise RuntimeError("a launch is already in progress")
        self._claimed = True
        try:
            model, binary, port, mode, stop = await asyncio.to_thread(
                self._plan, model_name, backend, replace, relaunch, force)
            self.buffer.clear()
            self.running = True
        finally:
            self._claimed = False
        self.returncode = None
        self.started = time.time()
        self.say(f"launching {model['name']} · backend={backend} · slots={slots}"
                 f" · ctx={ctx} · thinking={thinking} · spec={spec}"
                 f" · port={port} · mode={mode}")
        self._task = asyncio.create_task(
            self._run(model, binary, backend, port, ctx, slots, thinking, spec,
                      replace, stop, force))
        return {"launching": True, "model": model["name"], "backend": backend,
                "port": port, "mode": mode}

    def _plan(self, model_name, backend, replace, relaunch, force) -> tuple:
        """(model, binary, port, mode, pids to stop first). Sync: runs in a
        thread, and raises what the API turns into 400/404/409/500."""
        model = models.by_name(model_name)
        if model is None:
            raise LookupError(f"unknown model '{model_name}'")
        binary = config.BACKENDS.get(backend)
        if not binary:
            raise LookupError(f"unknown backend '{backend}'")
        kind = config.backend_kind(binary)
        fmt = model.get("format", "gguf")
        # ValueError is the API's 400: a plain mismatch the user can fix by
        # picking the other backend
        if fmt == "hf" and kind != "vllm":
            raise ValueError(f"'{model['name']}' is a safetensors model — "
                             "pick a vLLM backend")
        if fmt != "hf" and kind == "vllm":
            raise ValueError(f"'{model['name']}' is a GGUF model — pick a "
                             "llama.cpp backend (vLLM serves safetensors "
                             "folders)")
        if not (os.path.isfile(binary) or Path(binary).name == binary):
            raise FileNotFoundError(
                f"{'vllm' if kind == 'vllm' else 'llama-server'} not found: "
                f"{binary}")

        instances = procs.llama_instances()
        # Same alias means same log and same model id at the proxy: a second
        # copy beside the first is never what was meant.
        same = [i for i in instances if i.get("alias") == model["name"]]
        if same and not (replace or relaunch):
            raise RuntimeError(
                f"'{model['name']}' is already serving on :{same[0]['port']} "
                "— use Relaunch, or tick replace")
        ours = [i for i in instances if i.get("port") in config.LLAMA_PORTS]
        stop = []
        if replace:
            mode = "replace-all"
            # stop_all_llama covers the deck's ports; a copy of this very
            # model elsewhere would still collide on alias and log
            stop = [i["pid"] for i in same if i.get("pid")]
            # Provisional: the first deck port, unless something that is not
            # one of our instances holds it. Re-checked once they are down.
            held = {i["port"] for i in ours}
            port = next((p for p in config.LLAMA_PORTS
                         if p in held or not procs.llama_port_busy(p)), None)
        elif same:
            mode = "relaunch"
            stop = [i["pid"] for i in same if i.get("pid")]
            port = same[0]["port"]
            if port not in config.LLAMA_PORTS:
                port = procs.next_free_llama_port(instances)
        elif instances:
            port, mode = procs.next_free_llama_port(instances), "additive"
            if port is not None and not force and kind != "vllm":
                self._check_vram(model)
        else:
            port, mode = procs.next_free_llama_port(instances), "first"
        if port is None:
            raise RuntimeError(f"no free llama port in {config.LLAMA_PORT_RANGE}")
        if kind == "vllm" and not stop and not replace:
            # Nothing will be stopped first, so what is free now is what it
            # gets: refuse before spawning rather than let vLLM fail late.
            vllm_budget(model, force)
        return model, binary, port, mode, stop

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
                   spec, replace, stop=(), force=False):
        try:
            if replace:
                n = await asyncio.to_thread(procs.stop_all_llama)
                self.say(f"stopped {n} running instance(s)")
            for pid in stop:
                ok = await asyncio.to_thread(procs.stop_pid, pid)
                self.say(f"stopped pid {pid}" if ok
                         else f"[warn] pid {pid} did not stop")
            if procs.llama_port_busy(port):
                if replace or stop:
                    self.say(f"waiting for port {port} to be released…")
                    await asyncio.to_thread(procs.wait_port_free, port, 30.0)
                if procs.llama_port_busy(port):
                    # Held by something that is not ours to stop: take the
                    # next free deck port rather than giving up.
                    alt = await asyncio.to_thread(procs.next_free_llama_port)
                    if alt is None:
                        self.say(f"[error] port {port} is still in use and no "
                                 f"other port in {config.LLAMA_PORT_RANGE} is "
                                 "free — aborting")
                        self.returncode = 1
                        return
                    self.say(f"port {port} is in use — using {alt} instead")
                    port = alt

            log_path = config.LOG_DIR / f"{model['name']}.log"
            vllm = config.backend_kind(binary) == "vllm"
            wait = {}
            if vllm:
                frac, note = await self._vllm_frac(model, force,
                                                   settle=bool(replace or stop))
                if frac is None:
                    self.say(f"[error] {note}")
                    self.returncode = 1
                    return
                self.say(note)
                self.say("reading `vllm serve --help=all` (imports torch; "
                         "cached after the first time)…")
                argv, notes = await asyncio.to_thread(
                    build_vllm_argv, model, binary=binary, port=port, ctx=ctx,
                    slots=slots, thinking=thinking, spec=spec, gpu_frac=frac)
                env = await asyncio.to_thread(vllm_env, binary)
                if env.get("CUDA_HOME"):
                    self.say(f"CUDA_HOME={env['CUDA_HOME']}")
                wait = {"served": model["name"], "max_wait": 1200.0,
                        "env": env, "what": "vllm",
                        "header": f"{procs.VLLM_LOG_MARK} "
                                  f"model={model['name']}"}
            else:
                argv, notes = await asyncio.to_thread(
                    build_argv, model, binary=binary, port=port, ctx=ctx,
                    slots=slots, thinking=thinking, spec=spec)
            for n in notes:
                self.say(n)
            if vllm:
                self.say(f"max-model-len {ctx} per request · up to {slots} "
                         "concurrent")
            else:
                self.say(f"ctx {ctx} total / {ctx // max(1, slots)} per slot")
            self.say(f"log: {log_path}")
            await asyncio.to_thread(_rotate, log_path)

            ok, exited = await self._spawn_and_wait(argv, log_path, port, **wait)
            if not ok and exited and _has_spec(argv):
                # The MTP/draft head is an optimisation; if the runtime could
                # not load it, serve the model plainly rather than nothing.
                # Only after the first attempt has really gone — a second
                # copy beside a live one would fight it for the port.
                plain = _strip_spec(argv)
                self.say("retrying without speculative decoding")
                if vllm:
                    # the first attempt's engine may still be handing its
                    # memory back; vLLM refuses to start until it is free
                    await asyncio.to_thread(procs.wait_port_free, port, 30.0)
                ok, _ = await self._spawn_and_wait(plain, log_path, port, **wait)
            self.returncode = 0 if ok else 1
            self.say("ready" if ok else "[error] launch failed")
        except Exception as e:                       # never leave it "running"
            self.say(f"[error] {e}")
            self.returncode = 1
        finally:
            self.running = False

    async def _vllm_frac(self, model: dict, force: bool, settle: bool) -> tuple:
        """(gpu_memory_utilization, note), or (None, why) when it cannot
        fit. After stopping instances the driver takes a moment to hand
        their memory back (and the GPU probe is cached), so a short wait
        for it to show up comes before giving up."""
        deadline = time.monotonic() + (20.0 if settle else 0.0)
        while True:
            try:
                return await asyncio.to_thread(vllm_budget, model, force)
            except MemoryError as e:
                if time.monotonic() >= deadline:
                    return None, str(e)
            await asyncio.sleep(2.0)

    async def _spawn_and_wait(self, argv: list, log_path: Path, port: int,
                              max_wait: float = 600.0, served: str = None,
                              env: dict = None, what: str = "llama-server",
                              header: str = None) -> tuple:
        """(ready, exited_by_itself). A server that is not ready in time is
        stopped here, never left loading beside whatever comes next.

        llama-server is ready when /health says "ok". vLLM answers /health
        with an empty 200 once its engine is up, and is ready when
        /v1/models also lists the served name.
        """
        self.say("$ " + _display(argv))
        try:
            proc = await asyncio.to_thread(_spawn, argv, log_path, env, header)
        except Exception as e:
            self.say(f"[error] could not start {what}: {e}")
            return False, True
        self.say(f"pid {proc.pid} — waiting for the server to become ready")
        deadline = time.monotonic() + max_wait
        host = procs.llama_probe_host()
        base = f"http://{'[%s]' % host if ':' in host else host}:{port}"
        url = f"{base}/health"
        ok = False
        # vLLM's engine runs in child processes. Should the server die,
        # they can outlive it holding VRAM; remember them so they go too.
        kids: dict = {}
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                while time.monotonic() < deadline:
                    if served is not None:
                        kids.update(await asyncio.to_thread(_children, proc.pid))
                    if proc.poll() is not None:
                        tail = procs.tail_text(log_path, 8192).splitlines()
                        tail = tail[-(14 if served else 8):]
                        self.say(f"[error] {what} exited "
                                 f"(code {proc.returncode}):")
                        for ln in tail:
                            self.say("    " + ln)
                        return False, True
                    try:
                        r = await client.get(url)
                        if served is not None:
                            if r.status_code == 200 and \
                                    await _lists_model(client, base, served):
                                ok = True
                                return True, False
                        elif r.status_code == 200 and '"ok"' in r.text:
                            ok = True
                            return True, False
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(2.0)
                    waited = int(max_wait - (deadline - time.monotonic()))
                    if waited and waited % 20 == 0:
                        # vLLM loads in long quiet phases (weights, compile,
                        # CUDA graphs): say which one it is in
                        last = ""
                        if served is not None:
                            lines = procs.tail_text(log_path, 2048).splitlines()
                            last = f" — {lines[-1].strip()[-160:]}" if lines else ""
                        self.say(f"still loading… {waited}s{last}")
            self.say(f"[error] not ready within {int(max_wait)}s; last log lines:")
            for ln in procs.tail_text(log_path, 4096).splitlines()[-8:]:
                self.say("    " + ln)
            return False, False
        finally:
            if not ok and proc.poll() is None:
                self.say(f"stopping pid {proc.pid}")
                await asyncio.to_thread(procs.stop_pid, proc.pid)
            if not ok and kids:
                n = await asyncio.to_thread(_reap_orphans, list(kids.values()))
                if n:
                    self.say(f"stopped {n} leftover engine process(es)")


def _children(pid: int) -> dict:
    """pid -> psutil.Process for every descendant. The Process objects
    remember their start time, so a recycled pid is never mistaken later."""
    try:
        return {c.pid: c for c in psutil.Process(pid).children(recursive=True)}
    except Exception:
        return {}


def _reap_orphans(kids: list) -> int:
    """Stop the children a failed launch left running."""
    alive = []
    for c in kids:
        try:
            if c.is_running() and c.status() != psutil.STATUS_ZOMBIE:
                c.terminate()
                alive.append(c)
        except Exception:
            continue
    if not alive:
        return 0
    _, still = psutil.wait_procs(alive, timeout=10.0)
    for c in still:
        try:
            c.kill()
        except Exception:
            pass
    return len(alive)


async def _lists_model(client, base: str, name: str) -> bool:
    try:
        r = await client.get(f"{base}/v1/models")
        return r.status_code == 200 and any(
            m.get("id") == name for m in r.json().get("data") or [])
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


_SPEC_FLAGS = {"--spec-type", "-md", "--model-draft", "--spec-draft-model",
               "-ngld", "--n-gpu-layers-draft", "--spec-draft-n-max",
               "--speculative-config", "-sc"}


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
