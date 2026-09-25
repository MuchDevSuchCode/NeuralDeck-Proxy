# NeuralDeck-Proxy

A local LLM control room for [llama.cpp](https://github.com/ggml-org/llama.cpp)
and [vLLM](https://github.com/vllm-project/vllm), in one Python package that
runs on **Linux and Windows** (macOS for everything but GPU telemetry).

* **Dashboard**: live CPU/GPU/VRAM/thermal telemetry, every running
  model server with the flags it was launched with, KV-cache occupancy, a
  roofline estimate, per-service start/stop, logs, and model management:
  launch, relaunch, delete, and download from Hugging Face.
* **Prompt Lab**: send a prompt to any loaded model and watch it stream,
  with time-to-first-token, prefill/decode throughput and draft acceptance
  measured per run.
* **Benchmarks**: every request through the stack is recorded, from the
  Prompt Lab, from a standard prompt, or from any other client on your
  network. Compare models, backends and speculation settings on medians.
* **Settings**: point NeuralDeck at your model folders and server builds
  from the page itself, with a file picker, validation, and an honest
  account of which changes apply at once and which need a restart.
* **Multimodal proxy**: one OpenAI- and Anthropic-compatible endpoint in
  front of every model you have running. It routes by model name,
  transcribes audio through whisper.cpp, turns video into frames for vision
  models, and relays speech synthesis.

> **No authentication, by design.** Run it on a network you trust, or bind
> it to `127.0.0.1` (see [Security](#security)).

---

## Contents

* [Quick start](#quick-start)
* [How it fits together](#how-it-fits-together)
* [Requirements](#requirements)
* [Install](#install)
* [Backends: llama.cpp and vLLM](#backends-llamacpp-and-vllm)
* [The dashboard](#the-dashboard)
* [Getting models](#getting-models)
* [Choosing a quantisation](#choosing-a-quantisation)
* [Prompt Lab](#prompt-lab)
* [Benchmarks](#benchmarks)
* [Using the proxy](#using-the-proxy)
* [How-to recipes](#how-to-recipes)
* [Settings reference](#settings-reference)
* [Configuring without the page](#configuring-without-the-page)
* [Command line](#command-line)
* [Where things are stored](#where-things-are-stored)
* [Dashboard API](#dashboard-api)
* [Troubleshooting](#troubleshooting)
* [Security](#security)
* [Uninstall](#uninstall)
* [Notes on portability](#notes-on-portability)
* [Documentation site](#documentation-site)

---

## Quick start

```bash
git clone https://github.com/MuchDevSuchCode/NeuralDeck-Proxy.git
cd NeuralDeck-Proxy
./install.sh                     # Windows: powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The installer builds a virtual environment, installs NeuralDeck, prints
what it found on your machine and opens the dashboard at
**http://localhost:8770**. Then:

1. **Settings → llama-server builds**: add the path to your `llama-server`
   binary if it wasn't found automatically.
2. **Settings → Model directories**: add the folders that hold your models.
3. **Dashboard → Models**: press **Launch** next to a model.
4. Point any OpenAI-compatible client at **`http://localhost:8080/v1`**,
   with `model` set to the name shown in the Models panel.

Afterwards, start it again any time with `neuraldeck up --open`.

## How it fits together

```
     your apps / Claude Code / the dashboard's Prompt Lab
                          │
                    :8080 │  multimodal proxy      ← the port clients use
                          │  (routes by model name)
        ┌─────────────────┼──────────────────┬─────────────────┐
        │                 │                  │                 │
  :8081 llama-server  :8082 llama-server  :8083 vllm serve  :8090 whisper-server
        │                 │                  │
        └──────── :8770 NeuralDeck dashboard ┘   ← watches, launches, benchmarks
```

* The **proxy** owns the port your clients point at. It finds every model
  server in the llama port range (`8081-8089` by default) and routes each
  request by its `model` field.
* **Model servers** (`llama-server` or `vllm serve`) are launched by the
  dashboard into that port range. They keep running if the dashboard
  restarts.
* The **dashboard** is a separate port that observes the whole stack,
  launches and stops models, and records benchmarks.

## Requirements

* **Python 3.10+**.
* **A model server**, at least one of:
  * a built `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp),
    or a fork of it. Any recent build works: the launcher reads each
    binary's `--help` and only passes flags that build supports;
  * a [vLLM](https://github.com/vllm-project/vllm) install, in its own
    environment, to serve Hugging Face safetensors models (see
    [vLLM](#vllm)).
* **Optional extras:**
  * `whisper-server` from [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
    for speech input;
  * `ffmpeg` for video input;
  * `huggingface_hub` for the in-dashboard model downloader (installed by
    the installer);
  * an OpenAI-compatible TTS server for speech output;
  * ComfyUI, if you want to start and stop it from the dashboard.
* **GPU telemetry** comes from `nvidia-smi` (NVIDIA), `amd-smi` /
  `rocm-smi` / amdgpu sysfs (AMD), or Windows CIM. Without any of them the
  dashboard still works; the GPU panels just hide what nobody can report.

## Install

### With the installer (recommended)

**Linux / macOS**

```bash
git clone https://github.com/MuchDevSuchCode/NeuralDeck-Proxy.git
cd NeuralDeck-Proxy
./install.sh
```

**Windows** (PowerShell, in the cloned folder)

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The installer:

1. finds a suitable Python 3.10+;
2. builds a virtual environment (default: `./venv`) and installs
   NeuralDeck into it, with the Hugging Face extras;
3. adds a launcher: `~/.local/bin/neuraldeck` on Linux/macOS, a Start Menu
   shortcut on Windows;
4. runs `neuraldeck doctor` to show what it found;
5. starts NeuralDeck and opens the dashboard.

Re-running it is safe. A virtual environment left half-built by an
interrupted run is detected and rebuilt.

| Installer flag | Effect |
| --- | --- |
| `--no-launch` / `-NoLaunch` | install only, don't start it |
| `--no-link` / `-NoShortcut` | skip the `~/.local/bin` launcher / Start Menu shortcut |
| `--dir <path>` / `-Dir <path>` | put the virtual environment somewhere else (relative paths are resolved) |

Nothing is installed system-wide: everything lives in the virtual
environment, plus a config file, logs and benchmark history in your user
data directory.

### Manually, or into an existing environment

```bash
pip install -e ".[hub]"     # or: pip install -r requirements.txt
neuraldeck doctor           # what this machine looks like to NeuralDeck
neuraldeck up --open        # start the proxy and the dashboard
```

From a checkout, without installing anything, the wrapper scripts build a
virtual environment on first run:

```bash
./scripts/neuraldeck.sh            # Linux / macOS
.\scripts\neuraldeck.ps1           # Windows PowerShell
```

### Updating

```bash
cd NeuralDeck-Proxy
git pull
```

Then restart: press **Ctrl+C** in the terminal running NeuralDeck and start
it again, and restart the proxy from the **Servers** panel (**Stop**, then
**Start**). The proxy runs as its own process and survives dashboard
restarts, so it keeps the old code until you restart it. Model servers
keep running throughout.

## Backends: llama.cpp and vLLM

A **backend** is a labelled path to a model server executable, set in
**Settings → llama-server builds**. The launch form picks one per launch,
so you can keep several and switch between them.

### llama.cpp builds

Add a label and the path to a `llama-server` binary. Examples of why you
might keep more than one:

| Label (example) | Why |
| --- | --- |
| `llama.cpp` | upstream, for most GGUF models |
| a fork with speculative-decoding extras | MTP heads (e.g. Gemma 4 assistant heads, Qwen NextN) and extra KV-cache types that upstream lacks |
| a fork for a special quant format | some formats (for example ternary `PTQ1_0`/`PQ2_0` models) only load on the fork that introduced them |

The launcher reads each binary's `--help` once and only passes flags that
build accepts, so upstream and forks both work without configuration. The
A/B workflow is simple: launch the same model on two backends and compare
them on the Benchmarks tab.

If NeuralDeck can't find a `llama-server` at all, it looks on `PATH`, then
in `~/llama.cpp/build/bin/` (and the usual Windows build folders). Add the
path in Settings if yours lives elsewhere.

### vLLM

vLLM serves Hugging Face **safetensors** models (including AWQ, GPTQ, FP8,
compressed-tensors, and EXL3 via a plugin). NeuralDeck treats any backend
whose executable is named `vllm` as a vLLM backend.

**1. Install vLLM in its own environment.** vLLM pins its own PyTorch and
CUDA build, so keep it out of NeuralDeck's environment. With conda
(using conda-forge avoids Anaconda's channel terms prompt):

```bash
conda create -y -p ~/vllm-env --override-channels -c conda-forge python=3.12 pip
~/vllm-env/bin/python -m pip install -U vllm huggingface_hub
~/vllm-env/bin/python -c "import vllm, torch; print(vllm.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A plain `python3 -m venv ~/vllm-env` works just as well if your system
Python is a version vLLM supports.

**2. Register it.** In **Settings → llama-server builds**, add a label
(e.g. `vLLM`) with the path `~/vllm-env/bin/vllm`. `neuraldeck doctor`
runs `vllm --version` to confirm it works.

**3. Launch a safetensors model.** Folders holding `config.json` and
`*.safetensors` appear in the Models panel with a **safetensors** badge.
Press **Launch** and the form switches to your vLLM backend automatically.
The first start takes a few minutes (vLLM compiles kernels and captures
CUDA graphs); later starts are faster.

What NeuralDeck does for a vLLM launch:

* **Flags come from vLLM's own help** (`vllm serve --help=all`, read once
  and cached): `--served-model-name`, `--max-model-len` (your context),
  `--max-num-seqs` (your slots), `--gpu-memory-utilization`,
  `--enable-prefix-caching`.
* **Model-specific options are detected:** reasoning and tool-call parsers
  for Qwen-family models (`qwen3_xml` when the chat template writes
  `<function=…>` calls, otherwise `hermes`), the model's own MTP layers as
  `--speculative-config` when speculation is `auto`, and
  `enable_thinking: false` as the default template argument when thinking
  is off. If a launch fails with speculation on, it's retried without it.
* **VRAM is sized to what's free**, so vLLM can run beside llama-servers.
  `--gpu-memory-utilization` is set to the free share of the card less a
  512 MiB margin, capped by `vllm_gpu_frac_max`. A model that plainly
  can't fit is refused before anything starts, with a message saying how
  much it needs and how much is free.
* **The environment is prepared:** the vLLM environment's `bin` goes first
  on `PATH`, `CUDA_HOME` points at the CUDA toolkit pip installed into it
  (flashinfer compiles kernels with it), and
  `VLLM_USE_FLASHINFER_SAMPLER=0` is set unless you set it yourself.
* **Readiness** is `/health` plus the served name in `/v1/models`, with up
  to 20 minutes allowed for the first start.

Things that differ from llama.cpp:

* **Context is per request.** vLLM's `--max-model-len` applies to each
  request; it isn't split across slots the way llama.cpp splits `ctx`.
* **vLLM reserves memory up front.** It claims its whole
  `--gpu-memory-utilization` share at start-up for its KV cache, so the
  memory pressure panel reads *tight* while it runs. That's expected, and
  it doesn't grow further.
* **Background traffic isn't benchmarked.** vLLM logs no per-request
  timings, so its runs are recorded from the Prompt Lab and the standard
  bench only.

#### EXL3 models on vLLM

EXL3 (ExLlamaV3) checkpoints, such as `orcarouter/OrcaSAQ-2-27B`, need
`quant_method: exl3` support, which stock vLLM doesn't have. The model's
publisher supplies a vLLM plugin; it also needs the `exllamav3` CUDA
extension, which isn't on PyPI and must match your PyTorch and CUDA
versions exactly:

```bash
PY=~/vllm-env/bin/python
$PY -c "import torch; print(torch.__version__, torch.version.cuda)"   # e.g. 2.13.0 13.0
# pick the matching wheel from https://github.com/turboderp-org/exllamav3/releases
#   (cu128.* for CUDA 12.x, cu132.* for CUDA 13.x; torchX.Y.0; cp312 for Python 3.12)
$PY -m pip install --no-deps ./exllamav3-<version>+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl
$PY -m pip install marisa_trie
# flashinfer's JIT needs nvcc at PyTorch's CUDA minor version, not the newest
$PY -m pip install "nvidia-cuda-nvcc==13.0.*"
# then the model's plugin, per its README (pin a commit you've reviewed)
```

Plugins from model publishers are third-party code that runs inside the
server process. Read one before installing it, and pin the commit you read
rather than tracking a branch.

## The dashboard

The dashboard is at `http://localhost:8770`. It has four views: Dashboard,
Prompt Lab, Benchmarks and Settings.

### Dashboard panels

| Panel | Shows |
| --- | --- |
| **CPU** | model, frequency, temperature, core/thread count, load |
| **Per-core utilisation** | load per logical core |
| **GPU / Accelerator** | device, temperature, clocks, power, fan, utilisation, VRAM (plus the shared GTT pool on AMD APUs) |
| **Memory pressure** | a verdict (ok / watch / tight) from RAM, VRAM and swap, with advice |
| **Roofline / Bottleneck** | an estimated decode ceiling from memory bandwidth ÷ weight bytes, and how close the live decode speed gets to it |
| **Storage** | usage of each model folder's disk, and disk read/write |
| **Network / Processes** | network rates, thread count, busiest processes |
| **History** | charts of CPU/GPU %, prefill/decode throughput, VRAM, GPU temperature and KV-cache fill per instance |
| **Models** | every discovered model, with launch controls (below) |
| **Servers** | every running model server and service, with Stop/Start |
| **Multimodal stack** | whisper model and transcriptions, proxy requests and errors |
| **Inference** | live stats for a running instance: health, requests, prefill/decode t/s, context use |
| **Logs** | tabs for the newest model log, the proxy, whisper and the launch log |

Panels can be **dragged to reorder** by their heading (a ⠿ grip appears on
hover or keyboard focus; ↑/↓ move a focused panel). **Reset layout** in
the header restores the default arrangement.

**Themes:** the header's theme picker offers **NeuralDeck** (the default
cyberpunk look), **Dark** and **Light**. The choice is remembered per
browser.

### Launching a model

The Models panel lists every model found in your model directories. Icons
show what each can do:

| Icon / badge | Meaning |
| --- | --- |
| 👁 | vision (an mmproj projector is paired) |
| 🎬 | video, as frames through the proxy |
| ⚡ | an MTP/NextN head for speculative decoding |
| 💭 | a chat template that supports thinking |
| 🧬 | an embedding model |
| **safetensors** | a Hugging Face format model that needs a vLLM backend |

The launch controls above the list apply to the next launch:

| Control | Meaning |
| --- | --- |
| **backend** | which server build to use (vLLM backends are labelled) |
| **slots** | parallel requests. llama.cpp splits the context across them; vLLM uses it as `--max-num-seqs` |
| **ctx** | context size in tokens. The note beside it shows the per-request budget |
| **spec** | `auto` uses the model's MTP head when it has one; `ngram` drafts from the context and needs no draft model; `off` disables speculation (useful for A/B timing) |
| **thinking** | reasoning budget for models that support it: `off` (0), `low` (1024 tokens), `medium` (4096), `high` (unlimited) |
| **replace running instances** | stop what's running first, instead of launching alongside it |

Buttons on each model row:

* **Launch** starts the model on the next free port in the llama port
  range. If the model's format doesn't suit the selected backend, the form
  switches to a compatible backend and says so.
* **Relaunch** (on a model that's already serving) stops that instance and
  starts it again on the same port with the current settings.
* **✕** deletes the model's files. For a GGUF model that's only its own
  files (its shards, mmproj and draft head); for a safetensors model it's
  the whole folder, since the folder is the model. It never touches
  anything outside your model directories, a symlinked folder loses only
  the link, and it's hidden while the model is serving.

Before launching alongside other models, NeuralDeck checks free VRAM and
refuses with an explanation if the model won't fit.

### Servers and logs

The **Servers** panel lists every running model server with its port,
backend, context, slots, KV cache, speed and pid, plus the proxy,
whisper-server and any optional services. **Stop** and **Start** act on
them directly. The **Logs** panel follows the newest model log without
jumping to the bottom while you scroll up to read, or while you select
text.

## Getting models

### The Hugging Face downloader

Press **⬇ Get models** in the Models panel.

* **Search** by name, or leave the box empty to browse. Sort by trending,
  downloads or likes.
* **Hover over a result** (or tab to it) to see a short description from
  its model card. For quantised repos, whose cards are mostly packaging
  notes, the base model's description is shown. Open a repo and the same
  description appears above its file list.
* **GGUF / vLLM toggle.** GGUF mode (the default) lists GGUF repos for
  llama.cpp. vLLM mode lists safetensors repos and downloads the whole
  repo; it's available once a vLLM backend is configured.
* Downloads **queue**, show progress next to the ⬇ Get models button even
  with the dialog closed, can be cancelled, and refresh the model list when
  they finish.

**In GGUF mode, what to tick:**

| File | Do you need it? |
| --- | --- |
| **one quant** (e.g. `…-Q4_K_M.gguf`) | **Yes.** This is the model. See [Choosing a quantisation](#choosing-a-quantisation) |
| an **mmproj** (`mmproj-F16.gguf`) | Only for image input. It's the vision projector |
| an **MTP head** (`mtp-…gguf`, `…-assistant…`) | Optional, for faster decoding with speculation on a backend that supports it |
| `BF16/…`, `F16` | Usually not. These are the full-precision originals |

The file list labels each helper file's role and marks the recommended
mmproj (F16 first) and MTP head (Q8_0 first) with ★. **When you tick your
first quant, the recommended mmproj and MTP head are ticked too**, with a
note saying so. Untick either to skip it. Split models (`-00001-of-00003`)
tick and untick all their parts together.

Everything you tick lands in one folder named after the main weights, so
discovery pairs the model with its mmproj and head automatically. Files
from repo subfolders are moved up into that folder.

### Adding models by hand

A model is a folder under a model directory:

```
~/models/
├── Qwen3-8B-Q4_K_M/                      ← GGUF model: folder name = model name
│   ├── Qwen3-8B-Q4_K_M.gguf
│   └── mmproj-F16.gguf                   ← optional: vision
├── gemma-4-26B-A4B-it-UD-Q4_K_XL/
│   ├── gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
│   ├── mmproj-F16.gguf
│   └── mtp-gemma-4-26B-A4B-it.gguf       ← optional: MTP draft head
├── Big-Model-Q4_K_M/
│   ├── Big-Model-Q4_K_M-00001-of-00002.gguf   ← split models: all parts together
│   └── Big-Model-Q4_K_M-00002-of-00002.gguf
└── OrcaSAQ-2-27B/                        ← safetensors model (vLLM)
    ├── config.json
    ├── model-00001-of-00004.safetensors …
    └── tokenizer.json …
```

A lone `.gguf` directly in a model directory is listed too, named after
the file.

NeuralDeck reads each GGUF's **header**, not its filename, to decide
whether it carries an MTP/NextN head and whether its chat template does
reasoning, because filenames lie about both. A file is treated as a draft
head if its name marks it as one (`mtp`, `nextn`, `assistant`, `draft` as
a separate word) or its header says it's an assistant model.

For a safetensors folder, the quantisation (`exl3 3.21bpw`, `gptq 4bit`,
`awq`, `fp8` …), architecture, vision support, MTP layers and reasoning
template come from `config.json`, `quantization_config.json` and the chat
template.

### mmproj files: which and when

* The **F16** mmproj is the right default. BF16 is the same size and fine
  on recent GPUs; F32 is twice the size for no visible gain.
* An mmproj is tied to its **base model**. It works with every quant of
  that model and usually with fine-tunes of the same base. It doesn't work
  with other sizes of the same family, or other families: llama.cpp
  refuses it with an embedding-size mismatch.
* To reuse one across several quants without copies, symlink it into each
  model's folder. NeuralDeck follows the link, and deleting the model
  removes only the link.

```bash
ln -s ~/models/gemma-4-26B-A4B-it-UD-Q4_K_XL/mmproj-F16.gguf ~/models/gemma-4-26B-A4B-it-UD-Q5_K_XL/
```

## Choosing a quantisation

| Format | What it is | Bits/weight | When to pick it |
| --- | --- | --- | --- |
| **Q4_K_M / Q5_K_M / Q6_K** (K-quants) | blocks of small integers with per-block scales; S/M/L/XL bump the sensitive tensors to more bits | ~4.8 / 5.7 / 6.6 | the safe default on any hardware |
| **IQ4_XS / IQ3_XXS / IQ2_M** (I-quants) | blocks coded against a lookup table; more quality per bit | ~4.25 / 3.1 / 2.7 | when memory is tight. Fast on GPU, slower on CPU |
| **UD-…** (Unsloth Dynamic) | per-layer precision chosen by calibration | varies | consistently good; a strong first choice when offered |
| **imatrix / i1** | calibration data guided which weights keep precision | – | prefer it at 4 bits and below |
| **MXFP4** | 4-bit floats with a shared power-of-two scale per 32 values | 4.25 | for models *released* in MXFP4 (e.g. gpt-oss), or on GPUs with FP4 hardware (NVIDIA Blackwell and newer). Elsewhere, K- or I-quants are usually better at the same size |
| **Q8_0** | plain 8-bit | 8.5 | near-lossless, if it fits |
| **BF16 / F16** | the original weights | 16 | reference quality; rarely practical |

A rule of thumb: pick the largest quant that leaves room for the context
you want. Weights plus KV cache must fit in VRAM; the KV cache grows with
context length and slots.

## Prompt Lab

Pick a running model (or *auto* for the default instance), set the
temperature and max tokens, optionally a system prompt, and press **Send**
(or Ctrl+Enter). The reply streams in, with any thinking shown in its own
pane and code blocks carrying copy buttons. The panel beside it shows:

* **TTFT**: time to first token;
* **prefill** and **decode** throughput in tokens per second;
* **draft acceptance**, when speculation is active;
* the prompt and completion token counts.

llama.cpp reports exact timings with each response. For backends that
don't (vLLM), the numbers are measured in the page from the stream, and
marked as such. Every Prompt Lab run is added to the benchmarks.

## Benchmarks

* **Run standard bench** sends a fixed ~250-word prompt at temperature 0
  with 320 max tokens to the chosen model, so runs are comparable. With
  one model running, it's selected automatically.
* Runs come from three sources: the Prompt Lab, the standard bench, and a
  collector that reads each llama-server's log, so requests from **any
  client** are measured too.
* The chart shows **medians per model × backend × speculation setting**,
  with decode and prefill each on their own scale. Filter by backend and
  model, or show standard runs only.
* The runs table lists every record, newest first. **✕** deletes a run,
  with a few seconds to undo.

History is kept in `bench.jsonl` in the data directory, capped at the
newest 5,000 runs.

## Using the proxy

Point any OpenAI-compatible client at **`http://localhost:8080/v1`** (or
the machine's address from elsewhere on your network). There's no API key;
clients that insist on one can send any value.

### Endpoints

| Endpoint | Behaviour |
| --- | --- |
| `POST /v1/chat/completions` | routed by `model`. `input_audio` parts are transcribed; `video_url` parts become frames for vision models |
| `POST /v1/messages` | the Anthropic Messages API, routed the same way (llama-server and vLLM both speak it) |
| `POST /v1/messages/count_tokens` | Anthropic token counting |
| `POST /v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/rerank` … | routed by `model` when the body names one |
| `POST /v1/multimodal` | one multipart form with `text`, `images`, `videos`, `audio`, `system`, `model` and `stream` fields |
| `POST /v1/audio/speech`, `GET /v1/audio/voices` | relayed to the TTS backend |
| `GET /v1/models` | every model being served, across all instances |
| `GET /props`, `GET /health` | the default instance's properties; the proxy's health |
| anything else | relayed to the default llama-server untouched |

### How `model` is matched

Names are compared ignoring case, with `.` and `_` treated like `-`, and a
trailing `.gguf` dropped.

1. **An exact match wins.**
2. Otherwise, a model whose name **extends the request at a `-`** matches,
   but only if **exactly one** does. `gemma-4-26B` would find
   `gemma-4-26B-A4B-it-UD-Q4_K_XL` if it's the only one, but
   `Ternary-Bonsai-2-27B` with both `…-PTQ1_0` and `…-PQ2_0` running is
   ambiguous and returns a 404 that lists both.
3. Partial words never match: `Orca` doesn't find `OrcaSAQ-2-27B`.
4. With **strict model routing** on (the default), a name nothing is
   serving is a 404 that lists what *is* serving. With it off, such
   requests go to the default instance, which misattributes benchmarks.

Upstream errors keep their real status code (a prompt that's too long is a
400, not a 502), so SDKs don't retry requests that can't succeed. Errors
on `/v1/messages` use the Anthropic error format.

### Examples

**curl**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "Qwen3-8B-Q4_K_M",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

**OpenAI Python SDK**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
reply = client.chat.completions.create(
    model="Qwen3-8B-Q4_K_M",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)
```

**Claude Code** (or any Anthropic client)

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=none
export ANTHROPIC_MODEL=OrcaSAQ-2-27B                  # a served model name
export ANTHROPIC_DEFAULT_HAIKU_MODEL=OrcaSAQ-2-27B    # background tasks too
claude
```

Claude Code also sends background requests under a small-model name.
Point those at a served model as well, or strict routing answers them with
a 404.

**Images, audio and video** go in chat messages as usual: `image_url`
parts for images, `input_audio` parts for speech (transcribed by
whisper-server and replaced with the text), and `video_url` parts for
video (sampled into frames for vision models; needs `ffmpeg`). Long videos
are sampled evenly across the whole clip, up to `max_frames` frames.

## How-to recipes

**Keep everything on this machine.** Set **Dashboard bind address**,
**Proxy bind address** and **llama bind address** to `127.0.0.1` in
Settings. The proxy restarts itself; the dashboard offers a restart button.

**A/B a fork against upstream.** Add both builds under llama-server
builds, launch the model on one, run the standard bench a few times,
**Relaunch** it on the other (pick the backend first), and bench again.
The Benchmarks chart puts the two side by side.

**Run two models at once.** Launch one, then another. Each gets its own
port and the proxy routes by name. NeuralDeck refuses a launch that won't
fit in free VRAM. Tick **replace running instances** to swap instead.

**Speed up decoding with MTP.** Download a model with its MTP head (the
downloader ticks it for you), launch it on a backend that supports
`draft-mtp` with spec **auto**, and compare against spec **off** on the
Benchmarks tab. The Prompt Lab shows draft acceptance.

**Use speculation without a draft model.** Set spec to **ngram**. It
drafts from text already in the context, which helps most on code and
repetitive output.

**Change ports.** Change **Proxy port** (the proxy restarts on the new
port) or **Dashboard port** (use the restart button; the page follows you
to the new port). Ports must not collide with each other or fall inside
the llama port range.

**Add speech input.** Build whisper.cpp, set **whisper-server binary** and
**Whisper model** in Settings, and start whisper-server from the Servers
panel. `input_audio` parts in chat requests are then transcribed.

**Reset the panel layout.** Press **Reset layout** in the header.

## Settings reference

Every setting is on the **Settings** tab. Each can also be set in
`config.json` (same key) or with an environment variable
`NEURALDECK_<KEY>` in upper case (e.g. `NEURALDECK_DECK_PORT`). Precedence
is environment, then `config.json`, then the default.

The **Applies** column says when a change takes effect:

* **now**: the dashboard re-reads settings on save (launch settings apply
  to the next launch, service commands to the next start);
* **proxy**: read by the proxy; saving restarts the proxy for you;
* **restart**: captured when the dashboard started; the page offers a
  restart button.

### Models

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `model_dirs` | `~/models` (and `/mnt/models` on Linux) | now | Folders scanned for models. A folder that isn't mounted right now is kept and flagged |
| `download_dir` | the first model directory | now | Where the Hugging Face downloader puts new models |

### Backends

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `backends` | the first `llama-server` found | now | Label → path to a `llama-server` binary or a `vllm` executable |
| `default_backend` | the first backend | now | Which backend the launch form starts on |
| `llama_port_range` | `8081-8089` | proxy | Ports models are launched into and the proxy discovers (at most 64). Running instances keep their ports |
| `llama_host` | `0.0.0.0` | now | What launched servers bind to. `127.0.0.1` keeps them off the network |

### Launch defaults

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `ctx` | `32768` | now | Context size (1,024–1,048,576). llama.cpp splits it across slots |
| `slots` | `1` | now | Parallel requests (1–64) |
| `spec` | `auto` | now | `auto` (MTP head when present), `ngram`, or `off` |
| `thinking` | `off` | now | `off`, `low`, `medium`, `high` |
| `kv_cache_type` | `q8_0` | now | llama.cpp `-ctk`/`-ctv`, e.g. `f16`, `q8_0`, `q4_0` (forks may add more) |
| `flash_attn` | `on` | now | `on`, `off`, `auto`; passed as `-fa` on builds that take a value |
| `n_gpu_layers` | `999` | now | `-ngl`. 999 offloads everything that fits |
| `threads` | `0` | now | CPU threads; 0 uses the physical core count |
| `extra_llama_args` | none | now | Appended to every llama-server launch, one argument per entry |

### vLLM

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `vllm_kv_cache_dtype` | `auto` | now | `--kv-cache-dtype`: `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`. fp8 halves KV memory at a small quality cost |
| `vllm_gpu_frac_max` | `0.92` | now | Ceiling for `--gpu-memory-utilization` (0.5–0.98). Each launch is sized to the VRAM actually free, never above this |
| `vllm_extra_args` | none | now | Appended to every `vllm serve`, one argument per entry |

### Sampling

Launch-time defaults; a request can always override them.

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `temp` | `0.7` | now | Temperature (0–5) |
| `top_p` | `0.95` | now | Top-p (0–1) |
| `min_p` | `0.05` | now | Min-p (0–1) |
| `repeat_penalty` | `1.05` | now | Repeat penalty (0–5) |

### Ports

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `deck_port` | `8770` | restart | The dashboard's port |
| `deck_host` | `0.0.0.0` | restart | The dashboard's bind address. `127.0.0.1` keeps it on this machine |
| `proxy_port` | `8080` | proxy | The port clients point at |
| `proxy_host` | `0.0.0.0` | proxy | The proxy's bind address |

### Speech & video

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `whisper_bin` | `whisper-server` on `PATH`, else `~/whisper.cpp/build/bin/whisper-server` | now | whisper.cpp's server, for speech input |
| `whisper_model` | `~/whisper.cpp/models/ggml-large-v3-turbo.bin` | now | A `ggml-*.bin` model |
| `whisper_port` | `8090` | proxy | Where whisper-server listens |
| `ffmpeg` | from `PATH` | proxy | Needed only for video input |
| `video_fps` | `1.0` | proxy | Frames per second sampled from video (0.1–30), lowered for long clips so frames cover the whole video |
| `max_frames` | `30` | proxy | Frames per video (1–500) |
| `tts_endpoint` | `http://127.0.0.1:<tts_port>` | proxy | An OpenAI-compatible `/v1/audio/speech` server |
| `tts_default_voice` | none | proxy | Voice used when a client doesn't name one |

### Proxy

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `strict_model_routing` | `true` | proxy | A request naming a model nothing serves is an error (on), or goes to the default instance (off) |
| `llama_timeout` | `600` | proxy | Seconds to wait on a model server (1–7200) |
| `whisper_timeout` | `300` | proxy | Seconds to wait on whisper-server (1–7200) |

### Dashboard

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `peak_bw_gbs` | `89.6` | now | Memory bandwidth in GB/s for the roofline panel. **Set it for your hardware** (e.g. about 1008 for an RTX 3090 Ti, 936 for an RTX 3090, 256 for a Strix Halo APU) |
| `vram_total_gb` | `0` | now | VRAM total, only used when no driver reports one. 0 leaves it unknown |
| `history_len` | `600` | restart | Points kept in the charts (60–86,400) |
| `sample_interval` | `1.0` | now | Seconds between telemetry samples (0.2–60) |

### Optional services

| Key | Default | Applies | Meaning |
| --- | --- | --- | --- |
| `tts_cmd` | none | now | Command that starts a TTS server, one argument per entry. Empty hides the service |
| `tts_port` | `8004` | proxy | Where the dashboard looks for the TTS server |
| `comfy_cmd` | none | now | Command that starts ComfyUI. Empty hides the service |
| `comfy_port` | `8188` | now | ComfyUI's port |

### Validation and safety

* Values that would stop NeuralDeck from starting are **refused before the
  file is written**: host names with a scheme or port, ports out of range
  or colliding with each other or the llama port range, a range of more
  than 64 ports, non-finite or fractional numbers.
* A `config.json` that isn't valid JSON (a hand edit gone wrong) is
  **never silently replaced**: the page shows the error and refuses to save
  until the file is fixed or removed. `neuraldeck doctor` reports it too.
* A setting fixed by an environment variable shows an **env** badge and is
  read-only on the page.
* **Clearing a field means "use the default"**, not "set it to empty". The
  *use default* button next to a stored value does the same.
* Unsaved edits survive switching views; the page warns before you leave
  with unsaved changes.

## Configuring without the page

`config.json` lives in the data directory (see
[Where things are stored](#where-things-are-stored)) and uses the same keys
with JSON types:

```json
{
  "backends": {
    "llama.cpp": "/home/me/llama.cpp/build/bin/llama-server",
    "fork":      "/home/me/my-fork/build/bin/llama-server",
    "vLLM":      "/home/me/vllm-env/bin/vllm"
  },
  "default_backend": "llama.cpp",
  "model_dirs": ["/home/me/models", "/mnt/big/models"],
  "peak_bw_gbs": 1008,
  "ctx": 65536
}
```

The same with environment variables:

```bash
export NEURALDECK_BACKENDS="llama.cpp=/home/me/llama.cpp/build/bin/llama-server"
export NEURALDECK_MODEL_DIRS="/home/me/models:/mnt/big/models"   # ';' on Windows
export NEURALDECK_PEAK_BW_GBS=1008
export NEURALDECK_TTS_CMD="python -m my_tts --port 8004"        # split like a shell would
```

* Path lists (`model_dirs`) are joined with the OS path separator (`:` on
  Linux/macOS, `;` on Windows).
* Commands (`tts_cmd`, `comfy_cmd`, `extra_llama_args`, `vllm_extra_args`)
  are split like a shell command line.
* `~` is expanded in paths. An empty variable counts as unset.

### Settings not on the page

These are rarely needed and can only be set in `config.json` or the
environment:

| Key | Default | Meaning |
| --- | --- | --- |
| `home` (env `NEURALDECK_HOME` only) | platform data directory | Moves the data directory itself, including `config.json` |
| `log_dir` | `<data dir>/logs` | Where model and service logs go |
| `bench_file` | `<data dir>/bench.jsonl` | Benchmark history |
| `proxy_log`, `whisper_log`, `tts_log`, `comfy_log` | in `log_dir` | Individual service logs |
| `llama_bin` | auto-detected | The `llama-server` used when `backends` isn't set |
| `max_upload_mb` | `512` | The proxy's request size limit, in MB (larger requests get a 413) |
| `tts_default_model` | `tts-1` | Model name sent to the TTS backend when a client doesn't give one |

## Command line

| Command | What it does |
| --- | --- |
| `neuraldeck` | same as `neuraldeck up` |
| `neuraldeck up` | starts the proxy (in the background, if it isn't running) and the dashboard (in the foreground) |
| `neuraldeck up --open` | the same, and opens the dashboard in a browser once it answers |
| `neuraldeck up --no-proxy` | dashboard only; leaves the proxy alone |
| `neuraldeck up --stop-proxy` | also stops the proxy when the dashboard exits |
| `neuraldeck deck [--open]` | dashboard only |
| `neuraldeck proxy` | the proxy only, in the foreground |
| `neuraldeck doctor` | resolved config, detected hardware, backends (with their kind and version), models, and what's running |

Run `neuraldeck doctor` first on a new machine: it tells you what was
found and what's missing before anything tries to start.

**Stopping:** press **Ctrl+C** in the terminal running the dashboard. The
proxy keeps running in the background unless you started with
`--stop-proxy`; stop it from the Servers panel. Model servers keep running
until you stop them.

## Where things are stored

The **data directory** is `~/.local/share/neuraldeck` on Linux/macOS
(`$XDG_DATA_HOME/neuraldeck` if set) and `%LOCALAPPDATA%\NeuralDeck` on
Windows, or wherever `NEURALDECK_HOME` points. The Settings tab and
`neuraldeck doctor` print the exact paths.

```
neuraldeck/
├── config.json            ← your settings
├── bench.jsonl            ← benchmark history (newest 5,000 runs)
├── last-model.txt         ← the last model launched
├── logs/
│   ├── <model>.log        ← one per model; the previous run is kept as <model>.log.1
│   ├── proxy.log
│   └── whisper.log
└── run/
    ├── proxy.pid          ← pid files for the services NeuralDeck started
    └── vllm-help-*.txt    ← cached `vllm serve --help=all` output
```

## Dashboard API

The dashboard is driven by a JSON API on its own port, which you can
script against too. There's no authentication.

| Method and path | Purpose |
| --- | --- |
| `GET /api/config` | ports, backends (and their kinds), defaults and links for the page |
| `GET /api/state` | the latest telemetry snapshot plus chart history |
| `GET /api/stream` | server-sent events: one snapshot per sample interval |
| `GET /api/models` | discovered models |
| `POST /api/models/delete` | delete a model's files: `{"name": …}` |
| `POST /api/llama/launch` | launch: `{"model", "backend", "ctx", "slots", "spec", "thinking", "replace", "relaunch"}` |
| `GET /api/launch/status` | progress and log of the current launch |
| `POST /api/llama/{port}/stop` | stop the instance on a port |
| `POST /api/service/{name}/start`, `…/stop` | start or stop `proxy`, `whisper`, `tts`, `comfy` |
| `GET /api/logs/{service}?lines=N` | the tail of a log (1–2000 lines) |
| `GET /api/settings`, `POST /api/settings` | read or save settings |
| `GET /api/browse?path=&mode=dir\|file&ext=` | directory listing for the file picker |
| `POST /api/restart` | restart the dashboard |
| `POST /api/chat` | same-origin relay to the proxy (used by the Prompt Lab) |
| `GET /api/bench`, `POST /api/bench`, `POST /api/bench/delete`, `POST /api/bench/clear` | benchmark history |
| `GET /api/hf/search`, `/api/hf/files`, `/api/hf/card` | Hugging Face search, file lists and model-card descriptions (`format=gguf\|hf`) |
| `POST /api/hf/download`, `GET /api/hf/status`, `POST /api/hf/cancel` | the download queue |

## Troubleshooting

Start with `neuraldeck doctor`. It shows the resolved settings, what was
detected, and any problem with `config.json`.

### Starting and installing

**`could not start llama-server: [Errno 2] No such file or directory: 'llama-server'`**
NeuralDeck didn't find a `llama-server` binary. Add its full path under
**Settings → llama-server builds**. The related warning *could not read
`llama-server --help`* has the same cause.

**Settings I changed don't take effect.**
Check the setting's **Applies** column in the
[reference](#settings-reference). Settings the proxy reads restart the
proxy automatically; dashboard ports and history length need the restart
button. A setting fixed by an environment variable (shown with an **env**
badge) can't be changed from the page.

**The Settings page shows a config error and won't save.**
`config.json` isn't valid JSON, usually after a hand edit. Fix the file (or
remove it to start from defaults) and reload. NeuralDeck never overwrites
a file it can't read, so your other settings aren't lost.

**After updating, a feature is missing or the proxy misbehaves.**
The proxy runs as its own process and keeps the old code until restarted.
Restart the dashboard *and* click **Stop**, then **Start**, on the proxy
in the Servers panel.

**The Windows Start Menu shortcut opens nothing.**
The shortcut runs without a console, so output goes to `deck.log` in the
logs folder. Check it for the error.

**Ctrl+C prints tracebacks or takes several seconds.**
That was fixed; update to the latest version. A clean shutdown now takes
well under a second, even with dashboard tabs open.

### Launching models

**A launch is refused for lack of VRAM.**
Other models are using the memory. Stop one, tick **replace running
instances**, or pick a smaller quant or context. The message says how much
the model needs and how much is free.

**A model with an MTP head launches without speculation.**
The backend doesn't support that head. Gemma 4 assistant heads and some
NextN heads need a fork with `draft-mtp` support; the launch log says
*retrying without speculative decoding* when a head failed to load. The
model still runs normally.

**`unknown model architecture` or `unknown quantization type` in the launch log.**
The build is too old for that model, or the format needs a specific fork
(for example ternary `PTQ1_0`/`PQ2_0` models need the fork that
introduced them). Add a suitable build as another backend.

**`mismatch between text model n_embd and mmproj n_embd`** (or similar).
The mmproj belongs to a different model. Use the mmproj from the same
repo as the model; see [mmproj files](#mmproj-files-which-and-when).

**The launch times out.**
Large models on slow disks can take minutes to load. The launcher waits up
to 10 minutes for llama-server and 20 for vLLM, then stops the process it
started (it never leaves a half-started server behind). Check the model's
log in the Logs panel for the real error.

### Clients and the proxy

**`model 'X' is not being served`.**
The name doesn't match a running model. `GET /v1/models` lists the exact
names. See [How `model` is matched](#how-model-is-matched).

**`model 'X' is ambiguous`.**
More than one running model starts with that name. Send the exact name the
error lists.

**Claude Code shows an error on some requests.**
Claude Code sends background requests under a separate small-model name.
Set `ANTHROPIC_DEFAULT_HAIKU_MODEL` (as well as `ANTHROPIC_MODEL`) to a
served model; see [Examples](#examples).

**A newly launched model isn't found for a few seconds.**
The proxy caches discovery briefly but re-checks on a miss, so this should
resolve itself on the next request. If it persists, the proxy may be an
old version: restart it from the Servers panel.

**Video requests fail.**
`ffmpeg` isn't installed or configured. Install it or set its path under
**Settings → Speech & video**. Everything else keeps working without it.

**A request is rejected with 413.**
It's larger than `max_upload_mb` (512 MB by default). Raise it in
`config.json` if you really mean to send that much.

### Models and downloads

**A repo I can see on Hugging Face doesn't appear in the downloader.**
The downloader shows GGUF repos in GGUF mode and safetensors repos in vLLM
mode. A repo in another format (EXL3, AWQ, GPTQ, MLX …) only appears in the
mode that can serve it. EXL3 and similar pre-quantised formats can't be
converted to GGUF; look for a GGUF quant of the same base model instead,
or serve it with vLLM.

**Hugging Face downloads are slow or rate-limited.**
Set an `HF_TOKEN` environment variable (a free read token from your
Hugging Face account) before starting NeuralDeck.

**A downloaded model doesn't show its vision or MTP icon.**
Its mmproj or head isn't in the same folder, or wasn't downloaded. Download
it into the model's folder (or symlink an existing one). The icons come
from the files' headers, so a head that doesn't declare MTP layers isn't
shown as one.

**Deleting a GGUF model left files behind.**
Delete removes only the files belonging to that model. Anything else in its
folder (notes, other quants) is left alone, and the folder is removed only
once it's empty. (A safetensors model's whole folder is removed.)

### vLLM

**`CondaToSNonInteractiveError` when creating the environment.**
Conda wants Anaconda's channel terms accepted. Either accept them
yourself (`conda tos accept …`, as the message says) or create the
environment from conda-forge instead: `conda create -p ~/vllm-env
--override-channels -c conda-forge python=3.12 pip`.

**The first launch takes several minutes.**
Expected: vLLM compiles kernels and captures CUDA graphs on first start.
Later starts reuse the cache.

**flashinfer fails to compile kernels, or complains about PTX versions.**
The `nvcc` in the vLLM environment is newer than PyTorch's CUDA. Pin it:
`~/vllm-env/bin/python -m pip install "nvidia-cuda-nvcc==<torch CUDA>.*"`
(e.g. `13.0.*` for PyTorch built with CUDA 13.0).

**An EXL3 model fails with `quant_method exl3` not supported, or a missing `exllamav3_ext`.**
The EXL3 plugin or the `exllamav3` extension isn't installed, or the wheel
doesn't match PyTorch and CUDA. See [EXL3 models on vLLM](#exl3-models-on-vllm).

**Memory pressure shows *tight* while vLLM runs.**
Expected. vLLM reserves its memory share up front for the KV cache. It
won't grow further, but there's no room for another model beside it.

**vLLM KV cache shows a percentage but no token count.**
Hybrid models (with Mamba-style layers) allocate in large blocks, so a
token count would be misleading; the percentage is vLLM's own figure.

**A vLLM model doesn't appear in the Prompt Lab or Benchmarks.**
Those lists show running models only; launch it first. If requests to it
then fail with *not being served*, restart the proxy: an older proxy
can't discover vLLM servers.

### Telemetry

**The Roofline panel shows over 100%.**
`peak_bw_gbs` is set below your hardware's real memory bandwidth (the
default suits a desktop DDR5 system, not a GPU). Set it in **Settings →
Dashboard**; the panel links there when it detects this.

**GPU readings are blank.**
No supported tool answered: install `nvidia-smi` (NVIDIA driver) or
`amd-smi`/`rocm-smi` (AMD). A tool that fails occasionally is retried with
back-off; one that fails on its very first call is treated as absent until
the dashboard restarts.

**The CPU temperature looks wrong.**
NeuralDeck prefers the CPU package sensor (`coretemp`, `k10temp`,
`zenpower`) over generic ACPI zones, which often report a constant.
If your board exposes only an ACPI zone, that's what's shown.

**KV cache shows — for a llama.cpp model.**
The model hasn't served a request yet, or its log doesn't contain the
lines the dashboard reads. It fills in after the first request.

## Security

There's no authentication anywhere: the dashboard can launch processes and
delete model files, and the proxy serves your models to anyone who can
reach it. By default all three bind to `0.0.0.0` so other machines on your
network can use them.

* On an untrusted network, set `deck_host`, `proxy_host` and `llama_host`
  to `127.0.0.1`.
* Don't expose these ports to the internet. If you need remote access, put
  them behind a VPN or an authenticating reverse proxy.
* Model-publisher plugins (such as vLLM quantisation plugins) run inside
  the server process. Review them before installing, and pin the version
  you reviewed.

## Uninstall

Stop NeuralDeck, then delete:

* the checkout folder, which holds the default `venv`;
* the virtual environment you chose with `--dir` / `-Dir`, if you did;
* `~/.local/bin/neuraldeck`, the launcher `install.sh` writes (Linux/macOS);
* the Start Menu shortcut `NeuralDeck.lnk` in
  `%APPDATA%\Microsoft\Windows\Start Menu\Programs` (Windows);
* the data directory (`~/.local/share/neuraldeck` on Linux,
  `%LOCALAPPDATA%\NeuralDeck` on Windows, or wherever `NEURALDECK_HOME`
  points), which holds `config.json`, the logs and the benchmark history;
* a vLLM environment such as `~/vllm-env`, if you made one for NeuralDeck.

Your model files are never touched by uninstalling.

## Notes on portability

* Process control goes through pids, never through pattern matching: no
  `pkill`, no `lsof`, and nothing that could match the dashboard itself.
* GPU telemetry is merged from whatever answers. Fields nothing can report
  are left unknown, and the panels that need them hide themselves rather
  than showing a permanent dash.
* The launcher reads each server's `--help` and only passes flags that
  build accepts, so upstream builds, forks and vLLM all work.

## Documentation site

`docs/` holds a web version of this README for GitHub Pages
(**Settings → Pages → Deploy from a branch → `main` / `/docs`**). It's
generated, so edit the README and rebuild rather than editing the HTML:

```bash
python3 docs/build.py        # needs markdown-it-py: pip install markdown-it-py
```

## Credit

The dashboard, prompt lab and benchmark UI come from the NeuralDeck deck
built for a Strix Halo box; this package is that work made standalone and
cross-platform.

## Licence

MIT. See [LICENSE](LICENSE).
