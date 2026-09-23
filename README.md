# NeuralDeck-Proxy

A local LLM control room for [llama.cpp](https://github.com/ggml-org/llama.cpp),
in one Python package that runs on **Windows or Linux**:

* **Dashboard** — live CPU/GPU/VRAM/thermal telemetry, every running
  `llama-server` with the flags it was launched with, KV-cache occupancy,
  a roofline estimate, per-service start/stop, logs, and model management
  (launch, delete, and download from Hugging Face).
* **Prompt Lab** — send a prompt to any loaded model and watch it stream,
  with TTFT, prefill/decode throughput and draft-acceptance measured per run.
* **Benchmarks** — every request through the stack is recorded, from the
  Prompt Lab, from a standard prompt, or from any other client on your
  network. Compare models, backends and speculation settings on medians.
* **Settings** — point NeuralDeck at your model folders and llama-server
  builds from the page itself, with a folder picker, validation, and an
  honest account of which changes apply at once and which need a restart.
* **Multimodal proxy** — one OpenAI-compatible endpoint in front of every
  `llama-server` you have running. It routes by model name, transcribes
  audio through whisper.cpp, turns video into frames for vision models,
  relays speech synthesis, and speaks the Anthropic message format too.

<sub>No authentication, by design. Run it on a network you trust.</sub>

---

## How it fits together

```
     your apps / Claude Code / the dashboard's Prompt Lab
                          │
                    :8080 │  multimodal proxy      ← the client-facing port
                          │  (routes by model name)
        ┌─────────────────┼──────────────────┐
        │                 │                  │
  :8081 llama-server  :8082 llama-server   :8090 whisper-server
        │                 │
        └────── :8770 NeuralDeck dashboard ─┘   ← watches, launches, benchmarks
```

The proxy owns the port your clients point at; `llama-server` instances sit
behind it. The dashboard is a third port that observes the whole thing and
launches models into it.

## Requirements

* Python 3.10+
* A built `llama-server` binary ([llama.cpp](https://github.com/ggml-org/llama.cpp) —
  any recent build; the launcher checks each binary's `--help` and only
  passes flags it supports)
* Optional: `whisper-server` from [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
  for speech input, `ffmpeg` for video input, `huggingface_hub` for the
  in-dashboard model downloader

## Install

One script does everything: it finds a suitable Python, builds a virtual
environment, installs NeuralDeck into it, prints what it found on your
machine, and starts the dashboard in your browser.

**Linux / macOS**

```bash
git clone git@github.com:MuchDevSuchCode/NeuralDeck-Proxy.git
cd NeuralDeck-Proxy
./install.sh
```

**Windows** (PowerShell, in the cloned folder)

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The dashboard opens at **http://localhost:8770**. Point it at your models
from the **Settings** tab — nothing else needs editing to get started.

| Installer flag | Effect |
| --- | --- |
| `--no-launch` / `-NoLaunch` | install only, do not start it |
| `--no-link` / `-NoShortcut` | skip the `~/.local/bin` launcher / Start Menu shortcut |
| `--dir <path>` / `-Dir <path>` | put the virtual environment somewhere else |

Nothing is installed system-wide: everything lives in the virtual
environment, plus a config file and logs under your user data directory.
Deleting the folder and that directory removes it completely.

<details>
<summary>Manual install, or into an existing environment</summary>

```bash
pip install -e ".[hub]"     # or: pip install -r requirements.txt
neuraldeck doctor           # what this machine looks like to NeuralDeck
neuraldeck                  # start the proxy and the dashboard
```

From a checkout without installing, the wrapper scripts build a venv on
first run:

```bash
./scripts/neuraldeck.sh            # Linux / macOS
.\scripts\neuraldeck.ps1           # Windows PowerShell
```
</details>

### Commands

| Command | What it does |
| --- | --- |
| `neuraldeck` (or `up`) | proxy as a child process, dashboard in the foreground |
| `neuraldeck up --open` | the same, and open the dashboard in a browser once it answers |
| `neuraldeck deck` | dashboard only |
| `neuraldeck proxy` | multimodal proxy only |
| `neuraldeck doctor` | resolved config, detected hardware, binaries, models, what is running |

Run `neuraldeck doctor` first. It tells you what was found and what is
missing before anything tries to start.

## Settings

Everything configurable is on the **Settings** tab, grouped: Models,
Backends, Launch defaults, Sampling, Ports, Speech & video, Proxy,
Dashboard, Optional services.

* **Model directories** — add or remove folders with a picker that shows
  how many `.gguf` files each subfolder holds, so you can see which one is
  the right level. A folder that is not mounted right now is kept and
  flagged rather than dropped.
* **llama-server builds** — label → path. Add more than one and the launch
  form lets you pick between them per launch, which is how you A/B a fork
  against upstream on the same model.
* Most changes **apply immediately** — the running process re-reads the
  file. Settings that were read when the process started (ports, history
  length, proxy behaviour) carry a `restart` badge, and the page offers a
  restart button that brings itself back, following the port if you changed
  it.
* A setting fixed by an environment variable carries an `env` badge and is
  read-only, because writing the file could not override it. The page says
  so instead of accepting a change that would do nothing.
* Clearing a field means "use the default", not "set it to empty". The
  `use default` button next to a stored value does the same.

Saved settings land in `config.json` in the data directory
(`%LOCALAPPDATA%\NeuralDeck` on Windows, `~/.local/share/neuraldeck` on
Linux), which also holds the instance logs and the benchmark history. The
Settings tab prints all three paths; so does `neuraldeck doctor`.

### Configuring without the page

Precedence is environment → `config.json` → defaults. The keys are the same
in both, and `config.json` uses lowercase names with JSON types:

```json
{
  "backends": {
    "upstream": "/home/me/llama.cpp/build/bin/llama-server",
    "fork":     "/home/me/my-fork/build/bin/llama-server"
  },
  "model_dirs": ["/home/me/models", "/mnt/big/models"],
  "peak_bw_gbs": 256,
  "ctx": 65536
}
```

```bash
export NEURALDECK_BACKENDS="upstream=/home/me/llama.cpp/build/bin/llama-server"
export NEURALDECK_MODEL_DIRS="/home/me/models:/mnt/big/models"   # ';' on Windows
export NEURALDECK_PEAK_BW_GBS=256
```

The settings most worth checking on a new machine:

| Key | Default | Meaning |
| --- | --- | --- |
| `model_dirs` | `~/models` (+ `/mnt/models`) | where to look for GGUFs |
| `backends` | first `llama-server` found | label → binary path |
| `deck_port` / `proxy_port` | `8770` / `8080` | the two listening ports |
| `deck_host` | `0.0.0.0` | `127.0.0.1` keeps the dashboard off the network |
| `llama_port_range` | `8081-8089` | ports the deck launches into and the proxy discovers |
| `ctx` / `slots` | `32768` / `1` | launch-form defaults |
| `kv_cache_type` | `q8_0` | `-ctk`/`-ctv` for launched instances |
| `peak_bw_gbs` | `89.6` | memory bandwidth for the roofline panel — set it to your real figure |
| `whisper_bin` / `whisper_model` | whisper.cpp defaults | speech input |
| `ffmpeg` | from `PATH` | needed only for video input |

`NEURALDECK_HOME` moves the data directory itself, which is the one setting
the page cannot change (it is where the page's settings are stored).

## Using the proxy

Point any OpenAI-compatible client at `http://localhost:8080/v1`. The
`model` field selects which running instance answers; a name that nothing
is serving is an error rather than a silent answer from the wrong model.

| Endpoint | Behaviour |
| --- | --- |
| `POST /v1/chat/completions` | routed by model; `input_audio` parts transcribed; `video_url` parts passed through or turned into frames |
| `POST /v1/messages` | Anthropic format, routed the same way |
| `POST /v1/multimodal` | one multipart call with text, images, videos and audio together |
| `POST /v1/audio/speech` | relayed to the TTS backend |
| `GET /v1/models`, `/props`, `/health` | aggregated across every instance |
| anything else | relayed to llama-server untouched |

## How models are found

A model is a directory under a model root holding a `.gguf`, plus the
helper files beside it. NeuralDeck reads each GGUF's header — not its
filename — to decide whether it carries an MTP/NextN draft head and whether
its chat template does reasoning, because filenames lie about both. A
projector (`*mmproj*.gguf`) in the same folder makes the model a vision
model; a separate draft head is used for speculative decoding when the
backend supports it.

## Benchmarks

Runs land in `bench.jsonl` in the data directory, from three sources: the
Prompt Lab, the Benchmarks tab's standard prompt, and a collector that
reads each instance's log so requests from *any* client are measured too.
The chart shows medians per model × backend × speculation setting, with
each metric on its own scale.

## Notes on portability

* Process control goes through pids, never through pattern matching — no
  `pkill`, no `lsof`, and nothing that could match the dashboard itself.
* GPU telemetry is merged from whatever answers: `nvidia-smi`, `amd-smi`,
  `rocm-smi`, the amdgpu sysfs nodes, or Windows CIM. Fields nothing can
  report are reported as unknown, and the panels that need them hide
  themselves rather than showing a permanent dash.
* The launcher reads each `llama-server`'s `--help` and only passes flags
  that build accepts, so a stock build and a fork both work. If a model's
  MTP head fails to load, the launch retries without it instead of failing.

## Credit

The dashboard, prompt lab and benchmark UI come from the NeuralDeck deck
built for a Strix Halo box; this package is that work made standalone and
cross-platform.

## Licence

MIT — see [LICENSE](LICENSE).
