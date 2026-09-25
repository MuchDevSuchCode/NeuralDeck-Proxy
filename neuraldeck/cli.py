"""Command line entry point: `neuraldeck <command>`."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def _doctor() -> int:
    """Print what this machine looks like to NeuralDeck, and what is missing."""
    from . import config, launcher, models, procs, services, sysinfo
    print("NeuralDeck configuration")
    print("=" * 56)
    for k, v in config.summary().items():
        if k == "config_error":
            if v:
                print(f"  {'CONFIG ERROR':<16} {v}")
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, indent=None)
        print(f"  {k:<16} {v}")

    print("\nHardware")
    print("=" * 56)
    g = sysinfo.gpu()
    print(f"  cpu              {sysinfo.cpu_name()}")
    temp, volts = sysinfo.cpu_metrics()
    print(f"  cpu temp         {temp if temp is not None else 'unavailable'}")
    print(f"  gpu              {g.get('name') or 'not detected'}")
    print(f"  gpu probes       {', '.join(g.get('sources') or []) or 'none'}")
    vt = g.get("vram_total_bytes")
    print(f"  vram             {vt / 1024**3:.1f} GiB" if vt else
          "  vram             unknown (set NEURALDECK_VRAM_TOTAL_GB)")

    print("\nBinaries")
    print("=" * 56)
    ok = True
    for label, binary in config.BACKENDS.items():
        found = os.path.isfile(binary) or shutil.which(binary)
        kind = config.backend_kind(binary)
        if kind == "vllm":
            # a vllm script that exists can still fail to import (a broken
            # env); running it is the only real check
            ver = launcher.vllm_version(binary) if found else None
            ok = ok and bool(ver)
            state = "ok " if ver else ("BROKEN" if found else "MISSING")
            print(f"  vllm             [{state}] {label}: {binary}")
            if found:
                print(f"                   version: "
                      f"{ver or '`vllm --version` failed or timed out'}")
            continue
        ok = ok and bool(found)
        print(f"  llama-server     [{'ok ' if found else 'MISSING'}] {label}: {binary}")
        if found:
            c = launcher.caps(binary)
            feats = [k for k in ("spec_type", "reasoning_budget", "mmproj",
                                 "embeddings", "jinja") if c.get(k)]
            print(f"                   supports: {', '.join(feats) or 'unknown'}")
    for label, path in (("whisper-server", config.WHISPER_BIN),
                        ("whisper model", config.WHISPER_MODEL),
                        ("ffmpeg", config.FFMPEG)):
        found = os.path.isfile(path) or shutil.which(path)
        note = "" if found else "  (optional)"
        print(f"  {label:<16} [{'ok ' if found else 'missing'}] {path}{note}")

    print("\nModels")
    print("=" * 56)
    found = models.discover()
    for m in found[:20]:
        flags = "".join(c if m[k] else "·" for c, k in
                        (("V", "vision"), ("M", "mtp"), ("T", "thinking"),
                         ("E", "embed")))
        fmt = (f"  (safetensors{', ' + m['quant'] if m.get('quant') else ''})"
               if m.get("format") == "hf" else "")
        print(f"  [{flags}] {m['vram_bytes'] / 1024**3:6.1f} GiB  {m['name']}{fmt}")
    if len(found) > 20:
        print(f"  … and {len(found) - 20} more")
    if not found:
        print(f"  none found in {config.MODEL_DIRS}")

    print("\nRunning")
    print("=" * 56)
    for inst in procs.llama_instances():
        print(f"  :{inst['port']} {inst['backend']:<16} {inst['alias']}"
              f"  ({inst.get('kind', 'llama.cpp')})")
    for name, svc in services.snapshot().items():
        print(f"  :{svc['port']} {name:<16} {'running' if svc['up'] else 'stopped'}")
    return 0 if ok else 1


def _open_when_ready(host: str, port: int, timeout: float = 90.0) -> None:
    """Open the dashboard in a browser once it answers.

    Used by the installers so the first run lands on the page instead of a
    URL the user has to copy out of a terminal. A wildcard bind is reached
    on localhost; a specific address is only reachable at that address.
    """
    import socket
    import threading
    import time
    import webbrowser

    host = (host or "").strip()
    if host in ("", "0.0.0.0", "::", "[::]"):
        probe, shown = "127.0.0.1", "localhost"
    else:
        probe = host
        shown = f"[{host}]" if ":" in host else host

    def wait():
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                socket.create_connection((probe, port), timeout=0.5).close()
            except OSError:
                time.sleep(0.5)
                continue
            webbrowser.open(f"http://{shown}:{port}/")
            return

    threading.Thread(target=wait, daemon=True).start()


def _run_all(args) -> int:
    """Proxy as a child, deck in the foreground: one command for the stack."""
    from . import config, services
    if getattr(args, "open", False):
        _open_when_ready(config.DECK_HOST, config.DECK_PORT)
    child = None
    if not args.no_proxy:
        try:
            info = services.start("proxy")
            print(f"proxy started (pid {info['pid']}) -> {info['log']}", flush=True)
            child = info["pid"]
        except RuntimeError:
            print("proxy already running", flush=True)
        except Exception as e:
            print(f"could not start the proxy: {e}", file=sys.stderr)
    try:
        from .deck import main as deck_main
        deck_main()
    finally:
        if child and args.stop_proxy:
            from . import procs
            procs.stop_pid(child)
    return 0


def _ensure_streams() -> None:
    """Give pythonw somewhere to write.

    Under pythonw.exe (the Start Menu shortcut) sys.stdout and sys.stderr
    are None, and the first print — or uvicorn configuring its log
    formatter — would kill the process before the page ever loads, with
    no window to show why. Send both to deck.log in the log directory.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        from . import config
        sink = open(config.LOG_DIR / "deck.log", "a", encoding="utf-8",
                    buffering=1)
    except Exception:
        sink = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = sink
    if sys.stderr is None:
        sys.stderr = sink


def main(argv=None) -> int:
    _ensure_streams()
    p = argparse.ArgumentParser(
        prog="neuraldeck",
        description="Local LLM dashboard, prompt lab, benchmarks and "
                    "multimodal proxy for llama.cpp.")
    sub = p.add_subparsers(dest="cmd")

    p_all = sub.add_parser("up", help="start the proxy and the dashboard (default)")
    p_all.add_argument("--no-proxy", action="store_true",
                       help="dashboard only; leave the proxy alone")
    p_all.add_argument("--stop-proxy", action="store_true",
                       help="stop the proxy again when the dashboard exits")
    p_all.add_argument("--open", action="store_true",
                       help="open the dashboard in a browser once it is up")

    p_deck = sub.add_parser("deck", help="dashboard only")
    p_deck.add_argument("--open", action="store_true",
                        help="open the dashboard in a browser once it is up")
    sub.add_parser("proxy", help="multimodal proxy only")
    sub.add_parser("doctor", help="show what this machine looks like to NeuralDeck")
    # bare `neuraldeck` means `up`, which reads these without a subparser
    p.set_defaults(no_proxy=False, stop_proxy=False, open=False)

    args = p.parse_args(argv)
    cmd = args.cmd or "up"

    if cmd == "deck":
        from . import config
        if getattr(args, "open", False):
            _open_when_ready(config.DECK_HOST, config.DECK_PORT)
        from .deck import main as deck_main
        deck_main()
        return 0
    if cmd == "proxy":
        from .proxy import main as proxy_main
        proxy_main()
        return 0
    if cmd == "doctor":
        return _doctor()
    return _run_all(args)
