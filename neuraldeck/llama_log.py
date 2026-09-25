"""llama.cpp server log parser.

llama-server's own /metrics endpoint reports counters, not per-request
timings, and says nothing at all about speculative-decode acceptance. Its
log does, in `print_timing` blocks, so the dashboard's headline inference
numbers come from parsing the log an instance writes.

Derived from llama_stats.py in the NeuralDeck scripts, reduced to the
parsing that the dashboard consumes.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── per-request timing block ───────────────────────────────────────────────
# Builds differ in where the numbers sit. Older ones print a
# `print_timing: id N | task N |` header and then bare value lines; newer
# ones repeat the header on every value line. Both are parsed: the header
# only opens a block, and values are read from that same line too.
RE_PRINT_TIMING = re.compile(r"slot print_timing: id\s+(\d+)\s*\|\s*task\s+(-?\d+)")
RE_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*"
    r"\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
RE_GEN_EVAL = re.compile(
    r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*"
    r"\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
RE_DRAFT = re.compile(r"draft acceptance rate\s*=\s*([\d.]+)\s*"
                      r"\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)")
RE_NEXTN = re.compile(
    r"statistics nextn:.*#gen drafts\s*=\s*(\d+),\s*#acc drafts\s*=\s*(\d+),"
    r"\s*#gen tokens\s*=\s*(\d+),\s*#acc tokens\s*=\s*(\d+)")

# ── in-flight state ────────────────────────────────────────────────────────
RE_NEW_PROMPT = re.compile(r"new prompt, n_ctx_slot\s*=\s*(\d+), n_keep\s*=\s*\d+,"
                           r" task\.n_tokens\s*=\s*(\d+)")
RE_PROGRESS = re.compile(r"prompt processing progress, n_tokens\s*=\s*(\d+),"
                         r" batch\.n_tokens\s*=\s*\d+, progress\s*=\s*([\d.]+)")
RE_LAUNCH = re.compile(r"launch_slot_: id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|"
                       r"\s*processing task")
RE_RELEASE = re.compile(r"slot\s+release: id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|"
                        r"\s*stop processing")
RE_ALL_IDLE = re.compile(r"all slots are idle")

# ── header ─────────────────────────────────────────────────────────────────
RE_MODEL = re.compile(r"load_model: loading model '([^']+)'")
RE_BUILD = re.compile(r"build_info:\s*(\S+)")
RE_DEVICE = re.compile(r"Device \d+:\s*(.+?),\s*(gfx\S+|\S+).*?VRAM:\s*(\d+)\s*MiB")
RE_NSLOTS = re.compile(r"n_slots\s*=\s*(\d+)")
# n_ctx_seq on older builds, n_ctx_slot on newer ones — same number
RE_NCTX_SEQ = re.compile(r"n_ctx_(?:seq|slot)\s*=\s*(\d+)")
RE_SERVER_UP = re.compile(r"listening on (\S+)")
RE_THREADS = re.compile(r"n_threads\s*=\s*(\d+)")


@dataclass
class Timing:
    slot: int = -1
    task: int = -1
    pp_ms: float = 0.0            # prefill (prompt eval)
    pp_tokens: int = 0
    pp_tps: float = 0.0
    tg_ms: float = 0.0            # decode (generation)
    tg_tokens: int = 0
    tg_tps: float = 0.0
    total_ms: float = 0.0
    total_tokens: int = 0
    draft_rate: Optional[float] = None
    draft_accepted: int = 0
    draft_generated: int = 0
    spec_gen_tokens: int = 0
    spec_acc_tokens: int = 0


@dataclass
class Header:
    model: Optional[str] = None
    build: Optional[str] = None
    devices: List[str] = field(default_factory=list)
    vram_mib: int = 0
    n_slots: Optional[int] = None
    n_ctx: Optional[int] = None
    n_threads: Optional[int] = None
    listen: Optional[str] = None


@dataclass
class LiveState:
    active: bool = False
    task: int = -1
    prompt_tokens: int = 0
    progress: float = 0.0


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def parse_lines(lines):
    """(timings, header, live) from log lines."""
    timings: List[Timing] = []
    header, live = Header(), LiveState()
    cur: Optional[Timing] = None

    def flush():
        nonlocal cur
        if cur is not None:
            timings.append(cur)
            cur = None

    for line in lines:
        if (m := RE_MODEL.search(line)):
            header.model = m.group(1)
        if (m := RE_BUILD.search(line)):
            header.build = m.group(1)
        if (m := RE_DEVICE.search(line)):
            header.devices.append(m.group(1).strip())
            header.vram_mib = max(header.vram_mib, int(m.group(3)))
        if (m := RE_NSLOTS.search(line)):
            header.n_slots = int(m.group(1))
        if (m := RE_NCTX_SEQ.search(line)):
            header.n_ctx = int(m.group(1))
        if (m := RE_THREADS.search(line)) and header.n_threads is None:
            header.n_threads = int(m.group(1))
        if (m := RE_SERVER_UP.search(line)):
            header.listen = m.group(1)

        if (m := RE_LAUNCH.search(line)):
            live.active = True
            live.task = int(m.group(2))
            live.progress = 0.0
            live.prompt_tokens = 0
        if (m := RE_NEW_PROMPT.search(line)):
            live.prompt_tokens = int(m.group(2))
        if (m := RE_PROGRESS.search(line)):
            live.progress = float(m.group(2))
        if RE_RELEASE.search(line) or RE_ALL_IDLE.search(line):
            live.active = False
            live.progress = 0.0

        if (m := RE_PRINT_TIMING.search(line)):
            task = int(m.group(2))
            if cur is None or cur.task != task:
                flush()
                cur = Timing(slot=int(m.group(1)), task=task)
            # deliberately no continue: on newer builds this very line
            # carries one of the values below
        if cur is None:
            continue
        if (m := RE_PROMPT_EVAL.search(line)):
            cur.pp_ms, cur.pp_tokens, cur.pp_tps = _f(m.group(1)), int(m.group(2)), _f(m.group(4))
        elif (m := RE_GEN_EVAL.search(line)):
            cur.tg_ms, cur.tg_tokens, cur.tg_tps = _f(m.group(1)), int(m.group(2)), _f(m.group(4))
        elif (m := RE_TOTAL.search(line)):
            cur.total_ms, cur.total_tokens = _f(m.group(1)), int(m.group(2))
        elif (m := RE_DRAFT.search(line)):
            cur.draft_rate = _f(m.group(1))
            cur.draft_accepted, cur.draft_generated = int(m.group(2)), int(m.group(3))
        elif (m := RE_NEXTN.search(line)):
            cur.spec_gen_tokens, cur.spec_acc_tokens = int(m.group(3)), int(m.group(4))
            flush()          # nextn is the last line of the block
    flush()
    return timings, header, live


def summarize(timings: List[Timing]) -> Dict:
    done = [t for t in timings
            if t.total_ms > 0 or t.tg_tokens > 0 or t.pp_tokens > 0]
    if not done:
        return {"requests": 0}
    pp_tps = [t.pp_tps for t in done if t.pp_tps > 0]
    tg_tps = [t.tg_tps for t in done if t.tg_tps > 0]
    total_pp = sum(t.pp_tokens for t in done)
    total_tg = sum(t.tg_tokens for t in done)
    total_pp_ms = sum(t.pp_ms for t in done)
    total_tg_ms = sum(t.tg_ms for t in done)
    drafts = [t for t in done if t.draft_generated > 0]
    draft_acc = sum(t.draft_accepted for t in drafts)
    draft_gen = sum(t.draft_generated for t in drafts)

    def stats(xs):
        return ({"min": min(xs), "max": max(xs), "avg": sum(xs) / len(xs)}
                if xs else {"min": 0.0, "max": 0.0, "avg": 0.0})

    return {
        "requests": len(done),
        "total_prompt_tokens": total_pp,
        "total_gen_tokens": total_tg,
        "total_gen_seconds": total_tg_ms / 1000.0,
        "total_busy_seconds": sum(t.total_ms for t in done) / 1000.0,
        "prefill_tps": stats(pp_tps),
        "decode_tps": stats(tg_tps),
        # token-weighted, which is more honest than a mean of rates
        "prefill_tps_weighted": (total_pp / total_pp_ms * 1000) if total_pp_ms else 0.0,
        "decode_tps_weighted": (total_tg / total_tg_ms * 1000) if total_tg_ms else 0.0,
        "draft_rate": (draft_acc / draft_gen) if draft_gen else None,
        "draft_accepted": draft_acc,
        "draft_generated": draft_gen,
    }


# Where one run of a server begins, in a log that may hold several.
RUN_START = "load_model: loading model"
TAIL_BYTES = 1 << 20


def _tail(path, n_bytes: int = TAIL_BYTES) -> Optional[str]:
    """The last n_bytes of a log, from a line boundary. None if unreadable."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - n_bytes))
            data = fh.read()
    except OSError:
        return None
    if size > n_bytes:
        data = data.split(b"\n", 1)[-1]
    return data.decode("utf-8", errors="replace")


def quick_stats(path) -> Dict:
    """Headline numbers for the dashboard, for the current run. Never raises.

    Called every second, so only a bounded tail is read, and only the part
    after the last startup counts — earlier launches into the same log must
    not blend into this one's totals.
    """
    text = _tail(path)
    if text is None:
        return {"ok": False}
    idx = text.rfind(RUN_START)
    timings, header, live = parse_lines(
        (text[idx:] if idx >= 0 else text).splitlines())
    if header.build is None and idx > 0:
        # build_info can print before the model loads
        builds = RE_BUILD.findall(text[:idx])
        header.build = builds[-1] if builds else None
    s = summarize(timings)
    s.update({
        "ok": True,
        "model": os.path.basename(header.model) if header.model else None,
        "n_ctx": header.n_ctx,
        "build": header.build,
        "live_active": live.active,
        "live_progress": live.progress,
        "live_prompt_tokens": live.prompt_tokens,
    })
    return s
