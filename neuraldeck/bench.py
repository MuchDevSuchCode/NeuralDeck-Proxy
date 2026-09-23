"""Benchmark history.

Three things write runs into one JSONL file:

  * the Prompt Lab, for every prompt you send      (kind "lab")
  * the Benchmarks tab's standard prompt            (kind "std")
  * the traffic collector below, for every request any client makes,
    including remote ones the dashboard never saw  (kind "traffic")

The collector tails each instance's log and reads the print_timing blocks,
which is the only place per-request numbers exist for a request the
dashboard did not itself issue.
"""

import json
import os
import re
import time
from collections import deque
from pathlib import Path

from . import config, procs

# The task id is on the print_timing header line; on newer builds it is
# repeated on every value line, on older ones the values follow bare. Track
# the open task per log and read values whichever way they arrive.
RE_PT_TASK = re.compile(r"print_timing: id\s+(\d+) \| task (-?\d+)")
RE_PT_PROMPT = re.compile(
    r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens"
    r".*?([\d.]+) tokens per second")
RE_PT_EVAL = re.compile(
    r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens"
    r".*?([\d.]+) tokens per second")
RE_PT_DRAFT = re.compile(
    r"draft acceptance(?: rate)? =\s*[\d.]+ "
    r"\(\s*(\d+) accepted /\s*(\d+) generated")

_tail: dict = {}      # log path -> {"off": bytes consumed, "rem": partial line}
_pending: dict = {}   # (path, task) -> partial record
# Runs the browser posted itself, so the collector does not record them
# twice from the log a moment later.
_client_posts: deque = deque(maxlen=30)


def _new_id() -> str:
    return f"{int(time.time() * 1000)}-{os.urandom(3).hex()}"


def read(limit: int = None) -> list:
    recs = []
    try:
        with open(config.BENCH_FILE) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return recs[-limit:] if limit else recs


def append(rec: dict) -> None:
    try:
        config.BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(config.BENCH_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def add_client_run(body: dict, instances: list) -> dict:
    """Record a run the dashboard measured in the browser."""
    model = str(body.get("model") or "")[:120]
    rec = {"id": _new_id(), "ts": round(time.time(), 1),
           "kind": "std" if body.get("kind") == "std" else "lab",
           "model": model}
    inst = next((i for i in instances if i.get("alias") == model), None)
    rec["backend"] = inst.get("backend") if inst else None
    rec["spec"] = inst.get("spec") if inst else None
    for k in ("prompt_n", "predicted_n", "draft_n", "draft_acc", "max_tokens",
              "ttft_ms"):
        if isinstance(body.get(k), (int, float)):
            rec[k] = int(body[k])
    for k in ("prompt_tps", "decode_tps", "wall_s", "temp"):
        if isinstance(body.get(k), (int, float)):
            rec[k] = round(float(body[k]), 2)
    append(rec)
    _client_posts.append((time.time(), model, int(body.get("predicted_n") or -1)))
    return rec


def delete(rec_id: str) -> bool:
    recs = read()
    kept = [r for r in recs if r.get("id") != rec_id]
    if len(kept) == len(recs):
        return False
    with open(config.BENCH_FILE, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept)
    return True


def clear() -> None:
    open(config.BENCH_FILE, "w").close()


def collect_traffic(instances: list) -> None:
    """Read what is new in each instance's log and bank finished requests."""
    now = time.time()
    for inst in instances or []:
        alias = inst.get("alias")
        if not alias:
            continue
        path = str(config.LOG_DIR / f"{alias}.log")
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        st = _tail.get(path)
        if st is None or st["off"] > size:
            # First sighting, or the log was truncated: start at the end.
            # This collector records live traffic; it does not replay history.
            _tail[path] = {"off": size, "rem": "", "task": None}
            continue
        if size == st["off"]:
            continue
        try:
            with open(path, "rb") as f:
                f.seek(st["off"])
                data = f.read(min(size - st["off"], 1 << 20))
        except OSError:
            continue
        st["off"] += len(data)
        lines = (st["rem"] + data.decode(errors="replace")).split("\n")
        st["rem"] = lines.pop()
        task = st.get("task")
        for ln in lines:
            if (m := RE_PT_TASK.search(ln)):
                task = st["task"] = m.group(2)
            if task is None:
                continue
            key = (path, task)
            if (m := RE_PT_PROMPT.search(ln)):
                p = _pending.setdefault(key, _blank(alias, inst, now))
                p.update(prompt_ms=float(m.group(1)), prompt_n=int(m.group(2)),
                         prompt_tps=float(m.group(3)))
            elif (m := RE_PT_EVAL.search(ln)):
                p = _pending.setdefault(key, _blank(alias, inst, now))
                p.update(eval_ms=float(m.group(1)), predicted_n=int(m.group(2)),
                         decode_tps=float(m.group(3)))
            elif (m := RE_PT_DRAFT.search(ln)):
                p = _pending.get(key)
                if p is not None:
                    p.update(draft_acc=int(m.group(1)), draft_n=int(m.group(2)))
            else:
                continue
            p = _pending[key]
            if "prompt_ms" in p and "eval_ms" in p:
                # Linger a moment so the draft line lands, and so a Prompt
                # Lab post can claim the run first.
                p.setdefault("done_at", now + 3.0)
    _flush_pending(now)


def _blank(alias, inst, now) -> dict:
    return {"model": alias, "backend": inst.get("backend"),
            "spec": inst.get("spec"), "seen": now}


def _flush_pending(now: float) -> None:
    for key in list(_pending):
        p = _pending[key]
        if "done_at" not in p:
            if now - p.get("seen", now) > 900:
                _pending.pop(key, None)
            continue
        if now < p["done_at"]:
            continue
        _pending.pop(key, None)
        pn = p.get("predicted_n", 0)
        if pn < 1:
            continue
        if any(m == p["model"] and abs(n - pn) <= 2 and now - t < 15
               for t, m, n in _client_posts):
            continue  # the browser already recorded this one
        rec = {"id": _new_id(), "ts": round(now, 1), "kind": "traffic",
               "model": p["model"], "backend": p.get("backend"),
               "spec": p.get("spec"), "prompt_n": p.get("prompt_n"),
               "predicted_n": pn,
               "prompt_tps": round(p.get("prompt_tps", 0), 2),
               "decode_tps": round(p.get("decode_tps", 0), 2),
               "wall_s": round((p.get("prompt_ms", 0) + p.get("eval_ms", 0)) / 1000, 2)}
        for k in ("draft_n", "draft_acc"):
            if k in p:
                rec[k] = p[k]
        append(rec)
