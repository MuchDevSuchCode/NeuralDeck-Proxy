"""Scheduled prompts: the deck runs a prompt for an account on a timer.

A task is a prompt, a model, and either an interval in minutes or a time
of day. Tasks live in ``tasks/<account>.json`` in the data directory.
Every half minute the loop runs whatever is due: the prompt goes to the
proxy with the deck's own tools, tool calls are executed here (no
browser is involved, so nothing asks for permission), and the exchange
is saved as a new conversation in the account's chats, where the Chat
tab shows it on its next load. The task keeps its last result too.

The chat makes tasks through the schedule_task tool; this module owns
them.
"""

import asyncio
import datetime as _dt
import json
import os
import re
import secrets
import tempfile
import threading
import time

import httpx

from . import chats, config, tools, users

ROOT = config.DATA_DIR / "tasks"
TICK = 30
MAX_TURNS = 8
MAX_TASKS = 50
_lock = threading.Lock()
_running: set = set()


# ── storage ─────────────────────────────────────────────────────────────────

def _file(account: str):
    return ROOT / (account + ".json")


def _read(account: str) -> list:
    try:
        d = json.loads(_file(account).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [t for t in (d.get("tasks") if isinstance(d, dict) else d) or [] if isinstance(t, dict)]


def _write(account: str, items: list) -> None:
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=str(ROOT), prefix="." + account + "-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"tasks": items}, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, _file(account))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _accounts() -> list:
    try:
        return [n[:-5] for n in os.listdir(ROOT) if n.endswith(".json") and not n.startswith(".")]
    except OSError:
        return []


def _next_run(every, at, after: float) -> float:
    if every:
        return after + every * 60
    hh, mm = (int(x) for x in at.split(":"))
    now = _dt.datetime.fromtimestamp(after)
    cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if cand.timestamp() <= after:
        cand += _dt.timedelta(days=1)
    return cand.timestamp()


def _public(t: dict) -> dict:
    fmt = lambda ts: _dt.datetime.fromtimestamp(ts).isoformat(timespec="minutes") if ts else None
    return {"id": t["id"], "prompt": t["prompt"], "model": t.get("model"),
            "every_minutes": t.get("every"), "daily_at": t.get("at"),
            "created": fmt(t.get("created")), "next_run": fmt(t.get("next")),
            "last_run": fmt(t.get("last")), "runs": t.get("runs", 0),
            "last_result": (t.get("last_result") or "")[:600], "last_error": t.get("last_error")}


# ── the API the tools use ───────────────────────────────────────────────────

def add(account: str, *, prompt: str, every, at, model) -> dict:
    with _lock:
        items = _read(account)
        if len(items) >= MAX_TASKS:
            raise tools.ToolError(f"an account can have at most {MAX_TASKS} tasks")
        now = time.time()
        t = {"id": secrets.token_hex(4), "prompt": prompt[:2000], "model": model or None,
             "every": every, "at": at, "created": int(now), "next": _next_run(every, at, now),
             "last": None, "runs": 0, "last_result": None, "last_error": None}
        items.append(t)
        _write(account, items)
    return _public(t)


def list_tasks(account: str) -> list:
    with _lock:
        return [_public(t) for t in _read(account)]


def remove(account: str, tid: str) -> bool:
    with _lock:
        items = _read(account)
        keep = [t for t in items if t.get("id") != tid]
        if len(keep) == len(items):
            return False
        _write(account, keep)
        return True


# ── running a task ──────────────────────────────────────────────────────────

SYSTEM = ("You are running a scheduled task for the user on their NeuralDeck. Do what the task "
          "says, using tools where they help, and finish with a clear, self-contained report "
          "of what you found or did. Nobody can answer questions, so make reasonable choices.")


def _user_for(account: str) -> dict:
    if account == chats.LOCAL:
        return {"name": "", "role": "admin", "access": list(users.VIEWS)}
    u = users.get(account)
    return users._public(u) if u else {"name": account, "role": "user", "access": ["chat"]}


def _model_for(t: dict):
    if t.get("model"):
        return t["model"]
    cur = tools.CONTEXT.get("current_model")
    return cur() if cur else None


async def _run(account: str, t: dict) -> tuple:
    """(result text, saved conversation) for one task."""
    user = _user_for(account)
    model = _model_for(t)
    if not model:
        raise tools.ToolError("no model is loaded to run the task with")
    defs = [d for d in tools.definitions(user)
            if d["function"]["name"] not in ("schedule_task", "cancel_task", "list_tasks")]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t["prompt"]}]
    headers = tools.CONTEXT["proxy_headers"]()
    cid = "task-" + secrets.token_hex(5)
    conv = {"id": cid, "title": "⏰ " + t["prompt"][:70], "created": int(time.time() * 1000),
            "updated": int(time.time() * 1000), "model": model, "messages": [
                {"id": secrets.token_hex(5), "role": "user", "content": t["prompt"]}]}
    final = ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0)) as c:
        for _ in range(MAX_TURNS):
            body = {"model": model, "messages": msgs, "stream": False, "temperature": 0.3}
            if defs:
                body["tools"] = defs
            r = await c.post(config.PROXY_CHAT, json=body, headers=headers)
            if r.status_code != 200:
                raise tools.ToolError(f"the proxy answered {r.status_code}: {r.text[:200]}")
            msg = ((r.json().get("choices") or [{}])[0].get("message")) or {}
            content = msg.get("content") or ""
            calls = msg.get("tool_calls") or []
            am = {"id": secrets.token_hex(5), "role": "assistant", "content": content, "model": model}
            if calls:
                am["toolCalls"] = []
            conv["messages"].append(am)
            if not calls:
                final = content
                break
            msgs.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            for call in calls:
                fn = call.get("function") or {}
                name, raw = fn.get("name") or "", fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except ValueError:
                    args = {}
                try:
                    out = await asyncio.to_thread(tools.run, name, args, user)
                    status = "done"
                except tools.ToolError as e:
                    out, status = str(e), "error"
                am["toolCalls"].append({"id": call.get("id") or "call_" + secrets.token_hex(4),
                                        "name": name, "args": raw if isinstance(raw, str) else json.dumps(raw),
                                        "status": status, "result": out[:20000], "source": "NeuralDeck"})
                msgs.append({"role": "tool", "tool_call_id": call.get("id"), "content": out[:20000]})
        else:
            final = content or "(stopped after too many tool rounds)"
    conv["updated"] = int(time.time() * 1000)
    await asyncio.to_thread(chats.put, account, cid, conv)
    return final, conv


async def _tick():
    now = time.time()
    for account in _accounts():
        with _lock:
            due = [t for t in _read(account) if (t.get("next") or 0) <= now and t["id"] not in _running]
        for t in due:
            _running.add(t["id"])
            try:
                try:
                    result, _ = await _run(account, t)
                    err = None
                except (tools.ToolError, httpx.HTTPError, ValueError) as e:
                    result, err = None, str(e)[:400]
                with _lock:
                    items = _read(account)
                    for x in items:
                        if x["id"] == t["id"]:
                            x.update({"last": int(time.time()), "runs": x.get("runs", 0) + 1,
                                      "next": _next_run(x.get("every"), x.get("at"), time.time()),
                                      "last_result": result, "last_error": err})
                    _write(account, items)
                print(f"[task] {account}/{t['id']}: " + ("ok" if err is None else "error: " + err), flush=True)
            finally:
                _running.discard(t["id"])


async def loop():
    """The background runner; cancelled when the deck stops."""
    await asyncio.sleep(10)
    while True:
        try:
            await _tick()
        except Exception as e:   # a bad task must not kill the runner
            print(f"[task] runner error: {e}", flush=True)
        await asyncio.sleep(TICK)
