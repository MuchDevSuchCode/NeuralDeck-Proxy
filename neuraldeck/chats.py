"""Chat conversations, kept per account.

Each account's conversations live under ``chats/<account>/`` in the data
directory, one JSON file per conversation, so one user never sees
another's history and a history follows its account to any browser.
With sign-in off there is a single implicit account, kept as ``_local``.

A conversation is whatever the chat tab saves: an ``id``, a ``title``,
``created``/``updated`` stamps, the ``model`` and a ``messages`` list.
The deck checks only the shape; the content is the client's.
"""

import json
import os
import re
import tempfile
import threading

from . import config, users

ROOT = config.DATA_DIR / "chats"
LOCAL = "_local"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{3,63}$")
MAX_BYTES = 64 * 1024 * 1024        # a conversation full of images
_lock = threading.Lock()


class ChatError(ValueError):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


def _dir(account):
    name = (account or "").strip().lower() or LOCAL
    if name != LOCAL and not users.NAME_RE.match(name):
        raise ChatError(400, "bad account name")
    return ROOT / name


def _check_id(cid) -> str:
    if not isinstance(cid, str) or not ID_RE.match(cid):
        raise ChatError(400, "a conversation id is 4–64 characters: letters, digits, '_' or '-'")
    return cid


def list_convs(account) -> list:
    """Every conversation of the account, newest first."""
    d = _dir(account)
    out = []
    with _lock:
        try:
            names = [n for n in os.listdir(d) if n.endswith(".json")]
        except OSError:
            return []
        for n in names:
            try:
                c = json.loads((d / n).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(c, dict) and isinstance(c.get("messages"), list) and c.get("id") == n[:-5]:
                out.append(c)
    out.sort(key=lambda c: c.get("updated") or 0, reverse=True)
    return out


def put(account, cid, conv: dict) -> None:
    cid = _check_id(cid)
    if not isinstance(conv, dict) or not isinstance(conv.get("messages"), list):
        raise ChatError(400, "a conversation is an object with a messages list")
    if conv.get("id") not in (None, cid):
        raise ChatError(400, "the id in the body differs from the one in the URL")
    conv = dict(conv, id=cid)
    conv.pop("draft", None)
    data = json.dumps(conv, ensure_ascii=False, separators=(",", ":"))
    if len(data) > MAX_BYTES:
        raise ChatError(413, f"this conversation is over {MAX_BYTES // (1024 * 1024)} MB")
    d = _dir(account)
    with _lock:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=str(d), prefix="." + cid + "-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.chmod(tmp, 0o600)
            os.replace(tmp, d / (cid + ".json"))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def delete(account, cid) -> bool:
    cid = _check_id(cid)
    with _lock:
        try:
            (_dir(account) / (cid + ".json")).unlink()
            return True
        except FileNotFoundError:
            return False


def clear(account) -> int:
    """Remove every conversation of the account; how many went."""
    d = _dir(account)
    n = 0
    with _lock:
        try:
            names = [x for x in os.listdir(d) if x.endswith(".json")]
        except OSError:
            return 0
        for x in names:
            try:
                (d / x).unlink()
                n += 1
            except OSError:
                pass
    return n
