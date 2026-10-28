"""Bearer tokens for the proxy.

The proxy is open until the first token is generated; from then on every
request (bar /health and the web UI's page) must carry one, as
``Authorization: Bearer <token>`` or ``x-api-key: <token>``. Tokens are
made on the Settings page, shown once, and only their SHA-256 is kept in
``proxy_tokens.json`` in the data directory. Revoking is deleting the row.

The dashboard talks to the proxy on the user's behalf (the Chat tab, the
Prompt Lab) and needs no token of its own: both processes run as the same
user, so they share ``proxy_internal.key`` (mode 0600), which the proxy
accepts whenever tokens are required. It is created on first use by
whichever process gets there first.

The proxy re-reads the token file when its mtime changes, so a token
works, or stops working, the moment it is saved — no restart.
"""

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time

from . import config

FILE = config.DATA_DIR / "proxy_tokens.json"
INTERNAL_FILE = config.DATA_DIR / "proxy_internal.key"
PREFIX = "nd_"
_lock = threading.Lock()
_cache = {"stamp": None, "tokens": []}
_touched: dict = {}   # token id -> when last_used was last written


def _stamp():
    try:
        st = FILE.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _read() -> list:
    """The stored rows, re-read only when the file changed."""
    stamp = _stamp()
    if stamp == _cache["stamp"]:
        return _cache["tokens"]
    tokens = []
    if stamp is not None:
        try:
            data = json.loads(FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("tokens", [])
            tokens = [t for t in data if isinstance(t, dict) and t.get("hash")]
        except (OSError, ValueError):
            tokens = []
    _cache["stamp"], _cache["tokens"] = stamp, tokens
    return tokens


def _write(tokens: list) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(FILE.parent), prefix=".tokens-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"tokens": tokens}, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _cache["stamp"], _cache["tokens"] = _stamp(), tokens


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _public(t: dict) -> dict:
    return {"id": t["id"], "name": t.get("name") or "", "prefix": t.get("prefix", ""),
            "created": t.get("created"), "last_used": t.get("last_used")}


def list_tokens() -> list:
    with _lock:
        return [_public(t) for t in _read()]


def required() -> bool:
    """Whether the proxy demands a token: it does once any exist."""
    return bool(_read())


def create(name: str) -> tuple:
    """A new token: (its public row, the secret). The secret is returned
    exactly once; only its hash is stored."""
    name = (name or "").strip()[:64]
    secret = PREFIX + secrets.token_urlsafe(32)
    row = {"id": secrets.token_hex(6), "name": name, "prefix": secret[:len(PREFIX) + 6],
           "hash": _hash(secret), "created": int(time.time()), "last_used": None}
    with _lock:
        tokens = list(_read())
        tokens.append(row)
        _write(tokens)
    return _public(row), secret


def revoke(token_id: str) -> bool:
    with _lock:
        tokens = list(_read())
        keep = [t for t in tokens if t.get("id") != token_id]
        if len(keep) == len(tokens):
            return False
        _write(keep)
        return True


def verify(secret: str):
    """The row the secret belongs to, or None. The dashboard's internal
    key verifies as a row of its own, never listed and never revocable."""
    if not secret:
        return None
    h = _hash(secret)
    for t in _read():
        if hmac.compare_digest(t.get("hash", ""), h):
            return t
    internal = _read_internal()
    if internal and hmac.compare_digest(internal, secret):
        return {"id": "internal", "name": "NeuralDeck dashboard", "internal": True}
    return None


def touch(token_id: str, every: float = 60.0) -> None:
    """Record a use, at most once a minute per token (it rewrites the file)."""
    if token_id == "internal":
        return
    now = time.time()
    if now - _touched.get(token_id, 0) < every:
        return
    _touched[token_id] = now
    with _lock:
        tokens = list(_read())
        for t in tokens:
            if t.get("id") == token_id:
                t["last_used"] = int(now)
                break
        else:
            return
        try:
            _write(tokens)
        except OSError:
            pass


def _read_internal():
    try:
        return INTERNAL_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def internal_key() -> str:
    """The dashboard's own key, created on first use."""
    key = _read_internal()
    if key:
        return key
    INTERNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    fresh = "ndi_" + secrets.token_urlsafe(32)
    try:
        fd = os.open(str(INTERNAL_FILE), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _read_internal() or fresh   # the other process won the race
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(fresh)
    return fresh
