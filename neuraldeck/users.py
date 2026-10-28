"""Dashboard accounts and sign-in sessions.

Users live in ``users.json`` in the data directory: a name, a role
(``admin`` or ``user``), and a scrypt hash of the password. Sessions live
beside it in ``sessions.json`` so a sign-in survives the dashboard
restarting itself from the Settings page. Both files are mode 0600.

On first start, with no users at all, an ``admin`` account is created
with a random password, which is printed to the dashboard's log and
written to ``admin-initial-password.txt`` in the data directory. That
account must change its password at first sign-in, and the file is
deleted when it does.

Access is per view. An admin can do everything; a user has a list of the
views they may use (Dashboard, Chat, Prompt Lab, Benchmarks, Settings),
each view carrying the API calls behind it. Accounts and proxy tokens are
always admin-only.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time

from . import config

FILE = config.DATA_DIR / "users.json"
SESSIONS = config.DATA_DIR / "sessions.json"
INITIAL = config.DATA_DIR / "admin-initial-password.txt"
ROLES = ("admin", "user")
VIEWS = ("dash", "chat", "prompt", "bench", "settings")
DEFAULT_ACCESS = ["dash", "chat", "prompt", "bench"]
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")
MIN_PASSWORD = 8
SESSION_TTL = 30 * 86400        # a month, sliding
_lock = threading.Lock()
_fails: dict = {}               # name -> [timestamps of recent failures]
LOCK_FAILS, LOCK_WINDOW = 10, 600


class UserError(ValueError):
    """A refusal to report to the caller, with an HTTP status."""

    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


# ── files ──────────────────────────────────────────────────────────────────

def _read(path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(data, dict):
        data = data.get("items", [])
    return [d for d in data if isinstance(d, dict)]


def _write(path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.stem + "-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"items": items}, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── passwords ──────────────────────────────────────────────────────────────

def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1,
                          dklen=32).hex()


def _check_password(password: str) -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise UserError(400, f"the password must be at least {MIN_PASSWORD} characters")
    if len(password) > 256:
        raise UserError(400, "the password is too long")


def _check_name(name) -> str:
    name = (name or "").strip().lower()
    if not NAME_RE.match(name):
        raise UserError(400, "a user name is 1–32 characters: letters, digits, '.', '_' or '-'")
    return name


def _access_of(u: dict) -> list:
    if u.get("role") == "admin":
        return list(VIEWS)
    acc = u.get("access")
    if not isinstance(acc, list):
        return list(DEFAULT_ACCESS)
    return [v for v in VIEWS if v in acc]


def _check_access(access) -> list:
    if not isinstance(access, list) or any(v not in VIEWS for v in access):
        raise UserError(400, f"access is a list drawn from {', '.join(VIEWS)}")
    return [v for v in VIEWS if v in access]


def _public(u: dict) -> dict:
    return {"name": u["name"], "role": u.get("role", "user"), "access": _access_of(u),
            "created": u.get("created"), "last_login": u.get("last_login"),
            "must_change": bool(u.get("must_change"))}


# ── users ──────────────────────────────────────────────────────────────────

def list_users() -> list:
    with _lock:
        return [_public(u) for u in _read(FILE)]


def count() -> int:
    return len(_read(FILE))


def get(name: str):
    for u in _read(FILE):
        if u.get("name") == name:
            return u
    return None


def ensure_admin():
    """Create the default admin when there are no accounts at all. Returns
    the generated password the one time it is made, else None."""
    with _lock:
        if _read(FILE):
            return None
        password = secrets.token_urlsafe(12)
        salt = os.urandom(16)
        _write(FILE, [{"name": "admin", "role": "admin", "hash": _hash(password, salt),
                       "salt": salt.hex(), "created": int(time.time()),
                       "last_login": None, "must_change": True}])
        try:
            fd = os.open(str(INITIAL), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(password + "\n")
        except OSError:
            pass
        return password


def initial_password_file():
    """Where the first admin password waits, while it has not been changed."""
    return str(INITIAL) if INITIAL.exists() else None


def create(name, password, role="user", must_change=False, access=None) -> dict:
    name = _check_name(name)
    _check_password(password)
    if role not in ROLES:
        raise UserError(400, f"role must be one of {', '.join(ROLES)}")
    access = _check_access(access) if access is not None else list(DEFAULT_ACCESS)
    with _lock:
        users = _read(FILE)
        if any(u.get("name") == name for u in users):
            raise UserError(409, f"user '{name}' already exists")
        salt = os.urandom(16)
        row = {"name": name, "role": role, "access": access, "hash": _hash(password, salt),
               "salt": salt.hex(), "created": int(time.time()), "last_login": None,
               "must_change": bool(must_change)}
        users.append(row)
        _write(FILE, users)
    return _public(row)


def delete(name: str, by: str) -> None:
    with _lock:
        users = _read(FILE)
        target = next((u for u in users if u.get("name") == name), None)
        if target is None:
            raise UserError(404, f"no user '{name}'")
        if name == by:
            raise UserError(400, "you cannot delete the account you are signed in as")
        if target.get("role") == "admin" and \
                sum(1 for u in users if u.get("role") == "admin") == 1:
            raise UserError(400, "that is the last admin; make another admin first")
        _write(FILE, [u for u in users if u.get("name") != name])
    drop_user_sessions(name)


def set_access(name: str, by: str, role=None, access=None) -> dict:
    """Change a user's role and/or the views they may use."""
    if role is not None and role not in ROLES:
        raise UserError(400, f"role must be one of {', '.join(ROLES)}")
    if access is not None:
        access = _check_access(access)
    with _lock:
        users = _read(FILE)
        target = next((u for u in users if u.get("name") == name), None)
        if target is None:
            raise UserError(404, f"no user '{name}'")
        if role is not None:
            if name == by and role != "admin":
                raise UserError(400, "you cannot remove your own admin role")
            target["role"] = role
        if access is not None:
            target["access"] = access
        _write(FILE, users)
        return _public(target)


def set_role(name: str, role: str, by: str) -> dict:
    return set_access(name, by, role=role)


def set_password(name: str, password: str, must_change: bool, current=None) -> dict:
    """Set a password. With `current` (the user changing their own) it must
    verify first; without (an admin reset) it need not."""
    _check_password(password)
    with _lock:
        users = _read(FILE)
        target = next((u for u in users if u.get("name") == name), None)
        if target is None:
            raise UserError(404, f"no user '{name}'")
        if current is not None and not _verify(target, current):
            raise UserError(403, "the current password is wrong")
        salt = os.urandom(16)
        target.update({"hash": _hash(password, salt), "salt": salt.hex(),
                       "must_change": bool(must_change)})
        _write(FILE, users)
        if name == "admin" and not must_change:
            try:
                INITIAL.unlink()
            except OSError:
                pass
        return _public(target)


def _verify(u: dict, password: str) -> bool:
    try:
        salt = bytes.fromhex(u.get("salt", ""))
    except ValueError:
        return False
    return hmac.compare_digest(_hash(password or "", salt), u.get("hash", ""))


def authenticate(name, password):
    """The user, or None. Ten failures in ten minutes lock the name for a
    while (a 429 the caller reports)."""
    name = (name or "").strip().lower()
    now = time.time()
    recent = [t for t in _fails.get(name, []) if now - t < LOCK_WINDOW]
    if len(recent) >= LOCK_FAILS:
        raise UserError(429, "too many failed sign-ins; try again in a few minutes")
    with _lock:
        users = _read(FILE)
        u = next((x for x in users if x.get("name") == name), None)
        if u is None or not _verify(u, password):
            _fails[name] = recent + [now]
            return None
        _fails.pop(name, None)
        u["last_login"] = int(now)
        try:
            _write(FILE, users)
        except OSError:
            pass
        return _public(u)


# ── sessions ───────────────────────────────────────────────────────────────

def _sessions() -> list:
    now = time.time()
    return [s for s in _read(SESSIONS) if s.get("expires", 0) > now]


def new_session(name: str) -> str:
    sid = secrets.token_urlsafe(32)
    with _lock:
        items = _sessions()
        items.append({"sid": sid, "name": name, "created": int(time.time()),
                      "expires": int(time.time() + SESSION_TTL)})
        _write(SESSIONS, items)
    return sid


def session_user(sid):
    """The user a session cookie belongs to, or None. A session that has
    aged a day is extended, so staying signed in keeps you signed in."""
    if not sid:
        return None
    now = time.time()
    with _lock:
        items = _sessions()
        s = next((x for x in items if hmac.compare_digest(x.get("sid", ""), sid)), None)
        if s is None:
            return None
        if s["expires"] - now < SESSION_TTL - 86400:
            s["expires"] = int(now + SESSION_TTL)
            try:
                _write(SESSIONS, items)
            except OSError:
                pass
    u = get(s["name"])
    return _public(u) if u else None


def drop_session(sid) -> None:
    if not sid:
        return
    with _lock:
        items = [s for s in _sessions() if s.get("sid") != sid]
        _write(SESSIONS, items)


def drop_user_sessions(name: str) -> None:
    with _lock:
        _write(SESSIONS, [s for s in _sessions() if s.get("name") != name])
