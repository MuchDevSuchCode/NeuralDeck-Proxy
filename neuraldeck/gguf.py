"""Minimal GGUF header reader.

Two questions about a model can only be answered by its header, and both
matter before launch:

  * does it carry an MTP/NextN draft head?  Combined builds hide the head
    inside the main weights file, so the filename cannot tell you.
  * does its chat template do reasoning?  That decides whether the thinking
    controls mean anything for this model.

Only the key/value block at the front of the file is read — never the
tensors — so this stays fast even on a 40 GiB shard.
"""

import os
import struct

# GGUF value type -> fixed byte width. Strings (8) and arrays (9) are
# variable and handled separately.
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

_cache: dict = {}  # path -> (mtime, {"mtp": bool, "thinking": bool})


def _read_str(f) -> bytes:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def _scan(path: str) -> dict:
    """Walk the KV block once, answering both questions in a single pass."""
    out = {"mtp": False, "thinking": False, "arch": None}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        f.read(4)                                   # version
        _, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if vtype == 8:
                val = _read_str(f)
                if key == b"tokenizer.chat_template":
                    out["thinking"] = b"think" in val.lower()
                elif key == b"general.architecture":
                    out["arch"] = val.decode(errors="replace")
            elif vtype in _SIZES:
                val = f.read(_SIZES[vtype])
                if key.endswith(b"nextn_predict_layers") \
                        or key.endswith(b"mtp_predict_layers"):
                    out["mtp"] = int.from_bytes(val, "little") > 0
            elif vtype == 9:                        # array
                atype, n = struct.unpack("<IQ", f.read(12))
                if atype == 8:
                    for _ in range(n):
                        _read_str(f)
                elif atype in _SIZES:
                    f.seek(_SIZES[atype] * n, 1)
                else:
                    break                           # unknown element type
            else:
                break                               # unknown value type
    return out


def info(path: str) -> dict:
    """Cached header facts for a GGUF. Never raises: unreadable means unknown."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {"mtp": False, "thinking": False, "arch": None}
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        out = _scan(path)
    except Exception:
        out = {"mtp": False, "thinking": False, "arch": None}
    _cache[path] = (mtime, out)
    return out


def has_mtp(path: str) -> bool:
    return info(path)["mtp"]


def supports_thinking(path: str) -> bool:
    return info(path)["thinking"]


def forget(path: str) -> None:
    _cache.pop(path, None)
