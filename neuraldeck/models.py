"""Model discovery.

A model is a directory under one of the configured model roots containing a
.gguf, plus the helper files that belong to it: a vision projector, and an
MTP/NextN draft head. Loose .gguf files sitting in a root count too.

Which file is the *model* takes some care — a folder can hold the weights,
an mmproj, a separate draft head and several shards, and picking the wrong
one loads a projector as a language model.
"""

import os
import re
from pathlib import Path

from . import config, gguf

SKIP_DIRS = {"venv", ".cache", "lost+found", "RyzenAdj", ".git", "__pycache__"}
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}", re.IGNORECASE)
# Files that live beside a model without being one.
HELPER_RE = re.compile(r"mmproj", re.IGNORECASE)
DRAFT_RE = re.compile(r"(assistant|[-_]mtp|nextn|draft)", re.IGNORECASE)


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _weights_bytes(main: str) -> int:
    """Total bytes loaded for a model: every shard, not just the first."""
    m = SHARD_RE.search(os.path.basename(main))
    if not m:
        return _size(main)
    stem = os.path.basename(main)[:m.start()]
    folder = os.path.dirname(main)
    return sum(_size(os.path.join(folder, f)) for f in os.listdir(folder)
               if f.startswith(stem) and f.lower().endswith(".gguf")
               and SHARD_RE.search(f))


def _ggufs(folder: Path) -> list:
    try:
        return sorted(p for p in folder.iterdir()
                      if p.is_file() and p.suffix.lower() == ".gguf")
    except OSError:
        return []


def _pick(folder: Path):
    """(main_weights, mmproj, draft_head) for one model folder.

    Pass 1 takes the first .gguf that is neither a projector nor a draft
    head. Pass 2 covers combined builds, where the MTP head is inside the
    weights file and its name says so — there, the one file is both.
    """
    files = _ggufs(folder)
    if not files:
        return None, None, None
    mmproj = next((p for p in files if HELPER_RE.search(p.name)), None)
    plain = [p for p in files if not HELPER_RE.search(p.name)
             and not DRAFT_RE.search(p.name)]
    main = plain[0] if plain else None
    draft = None
    if main is None:
        combined = [p for p in files if not HELPER_RE.search(p.name)
                    and DRAFT_RE.search(p.name)]
        if not combined:
            return None, None, None
        main = combined[0]
        draft = main                      # one file: weights + head
    else:
        draft = next((p for p in files if p is not main
                      and not HELPER_RE.search(p.name)
                      and DRAFT_RE.search(p.name)), None)
    # Only advertise a head the file actually declares. Keying on the name
    # trusts a label: plenty of repos ship "-MTP" in a filename with no
    # nextn layers in the header, and vice versa.
    if draft is not None and not gguf.has_mtp(str(draft)):
        draft = None
    return main, mmproj, draft


def _entry(main: Path, mmproj, draft, root: Path) -> dict:
    name = main.parent.name if main.parent != root else main.stem
    vram = _weights_bytes(str(main))
    if mmproj is not None:
        vram += _size(str(mmproj))
    if draft is not None and draft != main:
        vram += _size(str(draft))
    info = gguf.info(str(main))
    return {
        "name": name,
        "path": str(main),
        "mmproj_path": str(mmproj) if mmproj else None,
        "draft_path": str(draft) if draft else None,
        "source": _short(str(root)),
        "vram_bytes": vram,
        "vision": mmproj is not None,
        "mtp": draft is not None,
        "embed": "embed" in name.lower(),
        "thinking": info["thinking"],
        "arch": info["arch"],
    }


def _short(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home) else path


def discover() -> list:
    """Every model under every configured root, newest roots last."""
    out, seen = [], set()
    for root_str in config.MODEL_DIRS:
        root = Path(root_str)
        if not root.is_dir():
            continue
        try:
            children = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            children = []
        for folder in children:
            if folder.name in SKIP_DIRS or folder.name.startswith("."):
                continue
            main, mmproj, draft = _pick(folder)
            if main is None:
                continue
            e = _entry(main, mmproj, draft, root)
            if e["name"] in seen:
                continue
            seen.add(e["name"])
            out.append(e)
        for loose in _ggufs(root):
            if HELPER_RE.search(loose.name) or SHARD_RE.search(loose.name):
                continue
            e = _entry(loose, None, None, root)
            if e["name"] in seen:
                continue
            seen.add(e["name"])
            out.append(e)
    return out


def by_name(name: str):
    return next((m for m in discover() if m["name"] == name), None)


def fuzzy_eq(a: str, b: str) -> bool:
    """Match a llama alias against a discovery name.

    Instance logs report the .gguf filename; discovery reports the folder
    name. Normalise both and accept a prefix match either way.
    """
    norm = lambda s: re.sub(r"[._]", "-", re.sub(
        r"(\.gguf|-gguf)$", "", (s or "").split("/")[-1].lower()))
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))
