"""Model discovery.

A model is a directory under one of the configured model roots containing a
.gguf, plus the helper files that belong to it: a vision projector, and an
MTP/NextN draft head. Loose .gguf files sitting in a root count too.

A folder holding config.json and *.safetensors instead is a Hugging Face
format model ("format": "hf"), served by a vLLM backend; its capabilities
are read from config.json and the chat template rather than a GGUF header.

Which file is the *model* takes some care — a folder can hold the weights,
an mmproj, a separate draft head and several shards, and picking the wrong
one loads a projector as a language model.
"""

import json
import os
import re
from pathlib import Path

from . import config, gguf

SKIP_DIRS = {"venv", ".cache", "lost+found", "RyzenAdj", ".git", "__pycache__"}
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}", re.IGNORECASE)
# Files that live beside a model without being one.
HELPER_RE = re.compile(r"mmproj", re.IGNORECASE)
# "mtp" as its own word, leading or not: unsloth ships Gemma 4's head as
# mtp-gemma-4-….gguf, which a "-mtp" pattern treated as a second model
DRAFT_RE = re.compile(r"(assistant|(?:^|[-_.])mtp(?:[-_.]|$)|nextn|draft)",
                      re.IGNORECASE)


def _is_head(p: Path) -> bool:
    """A draft/MTP head by name, or by a header that says it's an assistant
    model (gemma4-assistant) whatever the file happens to be called."""
    return bool(DRAFT_RE.search(p.name)) or \
        "assistant" in (gguf.info(str(p)).get("arch") or "")


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def shards(main: str) -> list:
    """Every file of a split model (just `main` if it is not split)."""
    m = SHARD_RE.search(os.path.basename(main))
    if not m:
        return [main]
    base = os.path.basename(main)
    stem, tail = base[:m.start()], base[m.end():]
    folder = os.path.dirname(main)
    try:
        names = os.listdir(folder)
    except OSError:
        return [main]
    # same stem, same suffix, only the shard number differs — a sibling
    # quant whose name merely starts the same is not part of this model
    found = sorted(os.path.join(folder, f) for f in names
                   if f[:m.start()] == stem and f[m.end():] == tail
                   and SHARD_RE.fullmatch(f[m.start():m.end()]))
    return found or [main]


def _weights_bytes(main: str) -> int:
    """Total bytes loaded for a model: every shard, not just the first."""
    return sum(_size(f) for f in shards(main))


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
    weights file — whether or not its name says so, the one file is both.
    """
    files = _ggufs(folder)
    if not files:
        return None, None, None
    mmproj = next((p for p in files if HELPER_RE.search(p.name)), None)
    plain = [p for p in files if not HELPER_RE.search(p.name)
             and not _is_head(p)]
    main = plain[0] if plain else None
    draft = None
    if main is None:
        combined = [p for p in files if not HELPER_RE.search(p.name)
                    and _is_head(p)]
        if not combined:
            return None, None, None
        main = combined[0]
        draft = main                      # one file: weights + head
    else:
        draft = next((p for p in files if p is not main
                      and not HELPER_RE.search(p.name)
                      and _is_head(p)), None)
        if draft is None and gguf.has_mtp(str(main)):
            draft = main                  # head declared in the weights' header
    # Only advertise a head the file actually declares. Keying on the name
    # trusts a label: plenty of repos ship "-MTP" in a filename with no
    # nextn layers in the header, and vice versa.
    if draft is not None and not gguf.has_mtp(str(draft)):
        draft = None
    return main, mmproj, draft


def _entry(main: Path, mmproj, draft, root: Path) -> dict:
    name = (main.parent.name if main.parent != root
            else SHARD_RE.sub("", main.stem))
    vram = _weights_bytes(str(main))
    if mmproj is not None:
        vram += _size(str(mmproj))
    if draft is not None and draft != main:
        vram += _size(str(draft))
    info = gguf.info(str(main))
    return {
        "name": name,
        "format": "gguf",
        "path": str(main),
        "mmproj_path": str(mmproj) if mmproj else None,
        "draft_path": str(draft) if draft else None,
        "source": _short(str(root)),
        "root": str(root),
        # everything that belongs to this model and nothing else — what a
        # delete removes
        "files": shards(str(main))
                 + ([str(mmproj)] if mmproj else [])
                 + ([str(draft)] if draft is not None and draft != main else []),
        "vram_bytes": vram,
        "vision": mmproj is not None,
        "mtp": draft is not None,
        "embed": "embed" in name.lower(),
        "thinking": info["thinking"],
        "arch": info["arch"],
    }


# ── Hugging Face (safetensors) folders ─────────────────────────────────────

# Keys a config uses to declare multi-token-prediction layers; which one
# depends on the family (DeepSeek/GLM, Qwen3.5+, others).
MTP_KEYS = ("num_nextn_predict_layers", "mtp_num_hidden_layers",
            "num_mtp_layers", "mtp_num_layers", "n_mtp_layers")

_hf_cache: dict = {}   # folder -> (stamp, info)


def _read_json(path, head: int = None):
    """A JSON file as a dict, {} when absent or unreadable. With `head`,
    only that many bytes are read and parsed leniently — exl3's
    quantization_config.json carries a per-tensor table of ~600 KB after
    the few keys that matter."""
    try:
        with open(path, "rb") as f:
            raw = f.read(head) if head else f.read()
    except OSError:
        return {}
    return parse_json(raw, lenient=bool(head))


def parse_json(raw: bytes, lenient: bool = False) -> dict:
    """A JSON object from bytes, {} if it is not one. `lenient` accepts a
    truncated head and picks the top-level quantisation keys out of it."""
    text = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except ValueError:
        if not lenient:
            return {}
    # the per-tensor table nests the same key names one level down, so
    # only matches at the top level count
    out = {}
    for m in re.finditer(r'"(quant_method|bits|bits_per_weight|w_bit|'
                         r'load_in_4bit|load_in_8bit|fmt)"\s*:\s*'
                         r'("([^"]*)"|[\d.]+|true|false)', text):
        before = text[:m.start()]
        if before.count("{") - before.count("}") != 1:
            continue
        key, val = m.group(1), m.group(3) if m.group(3) is not None else m.group(2)
        out.setdefault(key, val)
    return out


def hf_quant(cfg: dict, qcfg: dict = None):
    """A short quantisation label from config.json's quantization_config
    (or quantization_config.json beside it), or None for plain weights."""
    q = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    q = q if isinstance(q, dict) and q else (qcfg or {})
    if not isinstance(q, dict) or not q:
        return None
    method = str(q.get("quant_method") or q.get("method") or "").lower()
    if not method:
        return None

    def num(v):
        try:
            f = float(v)
            return f"{f:g}"
        except (TypeError, ValueError):
            return None
    if method in ("exl2", "exl3"):
        bpw = num(q.get("bits_per_weight")) or num(q.get("bits"))
        return f"{method} {bpw}bpw" if bpw else method
    if method == "gptq":
        bits = num(q.get("bits"))
        return f"gptq {bits}bit" if bits else "gptq"
    if method == "bitsandbytes":
        if str(q.get("load_in_4bit")).lower() == "true":
            return "bnb 4bit"
        if str(q.get("load_in_8bit")).lower() == "true":
            return "bnb 8bit"
        return "bnb"
    return method


def _sub(cfg: dict) -> dict:
    t = cfg.get("text_config") if isinstance(cfg, dict) else None
    return t if isinstance(t, dict) else {}


def hf_arch(cfg: dict):
    archs = cfg.get("architectures") if isinstance(cfg, dict) else None
    return (cfg.get("model_type")
            or (archs[0] if isinstance(archs, list) and archs else None)
            or _sub(cfg).get("model_type"))


def hf_mtp(cfg: dict) -> bool:
    """True when config.json declares MTP layers, at the top or in its
    text_config (multimodal configs nest the language model there)."""
    for d in (cfg, _sub(cfg)):
        for k in MTP_KEYS:
            try:
                if int(d.get(k) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def hf_template(folder: Path) -> str:
    """The chat template text: chat_template.jinja, chat_template.json, or
    tokenizer_config.json's chat_template, whichever is present."""
    try:
        return (folder / "chat_template.jinja").read_text(encoding="utf-8",
                                                          errors="replace")
    except OSError:
        pass
    for name in ("chat_template.json", "tokenizer_config.json"):
        tpl = _read_json(folder / name).get("chat_template")
        if isinstance(tpl, list):          # named templates: [{name, template}]
            tpl = " ".join(str(t.get("template", "")) for t in tpl
                           if isinstance(t, dict))
        if isinstance(tpl, str) and tpl:
            return tpl
    return ""


def template_thinking(tpl: str) -> bool:
    return "enable_thinking" in tpl or "<think>" in tpl


def _safetensors(folder: Path) -> list:
    try:
        return sorted(p for p in folder.iterdir()
                      if p.is_file() and p.suffix.lower() == ".safetensors")
    except OSError:
        return []


def is_hf_folder(folder: Path) -> bool:
    return (folder / "config.json").is_file() and bool(_safetensors(folder))


def _all_files(folder: Path) -> list:
    """Every file under a model folder — what deleting it removes. Links
    are listed, never followed."""
    out = []
    for dirpath, dirnames, names in os.walk(folder):
        dirnames[:] = [d for d in dirnames
                       if not os.path.islink(os.path.join(dirpath, d))]
        out.extend(os.path.join(dirpath, n) for n in names)
    return sorted(out)


def _hf_info(folder: Path) -> dict:
    """Capabilities of an HF folder, cached until one of the files they
    come from changes."""
    watched = ("config.json", "quantization_config.json", "chat_template.jinja",
               "chat_template.json", "tokenizer_config.json",
               "preprocessor_config.json")
    stamp = []
    for name in watched:
        try:
            stamp.append(os.path.getmtime(folder / name))
        except OSError:
            stamp.append(None)
    stamp = tuple(stamp)
    hit = _hf_cache.get(str(folder))
    if hit and hit[0] == stamp:
        return hit[1]
    cfg = _read_json(folder / "config.json")
    qcfg = None
    if not isinstance(cfg.get("quantization_config"), dict):
        qcfg = _read_json(folder / "quantization_config.json", head=16384)
    tpl = hf_template(folder)
    info = {
        "arch": hf_arch(cfg),
        "architectures": [str(a) for a in (cfg.get("architectures") or [])
                          if isinstance(a, str)],
        "quant": hf_quant(cfg, qcfg),
        "vision": bool(cfg.get("vision_config"))
                  or (folder / "preprocessor_config.json").is_file(),
        "mtp": hf_mtp(cfg),
        "thinking": template_thinking(tpl),
        # Qwen3.5+ templates emit tool calls as <function=name>…</function>
        # rather than Hermes JSON, which needs vLLM's qwen3_xml parser
        "tool_xml": "<function=" in tpl,
    }
    _hf_cache[str(folder)] = (stamp, info)
    return info


def _hf_entry(folder: Path, root: Path) -> dict:
    info = _hf_info(folder)
    return {
        "name": folder.name,
        "format": "hf",
        "path": str(folder),
        "mmproj_path": None,
        "draft_path": None,
        "source": _short(str(root)),
        "root": str(root),
        "files": _all_files(folder),
        "vram_bytes": sum(_size(str(p)) for p in _safetensors(folder)),
        "vision": info["vision"],
        "mtp": info["mtp"],
        "embed": "embed" in folder.name.lower(),
        "thinking": info["thinking"],
        "arch": info["arch"],
        "architectures": info["architectures"],
        "quant": info["quant"],
        "tool_xml": info["tool_xml"],
    }


def _short(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home) else path


def _unique(name: str, root: Path, seen: set) -> str:
    """A second model of the same name (another root, or a loose file beside
    a folder) keeps its own entry under a name that says where it lives, so
    launch and delete never silently hit the first one."""
    cand = f"{name}@{root.name or 'root'}"
    n = 2
    while cand in seen:
        cand = f"{name}@{root.name or 'root'}-{n}"
        n += 1
    return cand


def discover() -> list:
    """Every model under every configured root, newest roots last."""
    out, seen = [], set()

    def add(e, root):
        if e["name"] in seen:
            e["name"] = _unique(e["name"], root, seen)
        seen.add(e["name"])
        out.append(e)

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
                if is_hf_folder(folder):
                    add(_hf_entry(folder, root), root)
                continue
            add(_entry(main, mmproj, draft, root), root)
        for loose in _ggufs(root):
            if HELPER_RE.search(loose.name):
                continue
            # a split model loose in the root is listed by its first shard
            m = SHARD_RE.search(loose.name)
            if m and not m.group(0).startswith("-00001-"):
                continue
            add(_entry(loose, None,
                       loose if gguf.has_mtp(str(loose)) else None, root), root)
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
