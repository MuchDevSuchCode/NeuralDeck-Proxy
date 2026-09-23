#!/usr/bin/env bash
# NeuralDeck installer — Linux and macOS.
#
#   ./install.sh                 install into ./venv, then start it
#   ./install.sh --no-launch     install only
#   ./install.sh --dir ~/apps/nd install somewhere else
#   ./install.sh --no-link       skip the ~/.local/bin/neuraldeck launcher
#
# Everything lands in a virtual environment; nothing is installed system-wide.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
LAUNCH=1
LINK=1
MIN_PY="3.10"

while [ $# -gt 0 ]; do
    case "$1" in
        --no-launch) LAUNCH=0 ;;
        --no-link)   LINK=0 ;;
        --dir)       VENV="${2:?--dir needs a path}"; shift ;;
        -h|--help)   sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

say()  { printf '\033[96m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[93m  ! \033[0m%s\n' "$*"; }
die()  { printf '\033[91m  x \033[0m%s\n' "$*" >&2; exit 1; }

# ── python ────────────────────────────────────────────────────────────────
PY=""
for cand in python3 python3.13 python3.12 python3.11 python3.10 python; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)" 2>/dev/null; then
        PY="$cand"; break
    fi
done
[ -n "$PY" ] || die "no Python >= $MIN_PY found. Install one and run this again."
say "using $("$PY" --version) at $(command -v "$PY")"

# ── venv ──────────────────────────────────────────────────────────────────
if [ ! -x "$VENV/bin/python" ]; then
    say "creating the virtual environment at $VENV"
    "$PY" -m venv "$VENV" || die "could not create a venv. On Debian/Ubuntu: sudo apt install python3-venv"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip >/dev/null 2>&1 || warn "could not upgrade pip; continuing"

# ── the package ───────────────────────────────────────────────────────────
say "installing NeuralDeck and its dependencies"
if ! "$VPY" -m pip install -e "$ROOT[hub]"; then
    warn "editable install failed — falling back to requirements.txt"
    "$VPY" -m pip install -r "$ROOT/requirements.txt" \
        || die "dependency install failed. Are you online?"
fi

# ── optional launcher on PATH ─────────────────────────────────────────────
if [ "$LINK" = 1 ]; then
    BIN="$HOME/.local/bin"
    mkdir -p "$BIN"
    cat > "$BIN/neuraldeck" <<LAUNCHER
#!/usr/bin/env bash
exec "$VPY" -m neuraldeck "\$@"
LAUNCHER
    chmod +x "$BIN/neuraldeck"
    say "installed launcher: $BIN/neuraldeck"
    case ":$PATH:" in
        *":$BIN:"*) ;;
        *) warn "$BIN is not on your PATH — add it, or run $VPY -m neuraldeck" ;;
    esac
fi

# ── what this machine looks like ──────────────────────────────────────────
echo
"$VPY" -m neuraldeck doctor || warn "doctor reported something missing (see above)"
echo
say "installed."
say "settings live in the dashboard's Settings tab — point it at your models there."

if [ "$LAUNCH" = 1 ]; then
    PORT="$("$VPY" -c 'from neuraldeck import config; print(config.DECK_PORT)' 2>/dev/null || echo 8770)"
    say "starting NeuralDeck — the dashboard will open at http://localhost:$PORT"
    say "press Ctrl-C to stop it"
    echo
    exec "$VPY" -m neuraldeck up --open
else
    say "start it with:  neuraldeck        (or: $VPY -m neuraldeck)"
fi
