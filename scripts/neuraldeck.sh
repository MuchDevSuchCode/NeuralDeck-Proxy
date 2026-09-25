#!/usr/bin/env bash
# Start NeuralDeck (proxy + dashboard) from a checkout, creating the venv
# on first run. Any arguments are passed through to the CLI, e.g.
#   ./scripts/neuraldeck.sh doctor
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${NEURALDECK_VENV:-$ROOT/venv}"
PY="$VENV/bin/python"

# Checked on every run, not just the first: a setup interrupted half-way
# leaves a python behind with no pip or no dependencies, and that is
# repaired here rather than failing later with an ImportError.
if [ ! -x "$PY" ] || ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "Creating venv at $VENV ..."
    CLEAR=""
    [ -f "$VENV/pyvenv.cfg" ] && CLEAR=1
    python3 -m venv ${CLEAR:+--clear} "$VENV"
fi
if ! "$PY" -c "import fastapi, uvicorn, httpx, psutil, multipart" >/dev/null 2>&1; then
    echo "Installing dependencies into $VENV ..."
    "$PY" -m pip install --upgrade pip >/dev/null || true
    "$PY" -m pip install -r "$ROOT/requirements.txt"
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m neuraldeck "$@"
