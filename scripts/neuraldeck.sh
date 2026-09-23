#!/usr/bin/env bash
# Start NeuralDeck (proxy + dashboard) from a checkout, creating the venv
# on first run. Any arguments are passed through to the CLI, e.g.
#   ./scripts/neuraldeck.sh doctor
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${NEURALDECK_VENV:-$ROOT/venv}"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip >/dev/null
    "$VENV/bin/pip" install -r "$ROOT/requirements.txt"
fi

cd "$ROOT"
exec "$PY" -m neuraldeck "$@"
