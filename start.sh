#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

detect_venv_dir() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        printf '%s\n' "$VIRTUAL_ENV"
    elif [ -n "${VENV_DIR:-}" ]; then
        printf '%s\n' "$VENV_DIR"
    elif [ -d "venv" ]; then
        printf '%s\n' "venv"
    else
        printf '%s\n' ".venv"
    fi
}

VENV_DIR="$(detect_venv_dir)"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -d "$VENV_DIR" ]; then
        echo "Virtual environment not found. Run ./install.sh first."
        exit 1
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

if [ -z "${VLM_BACKEND:-}" ]; then
    if python - <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
urlopen("http://localhost:8000/v1/models", timeout=1)
PY
    then
        export VLM_BACKEND=qwen
        export QWEN_VLM_BASE_URL="${QWEN_VLM_BASE_URL:-http://localhost:8000/v1}"
    else
        export VLM_BACKEND=heuristic
    fi
fi

export LOCATOR_BACKEND="${LOCATOR_BACKEND:-labels}"

GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
GRADIO_SERVER_PORT="$(
    python - <<'PY'
import os
import socket

start_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
host = "127.0.0.1"

for port in range(start_port, start_port + 100):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit(f"No free port found from {start_port} to {start_port + 99}")
PY
)"

export GRADIO_SERVER_NAME
export GRADIO_SERVER_PORT

echo "VLM_BACKEND=${VLM_BACKEND}"
echo "LOCATOR_BACKEND=${LOCATOR_BACKEND}"
echo "Starting Industrial Robot Copilot at http://localhost:${GRADIO_SERVER_PORT}"
exec python app.py
