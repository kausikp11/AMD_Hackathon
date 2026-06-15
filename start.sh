#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run ./install.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

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

echo "Starting Industrial Robot Copilot at http://localhost:${GRADIO_SERVER_PORT}"
exec python app.py
