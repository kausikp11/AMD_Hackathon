#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
FRAME_ID="${FRAME_ID:-013342}"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Checking imports and source compilation..."
python -m py_compile app.py src/*.py

echo "Running pipeline smoke test for frame ${FRAME_ID}..."
python - <<PY
from src.pipeline import run_pipeline

result = run_pipeline("${FRAME_ID}")

required_keys = {
    "frame",
    "world",
    "decision",
    "scene",
    "graph",
    "plan",
    "explanation",
}

missing = required_keys - set(result)
if missing:
    raise SystemExit(f"Missing pipeline keys: {sorted(missing)}")

print("Frame:", result["frame"]["frame_id"])
print("Risk:", result["decision"]["risk_level"])
print("Action:", result["plan"]["action"])
print("Mode:", result["plan"]["mode"])
PY

echo "Install and smoke test complete."
