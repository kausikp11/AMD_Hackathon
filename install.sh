#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FRAME_ID="${FRAME_ID:-013342}"

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

system_torch_has_gpu() {
    "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
try:
    import torch
    raise SystemExit(0 if torch.cuda.is_available() else 1)
except Exception:
    raise SystemExit(1)
PY
}

VENV_DIR="$(detect_venv_dir)"

if [ -z "${VIRTUAL_ENV:-}" ] && [ ! -d "$VENV_DIR" ]; then
    if system_torch_has_gpu; then
        echo "Creating ${VENV_DIR} with system site packages to preserve ROCm PyTorch."
        "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
    else
        echo "Creating ${VENV_DIR}."
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Checking imports and source compilation..."
python -m py_compile app.py src/*.py scripts/check_env.py

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
