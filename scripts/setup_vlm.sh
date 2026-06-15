#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

python -m pip install -U pip

# Keep torch out of this install path. On AMD/ROCm systems the correct torch
# wheel is usually preinstalled or inherited through --system-site-packages.
python -m pip install \
    transformers==4.57.1 \
    accelerate \
    sentencepiece \
    einops \
    pillow \
    opencv-python-headless==4.11.0.86 \
    gradio \
    pandas \
    openai \
    peft \
    decord==0.6.0 \
    lmdb==1.7.5 \
    qwen-vl-utils

python -m pip install /opt/rocm-7.0.0/share/amd_smi 2>/dev/null || \
    python -m pip install amdsmi

echo
echo "Qwen server:"
echo "  ./scripts/start_qwen_vllm.sh"
echo
echo "Demo app:"
echo "  ./start.sh"
