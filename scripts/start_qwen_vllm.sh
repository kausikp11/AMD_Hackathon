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

if [ "${QWEN_VLLM_SKIP_ENV_CHECK:-0}" != "1" ]; then
    python3 scripts/check_vllm_env.py
fi

unset CUDA_VISIBLE_DEVICES
unset VLLM_USE_V1

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"

MODEL="${QWEN_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${QWEN_VLLM_PORT:-8000}"

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --max-model-len "${QWEN_VLLM_MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "${QWEN_VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-num-seqs "${QWEN_VLLM_MAX_NUM_SEQS:-8}"
