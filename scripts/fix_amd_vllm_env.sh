#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --force-reinstall \
    "huggingface-hub==0.36.0" \
    "fastapi==0.115.14" \
    "starlette==0.46.2" \
    "prometheus-fastapi-instrumentator==7.1.0" \
    --break-system-packages

python3 scripts/check_vllm_env.py
