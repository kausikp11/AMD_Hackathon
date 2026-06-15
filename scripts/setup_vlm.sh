#!/usr/bin/env bash
set -euo pipefail

pip install -U pip

pip install \
torch \
torchvision \
transformers==4.57.1 \
accelerate \
sentencepiece \
einops \
pillow \
opencv-python-headless==4.11.0.86 \
numpy==1.25.0 \
gradio \
pandas \
openai \
peft \
decord==0.6.0 \
lmdb==1.7.5 \
qwen-vl-utils

pip install vllm

cat <<'MSG'

Optional model servers:

Qwen scene understanding:
  vllm serve "Qwen/Qwen2.5-VL-7B-Instruct" --host 0.0.0.0 --port 8000
  VLM_BACKEND=qwen QWEN_VLM_BASE_URL=http://localhost:8000/v1 ./start.sh

NVIDIA LocateAnything localization:
  vllm serve "nvidia/LocateAnything-3B" --host 0.0.0.0 --port 8001
  LOCATOR_BACKEND=nvidia_vllm NVIDIA_LOCATE_ANYTHING_BASE_URL=http://localhost:8001/v1 ./start.sh

Local Transformers LocateAnything:
  LOCATOR_BACKEND=nvidia_transformers ./start.sh

MSG
