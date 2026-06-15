# Industrial Robot Copilot

Multimodal industrial robot copilot prototype using RGB labels/images, thermal labels/images, radar point clouds, and metadata to produce world state, risk assessment, robot planning, scene graph context, and an explainable Gradio dashboard.

## Quick Start

```bash
./install.sh
./start.sh
```

Open the URL printed by `start.sh`, usually `http://localhost:7860`.

The top stream plays all frames as a browser-side canvas stream. The lower controls
inspect one selected frame through the full copilot pipeline, including boxed
RGB/thermal views, robot command, world state, risk decision, scene, plan, and
explanation. Limit the stream/selector with `DEMO_FRAME_COUNT=10` if needed.
Use `Play Model-Synced` in the lower controls when you want each inspected
frame to render only after Qwen/control output is complete.

To show Qwen output in sync with the stream, precompute a lightweight cache:

```bash
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=labels \
python scripts/cache_qwen_outputs.py --frame-count 50
```

Then start the app. The stream will show cached Qwen scene/risk/action/human
status for frames present in `cache/qwen_stream.json`.

## Smoke Tests

Run the default five-frame check:

```bash
./scripts/run_smoke_tests.sh
```

Run a larger check:

```bash
FRAME_COUNT=20 ./scripts/run_smoke_tests.sh
```

## Runtime Modes

The AMD-safe final-demo default is:

```bash
VLM_BACKEND=heuristic
LOCATOR_BACKEND=labels
./start.sh
```

This uses the deterministic local pipeline and dataset RGB/thermal labels. It is the safest mode when only AMD resources are available and LocateAnything has not been validated on ROCm.

## Qwen VLM On AMD

Serve Qwen on the AMD resource using the helper:

```bash
./scripts/start_qwen_vllm.sh
```

Then run the app in another terminal:

```bash
./start.sh
```

`start.sh` checks `http://localhost:8000/v1/models`; if Qwen is available it uses `VLM_BACKEND=qwen`, otherwise it falls back to the heuristic scene backend. Localization remains label-based unless changed.

Manual Qwen app mode:

```bash
VLM_BACKEND=qwen QWEN_VLM_BASE_URL=http://localhost:8000/v1 LOCATOR_BACKEND=labels ./start.sh
```

## LocateAnything

The code supports `nvidia/LocateAnything-3B`, but the final AMD-only demo should keep:

```bash
LOCATOR_BACKEND=labels
```

until `nvidia/LocateAnything-3B` is validated on the AMD ROCm stack. The model is available on Hugging Face and is officially documented with NVIDIA-oriented hardware/runtime guidance.

If a compatible endpoint is available:

```bash
LOCATOR_BACKEND=nvidia_vllm \
NVIDIA_LOCATE_ANYTHING_BASE_URL=http://localhost:8001/v1 \
NVIDIA_LOCATE_ANYTHING_MODEL=nvidia/LocateAnything-3B \
./start.sh
```

If local Transformers inference is validated:

```bash
LOCATOR_BACKEND=nvidia_transformers \
NVIDIA_LOCATE_ANYTHING_ATTN_IMPLEMENTATION=sdpa \
./start.sh
```

## GroundingDINO Localization

GroundingDINO can be used as a LocateAnything-style open-vocabulary locator:

```bash
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=grounding_dino \
GROUNDING_DINO_MODEL=IDEA-Research/grounding-dino-tiny \
./start.sh
```

Bounding boxes from GroundingDINO appear on inspected RGB frames with `DINO`
labels. Qwen boxes appear with `Qwen` labels when Qwen returns approximate
`located_objects`; dataset RGB/thermal label boxes are always available in
`LOCATOR_BACKEND=labels`.

GroundingDINO is prompted for the tracked industrial objects: human,
industrial machine, workbench, pipe, cabinet, storage box, control panel,
forklift, and robot. The robot command panel labels its source separately:
scene output comes from Qwen/heuristics, while command/steering comes from the
rule-based control algorithm.

## Useful Files

- `app.py`: Gradio dashboard.
- `src/pipeline.py`: end-to-end orchestration.
- `src/qwen_vlm.py`: Qwen OpenAI-compatible scene backend.
- `src/locate_anything.py`: LocateAnything adapter plus label fallback.
- `scripts/check_env.py`: environment and dataset checks.
- `scripts/run_smoke_tests.sh`: compile, environment, and multi-frame pipeline smoke test.
- `.env.example`: runtime configuration template.
