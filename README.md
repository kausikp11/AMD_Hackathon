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
explanation. The selected RGB frame also overlays the estimated desired robot
path from Qwen when available, or a local aisle-based fallback path otherwise.
Limit the stream/selector with `DEMO_FRAME_COUNT=10` if needed.
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

## Live YOLO Localization On AMD

If GroundingDINO or NVIDIA LocateAnything are not reliable on the AMD VM, use
the live YOLO backend:

```bash
python -m pip install -r requirements-yolo.txt
VLM_BACKEND=heuristic \
LOCATOR_BACKEND=yolo_live \
YOLO_LIVE_MODEL=yolo11n.pt \
./start.sh
```

`yolo_live` runs Ultralytics YOLO on the RGB image and returns the same
`located_objects` shape as the other locators. Stock YOLO maps `person` to
`human`, so it is useful for live human detection and steering boxes. It is not
open-vocabulary; industrial classes such as `control_panel`, `workbench`, or
`industrial_machine` require a custom-trained YOLO model or a separate
open-vocabulary detector.

When `LOCATOR_BACKEND=yolo_live`, fusion can also use a live YOLO human box to
set `world["human"]["present"]` if no dataset RGB human label is available.
Distance remains dataset-derived when available; live YOLO detections currently
report `distance: null`.

## Qwen VLM On AMD

Serve Qwen on the AMD resource using the helper:

```bash
./scripts/start_qwen_vllm.sh
```

If the AMD image has the known vLLM dependency mismatch, the helper prints the
repair command. You can run the bundled repair explicitly:

```bash
./scripts/fix_amd_vllm_env.sh
```

This pins the versions that avoid the observed `huggingface-hub>=0.34,<1.0`
and `prometheus_fastapi_instrumentator` / FastAPI route crashes.

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

## locate-anything.cpp

The project also supports the `mudler/locate-anything.cpp` CLI backend:

Install/build example:

```bash
cd /workspace
git clone --recursive https://github.com/mudler/locate-anything.cpp
cd locate-anything.cpp
cmake -B build -DLA_BUILD_CLI=ON
cmake --build build -j
hf download mudler/locate-anything.cpp-gguf locate-anything-q8_0.gguf --local-dir models
```

Direct CLI smoke test:

```bash
/workspace/locate-anything.cpp/build/examples/cli/locate-anything-cli detect \
  --model /workspace/locate-anything.cpp/models/locate-anything-q8_0.gguf \
  --input /workspace/AMD_Hackathon/data/industrial_subset/05_rgb/calibrated/013342.jpg \
  --prompt "Locate all the instances that matches the following description: human</c>industrial machine</c>workbench</c>storage box." \
  --mode hybrid \
  --output /tmp/la_boxes.json
```

App mode:

```bash
LOCATOR_BACKEND=locate_anything_cpp \
LOCATE_ANYTHING_CPP_BIN=/path/to/locate-anything-cli \
LOCATE_ANYTHING_CPP_MODEL=/path/to/locate-anything-q8_0.gguf \
LOCATE_ANYTHING_CPP_MODE=hybrid \
./start.sh
```

The CLI prompt is built from `config/tracked_objects.json`; categories are
separated with `</c>` as required by locate-anything.cpp. Use
`LOCATE_ANYTHING_CPP_STRICT=1` to fail instead of falling back to dataset labels.

For repeated local runs, start the lightweight wrapper service. It caches
detections by model, mode, image path, image mtime, and prompt:

```bash
python scripts/locate_anything_cpp_server.py \
  --bin /workspace/locate-anything.cpp/build/examples/cli/locate-anything-cli \
  --model /workspace/locate-anything.cpp/models/locate-anything-q8_0.gguf \
  --mode fast \
  --host 127.0.0.1 \
  --port 8188
```

Then use:

```bash
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
./start.sh
```

This wrapper avoids recomputing frames it has already seen. For maximum
throughput on never-before-seen frames, use a native C-API server that keeps the
GGUF model loaded in memory; the current wrapper is an integration-safe bridge
around the CLI.

To cache Qwen plus locate-anything.cpp overlays for the top stream:

```bash
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp \
LOCATE_ANYTHING_CPP_BIN=/workspace/locate-anything.cpp/build/examples/cli/locate-anything-cli \
LOCATE_ANYTHING_CPP_MODEL=/workspace/locate-anything.cpp/models/locate-anything-q8_0.gguf \
LOCATE_ANYTHING_CPP_MODE=hybrid \
python scripts/cache_qwen_outputs.py --frame-count 621
```

The cache script is resumable: it writes `cache/qwen_stream.json` after every
frame and skips frames already present in the cache. Use `--force` to recompute
everything.

Then start with the same locator settings:

```bash
DEMO_FRAME_COUNT=621 \
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp \
LOCATE_ANYTHING_CPP_BIN=/workspace/locate-anything.cpp/build/examples/cli/locate-anything-cli \
LOCATE_ANYTHING_CPP_MODEL=/workspace/locate-anything.cpp/models/locate-anything-q8_0.gguf \
LOCATE_ANYTHING_CPP_MODE=hybrid \
GRADIO_SHARE=1 \
./start.sh
```

Benchmark latency and GPU snapshots:

```bash
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp \
LOCATE_ANYTHING_CPP_BIN=/workspace/locate-anything.cpp/build/examples/cli/locate-anything-cli \
LOCATE_ANYTHING_CPP_MODEL=/workspace/locate-anything.cpp/models/locate-anything-q8_0.gguf \
LOCATE_ANYTHING_CPP_MODE=hybrid \
python scripts/benchmark_pipeline.py --frame-count 10
```

Outputs are written to `outputs/benchmark_pipeline.json` and
`outputs/benchmark_pipeline.csv`.

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

## Detection Labels And Path Sources

Tracked detection/VLM labels are configured in `config/tracked_objects.json`.
Add or remove labels in `track_objects` to change the object prompts used by
Qwen and open-vocabulary locators. Dataset-label fallback still only returns
objects that exist in the dataset annotations.

The scene and plan JSON include:

- `desired_path_source`: `qwen_vlm` or `heuristic_floor_projection`.
- `floor_region_source`: `qwen_vlm` or `heuristic_floor_projection`.

The RGB overlay tints the estimated floor region and clips the speed-colored
path ribbon to that floor polygon.

## Useful Files

- `app.py`: Gradio dashboard.
- `src/pipeline.py`: end-to-end orchestration.
- `src/qwen_vlm.py`: Qwen OpenAI-compatible scene backend.
- `src/locate_anything.py`: LocateAnything adapter plus label fallback.
- `tests/`: pipeline, module, and backend checks.
- `analysis/`: dataset analysis and CSV generation utilities.
- `outputs/`: generated analysis CSVs.
- `docs/PROJECT_SPEC.md`: project specification and architecture notes.
- `scripts/check_env.py`: environment and dataset checks.
- `scripts/run_smoke_tests.sh`: compile, environment, and multi-frame pipeline smoke test.
- `.env.example`: runtime configuration template.
