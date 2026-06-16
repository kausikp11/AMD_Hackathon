# Industrial Robot Copilot - Committee Runbook

## Project Summary

Industrial Robot Copilot is a multimodal safety and navigation assistant for
industrial robots. It combines RGB, thermal, radar, VLM scene understanding,
open-vocabulary object localization, risk reasoning, and robot path planning in
a Gradio dashboard.

The demo shows a robot-perspective view with:

- detected humans and industrial objects
- risk level and action recommendation
- desired ground-plane path
- speed-coded path segments
- stop point behavior when the robot must halt
- Qwen or heuristic VLM scene reasoning
- locate-anything.cpp bounding boxes

## Main Demo Order

Run in this order on the AMD cloud machine:

1. Prepare the Qwen/vLLM environment.
2. Start Qwen VLM server.
3. Start locate-anything.cpp HTTP wrapper.
4. Run benchmark.
5. Build or resume the cached stream output.
6. Start the Gradio demo server.

## Terminal 1 - Qwen Environment And Server

From the project root:

```bash
cd /workspace/AMD_Hackathon
./scripts/fix_amd_vllm_env.sh
./scripts/start_qwen_vllm.sh
```

Expected Qwen endpoint:

```text
http://localhost:8000/v1
```

Health check:

```bash
curl http://localhost:8000/v1/models
```

If Qwen is still unstable, the app can run with:

```bash
VLM_BACKEND=heuristic
```

## Terminal 2 - Start locate-anything.cpp Wrapper

Start the wrapper around the locate-anything.cpp CLI:

```bash
cd /workspace/AMD_Hackathon
python scripts/locate_anything_cpp_server.py \
  --bin /workspace/locate-anything.cpp/build/examples/cli/locate-anything-cli \
  --model /workspace/locate-anything.cpp/models/locate-anything-q8_0.gguf \
  --mode fast \
  --host 127.0.0.1 \
  --port 8188
```

Health check:

```bash
curl http://127.0.0.1:8188/health
```

The wrapper caches repeated detections by model, mode, image path, image mtime,
and prompt. This makes repeated app and cache runs faster.

## Terminal 3 - Benchmark First

Run the benchmark before caching:

```bash
cd /workspace/AMD_Hackathon
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
python scripts/benchmark_pipeline.py --frame-count 10
```

Outputs:

```text
outputs/benchmark_pipeline.json
outputs/benchmark_pipeline.csv
```

The benchmark reports:

- `locator_cold_seconds`: locate-anything.cpp recomputed for the frame
- `locator_cached_seconds`: repeated wrapper call using cache when available
- `pipeline_seconds`: full app pipeline using normal cache behavior
- `pipeline_non_locator_seconds`: estimated non-locator pipeline time
- `estimated_cold_e2e_seconds`: cold locator plus non-locator pipeline estimate

Latest measured sample from AMD run:

```text
Average locator latency: 4.837s
Average pipeline latency: 2.823s
```

Interpretation:

- The locator is the current bottleneck on cold frames.
- Pipeline time can be lower than locator time when the wrapper cache is warm.
- For jury numbers, report both cold locator and warm/cached demo latency.

To measure a fully warm run:

```bash
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
python scripts/benchmark_pipeline.py --frame-count 10 --use-existing-locator-cache
```

## Monitor AMD GPU Use

In another terminal:

```bash
watch -n 1 rocm-smi
```

For one-time detailed info:

```bash
rocm-smi
```

For locate-anything.cpp build validation:

```bash
grep -R "GGML_HIP" /workspace/locate-anything.cpp/build/CMakeCache.txt
grep -R "LA_GGML" /workspace/locate-anything.cpp/build/CMakeCache.txt
ldd /workspace/locate-anything.cpp/build/examples/cli/locate-anything-cli | grep -iE "hip|roc|amd"
```

Expected HIP-related build flags include:

```text
GGML_HIP:BOOL=ON
GGML_HIP_GRAPHS:BOOL=ON
GGML_HIP_MMQ_MFMA:BOOL=ON
```

If `rocm-smi` shows high VRAM but `GPU%` stays near zero during detection, the
binary may still be mostly CPU-bound or only loading memory without meaningful
GPU compute.

## Build Or Resume Cached Stream

Build the Qwen plus locator cache:

```bash
cd /workspace/AMD_Hackathon
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
python scripts/cache_qwen_outputs.py --frame-count 621
```

The cache script saves after every frame and resumes automatically. If it stops,
run the same command again and it will skip completed frames.

To recompute everything:

```bash
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
python scripts/cache_qwen_outputs.py --frame-count 621 --force
```

Cache file:

```text
cache/qwen_stream.json
```

Cached records include:

- Qwen scene output
- risk/action/mode/target speed
- navigation desired path and floor region
- locate-anything.cpp bounding boxes
- path and floor source

## Start Gradio Demo

Start the full demo with public sharing:

```bash
cd /workspace/AMD_Hackathon
DEMO_FRAME_COUNT=621 \
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
GRADIO_SHARE=1 \
./start.sh
```

Expected output:

```text
Running on local URL: http://0.0.0.0:7860
Running on public URL: https://...gradio.live
```

If Qwen is unavailable, run the stable fallback:

```bash
DEMO_FRAME_COUNT=621 \
VLM_BACKEND=heuristic \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
GRADIO_SHARE=1 \
./start.sh
```

If locate-anything.cpp is unavailable, run the safest fallback:

```bash
DEMO_FRAME_COUNT=621 \
VLM_BACKEND=heuristic \
LOCATOR_BACKEND=labels \
GRADIO_SHARE=1 \
./start.sh
```

## Complete Command Set

Use this section as the single copy-paste reference during the demo.

### 1. Qwen VLM Server

```bash
cd /workspace/AMD_Hackathon
./scripts/fix_amd_vllm_env.sh
./scripts/start_qwen_vllm.sh
```

```bash
curl http://localhost:8000/v1/models
```

### 2. locate-anything.cpp HTTP Wrapper

```bash
cd /workspace/AMD_Hackathon
python scripts/locate_anything_cpp_server.py \
  --bin /workspace/locate-anything.cpp/build/examples/cli/locate-anything-cli \
  --model /workspace/locate-anything.cpp/models/locate-anything-q8_0.gguf \
  --mode fast \
  --host 127.0.0.1 \
  --port 8188
```

```bash
curl http://127.0.0.1:8188/health
```

### 3. Benchmark

```bash
cd /workspace/AMD_Hackathon
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
python scripts/benchmark_pipeline.py --frame-count 10
```

### 4. Warm Benchmark

```bash
cd /workspace/AMD_Hackathon
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
python scripts/benchmark_pipeline.py --frame-count 10 --use-existing-locator-cache
```

### 5. GPU Monitoring

```bash
watch -n 1 rocm-smi
```

```bash
grep -R "GGML_HIP" /workspace/locate-anything.cpp/build/CMakeCache.txt
grep -R "LA_GGML" /workspace/locate-anything.cpp/build/CMakeCache.txt
ldd /workspace/locate-anything.cpp/build/examples/cli/locate-anything-cli | grep -iE "hip|roc|amd"
```

### 6. Build Or Resume Cache

```bash
cd /workspace/AMD_Hackathon
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
python scripts/cache_qwen_outputs.py --frame-count 621
```

### 7. Rebuild Cache From Scratch

```bash
cd /workspace/AMD_Hackathon
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
python scripts/cache_qwen_outputs.py --frame-count 621 --force
```

### 8. Start Full Demo

```bash
cd /workspace/AMD_Hackathon
DEMO_FRAME_COUNT=621 \
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
GRADIO_SHARE=1 \
./start.sh
```

### 9. Start Stable Fallback Demo

```bash
cd /workspace/AMD_Hackathon
DEMO_FRAME_COUNT=621 \
VLM_BACKEND=heuristic \
LOCATOR_BACKEND=labels \
GRADIO_SHARE=1 \
./start.sh
```

### 10. Export This Runbook To PDF

```bash
cd /workspace/AMD_Hackathon
pandoc docs/COMMITTEE_RUNBOOK.md -o docs/COMMITTEE_RUNBOOK.pdf --pdf-engine=xelatex
```

## Jury Focus Points

- Multimodal safety reasoning: RGB, thermal, radar, VLM, and locator outputs are
  fused into one robot command.
- Robot path is grounded to the floor region instead of floating over the image.
- Path color encodes target speed.
- STOP actions render as a point from the robot perspective, not a misleading
  end line.
- Qwen can provide scene and path estimates; local heuristics keep the demo
  robust if the VLM endpoint fails.
- locate-anything.cpp boxes are shown in the stream and in the detailed frame
  pane.
- Caching makes the long stream practical for a live demo while retaining model
  output per frame.

## Operational Notes

### Qwen Dependency Alignment

Run:

```bash
./scripts/fix_amd_vllm_env.sh
```

This aligns compatible versions for `huggingface-hub`, `fastapi`, `starlette`,
and `prometheus-fastapi-instrumentator`.

### Qwen Server Restart

Run:

```bash
./scripts/fix_amd_vllm_env.sh
./scripts/start_qwen_vllm.sh
```

### Missing Cached Frame Output

Run or resume:

```bash
VLM_BACKEND=qwen \
QWEN_VLM_BASE_URL=http://localhost:8000/v1 \
LOCATOR_BACKEND=locate_anything_cpp_http \
LOCATE_ANYTHING_CPP_URL=http://127.0.0.1:8188/detect \
python scripts/cache_qwen_outputs.py --frame-count 621
```

### Full Stream Frame Count

Start with:

```bash
DEMO_FRAME_COUNT=621 ./start.sh
```

### locate-anything.cpp Runtime Check

Check:

```bash
watch -n 1 rocm-smi
grep -R "GGML_HIP" /workspace/locate-anything.cpp/build/CMakeCache.txt
```

Use wrapper cache for demo runs:

```bash
LOCATOR_BACKEND=locate_anything_cpp_http
```

For fastest uncached throughput, the next engineering step is replacing the CLI
wrapper with a native long-running C/C++ server that keeps the GGUF model loaded.
