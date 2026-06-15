#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

FRAME_COUNT="${FRAME_COUNT:-5}"

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

python -m py_compile app.py src/*.py scripts/check_env.py

python scripts/check_env.py

python - <<PY
from pathlib import Path

from src.pipeline import run_pipeline

frames = [
    line.strip()
    for line in Path("data/industrial_subset/frames.txt").read_text().splitlines()
    if line.strip()
][:${FRAME_COUNT}]

print()
print(f"Running pipeline smoke test for {len(frames)} frame(s)")
print("frame_id,risk,action,mode,human_present,distance,scene_source,located_count")

for frame_id in frames:
    result = run_pipeline(frame_id)
    world = result["world"]
    scene = result["scene"]
    plan = result["plan"]
    decision = result["decision"]

    distance = world["human"]["distance"]
    distance_text = "" if distance is None else f"{distance:.3f}"

    print(
        ",".join(
            [
                frame_id,
                decision["risk_level"],
                plan["action"],
                plan["mode"],
                str(world["human"]["present"]),
                distance_text,
                scene.get("scene_source", ""),
                str(len(scene.get("located_objects", []))),
            ]
        )
    )
PY
