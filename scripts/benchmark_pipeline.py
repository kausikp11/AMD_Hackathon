import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.loader import load_frame
from src.locate_anything import locate_objects
from src.pipeline import run_pipeline


def frame_ids(limit):

    frames = [
        line.strip()
        for line in Path("data/industrial_subset/frames.txt").read_text().splitlines()
        if line.strip()
    ]

    if limit and limit > 0:
        return frames[:limit]

    return frames


def rocm_smi_snapshot():

    if shutil.which(
        "rocm-smi"
    ) is None:
        return {
            "available":
                False
        }

    try:
        result = subprocess.run(
            [
                "rocm-smi",
                "--showuse",
                "--showmemuse",
                "--showmeminfo",
                "vram",
                "--json"
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10
        )
    except Exception as exc:
        return {
            "available":
                False,
            "error":
                f"{type(exc).__name__}: {exc}"
        }

    if result.returncode != 0:
        return {
            "available":
                False,
            "error":
                result.stderr.strip()
        }

    try:
        return {
            "available":
                True,
            "raw":
                json.loads(
                    result.stdout
                )
        }
    except json.JSONDecodeError:
        return {
            "available":
                True,
            "raw_text":
                result.stdout
        }


def summarize_gpu(snapshot):

    if not snapshot.get(
        "available"
    ):
        return {}

    raw = snapshot.get(
        "raw",
        {}
    )

    summary = {}

    for gpu, values in raw.items():
        if not isinstance(
            values,
            dict
        ):
            continue

        summary[gpu] = {
            key:
                value
            for key, value in values.items()
            if any(
                token in key.lower()
                for token in [
                    "use",
                    "memory",
                    "vram"
                ]
            )
        }

    return summary


def timed_call(fn):

    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start

    return value, elapsed


@contextmanager
def temporary_env(updates):

    previous = {
        key:
            os.environ.get(
                key
            )
        for key in updates
    }

    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(
                    key,
                    None
                )
            else:
                os.environ[key] = str(
                    value
                )

        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(
                    key,
                    None
                )
            else:
                os.environ[key] = value


def benchmark_frame(frame_id, bypass_locator_cache):

    frame = load_frame(
        frame_id
    )

    gpu_before = rocm_smi_snapshot()

    with temporary_env({
        "LOCATE_ANYTHING_CPP_BYPASS_CACHE":
            "1" if bypass_locator_cache else None
    }):
        located, locator_cold_seconds = timed_call(
            lambda: locate_objects(
                frame
            )
        )

    with temporary_env({
        "LOCATE_ANYTHING_CPP_BYPASS_CACHE":
            None
    }):
        cached_located, locator_cached_seconds = timed_call(
            lambda: locate_objects(
                frame
            )
        )

    result, pipeline_seconds = timed_call(
        lambda: run_pipeline(
            frame_id
        )
    )

    gpu_after = rocm_smi_snapshot()

    scene = result["scene"]
    decision = result["decision"]
    plan = result["plan"]

    return {
        "frame_id":
            frame_id,
        "locator_backend":
            os.getenv(
                "LOCATOR_BACKEND",
                "labels"
            ),
        "vlm_backend":
            os.getenv(
                "VLM_BACKEND",
                "heuristic"
            ),
        "scene_source":
            scene.get(
                "scene_source",
                "unknown"
            ),
        "risk_level":
            decision["risk_level"],
        "action":
            plan["action"],
        "target_speed":
            plan["target_speed"],
        "locator_seconds":
            round(
                locator_cold_seconds,
                4
            ),
        "locator_cold_seconds":
            round(
                locator_cold_seconds,
                4
            ),
        "locator_cached_seconds":
            round(
                locator_cached_seconds,
                4
            ),
        "pipeline_non_locator_seconds":
            round(
                max(
                    0.0,
                    pipeline_seconds - locator_cached_seconds
                ),
                4
            ),
        "estimated_cold_e2e_seconds":
            round(
                locator_cold_seconds
                + max(
                    0.0,
                    pipeline_seconds - locator_cached_seconds
                ),
                4
            ),
        "pipeline_seconds":
            round(
                pipeline_seconds,
                4
            ),
        "located_count":
            len(
                located
            ),
        "cached_located_count":
            len(
                cached_located
            ),
        "pipeline_located_count":
            len(
                scene.get(
                    "located_objects",
                    []
                )
            ),
        "path_source":
            plan["navigation"].get(
                "desired_path_source"
            ),
        "floor_source":
            plan["navigation"].get(
                "floor_region_source"
            ),
        "gpu_before":
            summarize_gpu(
                gpu_before
            ),
        "gpu_after":
            summarize_gpu(
                gpu_after
            )
    }


def write_csv(path, records):

    if not records:
        return

    fieldnames = [
        "frame_id",
        "locator_backend",
        "vlm_backend",
        "scene_source",
        "risk_level",
        "action",
        "target_speed",
        "locator_seconds",
        "locator_cold_seconds",
        "locator_cached_seconds",
        "pipeline_non_locator_seconds",
        "estimated_cold_e2e_seconds",
        "pipeline_seconds",
        "located_count",
        "cached_located_count",
        "pipeline_located_count",
        "path_source",
        "floor_source"
    ]

    with path.open(
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()

        for record in records:
            writer.writerow({
                key:
                    record.get(
                        key
                    )
                for key in fieldnames
            })


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-count",
        type=int,
        default=5
    )
    parser.add_argument(
        "--output-json",
        default="outputs/benchmark_pipeline.json"
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/benchmark_pipeline.csv"
    )
    parser.add_argument(
        "--use-existing-locator-cache",
        action="store_true",
        help=(
            "Do not force the locate-anything.cpp HTTP wrapper to recompute "
            "the first locator call for each frame."
        )
    )
    args = parser.parse_args()

    records = []

    for idx, frame_id in enumerate(
        frame_ids(
            args.frame_count
        ),
        start=1
    ):
        print(
            f"[{idx}/{args.frame_count}] benchmarking {frame_id}",
            flush=True
        )
        record = benchmark_frame(
            frame_id,
            not args.use_existing_locator_cache
        )
        records.append(
            record
        )
        print(
            json.dumps(
                {
                    "frame_id":
                        record["frame_id"],
                    "locator_seconds":
                        record["locator_seconds"],
                    "locator_cached_seconds":
                        record["locator_cached_seconds"],
                    "pipeline_seconds":
                        record["pipeline_seconds"],
                    "estimated_cold_e2e_seconds":
                        record["estimated_cold_e2e_seconds"],
                    "located_count":
                        record["located_count"]
                }
            ),
            flush=True
        )

    output_json = Path(
        args.output_json
    )
    output_csv = Path(
        args.output_csv
    )
    output_json.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    output_json.write_text(
        json.dumps(
            {
                "environment":
                    {
                        key:
                            os.getenv(
                                key,
                                ""
                            )
                        for key in [
                            "VLM_BACKEND",
                            "QWEN_VLM_BASE_URL",
                            "QWEN_VLM_MODEL",
                            "LOCATOR_BACKEND",
                            "LOCATE_ANYTHING_CPP_BIN",
                            "LOCATE_ANYTHING_CPP_MODEL",
                            "LOCATE_ANYTHING_CPP_MODE"
                        ]
                    },
                "records":
                    records
            },
            indent=2
        )
    )
    write_csv(
        output_csv,
        records
    )

    if records:
        locator_avg = sum(
            record["locator_seconds"]
            for record in records
        ) / len(records)
        pipeline_avg = sum(
            record["pipeline_seconds"]
            for record in records
        ) / len(records)
        cached_locator_avg = sum(
            record["locator_cached_seconds"]
            for record in records
        ) / len(records)
        cold_e2e_avg = sum(
            record["estimated_cold_e2e_seconds"]
            for record in records
        ) / len(records)
        print(
            f"Average locator latency: {locator_avg:.3f}s"
        )
        print(
            f"Average cached locator latency: {cached_locator_avg:.3f}s"
        )
        print(
            f"Average pipeline latency: {pipeline_avg:.3f}s"
        )
        print(
            f"Average estimated cold end-to-end latency: {cold_e2e_avg:.3f}s"
        )

    print(
        f"Wrote {output_json} and {output_csv}"
    )


if __name__ == "__main__":
    main()
