import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

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


def compact(frame_id, result):

    world = result["world"]
    decision = result["decision"]
    plan = result["plan"]
    scene = result["scene"]

    human = world["human"]
    distance = human["distance"]

    return {
        "frame_id":
            frame_id,
        "scene_source":
            scene.get(
                "scene_source",
                "unknown"
            ),
        "risk_level":
            decision["risk_level"],
        "action":
            plan["action"],
        "mode":
            plan["mode"],
        "target_speed":
            plan["target_speed"],
        "human_present":
            human["present"],
        "human_distance":
            distance,
        "environment_type":
            scene.get(
                "environment_type",
                "unknown"
            ),
        "hazards":
            scene.get(
                "hazards",
                []
            ),
        "located_count":
            len(
                scene.get(
                    "located_objects",
                    []
                )
            ),
        "located_objects":
            scene.get(
                "located_objects",
                []
            ),
        "navigation":
            plan.get(
                "navigation",
                {}
            ),
        "reasoning":
            decision["reasoning"],
    }


def load_existing_records(output_path):

    if not output_path.exists():
        return []

    try:
        payload = json.loads(
            output_path.read_text()
        )
    except json.JSONDecodeError:
        return []

    return payload.get(
        "records",
        []
    )


def write_records(output_path, records):

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    temp_path.write_text(
        json.dumps(
            {
                "records":
                    records
            },
            indent=2
        )
    )
    temp_path.replace(
        output_path
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-count",
        type=int,
        default=int(
            os.getenv(
                "QWEN_CACHE_FRAME_COUNT",
                "50"
            )
        )
    )
    parser.add_argument(
        "--output",
        default=os.getenv(
            "QWEN_STREAM_CACHE",
            "cache/qwen_stream.json"
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute frames even if they already exist in the cache."
    )
    args = parser.parse_args()

    os.environ.setdefault(
        "VLM_BACKEND",
        "qwen"
    )
    os.environ.setdefault(
        "LOCATOR_BACKEND",
        "labels"
    )

    output_path = Path(
        args.output
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    records = (
        []
        if args.force
        else load_existing_records(
            output_path
        )
    )
    existing = {
        record.get(
            "frame_id"
        )
        for record in records
    }

    frames = frame_ids(
        args.frame_count
    )

    for idx, frame_id in enumerate(
        frames,
        start=1
    ):

        if frame_id in existing:
            print(
                f"[{idx}/{len(frames)}] skip {frame_id} already cached",
                flush=True
            )
            continue

        print(
            f"[{idx}/{len(frames)}] caching {frame_id}",
            flush=True
        )
        start = time.perf_counter()
        result = run_pipeline(
            frame_id
        )
        records.append(
            compact(
                frame_id,
                result
            )
        )
        existing.add(
            frame_id
        )
        write_records(
            output_path,
            records
        )
        elapsed = time.perf_counter() - start
        print(
            f"[{idx}/{len(frames)}] wrote {frame_id} in {elapsed:.2f}s",
            flush=True
        )

    print(
        f"Wrote {len(records)} records to {output_path}"
    )


if __name__ == "__main__":
    main()
