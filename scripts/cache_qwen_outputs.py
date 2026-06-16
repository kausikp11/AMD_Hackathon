import argparse
import json
import os
from pathlib import Path
import sys

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
        "reasoning":
            decision["reasoning"],
    }


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

    records = []

    for idx, frame_id in enumerate(
        frame_ids(
            args.frame_count
        ),
        start=1
    ):
        print(
            f"[{idx}] caching {frame_id}",
            flush=True
        )
        result = run_pipeline(
            frame_id
        )
        records.append(
            compact(
                frame_id,
                result
            )
        )

    output_path.write_text(
        json.dumps(
            {
                "records":
                    records
            },
            indent=2
        )
    )

    print(
        f"Wrote {len(records)} records to {output_path}"
    )


if __name__ == "__main__":
    main()
