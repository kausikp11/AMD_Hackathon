from collections import Counter

from src.loader import load_frame
from src.world_state import build_world_state
from src.reasoner import reason


FRAME_FILE = "data/industrial_subset/frames.txt"


risk_counter = Counter()

with open(FRAME_FILE) as f:

    frame_ids = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Frames: {len(frame_ids)}")

for frame_id in frame_ids:

    try:

        frame = load_frame(frame_id)

        world = build_world_state(frame)

        decision = reason(world)

        risk_counter[
            decision["risk_level"]
        ] += 1

    except Exception as e:

        print(
            f"Failed {frame_id}: {e}"
        )

print("\nRisk Summary")
print("-" * 30)

for k, v in risk_counter.items():

    print(
        f"{k:<10} {v}"
    )