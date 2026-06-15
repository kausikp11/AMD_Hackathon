from pathlib import Path
import pandas as pd

from src.loader import load_frame
from src.world_state import build_world_state
from src.reasoner import reason


FRAME_FILE = "data/industrial_subset/frames.txt"

rows = []

with open(FRAME_FILE) as f:
    frame_ids = [line.strip() for line in f if line.strip()]

print(f"Processing {len(frame_ids)} frames...")

for i, frame_id in enumerate(frame_ids):

    try:

        frame = load_frame(frame_id)

        world = build_world_state(frame)

        decision = reason(world)

        rows.append({

            "frame_id":
                frame_id,

            "risk_level":
                decision["risk_level"],

            "action":
                decision["action"],

            "confidence":
                decision["confidence"],

            "human_present":
                world["human"]["present"],

            "human_distance":
                world["human"]["distance"],

            "motion_level":
                world["scene"]["motion_level"],

            "motion_ratio":
                world["scene"]["motion_ratio"],

            "tags":
                ",".join(world["scene_tags"])
        })

    except Exception as e:

        print(
            f"Failed frame {frame_id}: {e}"
        )

df = pd.DataFrame(rows)

df.to_csv(
    "decision_dataset.csv",
    index=False
)

print("\nSaved:")
print("decision_dataset.csv")

print("\nRisk distribution:")
print(
    df["risk_level"].value_counts()
)

print("\nAction distribution:")
print(
    df["action"].value_counts()
)

print("\nHuman present:")
print(
    df["human_present"].value_counts()
)

print("\nFirst 5 rows:")
print(df.head())