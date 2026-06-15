import pandas as pd

from src.loader import load_frame, nearest_human
from src.fusion import fuse_frame

FRAMES_FILE = "data/industrial_subset/frames.txt"

rows = []

with open(FRAMES_FILE, "r") as f:
    frame_ids = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Processing {len(frame_ids)} frames...")

for i, frame_id in enumerate(frame_ids):

    try:

        frame = load_frame(frame_id)

        state = fuse_frame(frame)

        radar = state["radar_summary"]

        rows.append({
            "frame_id": frame_id,

            "season":
                frame["environment"]["season"],

            "weather":
                frame["environment"]["weather"],

            "lighting":
                frame["environment"]["lighting"],

            "human_detected":
                state["human_detected"],

            "human_distance":
                state["human_distance"],

            "thermal_confirmed":
                state["thermal_confirmed"],

            "radar_activity":
                state["radar_activity"],

            "motion_ratio":
                state["motion_ratio"],

            "point_count":
                radar["point_count"],

            "moving_points":
                radar["moving_points"],

            "max_velocity":
                radar["max_velocity"],

            "mean_velocity":
                radar["mean_velocity"]
        })

    except Exception as e:

        print(
            f"Error on {frame_id}: {e}"
        )

    if i % 50 == 0:
        print(
            f"{i}/{len(frame_ids)}"
        )

df = pd.DataFrame(rows)

output_file = "industrial_analysis.csv"

df.to_csv(
    output_file,
    index=False
)

print()
print("Saved:", output_file)
print("Rows:", len(df))