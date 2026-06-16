# test_radar.py

from src.radar import (
    summarize_radar,
    motion_ratio
)

frame_id = "013342"

summary = summarize_radar("013342")

print(motion_ratio(summary))