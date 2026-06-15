from src.loader import (
    nearest_human,
    has_human
)
from src.radar import (
    summarize_radar,
    motion_ratio,
    radar_activity
)

def fuse_frame(
    frame,
    data_root="data/industrial_subset"
):

    frame_id = frame["frame_id"]

    human = nearest_human(frame)

    radar = summarize_radar(
        frame_id,
        data_root
    )

    return {

        "frame_id":
            frame_id,

        "human_detected":
            human is not None,

        "human_distance":
            None if human is None
            else human["distance"],

        "thermal_confirmed":
    has_human(
        frame["thermal_detections"]
    ),

        "radar_activity":
            radar_activity(radar),

        "motion_ratio":
            motion_ratio(radar),

        "radar_summary":
            radar
    }
