import os

from src.loader import (
    nearest_human,
    has_human
)
from src.locate_anything import locate_objects
from src.radar import (
    summarize_radar,
    motion_ratio,
    radar_activity
)


def live_human_detection(frame):

    if os.getenv(
        "LOCATOR_BACKEND",
        "labels"
    ).lower() != "yolo_live":
        return None

    detections = locate_objects(
        frame,
        [
            "human"
        ]
    )

    humans = [
        det
        for det in detections
        if det.get(
            "label"
        ) == "human"
    ]

    if not humans:
        return None

    return max(
        humans,
        key=lambda det: det.get(
            "confidence",
            0
        )
    )

def fuse_frame(
    frame,
    data_root="data/industrial_subset"
):

    frame_id = frame["frame_id"]

    human = nearest_human(frame)
    live_human = (
        None
        if human is not None
        else live_human_detection(
            frame
        )
    )

    radar = summarize_radar(
        frame_id,
        data_root
    )

    return {

        "frame_id":
            frame_id,

        "human_detected":
            human is not None
            or live_human is not None,

        "human_distance":
            None if human is None
            else human["distance"],

        "human_source":
            live_human.get(
                "source",
                "yolo_live"
            )
            if live_human is not None
            else "yolo_rgb"
            if human is not None
            else None,

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
