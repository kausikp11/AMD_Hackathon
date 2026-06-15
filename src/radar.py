from pathlib import Path

import numpy as np
import pandas as pd

def load_radar_csv(
    frame_id,
    data_root="data/industrial_subset"
):

    csv_path = (
        Path(data_root)
        / "04_radar"
        / "pointcloud"
        / "csv"
        / f"{frame_id}.csv"
    )

    if not csv_path.exists():
        return None

    return pd.read_csv(
        csv_path,
        header=None
    )

def summarize_radar(
    frame_id,
    data_root="data/industrial_subset"
):

    df = load_radar_csv(
        frame_id,
        data_root
    )

    if df is None:
        return None

    velocities = np.abs(df[4])

    return {

        "point_count":
            int(len(df)),

        "moving_points":
            int((velocities > 0).sum()),

        "max_velocity":
            float(velocities.max()),

        "mean_velocity":
            float(velocities.mean())
    }

def nearest_radar_target(
    frame_id,
    data_root="data/industrial_subset"
):

    df = load_radar_csv(
        frame_id,
        data_root
    )

    if df is None:
        return None

    xyz = df[[0,1,2]].values

    distances = np.sqrt(
        np.sum(
            xyz**2,
            axis=1
        )
    )

    idx = np.argmin(distances)

    return {
        "distance":
            float(distances[idx]),

        "velocity":
            float(df.iloc[idx,4])
    }

def radar_activity(summary):

    if summary is None:
        return False

    return (
        summary["moving_points"] > 0
    )

def motion_ratio(summary):

    if summary is None:
        return 0.0

    if summary["point_count"] == 0:
        return 0.0

    return (
        summary["moving_points"]
        / summary["point_count"]
    )

