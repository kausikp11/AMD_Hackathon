import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT_DIR / "outputs/industrial_analysis.csv")

print(
    pd.crosstab(
        df["human_detected"],
        df["radar_activity"]
    )
)

print(
    df.groupby("human_detected")[
        "motion_ratio"
    ].describe()
)

df = pd.read_csv(ROOT_DIR / "outputs/industrial_analysis.csv")

print(
    df.groupby(
        "human_detected"
    )[
        [
            "motion_ratio",
            "max_velocity",
            "moving_points"
        ]
    ].mean()
)
