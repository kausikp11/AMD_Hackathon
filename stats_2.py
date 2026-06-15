import pandas as pd

df = pd.read_csv("industrial_analysis.csv")

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

df = pd.read_csv("industrial_analysis.csv")

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