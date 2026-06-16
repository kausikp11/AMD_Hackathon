# analyze_decisions.py

import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT_DIR / "outputs/decision_dataset.csv")

print("\nRisk Levels")
print(df["risk_level"].value_counts())

print("\nActions")
print(df["action"].value_counts())

print("\nRisk vs Human Presence")
print(
    pd.crosstab(
        df["human_present"],
        df["risk_level"]
    )
)

print("\nDistance by Risk")
print(
    df.groupby("risk_level")
      ["human_distance"]
      .describe()
)
