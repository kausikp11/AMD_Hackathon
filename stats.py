import pandas as pd

df = pd.read_csv(
    "industrial_analysis.csv"
)

print(df.head())

print()
print("Frames:", len(df))

print()
print("Human detected:")
print(
    df["human_detected"]
    .value_counts()
)

print()
print("Distance stats:")
print(
    df["human_distance"]
    .describe()
)

print()
print("Motion ratio:")
print(
    df["motion_ratio"]
    .describe()
)

print()
print("Max velocity:")
print(
    df["max_velocity"]
    .describe()
)