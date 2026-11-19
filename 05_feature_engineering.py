
from pathlib import Path
import numpy as np
import pandas as pd

# ==================================================
# AeroPulse | 05 - Feature Engineering
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "aeropulse_2026_08_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(DATA_PATH, low_memory=False)

# 1. Keep completed flights with known delays
df = df.loc[
    (df["FlightStatus"] == "Completed")
    & df["DepDelay"].notna()
].copy()

# 2. Target variable
df["Delayed15"] = (
    df["DepDelay"] >= 15
).astype(int)

# 3. Date features
df["FlightDate"] = pd.to_datetime(df["FlightDate"])

df["DayOfMonth"] = df["FlightDate"].dt.day

df["IsWeekend"] = (
    df["DayOfWeek"].isin([6, 7])
).astype(int)

# 4. Scheduled departure hour
df["ScheduledDepHour"] = pd.to_numeric(
    df["ScheduledDepHour"], errors="coerce"
)

# 5. Departure period
def departure_period(hour):
    if pd.isna(hour):
        return "Unknown"
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

df["DepPeriod"] = df[
    "ScheduledDepHour"
].apply(departure_period)

# 6. Cyclical hour features
df["HourSin"] = np.sin(
    2 * np.pi * df["ScheduledDepHour"] / 24
)

df["HourCos"] = np.cos(
    2 * np.pi * df["ScheduledDepHour"] / 24
)

# 7. Distance transformation
df["LogDistance"] = np.log1p(df["Distance"])

# 8. Select model columns
features = [
    "FlightDate",
    "DayOfMonth",
    "DayOfWeek",
    "IsWeekend",
    "ScheduledDepHour",
    "DepPeriod",
    "HourSin",
    "HourCos",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Distance",
    "LogDistance",
    "Delayed15"
]

model_df = df[features].copy()

# 9. Basic validation
print("Model dataset shape:", model_df.shape)

print("\nTarget distribution:")
print(model_df["Delayed15"].value_counts())

print("\nTarget percentage:")
print(
    model_df["Delayed15"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nDeparture periods:")
print(model_df["DepPeriod"].value_counts())

print("\nMissing values:")
print(
    model_df.isna().sum()[
        model_df.isna().sum() > 0
    ]
)

# 10. Save
output_path = (
    OUTPUT_DIR / "aeropulse_model_features.csv"
)

model_df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("AeroPulse | Feature Engineering Completed!")
