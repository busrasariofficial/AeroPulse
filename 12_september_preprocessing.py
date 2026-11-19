
from pathlib import Path
import numpy as np
import pandas as pd

# ==================================================
# AeroPulse | 12 - September Preprocessing
# External Validation Dataset
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / (
    "On_Time_Reporting_Carrier_On_Time_Performance_"
    "(1987_present)_2026_9.csv"
)

OUTPUT_PATH = (
    DATA_DIR / "aeropulse_2026_09_model_features.csv"
)

AUGUST_PATH = (
    DATA_DIR / "aeropulse_model_features.csv"
)

# 1. Load September data
df = pd.read_csv(INPUT_PATH, low_memory=False)

print("=" * 55)
print("AeroPulse | September Preprocessing")
print("=" * 55)

print("\nOriginal shape:", df.shape)

# 2. Validate essential columns
required = [
    "FlightDate", "DayOfWeek",
    "CRSDepTime", "Reporting_Airline",
    "Origin", "Dest", "Distance",
    "DepDelay", "Cancelled", "Diverted"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 3. Keep completed flights only
df = df.loc[
    (df["Cancelled"] == 0)
    & (df["Diverted"] == 0)
    & df["DepDelay"].notna()
].copy()

print("\nCompleted flights:", len(df))

# 4. Target: departure delay >= 15 minutes
df["Delayed15"] = (
    df["DepDelay"] >= 15
).astype(int)

# 5. Date features
df["FlightDate"] = pd.to_datetime(
    df["FlightDate"],
    errors="raise"
)

df["DayOfMonth"] = df["FlightDate"].dt.day

df["IsWeekend"] = (
    df["DayOfWeek"].isin([6, 7])
).astype(int)

# 6. Scheduled departure hour
crs_time = pd.to_numeric(
    df["CRSDepTime"],
    errors="coerce"
)

# BTS can encode midnight as 2400.
df["ScheduledDepHour"] = (
    (crs_time // 100) % 24
)

# 7. Departure period
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

# 8. Cyclical time features
df["HourSin"] = np.sin(
    2 * np.pi * df["ScheduledDepHour"] / 24
)

df["HourCos"] = np.cos(
    2 * np.pi * df["ScheduledDepHour"] / 24
)

# 9. Distance transformation
df["LogDistance"] = np.log1p(
    df["Distance"]
)

# 10. Match August model schema
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

# 11. Validate against August schema
august_columns = pd.read_csv(
    AUGUST_PATH,
    nrows=0
).columns.tolist()

if model_df.columns.tolist() != august_columns:
    raise ValueError(
        "September feature schema does not match August!"
    )

if model_df.isna().any().any():
    print("\nMissing values:")
    print(model_df.isna().sum())
    raise ValueError("Missing values detected!")

if not model_df["FlightDate"].between(
    "2026-09-01", "2026-09-30"
).all():
    raise ValueError("Unexpected flight dates!")

# 12. Summary
print("\nModel dataset shape:", model_df.shape)

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
print(model_df.isna().sum().sum())

# 13. Save dataset
model_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:", OUTPUT_PATH)
print("\nAeroPulse | September Preprocessing Completed!")
