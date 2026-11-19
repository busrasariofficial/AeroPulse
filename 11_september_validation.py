
from pathlib import Path
import pandas as pd

# ==================================================
# AeroPulse | 11 - September 2026 Data Validation
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# 1. Locate September BTS dataset
files = list(
    DATA_DIR.glob(
        "*2026_9.csv"
    )
)

if not files:
    raise FileNotFoundError(
        "September 2026 BTS CSV not found "
        "in data/processed"
    )

DATA_PATH = files[0]

print("=" * 55)
print("AeroPulse | September 2026 Data Validation")
print("=" * 55)

print("\nFile found:", DATA_PATH.name)

# 2. Load dataset
df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("\nDataset shape:", df.shape)

# 3. Validate required columns
required_columns = [
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "Distance",
    "DepDelay",
    "Cancelled",
    "Diverted"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print("\nRequired columns: ALL FOUND")

# 4. Validate flight dates
df["FlightDate"] = pd.to_datetime(
    df["FlightDate"],
    errors="coerce"
)

valid_dates = df["FlightDate"].between(
    "2026-09-01",
    "2026-09-30"
)

print("\nSeptember records:", valid_dates.sum())
print("Invalid/out-of-month dates:",
      (~valid_dates).sum())

# 5. Flight status
df["FlightStatus"] = "Completed"

df.loc[
    df["Cancelled"] == 1,
    "FlightStatus"
] = "Cancelled"

df.loc[
    df["Diverted"] == 1,
    "FlightStatus"
] = "Diverted"

print("\nFlight status:")
print(df["FlightStatus"].value_counts())

# 6. Missing values
print("\nMissing departure delays:")
print(df["DepDelay"].isna().sum())

print("\nMissing arrival delays:")
if "ArrDelay" in df.columns:
    print(df["ArrDelay"].isna().sum())
else:
    print("ArrDelay column not found")

# 7. Departure delay rate
completed = df[
    df["FlightStatus"] == "Completed"
]

delay_rate = (
    (completed["DepDelay"] >= 15).mean() * 100
)

print("\nCompleted flights delay rate (%):")
print(round(delay_rate, 2))

# 8. Scheduled departure hours
scheduled_time = pd.to_numeric(
    df["CRSDepTime"],
    errors="coerce"
)

df["ScheduledDepHour"] = (
    scheduled_time // 100
).astype("Int64")

print("\nScheduled departure hours:")
print(
    df["ScheduledDepHour"]
    .value_counts()
    .sort_index()
)

# 9. Data quality summary
print("\nDATA QUALITY SUMMARY")
print("-" * 45)

print("Total flights:", len(df))
print("Completed flights:", len(completed))
print("Duplicate rows:", df.duplicated().sum())
print("Total columns:", df.shape[1])

print("\nAeroPulse | September Validation Completed!")
