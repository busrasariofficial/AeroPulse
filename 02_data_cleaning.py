
from pathlib import Path
import pandas as pd

# ==================================================
# AeroPulse | 02 - Data Cleaning
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / (
    "On_Time_Reporting_Carrier_On_Time_Performance_"
    "(1987_present)_2026_8.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load raw data
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Original shape:", df.shape)

# 2. Select relevant columns
columns = [
    "FlightDate",
    "DayOfWeek",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "OriginCityName",
    "Dest",
    "DestCityName",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDel15",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDel15",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "Distance",
    "AirTime",
    "TaxiOut",
    "TaxiIn",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay"
]

df = df[columns].copy()

# 3. Convert date
df["FlightDate"] = pd.to_datetime(
    df["FlightDate"], errors="coerce"
)

# 4. Convert numerical columns
numeric_cols = [
    "DepDelay", "DepDel15",
    "ArrDelay", "ArrDel15",
    "Cancelled", "Diverted"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5. Create scheduled departure hour
# BTS scheduled times are HHMM format
df["ScheduledDepHour"] = (
    df["CRSDepTime"] // 100
).astype("Int64")

# Midnight can be represented as 2400
df.loc[
    df["ScheduledDepHour"] == 24,
    "ScheduledDepHour"
] = 0

# 6. Separate operational statuses
df["FlightStatus"] = "Completed"

df.loc[
    df["Diverted"] == 1,
    "FlightStatus"
] = "Diverted"

df.loc[
    df["Cancelled"] == 1,
    "FlightStatus"
] = "Cancelled"

# 7. Create departure delay flag
# Keep missing delays as missing, not zero
df["Delayed15"] = pd.Series(
    pd.NA, index=df.index, dtype="Int64"
)

valid_delay = (
    (df["Cancelled"] == 0)
    & df["DepDelay"].notna()
)

df.loc[valid_delay, "Delayed15"] = (
    df.loc[valid_delay, "DepDelay"] >= 15
).astype(int)

# 8. Data quality checks
print("\nCleaned shape:", df.shape)

print("\nFlight status:")
print(df["FlightStatus"].value_counts())

print("\nMissing departure delays:")
print(df["DepDelay"].isna().sum())

print("\nDeparture delay rate (%):")
print(round(df["Delayed15"].mean() * 100, 2))

print("\nScheduled departure hours:")
print(
    df["ScheduledDepHour"]
    .value_counts()
    .sort_index()
)

# 9. Save processed data
output_path = OUTPUT_DIR / "aeropulse_2026_08_clean.csv"

df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("AeroPulse | Data Cleaning Completed!")
