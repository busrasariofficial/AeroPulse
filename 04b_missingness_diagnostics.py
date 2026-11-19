
from pathlib import Path
import pandas as pd

# AeroPulse | Missingness Diagnostics

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "aeropulse_2026_08_clean.csv"
)

df = pd.read_csv(DATA_PATH, low_memory=False)

print("=" * 55)
print("AeroPulse | Missingness Diagnostics")
print("=" * 55)

# 1. Flight status vs missing departure delay
print("\n1. FLIGHT STATUS VS MISSING DEPDELAY")

print(
    pd.crosstab(
        df["FlightStatus"],
        df["DepDelay"].isna(),
        margins=True
    )
)

# 2. Flight status vs missing arrival delay
print("\n2. FLIGHT STATUS VS MISSING ARRDELAY")

print(
    pd.crosstab(
        df["FlightStatus"],
        df["ArrDelay"].isna(),
        margins=True
    )
)

# 3. Delay reason availability
print("\n3. DELAY REASON AVAILABILITY")

delay_cols = [
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay"
]

print(
    df[delay_cols]
    .notna()
    .sum()
)

# 4. Cancellation code consistency
print("\n4. CANCELLATION CODE CHECK")

print(
    pd.crosstab(
        df["Cancelled"],
        df["CancellationCode"].notna(),
        margins=True
    )
)

print("\nDiagnostics completed!")
