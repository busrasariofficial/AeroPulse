
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# AeroPulse | 03 - Exploratory Data Analysis
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "aeropulse_2026_08_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Load cleaned data
df = pd.read_csv(DATA_PATH, low_memory=False)

# Valid departure delay observations
valid = df[
    (df["Cancelled"] == 0)
    & df["DepDelay"].notna()
].copy()

valid["Delayed15"] = (
    valid["DepDelay"] >= 15
).astype(int)

print("Total flights:", len(df))
print("Valid delay observations:", len(valid))

# ==================================================
# 2. Hourly Delay Rate
# ==================================================

hourly = (
    valid.groupby("ScheduledDepHour")
    .agg(
        Flights=("Delayed15", "size"),
        DelayRate=("Delayed15", "mean")
    )
    .reset_index()
)

hourly["DelayRate"] *= 100

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=hourly,
    x="ScheduledDepHour",
    y="DelayRate",
    marker="o"
)

plt.title("Departure Delay Rate by Scheduled Hour")
plt.xlabel("Scheduled Departure Hour")
plt.ylabel("Delay Rate (%)")
plt.xticks(range(24))
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_hourly_delay_rate.png",
    dpi=300
)
plt.show()

# ==================================================
# 3. Airline Delay Performance
# ==================================================

airlines = (
    valid.groupby("Reporting_Airline")
    .agg(
        Flights=("Delayed15", "size"),
        DelayRate=("Delayed15", "mean")
    )
    .query("Flights >= 1000")
    .sort_values("DelayRate", ascending=False)
)

airlines["DelayRate"] *= 100

plt.figure(figsize=(12, 6))

sns.barplot(
    x=airlines["DelayRate"],
    y=airlines.index,
    color="steelblue"
)

plt.title("Departure Delay Rate by Airline")
plt.xlabel("Delay Rate (%)")
plt.ylabel("Airline")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_airline_delay_rate.png",
    dpi=300
)
plt.show()

# ==================================================
# 4. Airport Delay Performance
# ==================================================

airports = (
    valid.groupby("Origin")
    .agg(
        Flights=("Delayed15", "size"),
        DelayRate=("Delayed15", "mean")
    )
    .query("Flights >= 1000")
    .sort_values("DelayRate", ascending=False)
    .head(15)
)

airports["DelayRate"] *= 100

plt.figure(figsize=(12, 7))

sns.barplot(
    x=airports["DelayRate"],
    y=airports.index,
    color="steelblue"
)

plt.title("Top 15 Airports by Departure Delay Rate")
plt.xlabel("Delay Rate (%)")
plt.ylabel("Origin Airport")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_airport_delay_rate.png",
    dpi=300
)
plt.show()

# ==================================================
# 5. Departure Delay Distribution
# ==================================================

plt.figure(figsize=(12, 6))

sns.histplot(
    valid["DepDelay"].clip(-30, 180),
    bins=60,
    color="steelblue"
)

plt.title("Departure Delay Distribution")
plt.xlabel("Departure Delay (minutes, clipped)")
plt.ylabel("Flight Count")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_delay_distribution.png",
    dpi=300
)
plt.show()

# ==================================================
# 6. Export Summary Tables
# ==================================================

SUMMARY_DIR = BASE_DIR / "outputs" / "tables"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

hourly.to_csv(
    SUMMARY_DIR / "hourly_delay.csv",
    index=False
)

airlines.to_csv(
    SUMMARY_DIR / "airline_delay.csv"
)

airports.to_csv(
    SUMMARY_DIR / "airport_delay.csv"
)

print("\nAeroPulse | EDA Completed!")
