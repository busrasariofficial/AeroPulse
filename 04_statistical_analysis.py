
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, spearmanr

# ====================================================
# AeroPulse | 04 - Statistical Analysis
# BTS Flight Data | August 2026
# ====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "aeropulse_2026_08_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)

# Use flights with known departure delays
valid = df.loc[
    (df["Cancelled"] == 0)
    & df["DepDelay"].notna()
].copy()

valid["Delayed15"] = (
    valid["DepDelay"] >= 15
).astype(int)

print("=" * 55)
print("AeroPulse | Statistical Analysis")
print("=" * 55)
print("Valid observations:", len(valid))


# ====================================================
# HELPER: Cramer's V
# ====================================================

def cramers_v(table, chi2):
    n = table.to_numpy().sum()
    r, k = table.shape

    return np.sqrt(
        (chi2 / n) / min(r - 1, k - 1)
    )


# ====================================================
# TEST 1: Scheduled Hour vs Departure Delay
# ====================================================

print("\n" + "=" * 55)
print("TEST 1: DEPARTURE HOUR VS DELAY")
print("=" * 55)

hour_data = valid.dropna(
    subset=["ScheduledDepHour"]
).copy()

hour_table = pd.crosstab(
    hour_data["ScheduledDepHour"],
    hour_data["Delayed15"]
)

chi2_hour, p_hour, dof_hour, expected_hour = (
    chi2_contingency(hour_table)
)

v_hour = cramers_v(hour_table, chi2_hour)

print(f"Chi-square: {chi2_hour:.4f}")
print(f"Degrees of freedom: {dof_hour}")
print(f"P-value: {p_hour:.6g}")
print(f"Cramer's V: {v_hour:.4f}")
print("Minimum expected count:",
      round(expected_hour.min(), 2))

hour_summary = (
    hour_data.groupby("ScheduledDepHour")
    .agg(
        Flights=("Delayed15", "size"),
        DelayRate=("Delayed15", "mean")
    )
    .reset_index()
)

hour_summary["DelayRate"] *= 100

print("\nHourly delay rates:")
print(hour_summary.round(2).to_string(index=False))

hour_summary.to_csv(
    OUTPUT_DIR / "hourly_delay_statistics.csv",
    index=False
)


# ====================================================
# TEST 2: Airline vs Departure Delay
# ====================================================

print("\n" + "=" * 55)
print("TEST 2: AIRLINE VS DELAY")
print("=" * 55)

airline_counts = valid[
    "Reporting_Airline"
].value_counts()

eligible_airlines = airline_counts[
    airline_counts >= 1000
].index

airline_data = valid[
    valid["Reporting_Airline"].isin(eligible_airlines)
].copy()

airline_table = pd.crosstab(
    airline_data["Reporting_Airline"],
    airline_data["Delayed15"]
)

chi2_airline, p_airline, dof_airline, expected_airline = (
    chi2_contingency(airline_table)
)

v_airline = cramers_v(
    airline_table, chi2_airline
)

print(f"Chi-square: {chi2_airline:.4f}")
print(f"Degrees of freedom: {dof_airline}")
print(f"P-value: {p_airline:.6g}")
print(f"Cramer's V: {v_airline:.4f}")
print("Minimum expected count:",
      round(expected_airline.min(), 2))

airline_summary = (
    airline_data.groupby("Reporting_Airline")
    .agg(
        Flights=("Delayed15", "size"),
        DelayRate=("Delayed15", "mean")
    )
    .sort_values("DelayRate", ascending=False)
)

airline_summary["DelayRate"] *= 100

print("\nAirline delay rates:")
print(airline_summary.round(2))

airline_summary.to_csv(
    OUTPUT_DIR / "airline_delay_statistics.csv"
)


# ====================================================
# TEST 3: Distance vs Departure Delay
# ====================================================

print("\n" + "=" * 55)
print("TEST 3: DISTANCE VS DEPARTURE DELAY")
print("=" * 55)

distance_data = valid[
    ["Distance", "DepDelay"]
].dropna()

rho, p_distance = spearmanr(
    distance_data["Distance"],
    distance_data["DepDelay"]
)

print(f"Spearman rho: {rho:.4f}")
print(f"P-value: {p_distance:.6g}")
print("Observations:", len(distance_data))

# Distance groups using quintiles
distance_data = distance_data.copy()

distance_data["DistanceGroup"] = pd.qcut(
    distance_data["Distance"],
    q=5,
    duplicates="drop"
)

distance_summary = (
    distance_data.groupby(
        "DistanceGroup", observed=True
    )
    .agg(
        Flights=("DepDelay", "size"),
        MeanDelay=("DepDelay", "mean"),
        MedianDelay=("DepDelay", "median"),
        DelayRate=(
            "DepDelay",
            lambda x: (x >= 15).mean() * 100
        )
    )
    .reset_index()
)

print("\nDistance group summary:")
print(distance_summary.round(2).to_string(index=False))

distance_summary.to_csv(
    OUTPUT_DIR / "distance_statistics.csv",
    index=False
)


# ====================================================
# VISUALIZATION: Distance vs Delay Rate
# ====================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 6))

sns.barplot(
    data=distance_summary,
    x="DistanceGroup",
    y="DelayRate",
    color="steelblue"
)

plt.title("Departure Delay Rate by Flight Distance")
plt.xlabel("Flight Distance Quintile (miles)")
plt.ylabel("Departure Delay Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "distance_delay_rate.png",
    dpi=300
)

plt.show()


# ====================================================
# SAVE TEST RESULTS
# ====================================================

results = pd.DataFrame([
    {
        "Test": "Hour vs Delay",
        "Method": "Chi-square",
        "Statistic": chi2_hour,
        "PValue": p_hour,
        "EffectSize": v_hour,
        "EffectType": "Cramers V"
    },
    {
        "Test": "Airline vs Delay",
        "Method": "Chi-square",
        "Statistic": chi2_airline,
        "PValue": p_airline,
        "EffectSize": v_airline,
        "EffectType": "Cramers V"
    },
    {
        "Test": "Distance vs Delay",
        "Method": "Spearman",
        "Statistic": rho,
        "PValue": p_distance,
        "EffectSize": rho,
        "EffectType": "Spearman rho"
    }
])

results.to_csv(
    OUTPUT_DIR / "statistical_test_results.csv",
    index=False
)

print("\n" + "=" * 55)
print("STATISTICAL TEST SUMMARY")
print("=" * 55)

print(results.to_string(index=False))

print("\nAeroPulse | Statistical Analysis Completed!")
