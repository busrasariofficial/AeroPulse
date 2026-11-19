
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# ==================================================
# AeroPulse | 08b - SHAP Analysis
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "outputs" / "models" / "xgboost.joblib"
)

DATA_PATH = (
    ROOT / "data" / "processed"
    / "aeropulse_model_features.csv"
)

OUT = ROOT / "outputs" / "explainability"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load model and data
model = joblib.load(MODEL_PATH)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["FlightDate"]
)

numeric = [
    "DayOfWeek",
    "IsWeekend",
    "ScheduledDepHour",
    "HourSin",
    "HourCos",
    "LogDistance"
]

categorical = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DepPeriod"
]

features = numeric + categorical

train = df[
    df["FlightDate"] <= "2026-08-20"
].copy()

test = df[
    df["FlightDate"] >= "2026-08-26"
].copy()

# 2. Sample test flights for explanation
sample = test.sample(
    n=min(3000, len(test)),
    random_state=42
).copy()

X = sample[features].copy()

# Match categorical encoding used in training
for col in categorical:
    dtype = pd.CategoricalDtype(
        categories=sorted(
            train[col].dropna().unique()
        )
    )
    X[col] = X[col].astype(dtype)

# 3. Calculate native TreeSHAP contributions
booster = model.get_booster()

dmatrix = xgb.DMatrix(
    X,
    enable_categorical=True
)

contributions = booster.predict(
    dmatrix,
    pred_contribs=True
)

# Last column is the baseline contribution
shap_values = contributions[:, :-1]
base_values = contributions[:, -1]

assert shap_values.shape[1] == len(features)

# 4. Global SHAP importance
importance = pd.DataFrame({
    "Feature": features,
    "MeanAbsSHAP": np.abs(
        shap_values
    ).mean(axis=0)
}).sort_values(
    "MeanAbsSHAP",
    ascending=False
)

print("\nGLOBAL SHAP IMPORTANCE")
print(importance.round(4).to_string(index=False))

importance.to_csv(
    OUT / "shap_global_importance.csv",
    index=False
)

# 5. Global SHAP importance chart
plot_data = importance.sort_values(
    "MeanAbsSHAP"
)

plt.figure(figsize=(11, 7))

plt.barh(
    plot_data["Feature"],
    plot_data["MeanAbsSHAP"],
    color="steelblue"
)

plt.title("AeroPulse - Global SHAP Importance")
plt.xlabel("Mean Absolute SHAP Value (log-odds)")
plt.tight_layout()

plt.savefig(
    OUT / "shap_global_importance.png",
    dpi=300
)
plt.show()

# 6. Explain one high-risk flight
probabilities = model.predict_proba(X)[:, 1]

high_risk_idx = int(np.argmax(probabilities))

flight = sample.iloc[high_risk_idx]

local = pd.DataFrame({
    "Feature": features,
    "SHAP": shap_values[high_risk_idx]
}).sort_values("SHAP")

print("\nHIGH-RISK FLIGHT")
print("Date:", flight["FlightDate"])
print("Airline:", flight["Reporting_Airline"])
print("Origin:", flight["Origin"])
print("Destination:", flight["Dest"])
print("Scheduled hour:", flight["ScheduledDepHour"])
print("Actual delayed:", flight["Delayed15"])
print("Predicted probability:",
      round(probabilities[high_risk_idx], 4))

print("\nLOCAL SHAP CONTRIBUTIONS")
print(local.round(4).to_string(index=False))

# 7. Local explanation chart
colors = [
    "tomato" if v > 0 else "steelblue"
    for v in local["SHAP"]
]

plt.figure(figsize=(11, 7))

plt.barh(
    local["Feature"],
    local["SHAP"],
    color=colors
)

plt.axvline(0, color="black", linewidth=1)

plt.title("AeroPulse - High Risk Flight Explanation")
plt.xlabel("SHAP Contribution (log-odds)")
plt.tight_layout()

plt.savefig(
    OUT / "shap_high_risk_flight.png",
    dpi=300
)
plt.show()

print("\nAeroPulse | SHAP Analysis Completed!")
