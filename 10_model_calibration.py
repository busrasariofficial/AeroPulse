
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score
)

# AeroPulse | 10 - Model Calibration

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "calibration"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "data" / "processed"
    / "aeropulse_model_features.csv",
    parse_dates=["FlightDate"]
)

model = joblib.load(
    ROOT / "outputs" / "models"
    / "xgboost.joblib"
)

numeric = [
    "DayOfWeek", "IsWeekend",
    "ScheduledDepHour", "HourSin",
    "HourCos", "LogDistance"
]

categorical = [
    "Reporting_Airline",
    "Origin", "Dest", "DepPeriod"
]

features = numeric + categorical

train = df[
    df["FlightDate"] <= "2026-08-20"
]

test = df[
    df["FlightDate"] >= "2026-08-26"
]

X_test = test[features].copy()
y_test = test["Delayed15"]

# Match training categorical encoding
for col in categorical:
    dtype = pd.CategoricalDtype(
        categories=sorted(
            train[col].dropna().unique()
        )
    )
    X_test[col] = X_test[col].astype(dtype)

# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]

print("\nAEROPULSE - CALIBRATION RESULTS")
print("=" * 45)

print(
    "Brier Score:",
    round(brier_score_loss(y_test, probs), 4)
)

print(
    "Log Loss:",
    round(log_loss(y_test, probs), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, probs), 4)
)

# Calibration curve
fraction_positive, mean_predicted = (
    calibration_curve(
        y_test,
        probs,
        n_bins=10,
        strategy="quantile"
    )
)

plt.figure(figsize=(8, 7))

plt.plot(
    [0, 1], [0, 1],
    "--", color="gray",
    label="Perfect Calibration"
)

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="XGBoost"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Delay Frequency")
plt.title("AeroPulse - Calibration Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "calibration_curve.png",
    dpi=300
)

plt.show()

# Save bin results
calibration_df = pd.DataFrame({
    "MeanPredictedProbability": mean_predicted,
    "ObservedDelayRate": fraction_positive
})

calibration_df.to_csv(
    OUT / "calibration_results.csv",
    index=False
)

print("\nAeroPulse | Calibration Analysis Completed!")
