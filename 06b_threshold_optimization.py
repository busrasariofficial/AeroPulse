
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score,
    f1_score, accuracy_score,
    precision_recall_curve
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "models"

df = pd.read_csv(
    ROOT / "data" / "processed"
    / "aeropulse_model_features.csv",
    parse_dates=["FlightDate"]
)

model = joblib.load(
    MODEL_DIR / "logistic_regression.joblib"
)

features = [
    "DayOfWeek", "IsWeekend",
    "ScheduledDepHour", "HourSin",
    "HourCos", "LogDistance",
    "Reporting_Airline", "Origin",
    "Dest", "DepPeriod"
]

val = df[
    df["FlightDate"].between(
        "2026-08-21", "2026-08-25"
    )
].copy()

test = df[
    df["FlightDate"] >= "2026-08-26"
].copy()

y_val = val["Delayed15"]
val_probs = model.predict_proba(
    val[features]
)[:, 1]

# Optimize F1 on validation only
thresholds = np.arange(0.10, 0.81, 0.02)

rows = []

for threshold in thresholds:
    preds = (val_probs >= threshold).astype(int)

    rows.append({
        "Threshold": round(threshold, 2),
        "Precision": precision_score(
            y_val, preds, zero_division=0
        ),
        "Recall": recall_score(
            y_val, preds, zero_division=0
        ),
        "F1": f1_score(
            y_val, preds, zero_division=0
        )
    })

results = pd.DataFrame(rows)

best = results.loc[
    results["F1"].idxmax()
]

best_threshold = float(best["Threshold"])

print("Best validation threshold:", best_threshold)
print("\nBest validation metrics:")
print(best.to_string())

# Evaluate on untouched test set
y_test = test["Delayed15"]

test_probs = model.predict_proba(
    test[features]
)[:, 1]

test_preds = (
    test_probs >= best_threshold
).astype(int)

print("\nTEST RESULTS - OPTIMIZED THRESHOLD")
print("Accuracy:", round(
    accuracy_score(y_test, test_preds), 4
))
print("Precision:", round(
    precision_score(y_test, test_preds), 4
))
print("Recall:", round(
    recall_score(y_test, test_preds), 4
))
print("F1:", round(
    f1_score(y_test, test_preds), 4
))

# Plot validation metrics
plt.figure(figsize=(10, 6))

for metric in ["Precision", "Recall", "F1"]:
    plt.plot(
        results["Threshold"],
        results[metric],
        marker="o",
        label=metric
    )

plt.axvline(
    best_threshold,
    linestyle="--",
    color="black",
    label=f"Best F1 threshold: {best_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("AeroPulse - Threshold Optimization")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "threshold_optimization.png",
    dpi=300
)
plt.show()

results.to_csv(
    MODEL_DIR / "threshold_metrics.csv",
    index=False
)

print("\nAeroPulse | Threshold Optimization Completed!")
