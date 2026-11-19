
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score
)

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

features = [
    "DayOfWeek", "IsWeekend",
    "ScheduledDepHour", "HourSin",
    "HourCos", "LogDistance",
    "Reporting_Airline", "Origin",
    "Dest", "DepPeriod"
]

categorical = [
    "Reporting_Airline", "Origin",
    "Dest", "DepPeriod"
]

train = df[df.FlightDate <= "2026-08-20"]

val = df[df.FlightDate.between(
    "2026-08-21", "2026-08-25"
)]

test = df[df.FlightDate >= "2026-08-26"]

def prepare(frame):
    X = frame[features].copy()

    for col in categorical:
        dtype = pd.CategoricalDtype(
            categories=sorted(
                train[col].dropna().unique()
            )
        )
        X[col] = X[col].astype(dtype)

    return X

X_val = prepare(val)
X_test = prepare(test)

y_val = val["Delayed15"]
y_test = test["Delayed15"]

val_probs = model.predict_proba(X_val)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]

# 1. Sigmoid calibration
# Use logit of original probabilities
eps = 1e-7

def to_logit(probs):
    p = np.clip(probs, eps, 1 - eps)
    return np.log(p / (1 - p)).reshape(-1, 1)

sigmoid = LogisticRegression(
    max_iter=1000
)

sigmoid.fit(
    to_logit(val_probs), y_val
)

sigmoid_probs = sigmoid.predict_proba(
    to_logit(test_probs)
)[:, 1]

# 2. Isotonic calibration
isotonic = IsotonicRegression(
    out_of_bounds="clip"
)

isotonic.fit(val_probs, y_val)

isotonic_probs = isotonic.predict(test_probs)

# 3. Evaluate all methods
predictions = {
    "Original XGBoost": test_probs,
    "Sigmoid": sigmoid_probs,
    "Isotonic": isotonic_probs
}

rows = []

for name, probs in predictions.items():
    rows.append({
        "Method": name,
        "Brier": brier_score_loss(y_test, probs),
        "LogLoss": log_loss(y_test, probs),
        "ROC_AUC": roc_auc_score(y_test, probs),
        "PR_AUC": average_precision_score(
            y_test, probs
        )
    })

results = pd.DataFrame(rows)

print("\nCALIBRATION COMPARISON")
print("=" * 60)
print(results.round(4).to_string(index=False))

results.to_csv(
    OUT / "calibration_comparison.csv",
    index=False
)

# 4. Calibration curves
plt.figure(figsize=(9, 7))

plt.plot(
    [0, 1], [0, 1],
    "--", color="gray",
    label="Perfect Calibration"
)

for name, probs in predictions.items():
    observed, predicted = calibration_curve(
        y_test,
        probs,
        n_bins=10,
        strategy="quantile"
    )

    plt.plot(
        predicted,
        observed,
        marker="o",
        label=name
    )

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Delay Frequency")
plt.title("AeroPulse - Calibration Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "calibration_comparison.png",
    dpi=300
)

plt.show()

# 5. Save calibrators
joblib.dump(
    sigmoid,
    OUT / "sigmoid_calibrator.joblib"
)

joblib.dump(
    isotonic,
    OUT / "isotonic_calibrator.joblib"
)

print("\nAeroPulse | Calibration Comparison Completed!")
