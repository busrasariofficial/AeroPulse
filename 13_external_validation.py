
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix
)

# ============================================================
# AeroPulse | 13 - September External Validation
# Frozen August XGBoost + Sigmoid Calibration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "models"
CALIBRATION_DIR = ROOT / "outputs" / "calibration"
OUTPUT_DIR = ROOT / "outputs" / "external_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEPTEMBER_PATH = (
    DATA_DIR / "aeropulse_2026_09_model_features.csv"
)

AUGUST_PATH = (
    DATA_DIR / "aeropulse_model_features.csv"
)

MODEL_PATH = MODEL_DIR / "xgboost.joblib"

CALIBRATOR_PATH = (
    CALIBRATION_DIR / "sigmoid_calibrator.joblib"
)

print("=" * 65)
print("AeroPulse | September 2026 External Validation")
print("=" * 65)

# ============================================================
# 1. LOAD DATA AND SAVED MODELS
# ============================================================

sep = pd.read_csv(SEPTEMBER_PATH)
aug = pd.read_csv(AUGUST_PATH)

model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

print("\nSeptember dataset:", sep.shape)
print("August dataset:", aug.shape)

# ============================================================
# 2. DEFINE MODEL FEATURES
# ============================================================

numeric_features = [
    "DayOfWeek",
    "IsWeekend",
    "ScheduledDepHour",
    "HourSin",
    "HourCos",
    "LogDistance"
]

categorical_features = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DepPeriod"
]

expected_features = (
    numeric_features + categorical_features
)

# Use saved model feature order whenever available.
if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
else:
    features = expected_features

if set(features) != set(expected_features):
    raise ValueError(
        "Saved XGBoost model has unexpected features: "
        f"{features}"
    )

X = sep[features].copy()
y = sep["Delayed15"].astype(int)

print("\nTarget distribution:")
print(y.value_counts())

print("\nSeptember delay rate:",
      round(y.mean() * 100, 2), "%")

# ============================================================
# 3. CATEGORICAL DRIFT CHECK
# ============================================================

print("\nCATEGORICAL DRIFT CHECK")
print("-" * 65)

drift_rows = []

for col in categorical_features:

    # Use August categories as the reference.
    august_categories = sorted(
        aug[col].dropna().astype(str).unique()
    )

    september_values = X[col].astype(str)

    unseen_mask = ~september_values.isin(
        august_categories
    )

    unseen_count = int(unseen_mask.sum())

    drift_rows.append({
        "Feature": col,
        "UnseenRows": unseen_count,
        "UnseenRate": float(unseen_mask.mean())
    })

    # Unseen categories become NaN.
    # XGBoost can route missing categorical values.
    X[col] = pd.Categorical(
        september_values,
        categories=august_categories
    )

drift_df = pd.DataFrame(drift_rows)

print(drift_df.to_string(index=False))

print("\nCategorical dtypes:")
for col in categorical_features:
    print(f"{col}: {X[col].dtype}")

total_unseen = int(
    drift_df["UnseenRows"].sum()
)

print("\nTotal unseen category occurrences:",
      total_unseen)

if total_unseen > 0:
    print(
        "Unseen categories encoded as missing values."
    )
    print(
        "Predictions will use the model's "
        "missing-value routing."
    )
else:
    print("No unseen categories detected.")

# ============================================================
# 4. XGBOOST PREDICTIONS
# ============================================================

print("\nRunning frozen August XGBoost...")

original_prob = model.predict_proba(X)[:, 1]

print("XGBoost predictions completed.")

# ============================================================
# 5. SIGMOID CALIBRATION
# ============================================================

print("\nApplying sigmoid calibration...")

# The saved calibrator is assumed to have been trained
# on logit-transformed validation probabilities.

eps = 1e-7

clipped_prob = np.clip(
    original_prob,
    eps,
    1 - eps
)

logits = np.log(
    clipped_prob / (1 - clipped_prob)
).reshape(-1, 1)

calibrated_prob = (
    calibrator.predict_proba(logits)[:, 1]
)

print("Sigmoid calibration completed.")

# ============================================================
# 6. PERFORMANCE METRICS
# ============================================================

def evaluate(y_true, probabilities, name):

    predicted = (
        probabilities >= 0.50
    ).astype(int)

    return {
        "Model": name,
        "ROC_AUC": roc_auc_score(
            y_true, probabilities
        ),
        "PR_AUC": average_precision_score(
            y_true, probabilities
        ),
        "Brier": brier_score_loss(
            y_true, probabilities
        ),
        "LogLoss": log_loss(
            y_true, probabilities
        ),
        "Accuracy_050": accuracy_score(
            y_true, predicted
        ),
        "Precision_050": precision_score(
            y_true, predicted,
            zero_division=0
        ),
        "Recall_050": recall_score(
            y_true, predicted,
            zero_division=0
        ),
        "F1_050": f1_score(
            y_true, predicted,
            zero_division=0
        )
    }


results = pd.DataFrame([
    evaluate(
        y,
        original_prob,
        "Original XGBoost"
    ),
    evaluate(
        y,
        calibrated_prob,
        "Sigmoid Calibrated"
    )
])

print("\nSEPTEMBER MODEL PERFORMANCE")
print("=" * 65)

print(
    results.round(4).to_string(index=False)
)

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

calibrated_pred = (
    calibrated_prob >= 0.50
).astype(int)

cm = confusion_matrix(
    y,
    calibrated_pred
)

print("\nCALIBRATED CONFUSION MATRIX")
print("(Threshold = 0.50)")

print(cm)

# ============================================================
# 8. AUGUST VS SEPTEMBER
# ============================================================

august_rate = (
    aug["Delayed15"].mean() * 100
)

september_rate = (
    y.mean() * 100
)

print("\nMONTHLY DELAY RATE COMPARISON")
print("-" * 65)

print(
    f"August 2026: {august_rate:.2f}%"
)

print(
    f"September 2026: {september_rate:.2f}%"
)

print(
    f"Difference: "
    f"{september_rate - august_rate:+.2f} pp"
)

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_DIR / "september_metrics.csv",
    index=False
)

drift_df.to_csv(
    OUTPUT_DIR / "september_category_drift.csv",
    index=False
)

predictions = sep[[
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Delayed15"
]].copy()

predictions["OriginalProbability"] = (
    original_prob
)

predictions["CalibratedProbability"] = (
    calibrated_prob
)

predictions["Predicted050"] = (
    calibrated_pred
)

predictions.to_csv(
    OUTPUT_DIR / "september_predictions.csv",
    index=False
)

print("\nOUTPUT FILES SAVED")
print("-" * 65)

print("1. september_metrics.csv")
print("2. september_category_drift.csv")
print("3. september_predictions.csv")

print("\nAeroPulse | External Validation Completed!")
