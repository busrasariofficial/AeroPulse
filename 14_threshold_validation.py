
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# ============================================================
# AeroPulse | 14 - Calibrated Threshold Validation
# August validation -> September external test
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "models"
CAL_DIR = ROOT / "outputs" / "calibration"
OUT_DIR = ROOT / "outputs" / "threshold_validation"

OUT_DIR.mkdir(parents=True, exist_ok=True)

AUG_PATH = DATA_DIR / "aeropulse_model_features.csv"
SEP_PATH = DATA_DIR / "aeropulse_2026_09_model_features.csv"

MODEL_PATH = MODEL_DIR / "xgboost.joblib"
CALIBRATOR_PATH = CAL_DIR / "sigmoid_calibrator.joblib"

# ------------------------------------------------------------
# 1. Load data and frozen models
# ------------------------------------------------------------

aug = pd.read_csv(AUG_PATH, parse_dates=["FlightDate"])
sep = pd.read_csv(SEP_PATH, parse_dates=["FlightDate"])

model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

print("=" * 65)
print("AeroPulse | Calibrated Threshold Validation")
print("=" * 65)

# ------------------------------------------------------------
# 2. Reconstruct feature schema
# ------------------------------------------------------------

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

expected_features = numeric_features + categorical_features

if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)
else:
    features = expected_features

if set(features) != set(expected_features):
    raise ValueError(
        f"Unexpected saved model features: {features}"
    )

# ------------------------------------------------------------
# 3. Use August validation dates only
# ------------------------------------------------------------

aug_val = aug.loc[
    aug["FlightDate"].between(
        "2026-08-21", "2026-08-25"
    )
].copy()

if aug_val.empty:
    raise ValueError("August validation dataset is empty!")

print("\nAugust validation flights:", len(aug_val))
print("September external flights:", len(sep))

# ------------------------------------------------------------
# 4. Prepare categorical features
# ------------------------------------------------------------

def prepare_features(df, reference):
    X = df[features].copy()

    for col in categorical_features:
        categories = sorted(
            reference[col].dropna().astype(str).unique()
        )

        X[col] = pd.Categorical(
            X[col].astype(str),
            categories=categories
        )

    return X


# IMPORTANT:
# Categories should ideally come from the original
# August training encoder, not from validation/test.
# Here we reconstruct them using August data, matching
# the convention used in script 13.

X_val = prepare_features(aug_val, aug)
X_sep = prepare_features(sep, aug)

y_val = aug_val["Delayed15"].astype(int)
y_sep = sep["Delayed15"].astype(int)

# ------------------------------------------------------------
# 5. Frozen model probabilities
# ------------------------------------------------------------

print("\nGenerating XGBoost probabilities...")

p_val_raw = model.predict_proba(X_val)[:, 1]
p_sep_raw = model.predict_proba(X_sep)[:, 1]

# ------------------------------------------------------------
# 6. Sigmoid calibration
# ------------------------------------------------------------

def apply_calibration(prob):
    eps = 1e-7
    clipped = np.clip(prob, eps, 1 - eps)

    logits = np.log(
        clipped / (1 - clipped)
    ).reshape(-1, 1)

    return calibrator.predict_proba(logits)[:, 1]


p_val = apply_calibration(p_val_raw)
p_sep = apply_calibration(p_sep_raw)

# ------------------------------------------------------------
# 7. Optimize threshold on August validation
# ------------------------------------------------------------

precision, recall, thresholds = precision_recall_curve(
    y_val, p_val
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (
        precision[:-1]
        + recall[:-1]
        + 1e-12
    )
)

best_idx = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_idx])

print("\nAUGUST VALIDATION THRESHOLD")
print("-" * 65)

print("Best threshold:", round(best_threshold, 4))
print("Validation F1:", round(float(f1_scores[best_idx]), 4))

# ------------------------------------------------------------
# 8. Evaluation function
# ------------------------------------------------------------

def evaluate(y_true, probabilities, threshold, name):

    pred = (
        probabilities >= threshold
    ).astype(int)

    return {
        "Dataset": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(
            y_true, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_true, probabilities
        ),
        "PR_AUC": average_precision_score(
            y_true, probabilities
        ),
        "Brier": brier_score_loss(
            y_true, probabilities
        )
    }


results = pd.DataFrame([
    evaluate(
        y_val, p_val, best_threshold,
        "August Validation"
    ),
    evaluate(
        y_sep, p_sep, best_threshold,
        "September External"
    ),
    evaluate(
        y_sep, p_sep, 0.50,
        "September Threshold 0.50"
    )
])

print("\nTHRESHOLD PERFORMANCE")
print("=" * 65)
print(results.round(4).to_string(index=False))

# ------------------------------------------------------------
# 9. September confusion matrix
# ------------------------------------------------------------

sep_pred = (
    p_sep >= best_threshold
).astype(int)

cm = confusion_matrix(y_sep, sep_pred)

print("\nSEPTEMBER CONFUSION MATRIX")
print("(August-optimized threshold)")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# ------------------------------------------------------------
# 10. Save results
# ------------------------------------------------------------

results.to_csv(
    OUT_DIR / "threshold_comparison.csv",
    index=False
)

pd.DataFrame({
    "FlightDate": sep["FlightDate"],
    "Delayed15": y_sep,
    "CalibratedProbability": p_sep,
    "PredictedOptimized": sep_pred
}).to_csv(
    OUT_DIR / "september_threshold_predictions.csv",
    index=False
)

joblib.dump(
    {"threshold": best_threshold},
    OUT_DIR / "calibrated_threshold.joblib"
)

print("\nSaved outputs:")
print(" - threshold_comparison.csv")
print(" - september_threshold_predictions.csv")
print(" - calibrated_threshold.joblib")

print("\nAeroPulse | Threshold Validation Completed!")
