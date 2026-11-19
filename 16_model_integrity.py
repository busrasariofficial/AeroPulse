
from pathlib import Path
import json
import hashlib
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    f1_score,
)

# ============================================================
# AeroPulse | 16 - Model Integrity & Reproducibility Audit
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "outputs" / "models"
CAL_DIR = ROOT / "outputs" / "calibration"
THRESHOLD_DIR = ROOT / "outputs" / "threshold_validation"
EXTERNAL_DIR = ROOT / "outputs" / "external_validation"
OUTPUT_DIR = ROOT / "outputs" / "integrity"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUG_PATH = DATA_DIR / "aeropulse_model_features.csv"
SEP_PATH = DATA_DIR / "aeropulse_2026_09_model_features.csv"

MODEL_PATH = MODEL_DIR / "xgboost.joblib"
CAL_PATH = CAL_DIR / "sigmoid_calibrator.joblib"

THRESHOLD_PATH = (
    THRESHOLD_DIR / "calibrated_threshold.joblib"
)

PRED_PATH = (
    THRESHOLD_DIR / "september_threshold_predictions.csv"
)

EXTERNAL_PRED_PATH = (
    EXTERNAL_DIR / "september_predictions.csv"
)

NUMERIC_FEATURES = [
    "DayOfWeek",
    "IsWeekend",
    "ScheduledDepHour",
    "HourSin",
    "HourCos",
    "LogDistance",
]

CATEGORICAL_FEATURES = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DepPeriod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "Delayed15"

checks = []


# ============================================================
# Helpers
# ============================================================

def add_check(name, status, details):
    checks.append({
        "Check": name,
        "Status": status,
        "Details": str(details),
    })

    print(f"[{status}] {name}")
    print(f"    {details}")


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def calculate_metrics(y, p, threshold):
    predictions = (p >= threshold).astype(int)

    return {
        "Flights": int(len(y)),
        "DelayRate": float(np.mean(y)),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(
            average_precision_score(y, p)
        ),
        "Brier": float(
            brier_score_loss(y, p)
        ),
        "LogLoss": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "F1": float(
            f1_score(y, predictions)
        ),
    }


print("=" * 70)
print("AeroPulse | Model Integrity Audit")
print("=" * 70)


# ============================================================
# 1. Required files
# ============================================================

required_files = [
    AUG_PATH,
    SEP_PATH,
    MODEL_PATH,
    CAL_PATH,
    THRESHOLD_PATH,
    PRED_PATH,
]

for path in required_files:
    require_file(path)

add_check(
    "Required artifacts",
    "PASS",
    "All required files exist."
)


# ============================================================
# 2. Load artifacts
# ============================================================

aug = pd.read_csv(
    AUG_PATH,
    parse_dates=["FlightDate"]
)

sep = pd.read_csv(
    SEP_PATH,
    parse_dates=["FlightDate"]
)

saved_predictions = pd.read_csv(
    PRED_PATH,
    parse_dates=["FlightDate"]
)

model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CAL_PATH)

threshold_artifact = joblib.load(
    THRESHOLD_PATH
)

if not isinstance(threshold_artifact, dict):
    raise TypeError(
        "Threshold artifact must be a dictionary."
    )

threshold = float(
    threshold_artifact["threshold"]
)

print("\nDATASET SUMMARY")
print("-" * 70)
print("August:", aug.shape)
print("September:", sep.shape)
print("Saved predictions:", saved_predictions.shape)
print("Threshold:", round(threshold, 6))


# ============================================================
# 3. Feature schema
# ============================================================

for name, frame in [
    ("August", aug),
    ("September", sep),
]:
    missing = [
        col for col in FEATURES + [TARGET, "FlightDate"]
        if col not in frame.columns
    ]

    if missing:
        add_check(
            f"{name} feature schema",
            "FAIL",
            f"Missing columns: {missing}"
        )
        raise ValueError(
            f"{name} missing columns: {missing}"
        )

    add_check(
        f"{name} feature schema",
        "PASS",
        "Required columns are present."
    )


if hasattr(model, "feature_names_in_"):
    model_features = list(
        model.feature_names_in_
    )

    if set(model_features) == set(FEATURES):
        add_check(
            "Saved model feature names",
            "PASS",
            "Feature names match the expected set."
        )
    else:
        add_check(
            "Saved model feature names",
            "FAIL",
            f"Model features: {model_features}"
        )
        raise ValueError(
            "Saved model feature mismatch."
        )
else:
    model_features = FEATURES

    add_check(
        "Saved model feature names",
        "REVIEW",
        "Model does not expose feature_names_in_."
    )


# ============================================================
# 4. Missing values and target validation
# ============================================================

for name, frame in [
    ("August", aug),
    ("September", sep),
]:
    missing_count = int(
        frame[FEATURES + [TARGET]].isna().sum().sum()
    )

    if missing_count == 0:
        add_check(
            f"{name} missing values",
            "PASS",
            "No missing values in model columns."
        )
    else:
        add_check(
            f"{name} missing values",
            "FAIL",
            f"{missing_count} missing values."
        )

    target_values = set(
        frame[TARGET].dropna().unique()
    )

    if target_values.issubset({0, 1}):
        add_check(
            f"{name} binary target",
            "PASS",
            f"Target values: {target_values}"
        )
    else:
        add_check(
            f"{name} binary target",
            "FAIL",
            f"Unexpected values: {target_values}"
        )


# ============================================================
# 5. Temporal integrity
# ============================================================

aug_dates = aug["FlightDate"]
sep_dates = sep["FlightDate"]

aug_train = aug.loc[
    aug_dates.between(
        "2026-08-01",
        "2026-08-20"
    )
]

aug_val = aug.loc[
    aug_dates.between(
        "2026-08-21",
        "2026-08-25"
    )
]

aug_test = aug.loc[
    aug_dates.between(
        "2026-08-26",
        "2026-08-31"
    )
]

date_sets = [
    set(aug_train["FlightDate"]),
    set(aug_val["FlightDate"]),
    set(aug_test["FlightDate"]),
    set(sep["FlightDate"]),
]

no_overlap = all(
    date_sets[i].isdisjoint(date_sets[j])
    for i in range(len(date_sets))
    for j in range(i + 1, len(date_sets))
)

all_august_covered = (
    len(aug_train)
    + len(aug_val)
    + len(aug_test)
    == len(aug)
)

if no_overlap and all_august_covered:
    add_check(
        "Temporal partition boundaries",
        "PASS",
        "Date partitions are disjoint and cover August."
    )
else:
    add_check(
        "Temporal partition boundaries",
        "FAIL",
        "Date overlap or uncovered August rows."
    )

add_check(
    "Training procedure verification",
    "REVIEW",
    "Date partitions are valid, but the saved model "
    "alone cannot prove which rows were used for fitting."
)


# ============================================================
# 6. Feature engineering consistency
# ============================================================

def check_engineered_features(frame, name):
    hour = frame["ScheduledDepHour"].to_numpy()

    expected_sin = np.sin(
        2 * np.pi * hour / 24
    )

    expected_cos = np.cos(
        2 * np.pi * hour / 24
    )

    sin_ok = np.allclose(
        frame["HourSin"].to_numpy(),
        expected_sin,
        atol=1e-6
    )

    cos_ok = np.allclose(
        frame["HourCos"].to_numpy(),
        expected_cos,
        atol=1e-6
    )

    if sin_ok and cos_ok:
        add_check(
            f"{name} cyclic time features",
            "PASS",
            "HourSin and HourCos are consistent."
        )
    else:
        add_check(
            f"{name} cyclic time features",
            "FAIL",
            "Cyclic feature mismatch."
        )

    expected_log_distance = np.log1p(
        frame["Distance"].to_numpy()
    )

    distance_ok = np.allclose(
        frame["LogDistance"].to_numpy(),
        expected_log_distance,
        atol=1e-6
    )

    add_check(
        f"{name} LogDistance",
        "PASS" if distance_ok else "FAIL",
        "log1p(Distance) consistency checked."
    )

    expected_weekend = (
        frame["DayOfWeek"].isin([6, 7])
    ).astype(int)

    weekend_ok = np.array_equal(
        frame["IsWeekend"].to_numpy(),
        expected_weekend.to_numpy()
    )

    add_check(
        f"{name} weekend indicator",
        "PASS" if weekend_ok else "FAIL",
        "IsWeekend consistency checked."
    )


check_engineered_features(aug, "August")
check_engineered_features(sep, "September")


# ============================================================
# 7. Category drift
# ============================================================

drift_rows = []

for col in CATEGORICAL_FEATURES:
    reference = set(
        aug[col].dropna().astype(str)
    )

    unseen = (
        ~sep[col].astype(str).isin(reference)
    )

    count = int(unseen.sum())

    drift_rows.append({
        "Feature": col,
        "UnseenRows": count,
        "UnseenRate": count / len(sep),
    })

drift_df = pd.DataFrame(drift_rows)

drift_df.to_csv(
    OUTPUT_DIR / "category_drift_audit.csv",
    index=False
)

print("\nCATEGORY DRIFT")
print("-" * 70)
print(drift_df.to_string(index=False))

add_check(
    "September category drift",
    "PASS",
    "Unseen category counts were measured. "
    "Unseen values are not automatically errors."
)


# ============================================================
# 8. Reconstruct model inputs
# ============================================================

def prepare_features(frame, reference):
    X = frame[model_features].copy()

    for col in CATEGORICAL_FEATURES:
        categories = sorted(
            reference[col]
            .dropna()
            .astype(str)
            .unique()
        )

        X[col] = pd.Categorical(
            X[col].astype(str),
            categories=categories
        )

    return X


X_sep = prepare_features(sep, aug_train)

# This reproduces the category convention used in
# scripts 13 and 14, but not necessarily the
# original training-time category mapping.

add_check(
    "Original category encoding provenance",
    "REVIEW",
    "August-wide sorted categories reproduce the "
    "validation convention, but original training "
    "category order must be checked in 07_tree_models.py."
)


# ============================================================
# 9. Model predictions
# ============================================================

print("\nREGENERATING SEPTEMBER PREDICTIONS")
print("-" * 70)

raw_prob = model.predict_proba(X_sep)[:, 1]

if (
    not np.isfinite(raw_prob).all()
    or not np.logical_and(
        raw_prob >= 0,
        raw_prob <= 1
    ).all()
):
    add_check(
        "Raw probability validity",
        "FAIL",
        "Invalid XGBoost probabilities."
    )
    raise ValueError(
        "Invalid raw model probabilities."
    )

add_check(
    "Raw probability validity",
    "PASS",
    "All XGBoost probabilities are finite and in [0,1]."
)


# ============================================================
# 10. Sigmoid calibration
# ============================================================

def apply_calibration(probabilities):
    eps = 1e-7

    clipped = np.clip(
        probabilities,
        eps,
        1 - eps
    )

    logits = np.log(
        clipped / (1 - clipped)
    ).reshape(-1, 1)

    return calibrator.predict_proba(logits)[:, 1]


calibrated_prob = apply_calibration(raw_prob)

add_check(
    "Original calibration input provenance",
    "REVIEW",
    "Logit input reproduces the convention in scripts "
    "13 and 14. Verify original fitting transformation "
    "in 10b_probability_calibration.py."
)


# ============================================================
# 11. Saved prediction alignment
# ============================================================

if len(saved_predictions) != len(sep):
    add_check(
        "Prediction row count",
        "FAIL",
        "Saved predictions and September data differ."
    )
    raise ValueError(
        "Prediction row count mismatch."
    )

dates_match = np.array_equal(
    sep["FlightDate"].to_numpy(),
    saved_predictions["FlightDate"].to_numpy()
)

targets_match = np.array_equal(
    sep[TARGET].to_numpy(),
    saved_predictions[TARGET].to_numpy()
)

if not dates_match or not targets_match:
    add_check(
        "Prediction row alignment",
        "FAIL",
        "Dates or targets do not match."
    )
    raise ValueError(
        "Saved prediction alignment failed."
    )

add_check(
    "Prediction row alignment",
    "PASS",
    "Dates and target values match row by row. "
    "Full flight-ID alignment is unavailable."
)


saved_prob = saved_predictions[
    "CalibratedProbability"
].to_numpy()

max_diff = float(
    np.max(
        np.abs(
            calibrated_prob - saved_prob
        )
    )
)

prob_match = np.allclose(
    calibrated_prob,
    saved_prob,
    atol=1e-6,
    rtol=0
)

add_check(
    "Prediction reproducibility",
    "PASS" if prob_match else "FAIL",
    f"Maximum probability difference: {max_diff:.10f}"
)


# ============================================================
# 12. Threshold integrity
# ============================================================

regenerated_labels = (
    calibrated_prob >= threshold
).astype(int)

saved_labels = saved_predictions[
    "PredictedOptimized"
].to_numpy()

label_match = np.array_equal(
    regenerated_labels,
    saved_labels
)

add_check(
    "Threshold prediction consistency",
    "PASS" if label_match else "FAIL",
    f"Threshold: {threshold:.6f}"
)


# ============================================================
# 13. External validation comparison
# ============================================================

if EXTERNAL_PRED_PATH.exists():
    external_pred = pd.read_csv(
        EXTERNAL_PRED_PATH
    )

    if (
        len(external_pred) == len(sep)
        and "CalibratedProbability"
        in external_pred.columns
    ):
        external_prob = external_pred[
            "CalibratedProbability"
        ].to_numpy()

        external_match = np.allclose(
            calibrated_prob,
            external_prob,
            atol=1e-6,
            rtol=0
        )

        add_check(
            "Script 13 probability consistency",
            "PASS" if external_match else "FAIL",
            "Compared regenerated probabilities "
            "against external validation output."
        )
    else:
        add_check(
            "Script 13 probability consistency",
            "REVIEW",
            "External prediction columns or "
            "row count are incompatible."
        )
else:
    add_check(
        "Script 13 probability consistency",
        "REVIEW",
        "External prediction CSV not available."
    )


# ============================================================
# 14. Recalculate September metrics
# ============================================================

metrics = calculate_metrics(
    sep[TARGET].to_numpy(),
    calibrated_prob,
    threshold
)

metrics_df = pd.DataFrame([metrics])

metrics_df.insert(
    0,
    "Dataset",
    "September 2026"
)

metrics_df.insert(
    1,
    "Threshold",
    threshold
)

metrics_df.to_csv(
    OUTPUT_DIR / "verified_september_metrics.csv",
    index=False
)

print("\nRECALCULATED SEPTEMBER METRICS")
print("-" * 70)

print(
    metrics_df.round(6).to_string(index=False)
)


# ============================================================
# 15. Artifact hashes
# ============================================================

artifact_paths = [
    AUG_PATH,
    SEP_PATH,
    MODEL_PATH,
    CAL_PATH,
    THRESHOLD_PATH,
    PRED_PATH,
]

hash_rows = []

for path in artifact_paths:
    hash_rows.append({
        "Artifact": str(path.relative_to(ROOT)),
        "SHA256": sha256_file(path)
    })

hash_df = pd.DataFrame(hash_rows)

hash_df.to_csv(
    OUTPUT_DIR / "artifact_hashes.csv",
    index=False
)

add_check(
    "Artifact fingerprints",
    "PASS",
    "SHA256 fingerprints recorded."
)


# ============================================================
# 16. Save audit summary
# ============================================================

audit_df = pd.DataFrame(checks)

audit_df.to_csv(
    OUTPUT_DIR / "integrity_audit.csv",
    index=False
)

status_counts = (
    audit_df["Status"]
    .value_counts()
    .to_dict()
)

summary = {
    "project": "AeroPulse",
    "audit": "Model Integrity",
    "august_rows": int(len(aug)),
    "september_rows": int(len(sep)),
    "threshold": threshold,
    "status_counts": status_counts,
    "metrics": metrics,
    "overall_status": (
        "FAIL"
        if "FAIL" in status_counts
        else "REVIEW"
        if "REVIEW" in status_counts
        else "PASS"
    )
}

with open(
    OUTPUT_DIR / "integrity_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\nAUDIT SUMMARY")
print("=" * 70)

print(audit_df["Status"].value_counts())

print("\nOverall status:", summary["overall_status"])

print("\nSaved outputs:")
print(" - integrity_audit.csv")
print(" - integrity_summary.json")
print(" - verified_september_metrics.csv")
print(" - category_drift_audit.csv")
print(" - artifact_hashes.csv")

print("\nAeroPulse | Model Integrity Audit Completed!")
