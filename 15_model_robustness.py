
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

# ============================================================
# AeroPulse | 15 - Model Robustness & Error Analysis
# September 2026 | Frozen XGBoost Predictions
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "outputs" / "threshold_validation"
OUTPUT_DIR = ROOT / "outputs" / "robustness"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = (
    DATA_DIR / "aeropulse_2026_09_model_features.csv"
)

PRED_PATH = (
    PRED_DIR / "september_threshold_predictions.csv"
)

THRESHOLD_PATH = (
    PRED_DIR / "calibrated_threshold.joblib"
)

# Minimum segment sizes for stable reporting
MIN_SEGMENT_SIZE = 500
MIN_POSITIVE = 30
MIN_NEGATIVE = 30

print("=" * 65)
print("AeroPulse | Model Robustness & Error Analysis")
print("=" * 65)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

pred = pd.read_csv(PRED_PATH)

print("\nModel dataset:", df.shape)
print("Prediction dataset:", pred.shape)

if len(df) != len(pred):
    raise ValueError(
        "Data and prediction row counts do not match!"
    )

required_prediction_columns = [
    "Delayed15",
    "CalibratedProbability",
    "PredictedOptimized"
]

missing = [
    col for col in required_prediction_columns
    if col not in pred.columns
]

if missing:
    raise ValueError(
        f"Missing prediction columns: {missing}"
    )

# The predictions were saved in the same row order.
# Validate labels and dates before combining.
if not np.array_equal(
    df["Delayed15"].to_numpy(),
    pred["Delayed15"].to_numpy()
):
    raise ValueError(
        "Target mismatch between data and predictions!"
    )

if not np.array_equal(
    pd.to_datetime(df["FlightDate"]).to_numpy(),
    pd.to_datetime(pred["FlightDate"]).to_numpy()
):
    raise ValueError(
        "Flight dates do not match!"
    )

df["Probability"] = pd.to_numeric(
    pred["CalibratedProbability"],
    errors="raise"
)

df["Predicted"] = pred[
    "PredictedOptimized"
].astype(int)

if not df["Probability"].between(0, 1).all():
    raise ValueError("Invalid probability values!")

# ============================================================
# 2. ERROR TYPES
# ============================================================

conditions = [
    (df["Delayed15"] == 1) & (df["Predicted"] == 1),
    (df["Delayed15"] == 0) & (df["Predicted"] == 0),
    (df["Delayed15"] == 0) & (df["Predicted"] == 1),
    (df["Delayed15"] == 1) & (df["Predicted"] == 0)
]

choices = ["TP", "TN", "FP", "FN"]

df["ErrorType"] = np.select(
    conditions,
    choices,
    default="Unknown"
)

print("\nOVERALL ERROR DISTRIBUTION")
print("-" * 65)
print(df["ErrorType"].value_counts())

# ============================================================
# 3. SEGMENT METRICS
# ============================================================

def calculate_segment_metrics(group):

    y = group["Delayed15"].to_numpy()
    p = group["Probability"].to_numpy()
    y_pred = group["Predicted"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y, y_pred, labels=[0, 1]
    ).ravel()

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))

    result = {
        "Flights": len(group),
        "ActualDelayRate": np.mean(y),
        "PredictedRiskRate": np.mean(p),
        "PredictedPositiveRate": np.mean(y_pred),
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(
            y, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y, y_pred, zero_division=0
        ),
        "Brier": brier_score_loss(y, p),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "ActualDelayed": positives,
        "ActualOnTime": negatives,
        "FalseAlarmRate": (
            fp / negatives if negatives else np.nan
        ),
        "MissRate": (
            fn / positives if positives else np.nan
        )
    }

    if (
        positives >= MIN_POSITIVE
        and negatives >= MIN_NEGATIVE
    ):
        result["ROC_AUC"] = roc_auc_score(y, p)
        result["PR_AUC"] = average_precision_score(
            y, p
        )
    else:
        result["ROC_AUC"] = np.nan
        result["PR_AUC"] = np.nan

    return pd.Series(result)


def segment_analysis(column, filename):

    rows = []

    for segment, group in df.groupby(
        column, observed=True
    ):
        if len(group) < MIN_SEGMENT_SIZE:
            continue

        metrics = calculate_segment_metrics(group)
        metrics[column] = segment
        rows.append(metrics)

    result = pd.DataFrame(rows)

    if result.empty:
        print(f"No eligible segments for {column}")
        return result

    result = result[
        [column] + [
            c for c in result.columns
            if c != column
        ]
    ]

    result = result.sort_values(
        "Flights",
        ascending=False
    )

    result.to_csv(
        OUTPUT_DIR / filename,
        index=False
    )

    print(f"\n{column.upper()} ROBUSTNESS")
    print("-" * 65)

    print(
        result[
            [
                column,
                "Flights",
                "ActualDelayRate",
                "Precision",
                "Recall",
                "F1",
                "Brier"
            ]
        ].head(15).round(4).to_string(index=False)
    )

    return result


# ============================================================
# 4. AIRLINE ROBUSTNESS
# ============================================================

airline_results = segment_analysis(
    "Reporting_Airline",
    "airline_robustness.csv"
)

# ============================================================
# 5. DEPARTURE PERIOD ROBUSTNESS
# ============================================================

period_results = segment_analysis(
    "DepPeriod",
    "departure_period_robustness.csv"
)

# ============================================================
# 6. HOURLY ROBUSTNESS
# ============================================================

hour_results = segment_analysis(
    "ScheduledDepHour",
    "hourly_robustness.csv"
)

# ============================================================
# 7. ORIGIN AIRPORT ROBUSTNESS
# ============================================================

origin_results = segment_analysis(
    "Origin",
    "origin_airport_robustness.csv"
)

# ============================================================
# 8. DESTINATION AIRPORT ROBUSTNESS
# ============================================================

destination_results = segment_analysis(
    "Dest",
    "destination_airport_robustness.csv"
)

# ============================================================
# 9. DAILY ROBUSTNESS
# ============================================================

df["FlightDate"] = pd.to_datetime(
    df["FlightDate"]
)

daily_results = segment_analysis(
    "FlightDate",
    "daily_robustness.csv"
)

# ============================================================
# 10. PLOTS
# ============================================================

def save_bar_plot(
    data,
    category,
    metric,
    title,
    filename,
    top_n=15
):
    if data.empty:
        return

    subset = data.sort_values(
        "Flights",
        ascending=False
    ).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        subset[category].astype(str),
        subset[metric]
    )

    ax.set_title(title)
    ax.set_xlabel(category)
    ax.set_ylabel(metric)

    if metric in [
        "Recall",
        "Precision",
        "F1",
        "ActualDelayRate",
        "FalseAlarmRate",
        "MissRate"
    ]:
        ax.set_ylim(0, 1)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=180
    )

    plt.close(fig)


save_bar_plot(
    airline_results,
    "Reporting_Airline",
    "Recall",
    "Recall by Airline",
    "airline_recall.png"
)

save_bar_plot(
    period_results,
    "DepPeriod",
    "F1",
    "F1 by Departure Period",
    "departure_period_f1.png"
)

save_bar_plot(
    hour_results,
    "ScheduledDepHour",
    "MissRate",
    "Miss Rate by Scheduled Departure Hour",
    "hourly_miss_rate.png",
    top_n=24
)

save_bar_plot(
    origin_results,
    "Origin",
    "FalseAlarmRate",
    "False Alarm Rate by Origin Airport",
    "origin_false_alarm_rate.png"
)

# ============================================================
# 11. CALIBRATION BY RISK GROUP
# ============================================================

risk_bins = np.linspace(0, 1, 11)

df["RiskBand"] = pd.cut(
    df["Probability"],
    bins=risk_bins,
    include_lowest=True
)

risk_summary = df.groupby(
    "RiskBand",
    observed=True
).agg(
    Flights=("Delayed15", "size"),
    ActualDelayRate=("Delayed15", "mean"),
    MeanPredictedRisk=("Probability", "mean"),
    ActualDelayed=("Delayed15", "sum")
).reset_index()

risk_summary["CalibrationGap"] = (
    risk_summary["MeanPredictedRisk"]
    - risk_summary["ActualDelayRate"]
)

risk_summary.to_csv(
    OUTPUT_DIR / "risk_band_calibration.csv",
    index=False
)

print("\nRISK BAND CALIBRATION")
print("-" * 65)
print(risk_summary.round(4).to_string(index=False))

# Calibration plot
fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

ax.plot(
    risk_summary["MeanPredictedRisk"],
    risk_summary["ActualDelayRate"],
    marker="o",
    label="September Calibration"
)

ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Observed Delay Rate")
ax.set_title("AeroPulse | September Calibration by Risk Band")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "september_risk_calibration.png",
    dpi=180
)

plt.close(fig)

# ============================================================
# 12. SAVE FULL ERROR ANALYSIS
# ============================================================

df.to_csv(
    OUTPUT_DIR / "september_error_analysis.csv",
    index=False
)

print("\nOUTPUTS SAVED TO:")
print(OUTPUT_DIR)

print("\nAeroPulse | Model Robustness Completed!")
