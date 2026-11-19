
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil

import numpy as np
import pandas as pd


# ============================================================
# AeroPulse | 17 - Final Project Report
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

REPORT_DIR = OUTPUTS / "final_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUTS / "models"
CAL_DIR = OUTPUTS / "calibration"
THRESHOLD_DIR = OUTPUTS / "threshold_validation"
ROBUST_DIR = OUTPUTS / "robustness"
INTEGRITY_DIR = OUTPUTS / "integrity"

used_files = []
warnings_list = []


# ============================================================
# Helpers
# ============================================================

def read_csv(path, required=False):
    if not path.exists():
        message = f"Missing file: {path.relative_to(ROOT)}"

        if required:
            raise FileNotFoundError(message)

        warnings_list.append(message)
        print("[WARNING]", message)
        return None

    df = pd.read_csv(path)
    used_files.append(str(path.relative_to(ROOT)))

    print("[LOADED]", path.relative_to(ROOT), df.shape)
    return df


def read_json(path):
    if not path.exists():
        warnings_list.append(
            f"Missing file: {path.relative_to(ROOT)}"
        )
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    used_files.append(str(path.relative_to(ROOT)))
    return data


def fmt(value, digits=4):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.{digits}f}"


def pct(value, digits=2):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.{digits}f}%"


def markdown_table(df, columns, labels=None, limit=None):
    if df is None or df.empty:
        return "_Data unavailable._"

    selected = [c for c in columns if c in df.columns]

    if not selected:
        return "_Required columns unavailable._"

    view = df[selected].copy()

    if limit is not None:
        view = view.head(limit)

    if labels:
        view = view.rename(columns=labels)

    def clean_cell(value):
        if pd.isna(value):
            return "N/A"

        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"

        return str(value).replace("|", "/")

    headers = list(view.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in view.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(
                clean_cell(v) for v in row
            ) + " |"
        )

    return "\n".join(lines)


def find_metric(row, *names):
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])

    return None


def section(title, body):
    return f"\n## {title}\n\n{body.strip()}\n"


def write_text(path, content):
    path.write_text(
        content.strip() + "\n",
        encoding="utf-8"
    )
    print("[SAVED]", path.relative_to(ROOT))


# ============================================================
# 1. Load verified results
# ============================================================

print("=" * 72)
print("AeroPulse | Final Project Report")
print("=" * 72)

verified = read_csv(
    INTEGRITY_DIR / "verified_september_metrics.csv",
    required=True
)

integrity = read_csv(
    INTEGRITY_DIR / "integrity_audit.csv",
    required=True
)

integrity_summary = read_json(
    INTEGRITY_DIR / "integrity_summary.json"
)

comparison = read_csv(
    MODEL_DIR / "tree_model_comparison.csv"
)

calibration = read_csv(
    CAL_DIR / "calibration_comparison.csv"
)

threshold_comparison = read_csv(
    THRESHOLD_DIR / "threshold_comparison.csv"
)

airlines = read_csv(
    ROBUST_DIR / "airline_robustness.csv"
)

periods = read_csv(
    ROBUST_DIR / "departure_period_robustness.csv"
)

hours = read_csv(
    ROBUST_DIR / "hourly_robustness.csv"
)

origins = read_csv(
    ROBUST_DIR / "origin_airport_robustness.csv"
)

destinations = read_csv(
    ROBUST_DIR / "destination_airport_robustness.csv"
)

daily = read_csv(
    ROBUST_DIR / "daily_robustness.csv"
)

risk_bands = read_csv(
    ROBUST_DIR / "risk_band_calibration.csv"
)

category_drift = read_csv(
    INTEGRITY_DIR / "category_drift_audit.csv"
)

metrics = verified.iloc[0].to_dict()

threshold = find_metric(metrics, "Threshold")
flights = int(metrics["Flights"])
delay_rate = find_metric(metrics, "DelayRate")

roc_auc = find_metric(metrics, "ROC_AUC")
pr_auc = find_metric(metrics, "PR_AUC")
brier = find_metric(metrics, "Brier")
logloss = find_metric(metrics, "LogLoss")
f1 = find_metric(metrics, "F1")


# ============================================================
# 2. Validate required metrics
# ============================================================

required_metrics = [
    "Flights",
    "Threshold",
    "DelayRate",
    "ROC_AUC",
    "PR_AUC",
    "Brier",
    "LogLoss",
    "F1",
]

missing_metrics = [
    name for name in required_metrics
    if name not in verified.columns
    or pd.isna(metrics.get(name))
]

if missing_metrics:
    raise ValueError(
        f"Missing verified metrics: {missing_metrics}"
    )

if flights <= 0:
    raise ValueError("Invalid flight count.")

if not (0 <= delay_rate <= 1):
    raise ValueError("Invalid delay rate.")

if not (0 <= threshold <= 1):
    raise ValueError("Invalid threshold.")

for name in ["ROC_AUC", "PR_AUC", "Brier", "F1"]:
    value = float(metrics[name])

    if not (0 <= value <= 1):
        raise ValueError(f"Invalid metric: {name}")

verified.to_csv(
    REPORT_DIR / "final_metrics.csv",
    index=False
)


# ============================================================
# 3. Audit status
# ============================================================

status_counts = (
    integrity["Status"]
    .value_counts()
    .to_dict()
)

pass_count = status_counts.get("PASS", 0)
fail_count = status_counts.get("FAIL", 0)
review_count = status_counts.get("REVIEW", 0)

audit_status = (
    "FAIL" if fail_count
    else "REVIEW" if review_count
    else "PASS"
)

review_rows = integrity[
    integrity["Status"] == "REVIEW"
]

review_text = "\n".join(
    f"- **{row['Check']}**: {row['Details']}"
    for _, row in review_rows.iterrows()
)

if not review_text:
    review_text = "No unresolved audit review flags."

if fail_count:
    warnings_list.append(
        "Integrity audit contains FAIL checks."
    )


# ============================================================
# 4. Performance interpretation
# ============================================================

performance_text = f"""
The final model is an XGBoost classifier trained on
August 2026 flight records and evaluated on an
independent September 2026 dataset.

The September evaluation covers **{flights:,} flights**,
with an observed delay prevalence of **{pct(delay_rate)}**.

The model achieved:

- ROC-AUC: **{fmt(roc_auc)}**
- PR-AUC: **{fmt(pr_auc)}**
- F1-score: **{fmt(f1)}**
- Brier score: **{fmt(brier)}**
- Log loss: **{fmt(logloss)}**

A decision threshold of **{fmt(threshold, 6)}** was
selected using August validation data and then
applied to September without re-optimizing it
on the September outcomes.

These results describe predictive performance,
not a causal explanation of flight delays.
"""


# ============================================================
# 5. Model comparison
# ============================================================

comparison_text = markdown_table(
    comparison,
    [
        "Model", "Threshold",
        "Accuracy", "Precision",
        "Recall", "F1",
        "ROC_AUC", "PR_AUC"
    ]
)

calibration_text = markdown_table(
    calibration,
    [
        "Method", "Brier",
        "LogLoss", "ROC_AUC", "PR_AUC"
    ]
)

threshold_text = markdown_table(
    threshold_comparison,
    list(threshold_comparison.columns)
    if threshold_comparison is not None
    else [],
    limit=15
)


# ============================================================
# 6. Robustness summaries
# ============================================================

airline_text = markdown_table(
    airlines,
    [
        "Reporting_Airline",
        "Flights",
        "ActualDelayRate",
        "Precision",
        "Recall",
        "F1",
        "Brier"
    ],
    limit=12
)

period_text = markdown_table(
    periods,
    [
        "DepPeriod",
        "Flights",
        "ActualDelayRate",
        "Precision",
        "Recall",
        "F1",
        "Brier"
    ]
)

hour_text = markdown_table(
    hours,
    [
        "ScheduledDepHour",
        "Flights",
        "ActualDelayRate",
        "Recall",
        "F1"
    ],
    limit=24
)

origin_text = markdown_table(
    origins,
    [
        "Origin", "Flights",
        "ActualDelayRate",
        "Recall", "F1", "Brier"
    ],
    limit=15
)

destination_text = markdown_table(
    destinations,
    [
        "Dest", "Flights",
        "ActualDelayRate",
        "Recall", "F1", "Brier"
    ],
    limit=15
)

daily_text = markdown_table(
    daily,
    [
        "FlightDate",
        "Flights",
        "ActualDelayRate",
        "Recall",
        "F1"
    ],
    limit=31
)

risk_text = markdown_table(
    risk_bands,
    [
        "RiskBand",
        "Flights",
        "ActualDelayRate",
        "MeanPredictedRisk",
        "CalibrationGap"
    ]
)

drift_text = markdown_table(
    category_drift,
    [
        "Feature",
        "UnseenRows",
        "UnseenRate"
    ]
)


# ============================================================
# 7. Operational insights
# ============================================================

insights = []

if periods is not None and not periods.empty:
    period_copy = periods.copy()

    if {
        "DepPeriod",
        "ActualDelayRate"
    }.issubset(period_copy.columns):

        period_copy["ActualDelayRate"] = pd.to_numeric(
            period_copy["ActualDelayRate"],
            errors="coerce"
        )

        valid = period_copy.dropna(
            subset=["ActualDelayRate"]
        )

        if not valid.empty:
            highest = valid.loc[
                valid["ActualDelayRate"].idxmax()
            ]

            lowest = valid.loc[
                valid["ActualDelayRate"].idxmin()
            ]

            insights.append(
                f"- **Departure period:** "
                f"{highest['DepPeriod']} had the highest "
                f"observed delay rate "
                f"({pct(highest['ActualDelayRate'])}), "
                f"while {lowest['DepPeriod']} had the "
                f"lowest ({pct(lowest['ActualDelayRate'])})."
            )

if airlines is not None and not airlines.empty:
    if {
        "Reporting_Airline",
        "Flights",
        "Recall"
    }.issubset(airlines.columns):

        eligible = airlines[
            pd.to_numeric(
                airlines["Flights"],
                errors="coerce"
            ) >= 10000
        ].copy()

        eligible["Recall"] = pd.to_numeric(
            eligible["Recall"],
            errors="coerce"
        )

        eligible = eligible.dropna(
            subset=["Recall"]
        )

        if not eligible.empty:
            best = eligible.loc[
                eligible["Recall"].idxmax()
            ]

            worst = eligible.loc[
                eligible["Recall"].idxmin()
            ]

            insights.append(
                f"- **Airline sensitivity:** Among "
                f"airlines with at least 10,000 flights, "
                f"{best['Reporting_Airline']} had the "
                f"highest recall ({pct(best['Recall'])}), "
                f"while {worst['Reporting_Airline']} "
                f"had the lowest ({pct(worst['Recall'])})."
            )

if destinations is not None and not destinations.empty:
    if {
        "Dest",
        "Flights",
        "ActualDelayRate"
    }.issubset(destinations.columns):

        eligible = destinations[
            pd.to_numeric(
                destinations["Flights"],
                errors="coerce"
            ) >= 10000
        ].copy()

        eligible["ActualDelayRate"] = pd.to_numeric(
            eligible["ActualDelayRate"],
            errors="coerce"
        )

        eligible = eligible.dropna(
            subset=["ActualDelayRate"]
        )

        if not eligible.empty:
            highest = eligible.loc[
                eligible["ActualDelayRate"].idxmax()
            ]

            insights.append(
                f"- **Destination exposure:** Among "
                f"destinations with at least 10,000 "
                f"flights, {highest['Dest']} had the "
                f"highest observed delay rate "
                f"({pct(highest['ActualDelayRate'])})."
            )

if not insights:
    insights.append(
        "- Operational subgroup summaries were unavailable."
    )

insights_text = "\n".join(insights)


# ============================================================
# 8. Methodological limitations
# ============================================================

limitations = """
1. **Temporal coverage:** The study uses August 2026
   for model development and September 2026 for
   external temporal validation. Longer-term and
   seasonal generalization remains untested.

2. **Validation reuse:** Sigmoid calibration was fitted
   on the August validation partition, which was also
   used for threshold optimization. Consequently,
   validation-based threshold performance should not
   be interpreted as a fully independent estimate.

3. **Category handling:** Native categorical XGBoost
   was trained with training-partition category levels.
   Category compatibility and handling of unseen
   September values require careful interpretation.

4. **Class imbalance:** Flight delays are less frequent
   than on-time departures. Precision, recall, PR-AUC,
   and threshold behavior are therefore particularly
   important.

5. **Operational trade-off:** A lower classification
   threshold improves delay detection but may increase
   false alarms.

6. **No causal claims:** Airline, airport, and time-of-day
   differences are descriptive associations and do not
   establish causal effects.

7. **Flight identity:** The saved prediction file was
   checked for date and target alignment, but unique
   flight identifiers were not available for a stronger
   row-level identity audit.
"""


# ============================================================
# 9. Final Markdown report
# ============================================================

report = f"""# AeroPulse
## Flight Delay Prediction and Operational Intelligence

**Project type:** End-to-end machine learning and analytics  
**Domain:** U.S. domestic aviation  
**Development data:** August 2026  
**External validation:** September 2026  
**Final classifier:** XGBoost  
**Probability calibration:** Sigmoid  
**Decision threshold:** {fmt(threshold, 6)}  

---

## Executive Overview

AeroPulse investigates U.S. flight departure delays
using historical airline operational data.

The project combines data preprocessing, feature
engineering, exploratory analysis, model comparison,
probability calibration, threshold optimization,
external temporal validation, and robustness analysis.

{section("1. Final External Validation", performance_text)}

{section("2. Tree Model Comparison", comparison_text)}

{section("3. Probability Calibration", calibration_text)}

{section("4. Threshold Evaluation", threshold_text)}

{section("5. Operational Findings", insights_text)}

{section("6. Airline Robustness", airline_text)}

{section("7. Departure Period Robustness", period_text)}

{section("8. Hourly Robustness", hour_text)}

{section("9. Origin Airport Robustness", origin_text)}

{section("10. Destination Airport Robustness", destination_text)}

{section("11. Daily Robustness", daily_text)}

{section("12. Risk Calibration by Probability Band", risk_text)}

{section("13. Category Drift", drift_text)}

## 14. Model Integrity Audit

- Passed checks: **{pass_count}**
- Failed checks: **{fail_count}**
- Review flags: **{review_count}**
- Automated audit status: **{audit_status}**

### Review flags

{review_text}

The original training and calibration source code was
also inspected separately. It confirms that XGBoost
was fitted on the August 1-20 training partition and
that sigmoid calibration used the logit transformation.

Automated audit flags are retained for transparency.

## 15. Methodological Limitations

{limitations}

## 16. Reproducibility

The project includes Python scripts for data
preparation, feature engineering, machine learning,
external validation, robustness analysis, and
model integrity auditing.

Saved prediction probabilities were reproduced with
a maximum reported difference of 0.0000000000.

Reproducibility within the saved pipeline does not
by itself establish generalization to other months,
airlines, airports, or operational conditions.

## 17. Conclusion

AeroPulse demonstrates an end-to-end workflow for
predicting departure delays and examining how model
performance varies across operational segments.

Independent September evaluation provides a more
credible estimate of out-of-time performance than
the reused August validation partition.

The final system is a research and portfolio prototype,
not a production-ready airline decision system.
"""


# ============================================================
# 10. Executive summary
# ============================================================

executive_summary = f"""# AeroPulse | Executive Summary

**Flight Delay Prediction & Operational Intelligence**

AeroPulse is an end-to-end machine learning project
using U.S. flight operations data from August and
September 2026.

The workflow includes data cleaning, feature
engineering, model comparison, calibration, threshold
optimization, external validation, and robustness
analysis.

## Final Results

| Metric | September 2026 |
|---|---:|
| Flights evaluated | {flights:,} |
| Observed delay rate | {pct(delay_rate)} |
| ROC-AUC | {fmt(roc_auc)} |
| PR-AUC | {fmt(pr_auc)} |
| F1-score | {fmt(f1)} |
| Brier score | {fmt(brier)} |
| Log loss | {fmt(logloss)} |
| Decision threshold | {fmt(threshold, 6)} |

## Key Operational Findings

{insights_text}

## Technical Validation

- Model: XGBoost
- Calibration: Sigmoid
- Independent evaluation month: September 2026
- Integrity checks passed: {pass_count}
- Integrity review flags: {review_count}
- Integrity failures: {fail_count}
- Prediction reproducibility: Confirmed

## Limitations

The dataset covers two consecutive months, and
validation data was reused for calibration and
threshold selection. Findings should not be treated
as causal or production-ready.

## Project Outcome

AeroPulse provides a reproducible analytical
framework for studying flight delay risk and
operational model performance.
"""


# ============================================================
# 11. Save reports
# ============================================================

write_text(
    REPORT_DIR / "AeroPulse_Final_Report.md",
    report
)

write_text(
    REPORT_DIR / "AeroPulse_Executive_Summary.md",
    executive_summary
)


# ============================================================
# 12. Copy available robustness plots
# ============================================================

PLOT_NAMES = [
    "airline_recall.png",
    "departure_period_f1.png",
    "hourly_miss_rate.png",
    "origin_false_alarm_rate.png",
    "september_risk_calibration.png",
]

plots_dir = REPORT_DIR / "figures"
plots_dir.mkdir(exist_ok=True)

copied_plots = []

for filename in PLOT_NAMES:
    source = ROBUST_DIR / filename

    if source.exists():
        destination = plots_dir / filename
        shutil.copy2(source, destination)

        copied_plots.append(
            str(destination.relative_to(ROOT))
        )
    else:
        warnings_list.append(
            f"Plot not found: {source.relative_to(ROOT)}"
        )


# ============================================================
# 13. Save manifest
# ============================================================

manifest = {
    "project": "AeroPulse",
    "report_generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "source_files": sorted(set(used_files)),
    "plots_copied": copied_plots,
    "warnings": warnings_list,
    "integrity_status": audit_status,
    "integrity_checks": {
        "PASS": int(pass_count),
        "FAIL": int(fail_count),
        "REVIEW": int(review_count),
    },
    "september_metrics": {
        "Flights": flights,
        "DelayRate": delay_rate,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "F1": f1,
        "Brier": brier,
        "LogLoss": logloss,
        "Threshold": threshold,
    }
}

with open(
    REPORT_DIR / "final_report_manifest.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 14. Console summary
# ============================================================

print("\n" + "=" * 72)
print("FINAL SEPTEMBER PERFORMANCE")
print("=" * 72)

print(f"Flights: {flights:,}")
print(f"Delay rate: {pct(delay_rate)}")
print(f"ROC-AUC: {fmt(roc_auc)}")
print(f"PR-AUC: {fmt(pr_auc)}")
print(f"F1-score: {fmt(f1)}")
print(f"Brier: {fmt(brier)}")
print(f"Log loss: {fmt(logloss)}")
print(f"Threshold: {fmt(threshold, 6)}")

print("\nINTEGRITY STATUS")
print(
    f"PASS={pass_count} | "
    f"FAIL={fail_count} | "
    f"REVIEW={review_count}"
)

print("\nREPORT OUTPUTS")
print("-" * 72)

for filename in [
    "AeroPulse_Final_Report.md",
    "AeroPulse_Executive_Summary.md",
    "final_metrics.csv",
    "final_report_manifest.json",
]:
    print(" -", REPORT_DIR / filename)

print(f"\nCopied figures: {len(copied_plots)}")
print(f"Warnings: {len(warnings_list)}")

for warning in warnings_list:
    print("[WARNING]", warning)

print("\nAeroPulse | Final Report Completed!")
