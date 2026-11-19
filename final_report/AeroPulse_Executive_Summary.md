# AeroPulse | Executive Summary

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
| Flights evaluated | 563,591 |
| Observed delay rate | 22.81% |
| ROC-AUC | 0.7142 |
| PR-AUC | 0.4249 |
| F1-score | 0.4588 |
| Brier score | 0.1577 |
| Log loss | 0.4861 |
| Decision threshold | 0.253331 |

## Key Operational Findings

- **Departure period:** Evening had the highest observed delay rate (34.62%), while Morning had the lowest (12.95%).
- **Airline sensitivity:** Among airlines with at least 10,000 flights, B6 had the highest recall (76.43%), while MQ had the lowest (27.31%).
- **Destination exposure:** Among destinations with at least 10,000 flights, BOS had the highest observed delay rate (42.51%).

## Technical Validation

- Model: XGBoost
- Calibration: Sigmoid
- Independent evaluation month: September 2026
- Integrity checks passed: 22
- Integrity review flags: 3
- Integrity failures: 0
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
