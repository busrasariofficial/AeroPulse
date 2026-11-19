
# AeroPulse ✈️
### U.S. Flight Delay Prediction & Operational Intelligence

An end-to-end data science project combining statistical analysis, machine learning, explainable AI, probability calibration, and temporal validation to investigate flight departure delays across the United States.

**Tech Stack:** Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn

---

## Project Overview

Flight delays create operational challenges for airlines, airports, and passengers.

AeroPulse investigates whether departure delays can be predicted using historical flight schedules, airline information, airport characteristics, and engineered temporal features.

The project goes beyond model training by evaluating probability calibration, decision thresholds, out-of-time generalization, and performance differences across operational segments.

### Research Questions

- Can scheduled flight information predict departure delays of 15 minutes or more?
- How do Logistic Regression, Random Forest, and XGBoost compare?
- Which features contribute most to delay predictions?
- Can probability calibration improve estimated delay risk?
- Does a model developed on August data generalize to September?
- How does predictive performance vary by airline, airport, and departure time?

---

## Dataset

**Source:** U.S. Department of Transportation, Bureau of Transportation Statistics (BTS)

**Dataset:** Airline On-Time Performance Data

**Period:** August–September 2026

| Dataset | Flights | Purpose |
|---|---:|---|
| August 2026 | 594,207 | Model development and internal evaluation |
| September 2026 | 563,591 | External temporal validation |
| **Total** | **1,157,798** | **Completed flights analyzed** |

The modeling datasets exclude cancelled and diverted flights.

### Target Variable

`Delayed15`

- `0`: Departure delay below 15 minutes
- `1`: Departure delay of 15 minutes or more

### Model Features

**Temporal features**
- Day of week
- Weekend indicator
- Scheduled departure hour
- Departure period
- Cyclical hour encoding (`HourSin`, `HourCos`)

**Flight characteristics**
- Reporting airline
- Origin airport
- Destination airport
- Log-transformed flight distance

The model uses scheduled and historical attributes rather than actual departure or arrival delay values as predictors.

---

## Methodology

The project follows a chronological machine learning workflow.

### 1. Data Preparation

- Raw BTS data inspection
- Data cleaning and feature selection
- Missing-value diagnostics
- Flight status analysis
- Feature engineering

### 2. Statistical and Exploratory Analysis

- Delay distributions
- Airline-level delay patterns
- Airport-level statistics
- Hourly departure patterns
- Flight distance analysis
- Statistical comparisons

### 3. Machine Learning

Three classification approaches were investigated:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost was selected for subsequent calibration and external validation.

### 4. Explainability

Feature importance, SHAP analysis, and ablation experiments were used to examine model behavior.

### 5. Probability Calibration

The project compared:

- Original XGBoost probabilities
- Sigmoid calibration
- Isotonic calibration

Sigmoid calibration was selected for the final evaluation workflow.

### 6. Threshold Optimization

A classification threshold was selected using August validation data.

**Final decision threshold: `0.253331`**

The selected threshold was subsequently applied to September data without re-optimizing it on September outcomes.

### 7. External Temporal Validation

The model was developed using August 2026 data and evaluated on September 2026 data.

This provides an out-of-time evaluation rather than relying exclusively on a random train-test split.

### 8. Robustness and Error Analysis

Performance was examined across:

- Airlines
- Departure periods
- Scheduled departure hours
- Origin airports
- Destination airports
- Flight dates
- Predicted probability bands

---

## Chronological Data Split

| Partition | Dates | Purpose |
|---|---|---|
| Training | August 1–20, 2026 | Model fitting |
| Validation | August 21–25, 2026 | Calibration and threshold selection |
| Internal test | August 26–31, 2026 | Internal evaluation |
| External validation | September 1–30, 2026 | Out-of-time evaluation |

The August validation partition was reused for calibration and threshold selection. Consequently, validation-based threshold results are not fully independent.

September serves as the external evaluation period.

---

## Final Model Performance

### September 2026 — External Validation

The final calibrated XGBoost model was evaluated on **563,591 completed flights**.

| Metric | Result |
|---|---:|
| ROC-AUC | **0.7142** |
| PR-AUC | **0.4249** |
| F1-score | **0.4588** |
| Precision | 0.3988 |
| Recall | 0.5401 |
| Accuracy | 0.7094 |
| Brier Score | **0.1577** |
| Log Loss | 0.4861 |
| Decision Threshold | 0.253331 |

The observed September departure delay rate was **22.81%**.

### Confusion Matrix

| | Predicted On Time | Predicted Delayed |
|---|---:|---:|
| Actually On Time | 330,366 | 104,667 |
| Actually Delayed | 59,120 | 69,438 |

At the selected threshold, the model detected approximately **54% of delayed flights**, while generating a substantial number of false alarms.

This illustrates the trade-off between delay detection and prediction precision.

---

## Key Findings

### 1. Departure Time Is an Important Predictor

Feature importance analysis identified temporal information as particularly influential.

The XGBoost gain-based analysis highlighted:

- `HourSin`: approximately 45.24%
- `DepPeriod`: approximately 27.61%

These importance values describe the fitted model and should not be interpreted as causal effects.

### 2. Delay Rates Vary by Departure Period

| Departure Period | Observed Delay Rate | Model Recall |
|---|---:|---:|
| Morning | 12.95% | 10.50% |
| Afternoon | 27.37% | 56.15% |
| Evening | 34.62% | 80.48% |
| Night | 22.23% | 57.12% |

Evening flights exhibited the highest observed delay rate.

The model detected a much larger share of evening delays than morning delays at the chosen decision threshold.

### 3. Performance Varies Across Airlines

Selected September results:

| Airline | Delay Rate | Recall |
|---|---:|---:|
| Southwest (WN) | 31.64% | 75.02% |
| Delta (DL) | 16.94% | 30.33% |
| American (AA) | 26.97% | 56.50% |
| United (UA) | 20.04% | 46.66% |
| JetBlue (B6) | 25.80% | 76.43% |

These differences highlight the importance of subgroup-level evaluation.

### 4. Calibration Improves Probability Quality

The calibrated model achieved a September Brier score of **0.1577**.

Risk-band analysis nevertheless indicated that predicted probabilities were slightly lower than observed delay frequencies across several populated risk groups.

Calibration should therefore be interpreted as an improvement in overall probability quality, not as perfect calibration for every subgroup.

### 5. Temporal Validation Matters

The project evaluated model performance on a subsequent month rather than relying solely on internal test results.

This provides evidence of short-term temporal generalization, although longer-term and seasonal performance remain untested.

---

## Visualizations

### Departure Period Robustness

![Departure Period F1](outputs/robustness/departure_period_f1.png)

### Airline Recall

![Airline Recall](outputs/robustness/airline_recall.png)

### Hourly Miss Rate

![Hourly Miss Rate](outputs/robustness/hourly_miss_rate.png)

### Airport False Alarm Analysis

![Origin Airport False Alarm Rate](outputs/robustness/origin_false_alarm_rate.png)

### Probability Calibration

![September Risk Calibration](outputs/robustness/september_risk_calibration.png)

---

## Model Integrity & Reproducibility

A dedicated integrity audit was performed to assess:

- Required artifacts and feature schemas
- Missing values and target validity
- Temporal partition boundaries
- Engineered feature consistency
- Category drift
- Probability reproducibility
- Threshold consistency
- Artifact fingerprints

### Audit Results

| Status | Checks |
|---|---:|
| PASS | 22 |
| FAIL | 0 |
| REVIEW | 3 |

The regenerated September probabilities matched the saved predictions with a reported maximum absolute difference of **0.0000000000**.

The original training and calibration scripts were also reviewed to confirm the training partition and sigmoid logit transformation.

The automated audit retains three review flags for transparency. Exact category-handling behavior and remaining methodological limitations should be considered when interpreting reproducibility.

---

## Project Structure

```text
AeroPulse/
|
|-- src/
|   |-- 01_initial_eda.py
|   |-- 02_data_cleaning.py
|   |-- 03_exploratory_analysis.py
|   |-- 04_statistical_analysis.py
|   |-- 04b_missingness_diagnostics.py
|   |-- 05_feature_engineering.py
|   |-- 06_logistic_regression.py
|   |-- 06b_threshold_optimization.py
|   |-- 07_tree_models.py
|   |-- 08_feature_importance.py
|   |-- 08b_shap_analysis.py
|   |-- 09_ablation_study.py
|   |-- 10_model_calibration.py
|   |-- 10b_probability_calibration.py
|   |-- 11_september_validation.py
|   |-- 12_september_preprocessing.py
|   |-- 13_external_validation.py
|   |-- 14_threshold_validation.py
|   |-- 15_model_robustness.py
|   |-- 16_model_integrity.py
|   `-- 17_final_report.py
|
|-- data/
|   `-- processed/
|
|-- outputs/
|   |-- calibration/
|   |-- experiments/
|   |-- explainability/
|   |-- external_validation/
|   |-- figures/
|   |-- final_report/
|   |-- integrity/
|   |-- models/
|   |-- robustness/
|   |-- statistics/
|   |-- tables/
|   `-- threshold_validation/
|
|-- README.md
|-- requirements.txt
`-- .gitignore
```

Large datasets and serialized model artifacts are excluded from version control.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/busrasariofficial/AeroPulse.git
cd AeroPulse
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The original BTS datasets must be obtained separately and placed in the expected data directories before running the pipeline.

The complete pipeline also requires generating intermediate artifacts in the intended script order.

---

## Limitations

- The project covers two consecutive months, limiting conclusions about seasonality.
- The validation partition was reused for probability calibration and threshold optimization.
- Class imbalance affects precision, recall, and threshold selection.
- Performance varies across airlines, airports, and departure periods.
- Some September airport categories were not present in the August reference data.
- The model uses historical and scheduled attributes; it does not incorporate real-time weather or air traffic conditions.
- The model is a research prototype rather than a production-ready airline decision system.
- Observed patterns do not establish causal relationships.

---

## Future Improvements

Potential extensions include:

- Multi-month and multi-year temporal validation
- Weather and airport congestion features
- More detailed subgroup calibration
- Model monitoring and drift detection
- Deployment as a prediction service

These extensions are outside the scope of the current project.

---

## Reports

Detailed project results are available in:

- [Final Technical Report](outputs/final_report/AeroPulse_Final_Report.md)
- [Executive Summary](outputs/final_report/AeroPulse_Executive_Summary.md)
- [Verified September Metrics](outputs/final_report/final_metrics.csv)

---

## Author

**Büşra Sarı**

Statistics Graduate | Data Analytics & Machine Learning

[GitHub Profile](https://github.com/busrasariofficial)

---

*Developed as an independent data science portfolio project.*
