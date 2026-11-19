
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report
)

# ==================================================
# AeroPulse | 06 - Logistic Regression Baseline
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "aeropulse_model_features.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["FlightDate"] = pd.to_datetime(df["FlightDate"])

# 1. Chronological split
train = df[df["FlightDate"] <= "2026-08-20"].copy()

val = df[
    (df["FlightDate"] >= "2026-08-21")
    & (df["FlightDate"] <= "2026-08-25")
].copy()

test = df[df["FlightDate"] >= "2026-08-26"].copy()

print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)

# 2. Feature selection
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

features = numeric_features + categorical_features
target = "Delayed15"

X_train = train[features]
y_train = train[target]

X_val = val[features]
y_val = val[target]

X_test = test[features]
y_test = test[target]

# 3. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# 4. Logistic Regression Pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        )
    )
])

# 5. Train model
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# 6. Evaluation function
def evaluate(name, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "Dataset": name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y, probabilities),
        "PR_AUC": average_precision_score(
            y, probabilities
        )
    }

    print(f"\n{name.upper()} RESULTS")
    print("-" * 45)

    for key, value in metrics.items():
        if key != "Dataset":
            print(f"{key}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y, predictions, zero_division=0
    ))

    return metrics, probabilities, predictions

# 7. Validation
val_metrics, val_probs, val_preds = evaluate(
    "Validation", X_val, y_val
)

# 8. Test
test_metrics, test_probs, test_preds = evaluate(
    "Test", X_test, y_test
)

# 9. Save metrics
results = pd.DataFrame([
    val_metrics,
    test_metrics
])

results.to_csv(
    OUTPUT_DIR / "logistic_metrics.csv",
    index=False
)

# 10. Confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_preds,
    display_labels=["On Time", "Delayed"],
    cmap="Blues",
    ax=ax
)

ax.set_title("AeroPulse - Logistic Regression")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "logistic_confusion_matrix.png",
    dpi=300
)
plt.show()

# 11. ROC curve
fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    test_probs,
    ax=ax
)

ax.set_title("AeroPulse - Logistic Regression ROC")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "logistic_roc_curve.png",
    dpi=300
)
plt.show()

# 12. Save predictions
predictions_df = test[
    ["FlightDate", "Delayed15"]
].copy()

predictions_df["PredictedProbability"] = test_probs
predictions_df["PredictedClass"] = test_preds

predictions_df.to_csv(
    OUTPUT_DIR / "logistic_test_predictions.csv",
    index=False
)

# 13. Save trained pipeline
joblib.dump(
    model,
    OUTPUT_DIR / "logistic_regression.joblib"
)

print("\nAeroPulse | Baseline Model Completed!")
