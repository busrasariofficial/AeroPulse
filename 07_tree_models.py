
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier

# AeroPulse | 07 - Tree Model Comparison

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "models"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "data" / "processed"
    / "aeropulse_model_features.csv",
    parse_dates=["FlightDate"]
)

numeric = [
    "DayOfWeek", "IsWeekend",
    "ScheduledDepHour", "HourSin",
    "HourCos", "LogDistance"
]

categorical = [
    "Reporting_Airline", "Origin",
    "Dest", "DepPeriod"
]

features = numeric + categorical

train = df[df.FlightDate <= "2026-08-20"]
val = df[df.FlightDate.between(
    "2026-08-21", "2026-08-25"
)]
test = df[df.FlightDate >= "2026-08-26"]

X_train, y_train = train[features].copy(), train["Delayed15"]
X_val, y_val = val[features].copy(), val["Delayed15"]
X_test, y_test = test[features].copy(), test["Delayed15"]

# Random Forest pipeline
rf = Pipeline([
    ("prep", ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(
            handle_unknown="ignore"
        ), categorical)
    ])),
    ("model", RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    ))
])

print("Training Random Forest...")
rf.fit(X_train, y_train)

# XGBoost native categorical support
X_train_xgb = X_train.copy()
X_val_xgb = X_val.copy()
X_test_xgb = X_test.copy()

for col in categorical:
    categories = pd.CategoricalDtype(
        categories=sorted(
            X_train[col].dropna().unique()
        )
    )
    for frame in [
        X_train_xgb, X_val_xgb, X_test_xgb
    ]:
        frame[col] = frame[col].astype(categories)

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost...")
xgb.fit(X_train_xgb, y_train)

# Select F1 threshold on validation data
def select_threshold(y, probabilities):
    thresholds = np.arange(0.10, 0.81, 0.02)

    scores = [
        f1_score(
            y, probabilities >= t,
            zero_division=0
        )
        for t in thresholds
    ]

    return float(thresholds[np.argmax(scores)])


def evaluate(name, model, Xv, Xt):
    val_probs = model.predict_proba(Xv)[:, 1]
    test_probs = model.predict_proba(Xt)[:, 1]

    threshold = select_threshold(
        y_val, val_probs
    )

    pred = (test_probs >= threshold).astype(int)

    return {
        "Model": name,
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test, test_probs
        ),
        "PR_AUC": average_precision_score(
            y_test, test_probs
        )
    }


results = [
    evaluate(
        "Random Forest", rf, X_val, X_test
    ),
    evaluate(
        "XGBoost", xgb,
        X_val_xgb, X_test_xgb
    )
]

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    OUT / "tree_model_comparison.csv",
    index=False
)

joblib.dump(rf, OUT / "random_forest.joblib")
joblib.dump(xgb, OUT / "xgboost.joblib")

print("\nAeroPulse | Tree Models Completed!")
