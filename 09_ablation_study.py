
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

# ============================================
# AeroPulse | 09 - Ablation Study
# ============================================

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "data" / "processed"
    / "aeropulse_model_features.csv",
    parse_dates=["FlightDate"]
)

time_features = [
    "ScheduledDepHour",
    "HourSin",
    "HourCos",
    "DepPeriod"
]

other_features = [
    "DayOfWeek",
    "IsWeekend",
    "LogDistance",
    "Reporting_Airline",
    "Origin",
    "Dest"
]

experiments = {
    "Full Model": time_features + other_features,
    "No Time Features": other_features,
    "Time Only": time_features
}

categorical = [
    "DepPeriod",
    "Reporting_Airline",
    "Origin",
    "Dest"
]

train = df[df.FlightDate <= "2026-08-20"]
val = df[df.FlightDate.between(
    "2026-08-21", "2026-08-25"
)]
test = df[df.FlightDate >= "2026-08-26"]

def prepare_data(features):
    X_train = train[features].copy()
    X_val = val[features].copy()
    X_test = test[features].copy()

    for col in categorical:
        if col not in features:
            continue

        dtype = pd.CategoricalDtype(
            categories=sorted(
                X_train[col].dropna().unique()
            )
        )

        for X in [X_train, X_val, X_test]:
            X[col] = X[col].astype(dtype)

    return X_train, X_val, X_test


results = []

for name, features in experiments.items():

    print(f"\nTraining: {name}")

    X_train, X_val, X_test = prepare_data(
        features
    )

    model = XGBClassifier(
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

    model.fit(
        X_train,
        train["Delayed15"]
    )

    val_probs = model.predict_proba(
        X_val
    )[:, 1]

    thresholds = np.arange(
        0.10, 0.81, 0.02
    )

    f1_scores = [
        f1_score(
            val["Delayed15"],
            val_probs >= t,
            zero_division=0
        )
        for t in thresholds
    ]

    best_threshold = float(
        thresholds[np.argmax(f1_scores)]
    )

    test_probs = model.predict_proba(
        X_test
    )[:, 1]

    test_preds = (
        test_probs >= best_threshold
    ).astype(int)

    results.append({
        "Experiment": name,
        "Features": len(features),
        "Threshold": round(best_threshold, 2),
        "ROC_AUC": roc_auc_score(
            test["Delayed15"], test_probs
        ),
        "PR_AUC": average_precision_score(
            test["Delayed15"], test_probs
        ),
        "F1": f1_score(
            test["Delayed15"], test_preds
        )
    })

results_df = pd.DataFrame(results)

print("\nABLATION STUDY RESULTS")
print(
    results_df.round(4).to_string(index=False)
)

results_df.to_csv(
    OUT / "ablation_results.csv",
    index=False
)

print("\nAeroPulse | Ablation Study Completed!")
