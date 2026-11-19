
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# AeroPulse | 08 - Feature Importance
# ==============================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "outputs" / "models"
    / "xgboost.joblib"
)

OUTPUT_DIR = ROOT / "outputs" / "explainability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load trained XGBoost model
model = joblib.load(MODEL_PATH)

# 2. Extract gain-based importance
booster = model.get_booster()

importance = booster.get_score(
    importance_type="gain"
)

importance_df = pd.DataFrame(
    importance.items(),
    columns=["Feature", "Gain"]
)

importance_df = importance_df.sort_values(
    "Gain", ascending=False
)

# 3. Normalize importance
importance_df["ImportancePct"] = (
    importance_df["Gain"]
    / importance_df["Gain"].sum()
    * 100
)

print("\nAEROPULSE - FEATURE IMPORTANCE")
print("=" * 50)
print(
    importance_df.round(3).to_string(index=False)
)

# 4. Visualization
sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 7))

sns.barplot(
    data=importance_df,
    x="ImportancePct",
    y="Feature",
    color="steelblue"
)

plt.title(
    "AeroPulse - XGBoost Feature Importance"
)
plt.xlabel("Normalized Mean Gain (%)")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "xgboost_feature_importance.png",
    dpi=300
)

plt.show()

# 5. Save results
importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nFeature Importance Analysis Completed!")
