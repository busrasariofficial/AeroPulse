
from pathlib import Path
import pandas as pd

# =====================================================
# AeroPulse | 01 - Initial Data Exploration
# Dataset: BTS On-Time Performance | August 2026
# =====================================================

# 1. FILE PATH
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / (
    "On_Time_Reporting_Carrier_On_Time_Performance_"
    "(1987_present)_2026_8.csv"
)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"CSV dosyasi bulunamadi: {DATA_PATH}"
    )

# 2. LOAD DATA
print("Loading AeroPulse dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

# Remove unnecessary columns
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

print("\nDataset loaded successfully!")

# 3. DATASET OVERVIEW
print("\n" + "=" * 50)
print("1. DATASET SHAPE")
print("=" * 50)

print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# 4. FIRST FIVE ROWS
print("\n" + "=" * 50)
print("2. FIRST FIVE ROWS")
print("=" * 50)

print(df.head().to_string())

# 5. COLUMN NAMES
print("\n" + "=" * 50)
print("3. COLUMN NAMES")
print("=" * 50)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# 6. DATA TYPES
print("\n" + "=" * 50)
print("4. DATA TYPES")
print("=" * 50)

print(df.dtypes.value_counts())

# 7. MISSING VALUES
print("\n" + "=" * 50)
print("5. MISSING VALUES - TOP 20")
print("=" * 50)

missing_values = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": df.isnull().mean() * 100
})

missing_values = missing_values.sort_values(
    by="Missing Percentage",
    ascending=False
)

print(missing_values.head(20).round(2).to_string())

# 8. DUPLICATE ROWS
print("\n" + "=" * 50)
print("6. DUPLICATE ROWS")
print("=" * 50)

duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count:,}")

# 9. NUMERICAL SUMMARY
print("\n" + "=" * 50)
print("7. NUMERICAL SUMMARY")
print("=" * 50)

print(df.describe().round(2).to_string())

# 10. EXPORT DATA QUALITY REPORT
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

missing_values.to_csv(
    OUTPUT_DIR / "missing_values_report.csv",
    index_label="Column"
)

print("\n" + "=" * 50)
print("AeroPulse | Initial EDA Completed!")
print("=" * 50)
