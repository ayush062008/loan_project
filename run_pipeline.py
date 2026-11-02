"""
Part 1 as a stand-alone script (no web server needed).

    python run_pipeline.py data/loan_approval_sample.csv Loan_Status

Prints the preprocessing report and model metrics, and writes every chart plus
the trained model into ./outputs/
"""
import sys
from pathlib import Path

from mining.ml_pipeline import load_dataset, run_pipeline, suggest_target

path = sys.argv[1] if len(sys.argv) > 1 else "data/loan_approval_sample.csv"
df = load_dataset(path)
target = sys.argv[2] if len(sys.argv) > 2 else suggest_target(df)
res = run_pipeline(df, target, Path("outputs"))

c, m = res["clean"], res["model"]
print(f"\nTarget: {target}  ({res['task']})")
print(f"Rows {c['shape_before'][0]} -> {c['shape_after'][0]}, columns {c['shape_before'][1]} -> {c['shape_after'][1]}")
print(f"Duplicates removed: {c['duplicates_removed']}")
print(f"Dropped columns: {c['dropped_columns']}")
print(f"Missing values: {c['missing_before_total']} -> {c['missing_after_total']}")
print("Imputation:", c["imputation"])
print("Outliers:", [(o['column'], o['iqr_outliers'], o['zscore_outliers']) for o in c["outliers"]])
print("Redundant features removed:", res["feature_selection"]["removed_redundant"])
print("\nModel comparison:", m["comparison"])
print("Best model:", m["best_model"])
print("Test metrics:", m["metrics"])
if "confusion_matrix" in m:
    print("Confusion matrix:", m["confusion_matrix"])
print("Top features:", m["importance"][:5])
print("\nCharts and model saved in ./outputs/")
