import pandas as pd
import numpy as np
import os

INPUT_FILE = "results/analysis/ml_results_extended.csv"
OUTPUT_FILE = "results/analysis/ml_results_extended_aggregated.csv"

# =========================
# Load data
# =========================
df = pd.read_csv(INPUT_FILE)

print("Loaded:", INPUT_FILE)
print("Columns:", df.columns.tolist())


# Compute distribution gap 

if "train_pos_ratio" in df.columns and "test_pos_ratio" in df.columns:
    df["distribution_gap"] = abs(df["train_pos_ratio"] - df["test_pos_ratio"])
    print("Computed distribution_gap")
else:
    print("Warning: train/test ratios not found, skipping distribution_gap")


# Metrics to aggregate

METRICS = [
    "precision",
    "recall",
    "pr_auc",
    "train_pos_ratio",
    "test_pos_ratio",
    "distribution_gap"
]


# Grouping

GROUP_COLS = ["dataset", "model", "split"]

agg_rows = []

for keys, group in df.groupby(GROUP_COLS):
    row = dict(zip(GROUP_COLS, keys))

    for metric in METRICS:
        if metric in group.columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

    # Add sizes (use mean just for reporting)
    if "train_size" in group.columns:
        row["train_size_mean"] = group["train_size"].mean()
    if "test_size" in group.columns:
        row["test_size_mean"] = group["test_size"].mean()

    agg_rows.append(row)

agg_df = pd.DataFrame(agg_rows)


# Save

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
agg_df = agg_df.round(3)
agg_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved extended aggregated results to:", OUTPUT_FILE)
print(agg_df.head())