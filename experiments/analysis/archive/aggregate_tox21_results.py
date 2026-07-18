import pandas as pd
import numpy as np
import os

INPUT_FILE = "results/analysis/ml_results.csv"
OUTPUT_FILE = "results/analysis/ml_results_aggregated.csv"


# Load data

df = pd.read_csv(INPUT_FILE)

print("Loaded:", INPUT_FILE)
print("Columns:", df.columns.tolist())


# Metrics to aggregate

METRICS = ["mcc", "auc", "f1", "accuracy"]


# Grouping

GROUP_COLS = ["dataset", "model", "split"]

agg_rows = []

for keys, group in df.groupby(GROUP_COLS):
    row = dict(zip(GROUP_COLS, keys))

    for metric in METRICS:
        if metric in group.columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()

    agg_rows.append(row)

agg_df = pd.DataFrame(agg_rows)


# Save

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
agg_df = agg_df.round(2)
agg_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved aggregated results to:", OUTPUT_FILE)
print(agg_df.head())