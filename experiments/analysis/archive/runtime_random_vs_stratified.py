import os
import pandas as pd

BASE_DIR = "results/experiments"

DATASETS = ["bace", "bbbp", "tox21", "hiv"]

METHODS = [
    "runtime_scaled.txt",
    "runtime_scaled_stratified.txt"
]

results = []

for dataset in DATASETS:

    dataset_dir = os.path.join(BASE_DIR, dataset)

    for fname in METHODS:

        path = os.path.join(dataset_dir, fname)

        if not os.path.exists(path):
            continue

        values = {}

        with open(path) as f:

            for line in f:

                key, value = line.strip().split(":")

                values[key.strip()] = float(value)

        results.append({
            "dataset": dataset,
            "method":
                fname.replace(".txt", ""),
            **values
        })

df = pd.DataFrame(results)

print(df)

df.to_csv(
    "results/analysis/runtime_comparison.csv",
    index=False
)