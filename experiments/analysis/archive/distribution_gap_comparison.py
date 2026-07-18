import os
import pandas as pd
import numpy as np

BASE_DIR = "results/experiments/splits"

'''
DATASETS = ["tox21", "hiv"]

SPLITS = [
    "datasail_scaled_stratified_lambda_0",
    "datasail_scaled_stratified_lambda_0.5",
    "datasail_scaled_stratified_lambda_1",
    "datasail_scaled_stratified_lambda_2",
    "datasail_scaled_stratified_lambda_5"
]
'''
DATASETS = ["bace", "bbbp", "tox21", "hiv"]

SPLITS = [
    "datasail_scaled",
    "datasail_scaled_stratified"
]

LABEL_COLUMNS = {
    "bace": "Class",
    "bbbp": "p_np",
    "tox21": "NR-AR",     
    "hiv": "HIV_active"
}

results = []

for dataset in DATASETS:

    label_col = LABEL_COLUMNS[dataset]

    for split_type in SPLITS:

        path = os.path.join(BASE_DIR, dataset, split_type)

        train = pd.read_csv(os.path.join(path, "train.csv"))
        val = pd.read_csv(os.path.join(path, "val.csv"))
        test = pd.read_csv(os.path.join(path, "test.csv"))

        full = pd.concat([train, val, test])

        overall_ratio = full[label_col].mean()

        for name, df in {
            "train": train,
            "val": val,
            "test": test
        }.items():

            ratio = df[label_col].mean()

            results.append({
                "dataset": dataset,
                "split": split_type,
                "partition": name,
                "positive_ratio": ratio,
                "distribution_gap":
                    abs(ratio - overall_ratio)
            })

df = pd.DataFrame(results)

summary = (
    df.groupby(["dataset", "split"])
      ["distribution_gap"]
      .agg(["mean", "max"])
      .reset_index()
      .round(3)
)



print(summary)

summary.to_csv(
    "results/analysis/distribution_gap.csv",
    index=False
)