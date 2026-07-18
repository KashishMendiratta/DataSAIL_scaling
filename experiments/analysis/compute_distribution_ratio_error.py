import os
import pandas as pd
import numpy as np

BASE_DIR = "results/experiments/splits"

DATASETS = ["bace", "bbbp", "tox21", "hiv"]

SPLITS = [
    "datasail_scaled_random",
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

    print(f"\nProcessing {dataset}...")

    label_col = LABEL_COLUMNS[dataset]

    for split_type in SPLITS:

        path = os.path.join(
            BASE_DIR,
            dataset,
            split_type
        )

        train = pd.read_csv(
            os.path.join(path, "train.csv")
        )

        val = pd.read_csv(
            os.path.join(path, "val.csv")
        )

        test = pd.read_csv(
            os.path.join(path, "test.csv")
        )

        # Original prevalence
        full = pd.concat([
            train,
            val,
            test
        ])

        overall_ratio = (
            full[label_col].mean()
        )

        split_dres = []

        for split_name, df in {
            "train": train,
            "val": val,
            "test": test
        }.items():

            split_ratio = (
                df[label_col].mean()
            )

            # Avoid division by zero
            eps = 1e-12

            dre = (
                abs(
                    split_ratio
                    - overall_ratio
                )
                /
                (overall_ratio + eps)
            )

            split_dres.append(dre)

            results.append({

                "dataset": dataset,
                "method": split_type,
                "partition": split_name,

                "overall_ratio":
                    overall_ratio,

                "split_ratio":
                    split_ratio,

                "DRE":
                    dre
            })

df = pd.DataFrame(results)
summary = (
    df.groupby(
        ["dataset", "method"]
    )["DRE"]
    .agg(
        MeanDRE="mean",
        MaxDRE="max"
    )
    .reset_index()
)

# Round numerical columns to 3 decimals
summary[["MeanDRE", "MaxDRE"]] = (
    summary[["MeanDRE", "MaxDRE"]]
    .round(3)
)

print("\nDistribution Ratio Error")
print(summary)

os.makedirs(
    "results/analysis",
    exist_ok=True
)

summary.to_csv(
    "results/analysis/distribution_ratio_error.csv",
    index=False
)

print(
    "\nSaved to:"
)

print(
    "results/analysis/distribution_ratio_error.csv"
)