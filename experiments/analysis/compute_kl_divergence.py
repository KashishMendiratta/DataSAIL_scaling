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

EPS = 1e-12

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

        # Original label distribution
        full = pd.concat([
            train,
            val,
            test
        ])

        p = full[label_col].mean()

        for split_name, df in {
            "train": train,
            "val": val,
            "test": test
        }.items():

            q = df[label_col].mean()

            # Numerical stability
            p_safe = np.clip(p, EPS, 1 - EPS)
            q_safe = np.clip(q, EPS, 1 - EPS)

            kl = (
                p_safe * np.log(p_safe / q_safe)
                +
                (1 - p_safe)
                * np.log(
                    (1 - p_safe)
                    /
                    (1 - q_safe)
                )
            )

            results.append({

                "dataset": dataset,
                "method": split_type,
                "partition": split_name,

                "overall_ratio": p,
                "split_ratio": q,

                "KL": kl
            })

df = pd.DataFrame(results)

# Round detailed results
df["overall_ratio"] = (
    df["overall_ratio"].round(3)
)

df["split_ratio"] = (
    df["split_ratio"].round(3)
)

df["KL"] = (
    df["KL"].round(6)
)

summary = (
    df.groupby(
        ["dataset", "method"]
    )["KL"]
    .agg(
        MeanKL="mean",
        MaxKL="max"
    )
    .reset_index()
)

summary[["MeanKL", "MaxKL"]] = (
    summary[
        ["MeanKL", "MaxKL"]
    ].round(6)
)

print("\nKL Divergence")
print(summary)

os.makedirs(
    "results/analysis",
    exist_ok=True
)

summary.to_csv(
    "results/analysis/kl_divergence.csv",
    index=False
)

print(
    "\nSaved to:"
)

print(
    "results/analysis/kl_divergence.csv"
)