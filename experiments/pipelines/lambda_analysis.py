"""
Lambda sensitivity analysis.

varies only the balance parameter λ
used in balanced kNN assignment.

Datasets:
    - tox21
    - hiv

Lambda values:
    [0, 0.5, 1, 2, 5]
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from collections import Counter

from src.downsampling import stratified_downsample
from src.assignment import balanced_knn_assign
from src.utils import (
    compute_fingerprints,
    save_splits,
    compute_max_deviation
)



# DATA LOADING


def load_dataset(name):

    if name == "tox21":
        path = "data/raw/moleculenet/tox21.csv"
        smiles_col = "smiles"
        label_col = "NR-AR"

    elif name == "hiv":
        path = "data/raw/moleculenet/hiv.csv"
        smiles_col = "smiles"
        label_col = "HIV_active"

    else:
        raise ValueError(
            "Only tox21 and hiv are supported."
        )

    df = pd.read_csv(path)

    fps, valid_idx = compute_fingerprints(
        df[smiles_col].tolist()
    )

    df = df.iloc[valid_idx].reset_index(drop=True)

    return df, fps, smiles_col, label_col



# LAMBDA EXPERIMENT


def run_lambda_experiment(
    dataset_name,
    lambda_values,
    downsample_ratio=0.25,
    k=5
):

    print("\n" + "=" * 80)
    print(f"{dataset_name.upper()}")
    print("=" * 80)

    summary_rows = []

    
    # LOAD DATA
    

    print("\nLoading dataset...")

    df, fps, smiles_col, label_col = load_dataset(
        dataset_name
    )

    
    # RECREATE STRATIFIED SUBSET
    

    print("Recreating stratified subset...")

    sampled_df, remaining_df = stratified_downsample(
        df,
        df[label_col],
        downsample_ratio,
        seed=42
    )

    sampled_indices = sampled_df.index.tolist()
    remaining_indices = remaining_df.index.tolist()

    sampled_fps = fps[sampled_indices]
    remaining_fps = fps[remaining_indices]

   
    # LOAD EXISTING DATASAIL SPLITS
    

    print("Loading original DataSAIL assignments...")

    splits_dir = (
        Path("results/experiments")
        / dataset_name
        / "datasail_scaled_stratified"
        / "C1e"
    )

    split_files = list(
        splits_dir.glob(
            "Molecule_*_splits.tsv"
        )
    )

    if len(split_files) == 0:
        raise FileNotFoundError(
            f"No split file found in {splits_dir}"
        )

    splits_file = split_files[0]

    splits_df = pd.read_csv(
        splits_file,
        sep="\t"
    )

    sampled_splits = pd.Series(
        splits_df["Split"].values
    )

    
    # LOOP OVER LAMBDAS
    

    for lam in lambda_values:

        print("\n" + "-" * 60)
        print(f"λ = {lam}")
        print("-" * 60)

        start = time.time()

        
        # BALANCED KNN
        

        assignments, _ = balanced_knn_assign(
            sampled_fps,
            sampled_splits,
            remaining_fps,
            k=k,
            balance_weight=lam
        )

        assign_time = time.time() - start

        
        # COMBINE SPLITS
        

        final_splits = pd.Series(
            index=range(len(df)),
            dtype=str
        )

        for i, idx in enumerate(sampled_indices):
            final_splits.iloc[idx] = sampled_splits.iloc[i]

        for i, idx in enumerate(remaining_indices):
            final_splits.iloc[idx] = assignments[i]

        
        # DISTRIBUTION
        

        counts = Counter(final_splits)

        total = len(df)

        max_dev = compute_max_deviation(
            counts,
            total
        )

        print("\nFinal split distribution:")

        for split in ["train", "val", "test"]:

            count = counts.get(split, 0)

            print(
                f"{split}: "
                f"{count} "
                f"({100 * count / total:.2f}%)"
            )

        print(
            f"Max deviation: "
            f"±{max_dev:.2f}"
        )

        
        # SAVE SPLITS
       

        df_with_splits = df.copy()

        df_with_splits["split"] = (
            final_splits.values
        )

        train_df = df_with_splits[
            df_with_splits["split"] == "train"
        ].drop(columns=["split"])

        val_df = df_with_splits[
            df_with_splits["split"] == "val"
        ].drop(columns=["split"])

        test_df = df_with_splits[
            df_with_splits["split"] == "test"
        ].drop(columns=["split"])

        split_name = (
            f"datasail_scaled_"
            f"stratified_lambda_{lam}"
        )

        save_splits(
            train_df,
            val_df,
            test_df,
            dataset_name,
            split_name
        )

        
        # SUMMARY
        

        summary_rows.append({

            "dataset": dataset_name,
            "lambda": lam,

            "train_count":
                counts["train"],

            "val_count":
                counts["val"],

            "test_count":
                counts["test"],

            "train_pct":
                100 * counts["train"] / total,

            "val_pct":
                100 * counts["val"] / total,

            "test_pct":
                100 * counts["test"] / total,

            "max_deviation":
                max_dev,

            "assignment_time":
                assign_time
        })

    
    # SAVE SUMMARY
    

    summary_df = pd.DataFrame(
        summary_rows
    )

    out_dir = (
        Path("results/analysis")
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df.to_csv(
        out_dir /
        f"{dataset_name}_lambda_summary.csv",
        index=False
    )

    print(
        "\nSaved summary to:"
    )

    print(
        out_dir /
        f"{dataset_name}_lambda_summary.csv"
    )



# MAIN


def main():

    lambda_values = [
        0,
        0.5,
        1,
        2,
        5
    ]

    datasets = [
        "tox21",
        "hiv"
    ]

    for dataset in datasets:

        run_lambda_experiment(
            dataset,
            lambda_values=lambda_values
        )

    print(
        "\nLambda sensitivity "
        "analysis complete."
    )


if __name__ == "__main__":
    main()