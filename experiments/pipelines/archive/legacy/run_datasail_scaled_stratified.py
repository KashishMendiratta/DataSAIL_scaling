"""
Scaled DataSAIL pipeline
(stratified downsampling + balanced kNN assignment).

This is the proposed scalable approximation of DataSAIL.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datasail.sail import sail
from collections import Counter

from src.downsampling import stratified_downsample
from src.assignment import balanced_knn_assign
from src.utils import (
    compute_fingerprints,
    save_splits,
    compute_max_deviation
)


# DATA LOADING


def load_dataset(name="bace"):

    if name == "bace":
        path = "data/raw/moleculenet/bace.csv"
        smiles_col = "mol"
        label_col = "Class"

    elif name == "bbbp":
        path = "data/raw/moleculenet/bbbp.csv"
        smiles_col = "smiles"
        label_col = "p_np"

    elif name == "tox21":
        path = "data/raw/moleculenet/tox21.csv"
        smiles_col = "smiles"
        label_col = "NR-AR"

    elif name == "hiv":
        path = "data/raw/moleculenet/hiv.csv"
        smiles_col = "smiles"
        label_col = "HIV_active"

    else:
        raise ValueError(f"Unknown dataset: {name}")

    df = pd.read_csv(path)

    fps, valid_idx = compute_fingerprints(
        df[smiles_col].tolist()
    )

    df = df.iloc[valid_idx].reset_index(drop=True)

    return df, fps, smiles_col, label_col


# MAIN PIPELINE


def run_datasail_scaled_stratified(
    dataset_name,
    downsample_ratio=0.25,
    k=5,
    balance_weight=2.0
):

    print("=" * 70)
    print(f"Scaled DataSAIL STRATIFIED: {dataset_name.upper()}")
    print("=" * 70)

    total_start_time = time.time()

    base_dir = Path("results/experiments")
    dataset_dir = base_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    datasail_dir = dataset_dir / "datasail_scaled_stratified"
    datasail_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset

    print("\n1. Loading dataset...")
    t0 = time.time()

    df, fps, smiles_col, label_col = load_dataset(dataset_name)

    load_time = time.time() - t0
    print(f"Loaded {len(df)} molecules in {load_time:.2f}s")

    # 2. Stratified down-sampling

    print("\n2. Stratified down-sampling...")
    t0 = time.time()

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

    downsample_time = time.time() - t0

    print(
        f"Sampled: {len(sampled_df)} | "
        f"Remaining: {len(remaining_df)}"
    )
    print(
        f"Downsampling time: "
        f"{downsample_time:.2f}s"
    )

    # 3. Run DataSAIL

    print("\n3. Running DataSAIL on subset...")
    t0 = time.time()

    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_file = temp_dir / f"{dataset_name}_sampled.csv"

    datasail_input = pd.DataFrame({
        "id": [
            f"mol_{i}"
            for i in range(len(sampled_df))
        ],
        "smiles": sampled_df[smiles_col].values
    })

    datasail_input.to_csv(
        input_file,
        index=False
    )

    sail(
        e_type="M",
        e_data=str(input_file),
        techniques=["C1e"],
        e_sim="ecfp",
        e_clusters=10,
        splits=[0.7, 0.2, 0.1],
        names=["train", "val", "test"],
        output=str(datasail_dir),
        verbosity="W",
        runs=1,
        threads=1
    )

    datasail_time = time.time() - t0
    print(
        f"DataSAIL time: "
        f"{datasail_time:.2f}s"
    )

    # 4. Load splits

    print("\n4. Loading DataSAIL splits...")

    splits_dir = datasail_dir / "C1e"

    if not splits_dir.exists():
        raise FileNotFoundError(
            f"Missing DataSAIL output: {splits_dir}"
        )

    split_files = list(
        splits_dir.glob(
            "Molecule_*_splits.tsv"
        )
    )

    if len(split_files) == 0:
        raise FileNotFoundError(
            "No split file found"
        )

    splits_file = split_files[0]

    print(
        f"Using split file: "
        f"{splits_file}"
    )

    splits_df = pd.read_csv(
        splits_file,
        sep="\t"
    )

    sampled_splits = pd.Series(
        splits_df["Split"].values
    )

    # 5. Balanced kNN assignment

    print(
        "\n5. Assigning remaining "
        "samples (balanced kNN)..."
    )

    t0 = time.time()

    remaining_assignments, _ = balanced_knn_assign(
        sampled_fps,
        sampled_splits,
        remaining_fps,
        k=k,
        balance_weight=balance_weight
    )

    assign_time = time.time() - t0

    print(
        f"Assignment time: "
        f"{assign_time:.2f}s"
    )

    # 6. Combine splits

    final_splits = pd.Series(
        index=range(len(df)),
        dtype=str
    )

    for i, idx in enumerate(sampled_indices):
        final_splits.iloc[idx] = sampled_splits.iloc[i]

    for i, idx in enumerate(remaining_indices):
        final_splits.iloc[idx] = remaining_assignments[i]

    final_counts = Counter(final_splits)
    total = len(df)

    print("\nFinal distribution:")

    for split in ["train", "val", "test"]:

        count = final_counts.get(split, 0)
        pct = 100 * count / total

        print(
            f"{split}: "
            f"{count} "
            f"({pct:.2f}%)"
        )

    max_dev = compute_max_deviation(
        final_counts,
        total
    )

    print(
        f"Max deviation: "
        f"±{max_dev:.2f} "
        f"percentage points"
    )

    # 7. Save ML splits

    print(
        "\n6. Saving ML-ready splits..."
    )

    df_with_splits = df.copy()
    df_with_splits["split"] = final_splits.values

    train_df = df_with_splits[
        df_with_splits["split"] == "train"
    ].drop(columns=["split"])

    val_df = df_with_splits[
        df_with_splits["split"] == "val"
    ].drop(columns=["split"])

    test_df = df_with_splits[
        df_with_splits["split"] == "test"
    ].drop(columns=["split"])

    save_splits(
        train_df,
        val_df,
        test_df,
        dataset_name,
        "datasail_scaled_stratified"
    )

    print(
        "ML-ready splits saved successfully!"
    )

    # 8. Save runtime

    total_runtime = (
        time.time() -
        total_start_time
    )

    runtime_file = (
        dataset_dir /
        "runtime_scaled_stratified.txt"
    )

    with open(runtime_file, "w") as f:

        f.write(
            f"total_runtime_seconds: "
            f"{total_runtime}\n"
        )

        f.write(
            f"load_time: "
            f"{load_time}\n"
        )

        f.write(
            f"downsample_time: "
            f"{downsample_time}\n"
        )

        f.write(
            f"datasail_time: "
            f"{datasail_time}\n"
        )

        f.write(
            f"assignment_time: "
            f"{assign_time}\n"
        )

    print(
        f"\nRuntime saved to: "
        f"{runtime_file}"
    )

    print(
        f"Total runtime: "
        f"{total_runtime:.2f}s"
    )


# MAIN


def main():

    datasets = [
        "bace",
        "bbbp",
        "tox21",
        "hiv"
    ]

    for dataset in datasets:
        run_datasail_scaled_stratified(dataset)

    print(
        "\nAll stratified "
        "scaled DataSAIL "
        "experiments complete"
    )


if __name__ == "__main__":
    main()