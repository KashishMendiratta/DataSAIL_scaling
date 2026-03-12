"""
Full pipeline with balanced kNN assignment.
Runs experiments and saves results cleanly.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datasail.sail import sail
from collections import Counter

from src.downsampling import random_downsample
from src.assignment import balanced_knn_assign
from src.utils import compute_fingerprints, log_experiment


def load_dataset(name="bace"):
    """Load dataset and compute fingerprints."""

    if name == "bace":
        path = "data/raw/moleculenet/bace.csv"
        smiles_col = "mol"

    elif name == "bbbp":
        path = "data/raw/moleculenet/bbbp.csv"
        smiles_col = "smiles"

    elif name == "tox21":
        path = "data/raw/moleculenet/tox21.csv"
        smiles_col = "smiles"
    elif name == "hiv":
        path = "data/raw/moleculenet/hiv.csv"
        smiles_col = "smiles"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    df = pd.read_csv(path)

    fps, valid_idx = compute_fingerprints(df[smiles_col].tolist())
    df = df.iloc[valid_idx].reset_index(drop=True)

    return df, fps, smiles_col


def run_balanced_pipeline(dataset_name, downsample_ratio=0.25, k=5, balance_weight=2.0):

    print("=" * 70)
    print(f"Balanced Pipeline: {dataset_name.upper()}")
    print("=" * 70)

    output_dir = Path(f"results/experiments/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasail_dir = output_dir / "datasail"
    datasail_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n1. Loading dataset...")
    df, fps, smiles_col = load_dataset(dataset_name)
    print(f"Loaded {len(df)} molecules")

    # Down-sampling
    print("\n2. Down-sampling...")
    start = time.time()

    sampled_df, remaining_df = random_downsample(df, downsample_ratio, seed=42)

    sampled_indices = sampled_df.index.tolist()
    remaining_indices = remaining_df.index.tolist()

    sampled_fps = fps[sampled_indices]
    remaining_fps = fps[remaining_indices]

    downsample_time = time.time() - start

    print(f"Sampled: {len(sampled_df)}")
    print(f"Remaining: {len(remaining_df)}")

    # Prepare DataSAIL input
    print("\n3. Running DataSAIL...")

    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    datasail_input = pd.DataFrame({
        "id": [f"mol_{i}" for i in range(len(sampled_df))],
        "smiles": sampled_df[smiles_col].values
    })

    input_file = temp_dir / f"{dataset_name}_sampled.csv"
    datasail_input.to_csv(input_file, index=False)

    datasail_start = time.time()

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

    datasail_time = time.time() - datasail_start

    # Read DataSAIL splits
    print("\n4. Reading DataSAIL splits...")

    splits_file = datasail_dir / "C1e" / f"Molecule_{dataset_name}_sampled_splits.tsv"
    splits_df = pd.read_csv(splits_file, sep="\t")

    sampled_splits = pd.Series(splits_df["Split"].values)

    # Balanced kNN assignment
    print("\n5. Balanced kNN assignment...")

    assign_start = time.time()

    remaining_assignments, confidences = balanced_knn_assign(
        sampled_fps,
        sampled_splits,
        remaining_fps,
        k=k,
        balance_weight=balance_weight
    )

    assign_time = time.time() - assign_start

    # Combine splits
    final_splits = pd.Series(index=range(len(df)), dtype=str)

    for i, idx in enumerate(sampled_indices):
        final_splits.iloc[idx] = sampled_splits.iloc[i]

    for i, idx in enumerate(remaining_indices):
        final_splits.iloc[idx] = remaining_assignments[i]

    final_counts = Counter(final_splits)

    print("\nFinal distribution:")

    results = {}
    for split in ["train", "val", "test"]:

        count = final_counts.get(split, 0)
        pct = 100 * count / len(df)

        target = {"train": 70, "val": 20, "test": 10}[split]
        diff = pct - target

        print(f"{split}: {count} ({pct:.1f}%)")

        results[split] = {
            "count": count,
            "pct": pct,
            "diff": diff
        }

    # Save final splits
    result_df = pd.DataFrame({
        "ID": [f"mol_{i}" for i in range(len(df))],
        "Split": final_splits.values,
        "Source": ["sampled" if i in sampled_indices else "assigned" for i in range(len(df))]
    })

    result_df.to_csv(output_dir / "final_splits.tsv", sep="\t", index=False)

    # Save statistics
    stats = pd.DataFrame({
        "split": ["train", "val", "test"],
        "count": [results[s]["count"] for s in ["train", "val", "test"]],
        "percentage": [results[s]["pct"] for s in ["train", "val", "test"]],
        "deviation": [results[s]["diff"] for s in ["train", "val", "test"]],
    })

    stats.to_csv(output_dir / "split_statistics.csv", index=False)

    total_time = downsample_time + datasail_time + assign_time

    with open(output_dir / "runtime.txt", "w") as f:
        f.write(f"total_runtime_seconds: {total_time}\n")

    print(f"\nResults saved to: {output_dir}")

    # Log experiment summary
    log_experiment(
        dataset_name,
        len(df),
        total_time,
        results
    )

    return results, total_time


def main():

    datasets = ["bace", "bbbp", "tox21", "hiv"]

    all_results = {}

    for dataset in datasets:

        results, runtime = run_balanced_pipeline(dataset)

        all_results[dataset] = runtime

    print("\nAll experiments complete")


if __name__ == "__main__":
    main()