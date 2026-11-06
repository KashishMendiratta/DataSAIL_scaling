"""
Scaled DataSAIL pipeline (downsampling + naive kNN assignment).

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
from src.assignment.knn_assignment import knn_assign
from src.utils import compute_fingerprints, save_splits
from src.utils import compute_max_deviation

# =========================
# DATA LOADING
# =========================

def load_dataset(name="bace"):

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


# =========================
# MAIN PIPELINE
# =========================

def run_datasail_scaled(dataset_name, downsample_ratio=0.25, k=5, balance_weight=2.0):

    print("=" * 70)
    print(f"Scaled DataSAIL: {dataset_name.upper()}")
    print("=" * 70)

    # START TOTAL TIMER
    total_start_time = time.time()

    base_dir = Path("results/experiments")
    dataset_dir = base_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    datasail_dir = dataset_dir / "datasail_scaled"
    datasail_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. Load dataset
    # =========================
    print("\n1. Loading dataset...")
    t0 = time.time()

    df, fps, smiles_col = load_dataset(dataset_name)

    load_time = time.time() - t0
    print(f"Loaded {len(df)} molecules in {load_time:.2f}s")

    # =========================
    # 2. Down-sampling
    # =========================
    print("\n2. Down-sampling...")
    t0 = time.time()

    sampled_df, remaining_df = random_downsample(df, downsample_ratio, seed=42)

    sampled_indices = sampled_df.index.tolist()
    remaining_indices = remaining_df.index.tolist()

    sampled_fps = fps[sampled_indices]
    remaining_fps = fps[remaining_indices]

    downsample_time = time.time() - t0
    print(f"Sampled: {len(sampled_df)} | Remaining: {len(remaining_df)}")
    print(f"Downsampling time: {downsample_time:.2f}s")

    # =========================
    # 3. Run DataSAIL
    # =========================
    print("\n3. Running DataSAIL on subset...")
    t0 = time.time()

    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_file = temp_dir / f"{dataset_name}_sampled.csv"

    datasail_input = pd.DataFrame({
        "id": [f"mol_{i}" for i in range(len(sampled_df))],
        "smiles": sampled_df[smiles_col].values
    })

    datasail_input.to_csv(input_file, index=False)

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
    print(f"DataSAIL time: {datasail_time:.2f}s")

    # =========================
    # 4. LOAD SPLITS
    # =========================
    print("\n4. Loading DataSAIL splits...")

    splits_dir = datasail_dir / "C1e"

    if not splits_dir.exists():
        raise FileNotFoundError(f"Missing DataSAIL output directory: {splits_dir}")

    split_files = list(splits_dir.glob("Molecule_*_splits.tsv"))

    if len(split_files) == 0:
        raise FileNotFoundError("No split file found inside DataSAIL output")

    splits_file = split_files[0]
    print(f"Using split file: {splits_file}")

    splits_df = pd.read_csv(splits_file, sep="\t")
    sampled_splits = pd.Series(splits_df["Split"].values)

    # =========================
    # 5. Naive kNN assignment
    # =========================
    print("\n5. Assigning remaining samples (Naive kNN)...")
    t0 = time.time()

    

    remaining_assignments = knn_assign(
        sampled_fps,
        sampled_splits,
        remaining_fps,
        k=k,
        method='majority'
    )

    assign_time = time.time() - t0
    print(f"Assignment time: {assign_time:.2f}s")

    # =========================
    # 6. Combine splits
    # =========================
    final_splits = pd.Series(index=range(len(df)), dtype=str)

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
        print(f"{split}: {count} ({pct:.2f}%)")
    
    max_dev = compute_max_deviation(final_counts, total)

    print(f"Max deviation: ±{max_dev:.2f} percentage points")

    # =========================
    # 7. SAVE ML SPLITS
    # =========================
    print("\n6. Saving ML-ready splits...")

    df_with_splits = df.copy()
    df_with_splits["split"] = final_splits.values

    train_df = df_with_splits[df_with_splits["split"] == "train"].drop(columns=["split"])
    val_df = df_with_splits[df_with_splits["split"] == "val"].drop(columns=["split"])
    test_df = df_with_splits[df_with_splits["split"] == "test"].drop(columns=["split"])

    save_splits(train_df, val_df, test_df, dataset_name, "datasail_scaled")

    print("ML-ready splits saved successfully!")

    # =========================
    # 8. SAVE RUNTIME
    # =========================
    total_runtime = time.time() - total_start_time

    runtime_file = dataset_dir / "runtime_scaled.txt"

    with open(runtime_file, "w") as f:
        f.write(f"total_runtime_seconds: {total_runtime}\n")
        f.write(f"load_time: {load_time}\n")
        f.write(f"downsample_time: {downsample_time}\n")
        f.write(f"datasail_time: {datasail_time}\n")
        f.write(f"assignment_time: {assign_time}\n")

    print(f"\nRuntime saved to: {runtime_file}")
    print(f"Total runtime: {total_runtime:.2f}s")


# =========================
# MAIN
# =========================

def main():

    datasets = ["bace", "bbbp", "tox21", "hiv"]

    for dataset in datasets:
        run_datasail_scaled(dataset)

    print("\nAll scaled DataSAIL with Naive knn assignment experiments complete")


if __name__ == "__main__":
    main()