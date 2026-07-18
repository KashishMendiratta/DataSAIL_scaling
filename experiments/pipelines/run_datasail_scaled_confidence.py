"""
Scaled DataSAIL pipeline
(random downsampling + balanced kNN + confidence)
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
        raise ValueError(name)

    df = pd.read_csv(path)

    fps, valid_idx = compute_fingerprints(
        df[smiles_col].tolist()
    )

    df = df.iloc[valid_idx].reset_index(drop=True)

    return df, fps, smiles_col


# MAIN PIPELINE

def run_datasail_scaled_confidence(
    dataset_name,
    downsample_ratio=0.25,
    k=5,
    balance_weight=2.0
):

    print("=" * 70)
    print(f"Scaled DataSAIL CONFIDENCE: {dataset_name.upper()}")
    print("=" * 70)

    total_start_time = time.time()

    base_dir = Path("results/experiments")
    dataset_dir = base_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    datasail_dir = dataset_dir / "datasail_scaled_confidence"
    datasail_dir.mkdir(parents=True, exist_ok=True)

    # 1 Load

    print("\n1. Loading dataset...")
    t0 = time.time()

    df, fps, smiles_col = load_dataset(dataset_name)

    load_time = time.time() - t0
    print(f"Loaded {len(df)} molecules in {load_time:.2f}s")

    # 2 Downsample

    print("\n2. Down-sampling...")
    t0 = time.time()

    sampled_df, remaining_df = random_downsample(
        df,
        downsample_ratio,
        seed=42
    )

    sampled_idx = sampled_df.index.tolist()
    remaining_idx = remaining_df.index.tolist()

    sampled_fps = fps[sampled_idx]
    remaining_fps = fps[remaining_idx]

    downsample_time = time.time() - t0

    print(
        f"Sampled: {len(sampled_df)} | "
        f"Remaining: {len(remaining_df)}"
    )

    # 3 DataSAIL

    print("\n3. Running DataSAIL...")
    t0 = time.time()

    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_file = temp_dir / f"{dataset_name}_sampled.csv"

    pd.DataFrame({
        "id":[f"mol_{i}" for i in range(len(sampled_df))],
        "smiles":sampled_df[smiles_col].values
    }).to_csv(input_file,index=False)

    sail(
        e_type="M",
        e_data=str(input_file),
        techniques=["C1e"],
        e_sim="ecfp",
        e_clusters=10,
        splits=[0.7,0.2,0.1],
        names=["train","val","test"],
        output=str(datasail_dir),
        verbosity="W",
        runs=1,
        threads=1
    )

    datasail_time = time.time() - t0

    # 4 Load splits

    split_file = list(
        (datasail_dir/"C1e").glob(
            "Molecule_*_splits.tsv"
        )
    )[0]

    sampled_splits = pd.read_csv(
        split_file,
        sep="\t"
    )["Split"]

    # 5 Balanced kNN + confidence

    print(
        "\n5. Assigning remaining "
        "(Balanced kNN + confidence)..."
    )

    t0 = time.time()

    assignments, confidences = balanced_knn_assign(
        sampled_fps,
        sampled_splits,
        remaining_fps,
        k=k,
        balance_weight=balance_weight
    )

    assign_time = time.time() - t0

    # 6 Combine

    final_splits = pd.Series(
        index=range(len(df)),
        dtype=str
    )

    confidence_col = np.ones(len(df))

    for i, idx in enumerate(sampled_idx):
        final_splits.iloc[idx] = sampled_splits.iloc[i]

    for i, idx in enumerate(remaining_idx):
        final_splits.iloc[idx] = assignments[i]
        confidence_col[idx] = confidences[i]

    final_counts = Counter(final_splits)
    total = len(df)

    max_dev = compute_max_deviation(
        final_counts,
        total
    )

    print(
        f"Max deviation: ±{max_dev:.2f}"
    )

    # 7 Save

    print("\n6. Saving splits...")

    df["split"] = final_splits.values
    df["confidence"] = confidence_col

    train_df = df[df["split"]=="train"]
    val_df = df[df["split"]=="val"]
    test_df = df[df["split"]=="test"]

    save_splits(
        train_df.drop(columns=["split"]),
        val_df.drop(columns=["split"]),
        test_df.drop(columns=["split"]),
        dataset_name,
        "datasail_scaled_confidence"
    )

    # confidence metrics

    mean_conf = np.mean(confidences)
    min_conf = np.min(confidences)
    low_frac = np.mean(confidences < 0.5)

    with open(
        dataset_dir/"confidence_metrics_balanced.txt",
        "w"
    ) as f:

        f.write(f"mean_confidence={mean_conf}\n")
        f.write(f"min_confidence={min_conf}\n")
        f.write(f"low_conf_fraction={low_frac}\n")

    # 8 Runtime

    total_runtime = time.time() - total_start_time

    runtime_file = (
        dataset_dir /
        "runtime_scaled_confidence.txt"
    )

    with open(runtime_file,"w") as f:

        f.write(f"total_runtime_seconds:{total_runtime}\n")
        f.write(f"load_time:{load_time}\n")
        f.write(f"downsample_time:{downsample_time}\n")
        f.write(f"datasail_time:{datasail_time}\n")
        f.write(f"assignment_time:{assign_time}\n")

    print(
        f"Runtime={total_runtime:.2f}s "
        f"MeanConf={mean_conf:.3f}"
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
        run_datasail_scaled_confidence(
            dataset
        )

    print(
        "\nAll scaled "
        "DataSAIL confidence "
        "experiments complete"
    )


if __name__ == "__main__":
    main()