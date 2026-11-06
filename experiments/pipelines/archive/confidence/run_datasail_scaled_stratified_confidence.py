"""
Scaled DataSAIL pipeline
(stratified downsampling + balanced kNN + confidence)
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


# =========================
# DATA LOADING
# =========================

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
        raise ValueError(
            f"Unknown dataset: {name}"
        )

    df = pd.read_csv(path)

    fps, valid_idx = compute_fingerprints(
        df[smiles_col].tolist()
    )

    df = df.iloc[
        valid_idx
    ].reset_index(drop=True)

    return (
        df,
        fps,
        smiles_col,
        label_col
    )


# =========================
# MAIN PIPELINE
# =========================

def run_datasail_scaled_stratified_confidence(
    dataset_name,
    downsample_ratio=0.25,
    k=5,
    balance_weight=2.0
):

    print("=" * 70)
    print(
        f"Scaled DataSAIL "
        f"Stratified+Confidence: "
        f"{dataset_name.upper()}"
    )
    print("=" * 70)

    total_start_time = time.time()

    base_dir = Path(
        "results/experiments"
    )

    dataset_dir = (
        base_dir /
        dataset_name
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    datasail_dir = (
        dataset_dir /
        "datasail_scaled_stratified_confidence"
    )

    datasail_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================
    # 1 Load dataset
    # =========================

    print(
        "\n1. Loading dataset..."
    )

    t0 = time.time()

    (
        df,
        fps,
        smiles_col,
        label_col
    ) = load_dataset(
        dataset_name
    )

    load_time = (
        time.time() - t0
    )

    print(
        f"Loaded "
        f"{len(df)} molecules "
        f"in "
        f"{load_time:.2f}s"
    )

    # =========================
    # 2 Stratified downsampling
    # =========================

    print(
        "\n2. Stratified "
        "down-sampling..."
    )

    t0 = time.time()

    sampled_df, remaining_df = (
        stratified_downsample(
            df,
            df[label_col],
            downsample_ratio,
            seed=42
        )
    )

    sampled_indices = (
        sampled_df
        .index
        .tolist()
    )

    remaining_indices = (
        remaining_df
        .index
        .tolist()
    )

    sampled_fps = fps[
        sampled_indices
    ]

    remaining_fps = fps[
        remaining_indices
    ]

    downsample_time = (
        time.time() - t0
    )

    print(
        f"Sampled: "
        f"{len(sampled_df)} | "
        f"Remaining: "
        f"{len(remaining_df)}"
    )

    # =========================
    # 3 Run DataSAIL
    # =========================

    print(
        "\n3. Running "
        "DataSAIL..."
    )

    t0 = time.time()

    temp_dir = Path(
        "data/temp"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    input_file = (
        temp_dir /
        f"{dataset_name}_sampled.csv"
    )

    datasail_input = pd.DataFrame({

        "id":[
            f"mol_{i}"
            for i in range(
                len(sampled_df)
            )
        ],

        "smiles":
        sampled_df[
            smiles_col
        ].values
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
        splits=[0.7,0.2,0.1],
        names=[
            "train",
            "val",
            "test"
        ],
        output=str(datasail_dir),
        verbosity="W",
        runs=1,
        threads=1
    )

    datasail_time = (
        time.time() - t0
    )

    # =========================
    # 4 Load splits
    # =========================

    splits_file = list(
        (
            datasail_dir /
            "C1e"
        ).glob(
            "Molecule_*_splits.tsv"
        )
    )[0]

    sampled_splits = pd.read_csv(
        splits_file,
        sep="\t"
    )["Split"]

    # =========================
    # 5 Balanced kNN + confidence
    # =========================

    print(
        "\n5. Balanced "
        "kNN + confidence..."
    )

    t0 = time.time()

    (
        assignments,
        confidences
    ) = balanced_knn_assign(

        sampled_fps,
        sampled_splits,
        remaining_fps,
        k=k,
        balance_weight=balance_weight
    )

    assign_time = (
        time.time() - t0
    )

    # =========================
    # 6 Combine
    # =========================

    final_splits = pd.Series(
        index=range(len(df)),
        dtype=str
    )

    confidence_col = np.ones(
        len(df)
    )

    for i, idx in enumerate(
        sampled_indices
    ):

        final_splits.iloc[idx] = (
            sampled_splits.iloc[i]
        )

    for i, idx in enumerate(
        remaining_indices
    ):

        final_splits.iloc[idx] = (
            assignments[i]
        )

        confidence_col[idx] = (
            confidences[i]
        )

    final_counts = Counter(
        final_splits
    )

    total = len(df)

    max_dev = (
        compute_max_deviation(
            final_counts,
            total
        )
    )

    print(
        f"Max deviation: "
        f"±{max_dev:.2f}"
    )

    # =========================
    # 7 Save
    # =========================

    df["split"] = (
        final_splits.values
    )

    df["confidence"] = (
        confidence_col
    )

    train_df = df[
        df["split"]=="train"
    ]

    val_df = df[
        df["split"]=="val"
    ]

    test_df = df[
        df["split"]=="test"
    ]

    save_splits(
        train_df.drop(
            columns=["split"]
        ),
        val_df.drop(
            columns=["split"]
        ),
        test_df.drop(
            columns=["split"]
        ),
        dataset_name,
        "datasail_scaled_stratified_confidence"
    )

    # =========================
    # Confidence metrics
    # =========================

    mean_conf = np.mean(
        confidences
    )

    min_conf = np.min(
        confidences
    )

    low_frac = np.mean(
        confidences < 0.5
    )

    with open(
        dataset_dir /
        "confidence_metrics_stratified.txt",
        "w"
    ) as f:

        f.write(
            f"mean_confidence="
            f"{mean_conf}\n"
        )

        f.write(
            f"min_confidence="
            f"{min_conf}\n"
        )

        f.write(
            f"low_conf_fraction="
            f"{low_frac}\n"
        )

    print(
        f"Mean confidence: "
        f"{mean_conf:.3f}"
    )

    # =========================
    # 8 Runtime
    # =========================

    total_runtime = (
        time.time() -
        total_start_time
    )

    runtime_file = (
        dataset_dir /
        "runtime_scaled_stratified_confidence.txt"
    )

    with open(
        runtime_file,
        "w"
    ) as f:

        f.write(
            f"total_runtime_seconds:"
            f"{total_runtime}\n"
        )

        f.write(
            f"load_time:"
            f"{load_time}\n"
        )

        f.write(
            f"downsample_time:"
            f"{downsample_time}\n"
        )

        f.write(
            f"datasail_time:"
            f"{datasail_time}\n"
        )

        f.write(
            f"assignment_time:"
            f"{assign_time}\n"
        )

    print(
        f"Total runtime: "
        f"{total_runtime:.2f}s"
    )


# =========================
# MAIN
# =========================

def main():

    datasets = [
        "bace",
        "bbbp",
        "tox21",
        "hiv"
    ]

    for dataset in datasets:

        run_datasail_scaled_stratified_confidence(
            dataset
        )

    print(
        "\nAll stratified "
        "confidence experiments "
        "complete"
    )


if __name__ == "__main__":
    main()