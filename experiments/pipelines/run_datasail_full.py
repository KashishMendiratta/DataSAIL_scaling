"""
Full DataSAIL pipeline (reference method).
"""

import pandas as pd
import time
from pathlib import Path
from datasail.sail import sail
from collections import Counter

from src.utils import save_splits
from src.utils import compute_max_deviation

# =========================
# DATA LOADING
# =========================

def load_dataset(name):

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
    return df, smiles_col


# =========================
# MAIN PIPELINE
# =========================

def run_datasail_full(dataset_name):

    print("=" * 70)
    print(f"Full DataSAIL: {dataset_name.upper()}")
    print("=" * 70)

    # TOTAL TIMER
    total_start_time = time.time()

    base_dir = Path("results/experiments")
    dataset_dir = base_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    output_dir = dataset_dir / "datasail_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. Load dataset
    # =========================
    print("\n1. Loading dataset...")
    t0 = time.time()

    df, smiles_col = load_dataset(dataset_name)

    load_time = time.time() - t0
    print(f"Loaded {len(df)} molecules in {load_time:.2f}s")

    # =========================
    # 2. Prepare input
    # =========================
    print("\n2. Preparing input...")
    t0 = time.time()

    processed_file = Path(f"data/processed/{dataset_name}_full.csv")
    processed_file.parent.mkdir(parents=True, exist_ok=True)

    smiles_df = pd.DataFrame({
        "id": [f"mol_{i}" for i in range(len(df))],
        "smiles": df[smiles_col].values
    })

    smiles_df.to_csv(processed_file, index=False)

    prep_time = time.time() - t0
    print(f"Preparation time: {prep_time:.2f}s")

    # =========================
    # 3. Run DataSAIL
    # =========================
    print("\n3. Running full DataSAIL...")
    t0 = time.time()

    sail(
        e_type="M",
        e_data=str(processed_file),
        techniques=["C1e"],
        e_sim="ecfp",
        e_clusters=10,
        splits=[0.7, 0.2, 0.1],
        names=["train", "val", "test"],
        output=str(output_dir),
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

    splits_dir = output_dir / "C1e"

    if not splits_dir.exists():
        raise FileNotFoundError(f"Missing DataSAIL output directory: {splits_dir}")

    split_files = list(splits_dir.glob("Molecule_*_splits.tsv"))

    if len(split_files) == 0:
        raise FileNotFoundError("No split file found inside DataSAIL output")

    splits_file = split_files[0]
    print(f"Using split file: {splits_file}")

    splits_df = pd.read_csv(splits_file, sep="\t")

    # =========================
    # 5. Align indices
    # =========================
    def extract_index(id_str):
        return int(id_str.split("_")[1])

    splits_df["idx"] = splits_df["ID"].apply(extract_index)

    df_with_splits = df.iloc[splits_df["idx"]].copy()
    df_with_splits["split"] = splits_df["Split"].values

    # =========================
    # 6. Print distribution
    # =========================
    print("\nSplit distribution:")
    counts = Counter(df_with_splits["split"])
    total = len(df)

    for split in ["train", "val", "test"]:
        count = counts.get(split, 0)
        pct = 100 * count / total
        print(f"{split}: {count} ({pct:.2f}%)")

    
    max_dev = compute_max_deviation(counts, total)

    print(f"Max deviation: ±{max_dev:.2f} percentage points")

    # =========================
    # 7. SAVE ML SPLITS
    # =========================
    print("\n5. Saving ML-ready splits...")

    train_df = df_with_splits[df_with_splits["split"] == "train"].drop(columns=["split"])
    val_df = df_with_splits[df_with_splits["split"] == "val"].drop(columns=["split"])
    test_df = df_with_splits[df_with_splits["split"] == "test"].drop(columns=["split"])

    save_splits(train_df, val_df, test_df, dataset_name, "datasail_full")

    print("ML-ready splits saved successfully!")

    # =========================
    # 8. SAVE RUNTIME
    # =========================
    total_runtime = time.time() - total_start_time

    runtime_file = dataset_dir / "runtime_full.txt"

    with open(runtime_file, "w") as f:
        f.write(f"total_runtime_seconds: {total_runtime}\n")
        f.write(f"load_time: {load_time}\n")
        f.write(f"prep_time: {prep_time}\n")
        f.write(f"datasail_time: {datasail_time}\n")

    print(f"\nRuntime saved to: {runtime_file}")
    print(f"Total runtime: {total_runtime:.2f}s")


# =========================
# MAIN
# =========================

def main():

    datasets = ["bace", "bbbp", "tox21"]

    for dataset in datasets:
        run_datasail_full(dataset)

    print("\nAll full DataSAIL experiments complete")


if __name__ == "__main__":
    main()