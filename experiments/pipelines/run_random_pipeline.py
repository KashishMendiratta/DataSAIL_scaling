"""
Random split pipeline (baseline).

Creates train/val/test splits using random splitting
without considering molecular similarity.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import time
from sklearn.model_selection import train_test_split

from src.utils import save_splits
from src.utils import compute_max_deviation

# =========================
# DATA LOADING
# =========================

def load_dataset(name):

    if name == "bace":
        path = "data/raw/moleculenet/bace.csv"

    elif name == "bbbp":
        path = "data/raw/moleculenet/bbbp.csv"

    elif name == "tox21":
        path = "data/raw/moleculenet/tox21.csv"

    elif name == "hiv":
        path = "data/raw/moleculenet/hiv.csv"

    else:
        raise ValueError(f"Unknown dataset: {name}")

    df = pd.read_csv(path)
    return df


# =========================
# RANDOM SPLIT PIPELINE
# =========================

def run_random_pipeline(dataset_name):

    print("=" * 70)
    print(f"Random Split Pipeline: {dataset_name.upper()}")
    print("=" * 70)

    # TOTAL TIMER
    total_start_time = time.time()

    base_dir = Path("results/experiments")
    dataset_dir = base_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. Load dataset
    # =========================
    print("\n1. Loading dataset...")
    t0 = time.time()

    df = load_dataset(dataset_name)

    load_time = time.time() - t0
    print(f"Loaded {len(df)} molecules in {load_time:.4f}s")

    # =========================
    # 2. Random split
    # =========================
    print("\n2. Performing random split...")
    t0 = time.time()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1/3,
        random_state=42,
        shuffle=True
    )

    split_time = time.time() - t0
    print(f"Split time: {split_time:.4f}s")

    # =========================
    # 3. Print distribution
    # =========================
    print("\n3. Split distribution:")

    total = len(df)

    counts = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df)
    }

    for split in ["train", "val", "test"]:
        count = counts[split]
        pct = 100 * count / total
        print(f"{split}: {count} ({pct:.2f}%)")

    max_dev = compute_max_deviation(counts, total)

    print(f"Max deviation: ±{max_dev:.2f} percentage points")

    # =========================
    # 4. SAVE ML SPLITS
    # =========================
    print("\n4. Saving ML-ready splits...")

    save_splits(train_df, val_df, test_df, dataset_name, "random")

    print("ML-ready splits saved")

    # =========================
    # 5. SAVE RUNTIME
    # =========================
    total_runtime = time.time() - total_start_time

    runtime_file = dataset_dir / "runtime_random.txt"

    with open(runtime_file, "w") as f:
        f.write(f"total_runtime_seconds: {total_runtime}\n")
        f.write(f"load_time: {load_time}\n")
        f.write(f"split_time: {split_time}\n")

    print(f"\nRuntime saved to: {runtime_file}")
    print(f"Total runtime: {total_runtime:.4f}s")


# =========================
# MAIN
# =========================

def main():

    datasets = ["bace", "bbbp", "tox21", "hiv"]

    for dataset in datasets:
        run_random_pipeline(dataset)

    print("\nAll random split experiments complete")


if __name__ == "__main__":
    main()