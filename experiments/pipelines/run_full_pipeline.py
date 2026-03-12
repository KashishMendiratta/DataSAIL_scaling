"""
Full pipeline debug version.
Used to test pipeline configurations.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from collections import Counter
from datasail.sail import sail

from src.utils import load_bace_dataset
from src.downsampling import random_downsample
from src.assignment import assign_with_confidence


def run_pipeline(df, fps, downsample_ratio=0.25, k=5, output_dir=None):

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"Pipeline Debug: ratio={downsample_ratio}, k={k}")
    print("="*70)

    # Downsample
    sampled_df, remaining_df = random_downsample(df, downsample_ratio, seed=42)

    sampled_idx = sampled_df.index.tolist()
    remaining_idx = remaining_df.index.tolist()

    sampled_fps = fps[sampled_idx]
    remaining_fps = fps[remaining_idx]

    # Prepare DataSAIL
    temp_dir = Path("data/temp")
    temp_dir.mkdir(exist_ok=True)

    datasail_input = pd.DataFrame({
        "id":[f"mol_{i}" for i in range(len(sampled_df))],
        "smiles":sampled_df["mol"]
    })

    input_file = temp_dir / "sampled_for_datasail.csv"
    datasail_input.to_csv(input_file,index=False)

    sail(
        e_type="M",
        e_data=str(input_file),
        techniques=["C1e"],
        e_sim="ecfp",
        e_clusters=10,
        splits=[0.7,0.2,0.1],
        names=["train","val","test"],
        output=str(output_dir),
        verbosity="W",
        runs=1
    )

    splits_file = output_dir / "C1e" / "Molecule_sampled_for_datasail_splits.tsv"
    splits_df = pd.read_csv(splits_file,sep="\t")

    sampled_splits = pd.Series(splits_df["Split"].values)

    remaining_assignments,_ = assign_with_confidence(
        sampled_fps,
        sampled_splits,
        remaining_fps,
        k=k
    )

    final_splits = pd.Series(index=range(len(df)),dtype=str)

    for i,idx in enumerate(sampled_idx):
        final_splits.iloc[idx] = sampled_splits.iloc[i]

    for i,idx in enumerate(remaining_idx):
        final_splits.iloc[idx] = remaining_assignments[i]

    result_df = pd.DataFrame({
        "ID":[f"mol_{i}" for i in range(len(df))],
        "Split":final_splits.values
    })

    result_df.to_csv(output_dir/"final_splits.tsv",sep="\t",index=False)

    print(f"Saved debug results to {output_dir}")


def main():

    df, fps = load_bace_dataset()

    configs = [
        {"ratio":0.25,"k":3},
        {"ratio":0.25,"k":5},
        {"ratio":0.10,"k":5},
    ]

    debug_root = Path("results/pipeline_debug")

    for i,c in enumerate(configs):

        output_dir = debug_root / f"config_{i+1}"

        run_pipeline(
            df,
            fps,
            downsample_ratio=c["ratio"],
            k=c["k"],
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()