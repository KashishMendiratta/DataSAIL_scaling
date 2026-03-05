"""Test pipeline on BBBP dataset."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datasail.sail import sail
from collections import Counter

from src.utils import smiles_to_fingerprint
from src.downsampling import random_downsample
from src.assignment import assign_with_confidence, evaluate_assignment_quality


def load_bbbp_dataset(data_path="data/raw/moleculenet/bbbp.csv"):
    """Load BBBP dataset and compute fingerprints."""
    df = pd.read_csv(data_path)
    
    # Find SMILES column
    smiles_col = 'smiles' if 'smiles' in df.columns else 'mol'
    
    # Compute fingerprints
    fps = []
    valid_idx = []
    for i, smiles in enumerate(df[smiles_col]):
        fp = smiles_to_fingerprint(smiles)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    
    df = df.iloc[valid_idx].reset_index(drop=True)
    fps = np.array(fps)
    
    print(f"Loaded {len(df)} molecules with valid SMILES")
    return df, fps


def run_pipeline_on_bbbp():
    """Run pipeline on BBBP dataset."""
    print("="*70)
    print("Testing Pipeline on BBBP Dataset")
    print("="*70)
    
    # Load data
    print("\n1. Loading BBBP dataset...")
    df, fps = load_bbbp_dataset()
    print(f"   Total samples: {len(df)}")
    
    # Configuration
    downsample_ratio = 0.25
    k = 5
    
    print(f"\n2. Running pipeline with {downsample_ratio*100:.0f}% sampling, k={k}")
    
    # Down-sample
    print("\n   [1/4] Down-sampling...")
    start = time.time()
    sampled_df, remaining_df = random_downsample(df, downsample_ratio, seed=42)
    
    sampled_indices = sampled_df.index.tolist()
    remaining_indices = remaining_df.index.tolist()
    
    sampled_fps = fps[sampled_indices]
    remaining_fps = fps[remaining_indices]
    downsample_time = time.time() - start
    
    print(f"   - Sampled: {len(sampled_df)}")
    print(f"   - Remaining: {len(remaining_df)}")
    
    # Prepare DataSAIL input
    print("\n   [2/4] Running DataSAIL...")
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    smiles_col = 'smiles' if 'smiles' in sampled_df.columns else 'mol'
    datasail_input = pd.DataFrame({
        'id': [f"mol_{i}" for i in range(len(sampled_df))],
        'smiles': sampled_df[smiles_col].values
    })
    
    input_file = temp_dir / "bbbp_sampled.csv"
    datasail_input.to_csv(input_file, index=False)
    
    output_dir = Path("results/bbbp_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasail_start = time.time()
    sail(
        e_type='M',
        e_data=str(input_file),
        techniques=['C1e'],
        e_sim='ecfp',
        e_clusters=10,
        splits=[0.7, 0.2, 0.1],
        names=['train', 'val', 'test'],
        output=str(output_dir),
        verbosity='W',
        runs=1,
        threads=1,
        max_sec=120
    )
    datasail_time = time.time() - datasail_start
    
    # Read results
    print("\n   [3/4] Reading DataSAIL splits...")
    splits_file = output_dir / "C1e" / "Molecule_bbbp_sampled_splits.tsv"
    splits_df = pd.read_csv(splits_file, sep='\t')
    sampled_splits = pd.Series(splits_df['Split'].values)
    
    split_counts = Counter(sampled_splits)
    print(f"   Down-sampled split distribution:")
    for split in ['train', 'val', 'test']:
        count = split_counts.get(split, 0)
        pct = 100 * count / len(sampled_splits)
        print(f"     {split}: {count} ({pct:.1f}%)")
    
    # kNN assignment
    print("\n   [4/4] kNN assignment...")
    assignment_start = time.time()
    remaining_assignments, confidences = assign_with_confidence(
        sampled_fps, sampled_splits, remaining_fps, k=k
    )
    assignment_time = time.time() - assignment_start
    
    metrics = evaluate_assignment_quality(remaining_assignments, confidences)
    print(f"   - Assigned: {metrics['total_assigned']}")
    print(f"   - Mean confidence: {metrics['mean_confidence']:.3f}")
    
    # Combine
    final_splits = pd.Series(index=range(len(df)), dtype=str)
    for i, orig_idx in enumerate(sampled_indices):
        final_splits.iloc[orig_idx] = sampled_splits.iloc[i]
    for i, orig_idx in enumerate(remaining_indices):
        final_splits.iloc[orig_idx] = remaining_assignments[i]
    
    final_counts = Counter(final_splits)
    print(f"\n3. Final Results (all {len(df)} samples):")
    for split in ['train', 'val', 'test']:
        count = final_counts.get(split, 0)
        pct = 100 * count / len(df)
        target = {'train': 70, 'val': 20, 'test': 10}[split]
        diff = pct - target
        print(f"   {split}: {count} ({pct:.1f}%) [target: {target}%, diff: {diff:+.1f}%]")
    
    # Performance summary
    total_time = downsample_time + datasail_time + assignment_time
    print(f"\n4. Performance:")
    print(f"   Total runtime: {total_time:.2f}s")
    print(f"   - Down-sampling: {downsample_time:.2f}s")
    print(f"   - DataSAIL: {datasail_time:.2f}s")
    print(f"   - kNN: {assignment_time:.2f}s")
    
    print("\n" + "="*70)
    print("✓ BBBP Test Complete")
    print("="*70)


if __name__ == "__main__":
    run_pipeline_on_bbbp()
