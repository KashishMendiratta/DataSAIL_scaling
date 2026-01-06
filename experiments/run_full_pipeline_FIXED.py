"""
Full pipeline: Down-sample, DataSAIL split, kNN assignment
VERSION - correctly combines all samples
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datasail.sail import sail
from collections import Counter

from src.utils import load_bace_dataset
from src.downsampling import random_downsample
from src.assignment import knn_assign, assign_with_confidence, evaluate_assignment_quality


def run_pipeline(
    df: pd.DataFrame,
    fps: np.ndarray,
    downsample_ratio: float = 0.25,
    k: int = 5,
    output_dir: Path = None
):
    """
    Run full down-sampling pipeline.
    
    Args:
        df: Dataset DataFrame
        fps: Molecular fingerprints
        downsample_ratio: Fraction to down-sample
        k: Number of neighbors for assignment
        output_dir: Where to save results
    """
    print("="*70)
    print(f"Full Pipeline: {downsample_ratio*100:.0f}% down-sampling, k={k}")
    print("="*70)
    
    # Step 1: Down-sample
    print("\n[1/4] Down-sampling...")
    start_time = time.time()
    
    sampled_df, remaining_df = random_downsample(df, downsample_ratio, seed=42)
    
    # IMPORTANT: Store original indices before reset
    sampled_original_indices = sampled_df.index.tolist()
    remaining_original_indices = remaining_df.index.tolist()
    
    # Get fingerprints using ORIGINAL indices
    sampled_fps = fps[sampled_original_indices]
    remaining_fps = fps[remaining_original_indices]
    
    downsample_time = time.time() - start_time
    
    print(f" Completed in {downsample_time:.2f}s")
    print(f" Sampled: {len(sampled_df)} (original indices: {len(sampled_original_indices)})")
    print(f" Remaining: {len(remaining_df)} (original indices: {len(remaining_original_indices)})")
    
    # Step 2: Prepare DataSAIL input
    print("\n[2/4] Running DataSAIL on down-sampled data...")
    
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    smiles_col = 'mol' if 'mol' in sampled_df.columns else 'smiles'
    datasail_input = pd.DataFrame({
        'id': [f"mol_{i}" for i in range(len(sampled_df))],
        'smiles': sampled_df[smiles_col].values
    })
    
    input_file = temp_dir / "sampled_for_datasail.csv"
    datasail_input.to_csv(input_file, index=False)
    
    # Run DataSAIL
    datasail_output = output_dir / "datasail_splits"
    datasail_output.mkdir(parents=True, exist_ok=True)
    
    datasail_start = time.time()
    
    sail(
        e_type='M',
        e_data=str(input_file),
        techniques=['C1e'],
        e_sim='ecfp',
        e_clusters=10,
        splits=[0.7, 0.2, 0.1],
        names=['train', 'val', 'test'],
        output=str(datasail_output),
        verbosity='W',
        runs=1,
        threads=1,
        max_sec=120
    )
    
    datasail_time = time.time() - datasail_start
    
    print(f"   Completed in {datasail_time:.2f}s")
    
    # Step 3: Read DataSAIL results
    print("\n[3/4] Reading DataSAIL splits...")
    
    splits_file = datasail_output / "C1e" / f"Molecule_sampled_for_datasail_splits.tsv"
    splits_df = pd.read_csv(splits_file, sep='\t')
    
    # Map back (splits_df has sequential indices, we need original indices)
    sampled_splits = pd.Series(splits_df['Split'].values)
    
    split_counts = Counter(sampled_splits)
    print(f" Split distribution in down-sampled data:")
    for split in ['train', 'val', 'test']:
        count = split_counts.get(split, 0)
        pct = 100 * count / len(sampled_splits)
        print(f"     {split}: {count} ({pct:.1f}%)")
    
    # Step 4: kNN assignment
    print(f"\n[4/4] Assigning remaining samples with k={k}...")
    
    assignment_start = time.time()
    
    remaining_assignments, confidences = assign_with_confidence(
        sampled_fps, sampled_splits, remaining_fps, k=k
    )
    
    assignment_time = time.time() - assignment_start
    
    print(f" Completed in {assignment_time:.2f}s")
    
    metrics = evaluate_assignment_quality(remaining_assignments, confidences)
    print(f" Assigned: {metrics['total_assigned']}")
    print(f" Mean confidence: {metrics['mean_confidence']:.3f}")
    print(f" Distribution:")
    for split in ['train', 'val', 'test']:
        count = metrics['split_distribution'].get(split, 0)
        pct = 100 * count / len(remaining_assignments)
        print(f"     {split}: {count} ({pct:.1f}%)")
    
    # Step 5: Combine results
    print("\n[5/5] Combining final splits...")
    
    # Create final split assignments for all data using original indices
    final_splits = pd.Series(index=range(len(df)), dtype=str)
    
    # Assign sampled data using original indices
    for i, orig_idx in enumerate(sampled_original_indices):
        final_splits.iloc[orig_idx] = sampled_splits.iloc[i]
    
    # Assign remaining data using original indices
    for i, orig_idx in enumerate(remaining_original_indices):
        final_splits.iloc[orig_idx] = remaining_assignments[i]
    
    # Verify no NaN values
    if final_splits.isna().any():
        print(f"   WARNING: {final_splits.isna().sum()} samples have no assignment!")
    
    final_counts = Counter(final_splits)
    print(f"   - Final distribution (all {len(df)} samples):")
    for split in ['train', 'val', 'test']:
        count = final_counts.get(split, 0)
        pct = 100 * count / len(df)
        target_pct = {'train': 70, 'val': 20, 'test': 10}[split]
        diff = pct - target_pct
        print(f"     {split}: {count} ({pct:.1f}%) [target: {target_pct}%, diff: {diff:+.1f}%]")
    
    # Save results
    results_file = output_dir / "final_splits.tsv"
    result_df = pd.DataFrame({
        'ID': [f"mol_{i}" for i in range(len(df))],
        'Split': final_splits.values,
        'Source': ['sampled' if i in sampled_original_indices else 'assigned' for i in range(len(df))]
    })
    result_df.to_csv(results_file, sep='\t', index=False)
    
    print(f"\n   Saved to: {results_file}")
    
    # Summary
    total_time = downsample_time + datasail_time + assignment_time
    print("\n" + "="*70)
    print("Pipeline Summary")
    print("="*70)
    print(f"Total runtime: {total_time:.2f}s")
    print(f"  - Down-sampling: {downsample_time:.2f}s")
    print(f"  - DataSAIL: {datasail_time:.2f}s")
    print(f"  - kNN assignment: {assignment_time:.2f}s")
    
    return final_splits, total_time


def main():
    print("="*70)
    print("FULL PIPELINE TEST - FIXED VERSION")
    print("="*70)
    
    # Load data
    print("\nLoading BACE dataset...")
    df, fps = load_bace_dataset()
    print(f"Loaded {len(df)} molecules")
    
    # Test pipeline with different configurations
    configs = [
        {'ratio': 0.25, 'k': 3},
        {'ratio': 0.25, 'k': 5},
        {'ratio': 0.10, 'k': 5},
    ]
    
    results_dir = Path("results/pipeline_tests_fixed")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for i, config in enumerate(configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"Configuration {i}/{len(configs)}")
        print(f"{'#'*70}")
        
        output_dir = results_dir / f"config_{i}_ratio{config['ratio']}_k{config['k']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_pipeline(df, fps, 
                    downsample_ratio=config['ratio'],
                    k=config['k'],
                    output_dir=output_dir)
    
    print("\n\n" + "="*70)
    print("ALL PIPELINE TESTS COMPLETE")
    print("="*70)
    print(f"\nResults saved in: {results_dir}")


if __name__ == "__main__":
    main()
