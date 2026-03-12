"""Test balanced kNN assignment on BACE."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_bace_dataset
from src.downsampling import random_downsample
from src.assignment import balanced_knn_assign
from collections import Counter

# Load and downsample
df, fps = load_bace_dataset()
sampled_df, remaining_df = random_downsample(df, 0.25, seed=42)

sampled_indices = sampled_df.index.tolist()
remaining_indices = remaining_df.index.tolist()

sampled_fps = fps[sampled_indices]
remaining_fps = fps[remaining_indices]

# Mock DataSAIL splits (use the actual ones from your previous run)
# For now, let's create realistic mock splits
import numpy as np
np.random.seed(42)
sampled_splits = np.random.choice(['train', 'val', 'test'], 
                                  size=len(sampled_df),
                                  p=[0.675, 0.212, 0.113])
import pandas as pd
sampled_splits = pd.Series(sampled_splits)

print("="*70)
print("Comparing: Naive kNN vs Balanced kNN")
print("="*70)

print("\nDown-sampled splits:")
counts = Counter(sampled_splits)
for split in ['train', 'val', 'test']:
    count = counts[split]
    pct = 100 * count / len(sampled_splits)
    print(f"  {split}: {count} ({pct:.1f}%)")

# Test different balance weights
for balance_weight in [0.5, 1.0, 2.0]:
    print(f"\n{'='*70}")
    print(f"Balance weight: {balance_weight}")
    print(f"{'='*70}")
    
    assignments, confidences = balanced_knn_assign(
        sampled_fps, sampled_splits, remaining_fps,
        k=5, balance_weight=balance_weight
    )
    
    print(f"\nAssigned {len(assignments)} samples")
    print(f"Mean confidence: {confidences.mean():.3f}")
    
    # Combined splits
    total = len(df)
    final_counts = Counter(sampled_splits.tolist() + assignments.tolist())
    
    print(f"\nFinal distribution (all {total} samples):")
    for split in ['train', 'val', 'test']:
        count = final_counts[split]
        pct = 100 * count / total
        target = {'train': 70, 'val': 20, 'test': 10}[split]
        diff = pct - target
        print(f"  {split}: {count} ({pct:.1f}%) [target: {target}%, diff: {diff:+.1f}%]")
