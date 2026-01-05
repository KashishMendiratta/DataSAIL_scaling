"""
Test different down-sampling strategies on BACE dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import load_bace_dataset
from src.downsampling import (
    random_downsample,
    diversity_downsample
)

# ---------------------------------------------------------------------
# stratified downsampling implementation
# ---------------------------------------------------------------------
def stratified_downsample(df, labels, ratio, random_state=42):
    """
    Stratified downsampling using sklearn.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    labels : pd.Series
        Class labels aligned with df
    ratio : float
        Fraction to sample
    random_state : int
        Seed for reproducibility
    """
    assert len(df) == len(labels), "DF and labels length mismatch"

    sampled_df, remaining_df = train_test_split(
        df,
        train_size=ratio,
        stratify=labels,
        random_state=random_state
    )

    return sampled_df, remaining_df


def assert_stratification_ok(df, sampled_df, tol=0.05):
    """
    Assert that class proportions are preserved within tolerance.
    """
    orig_props = df['Class'].value_counts(normalize=True)
    samp_props = sampled_df['Class'].value_counts(normalize=True)

    diff = (orig_props - samp_props).abs()

    assert diff.max() < tol, (
        f"Stratified sampling failed.\n"
        f"Original proportions:\n{orig_props}\n"
        f"Sampled proportions:\n{samp_props}\n"
        f"Difference:\n{diff}"
    )


def main():
    print("=" * 70)
    print("Testing Down-sampling Strategies")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    print("\n1. Loading BACE dataset...")
    df, fps = load_bace_dataset()

    assert 'Class' in df.columns, "BACE dataset must contain 'Class' column"
    labels = df['Class']

    print(f"   - Total samples: {len(df)}")
    print(f"   - Fingerprint shape: {fps.shape}")
    print(f"   - Class distribution: {dict(labels.value_counts())}")

    # -----------------------------------------------------------------
    # Test different ratios
    # -----------------------------------------------------------------
    ratios = [0.1, 0.25, 0.5]

    for ratio in ratios:
        print(f"\n{'=' * 70}")
        print(f"Testing with {ratio * 100:.0f}% sampling ratio")
        print(f"{'=' * 70}")

        # -------------------------------------------------------------
        # 1. Random sampling
        # -------------------------------------------------------------
        print("\n1. Random Sampling:")
        sampled, remaining = random_downsample(df, ratio)

        print(f"   - Sampled: {len(sampled)} ({len(sampled) / len(df) * 100:.1f}%)")
        print(f"   - Remaining: {len(remaining)} ({len(remaining) / len(df) * 100:.1f}%)")
        print(f"   - Sampled class dist: {dict(sampled['Class'].value_counts())}")

        # -------------------------------------------------------------
        # 2. Stratified sampling
        # -------------------------------------------------------------
        print("\n2. Stratified Sampling:")
        sampled, remaining = stratified_downsample(df, labels, ratio)

        print(f"   - Sampled: {len(sampled)}")
        print(f"   - Sampled class dist: {dict(sampled['Class'].value_counts())}")

        orig_props = df['Class'].value_counts(normalize=True)
        samp_props = sampled['Class'].value_counts(normalize=True)

        print(f"   - Original proportions: {dict(orig_props.round(3))}")
        print(f"   - Sampled proportions:  {dict(samp_props.round(3))}")

        # Assertion: fail fast if stratification breaks
        assert_stratification_ok(df, sampled)

        # -------------------------------------------------------------
        # 3. Diversity sampling (k-means)
        # -------------------------------------------------------------
        print("\n3. Diversity Sampling (k-means):")
        sampled, remaining = diversity_downsample(
            fps,
            df,
            ratio,
            method="kmeans"
        )

        print(f"   - Sampled: {len(sampled)}")
        print(f"   - Sampled class dist: {dict(sampled['Class'].value_counts())}")

    # -----------------------------------------------------------------
    # Save a sample for next pipeline steps
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("All down-sampling methods working!")
    print("=" * 70)

    print("\nSaving 25% random sample for pipeline testing...")
    sampled, remaining = random_downsample(df, 0.25)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_path = output_dir / "bace_sampled_25pct.csv"
    remaining_path = output_dir / "bace_remaining_75pct.csv"

    sampled.to_csv(sampled_path, index=False)
    remaining.to_csv(remaining_path, index=False)

    print(f"   - Sampled:   {sampled_path}")
    print(f"   - Remaining: {remaining_path}")


if __name__ == "__main__":
    main()
