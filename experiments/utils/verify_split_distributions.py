"""
Verify and report correct split distributions from pipeline results.
This addresses Roman's feedback about correctly reporting combined splits.
"""
from pathlib import Path
import pandas as pd
from collections import Counter

def analyze_splits(config_dir: Path):
    """Analyze splits from a pipeline run."""
    splits_file = config_dir / "final_splits.tsv"
    
    if not splits_file.exists():
        print(f"File not found: {splits_file}")
        return None
    
    df = pd.read_csv(splits_file, sep='\t')
    
    # Overall statistics
    total = len(df)
    sampled_count = (df['Source'] == 'sampled').sum()
    assigned_count = (df['Source'] == 'assigned').sum()
    
    print(f"\nTotal samples: {total}")
    print(f"  - Down-sampled: {sampled_count} ({100*sampled_count/total:.1f}%)")
    print(f"  - kNN-assigned: {assigned_count} ({100*assigned_count/total:.1f}%)")
    
    # Splits in down-sampled data
    sampled_df = df[df['Source'] == 'sampled']
    sampled_splits = Counter(sampled_df['Split'])
    print(f"\nDown-sampled data splits ({len(sampled_df)} samples):")
    for split in ['train', 'val', 'test']:
        count = sampled_splits.get(split, 0)
        pct = 100 * count / len(sampled_df) if len(sampled_df) > 0 else 0
        print(f"  {split}: {count} ({pct:.1f}%)")
    
    # Splits in kNN-assigned data
    assigned_df = df[df['Source'] == 'assigned']
    assigned_splits = Counter(assigned_df['Split'])
    print(f"\nkNN-assigned data splits ({len(assigned_df)} samples):")
    for split in ['train', 'val', 'test']:
        count = assigned_splits.get(split, 0)
        pct = 100 * count / len(assigned_df) if len(assigned_df) > 0 else 0
        print(f"  {split}: {count} ({pct:.1f}%)")
    
    # Combined (final) splits - THIS IS WHAT MATTERS
    all_splits = Counter(df['Split'])
    print(f"\n*** COMBINED FINAL SPLITS (all {total} samples) ***")
    print("This is what will be used for ML training:")
    results = {}
    for split in ['train', 'val', 'test']:
        count = all_splits.get(split, 0)
        pct = 100 * count / total
        target_pct = {'train': 70, 'val': 20, 'test': 10}[split]
        diff = pct - target_pct
        print(f"  {split}: {count} ({pct:.1f}%) — Target: {target_pct}%, Deviation: {diff:+.1f}%")
        results[split] = {'count': count, 'pct': pct, 'diff': diff}
    
    return results

def main():
    print("="*70)
    print("VERIFYING SPLIT DISTRIBUTIONS")
    print("="*70)
    print("\nNote: The 'Combined Final Splits' are what matter for ML training.")
    print("These combine both down-sampled (DataSAIL) and kNN-assigned samples.\n")
    
    results_dir = Path("results/pipeline_tests")
    
    configs = [
        ("config_1_ratio0.25_k3", "Configuration 1: 25% down-sampling, k=3"),
        ("config_2_ratio0.25_k5", "Configuration 2: 25% down-sampling, k=5"),
        ("config_3_ratio0.1_k5", "Configuration 3: 10% down-sampling, k=5"),
    ]
    
    all_results = {}
    
    for config_name, title in configs:
        print("="*70)
        print(title)
        print("="*70)
        result = analyze_splits(results_dir / config_name)
        if result:
            all_results[config_name] = result
        print()
    
    # Summary for report
    print("="*70)
    print("SUMMARY FOR REPORT")
    print("="*70)
    print("\nCopy this into your report's 'Split Distribution Results' section:\n")
    
    targets = {'train': 70, 'val': 20, 'test': 10}
    
    for config_name, title in configs:
        if config_name in all_results:
            results = all_results[config_name]
            print(f"**{title}**")
            print(f"Combined final splits (all 1,513 samples):")
            for split in ['train', 'val', 'test']:
                r = results[split]
                target = targets[split]
                print(f"  - {split.capitalize()}: {r['count']} ({r['pct']:.1f}%) — Target: {target}%, Deviation: **{r['diff']:+.1f}%**")
            print()

if __name__ == "__main__":
    main()
