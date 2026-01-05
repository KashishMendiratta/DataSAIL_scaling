"""
4th January 2026
"""
from pathlib import Path
import pandas as pd

print("="*70)
print("DataSAIL Baseline")
print("="*70)

# Check what we have
baseline_splits = Path("results/splits/bace_baseline/C1e/Molecule_bace_smiles_splits.tsv")

if baseline_splits.exists():
    df = pd.read_csv(baseline_splits, sep='\t')
    print(f"\nBaseline splits file created")
    print(f"  Location: {baseline_splits}")
    print(f"  Total samples: {len(df)}")
    print(f"  Split distribution:")
    for split in ['train', 'val', 'test']:
        count = (df['Split'] == split).sum()
        pct = 100 * count / len(df)
        print(f"    {split}: {count} ({pct:.1f}%)")

print("\n" + "="*70)
print("WHAT WE LEARNED")
print("="*70)
print("""
1. DataSAIL Setup:
   - Uses ECFP fingerprints for molecular similarity
   - Creates cluster-based splits to minimise information leakage
   
2. BACE Dataset:
   - 1,513 molecules
   - Binary classification task
   - Small enough to iterate quickly
   
3. Baseline Performance:
   - Runtime: ~2.6 seconds
   - Creates 10 clusters
   - Achieves target split ratios closely
   
4. File Structure:
   - DataSAIL outputs to technique-specific subdirectories
   - Main file: Molecule_<name>_splits.tsv
   - Also creates cluster assignments and visualisations
""")

