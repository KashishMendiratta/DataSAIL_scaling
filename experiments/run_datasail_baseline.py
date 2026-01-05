"""
DataSAIL Baseline - Cluster-based split on BACE dataset
This establishes the baseline performance to compare against
"""
import pandas as pd
import time
from pathlib import Path
from datasail.sail import sail
from collections import Counter

def main():
    print("="*70)
    print("DataSAIL Baseline: Cluster-Based Split (C1e + ECFP)")
    print("="*70)
    
    # 1. Load and prepare data
    data_path = Path("data/raw/moleculenet/bace.csv")
    df = pd.read_csv(data_path)
    
    smiles_col = 'mol' if 'mol' in df.columns else 'smiles'
    label_col = 'Class' if 'Class' in df.columns else 'pIC50'
    
    processed_file = Path("data/processed/bace_smiles.csv")
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    
    smiles_df = pd.DataFrame({
        'id': [f"mol_{i}" for i in range(len(df))],
        'smiles': df[smiles_col].values
    })
    smiles_df.to_csv(processed_file, index=False)
    
    print(f"\n1. Dataset prepared")
    print(f"   - Total molecules: {len(smiles_df)}")
    print(f"   - Label column: {label_col}")
    if label_col in df.columns:
        print(f"   - Label distribution: {dict(df[label_col].value_counts())}")
    
    # 2. Run DataSAIL
    output_dir = Path("results/splits/bace_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n2. Running DataSAIL...")
    print(f"   - Technique: C1e (cluster-based, 1D)")
    print(f"   - Similarity: ECFP")
    print(f"   - Clusters: 10")
    print(f"   - Target splits: 70/20/10")
    
    start_time = time.time()
    
    sail(
        e_type='M',
        e_data=str(processed_file),
        techniques=['C1e'],
        e_sim='ecfp',
        e_clusters=10,
        splits=[0.7, 0.2, 0.1],
        names=['train', 'val', 'test'],
        output=str(output_dir),
        verbosity='W',  # Less verbose
        runs=1,
        threads=1,
        max_sec=120
    )
    
    runtime = time.time() - start_time
    
    print(f"\nDataSAIL completed in {runtime:.2f} seconds")
    
    # 3. Analyse results
    print(f"\n3. Analysing results...")
    
    # Find the splits file
    splits_file = output_dir / "C1e" / f"Molecule_{processed_file.stem}_splits.tsv"
    clusters_file = output_dir / "C1e" / f"Molecule_{processed_file.stem}_clusters.tsv"
    
    if not splits_file.exists():
        print(f"Could not find splits file: {splits_file}")
        return
    
    # Read splits
    splits_df = pd.read_csv(splits_file, sep='\t')
    print(f"   - Splits file: {splits_file.name}")
    print(f"   - Columns: {list(splits_df.columns)}")
    
    # Analyse split distribution
    split_counts = Counter(splits_df.iloc[:, 1])  # Second column is split assignment
    total = len(splits_df)
    
    print(f"\n4. Split distribution:")
    for split_name in ['train', 'val', 'test']:
        count = split_counts.get(split_name, 0)
        pct = 100 * count / total
        target_pct = {0: 70, 1: 20, 2: 10}[['train', 'val', 'test'].index(split_name)]
        diff = pct - target_pct
        print(f"   - {split_name:5s}: {count:4d} ({pct:5.1f}%) [target: {target_pct}%, diff: {diff:+.1f}%]")
    
    # Analyse clusters
    if clusters_file.exists():
        clusters_df = pd.read_csv(clusters_file, sep='\t')
        print(f"\n5. Cluster analysis:")
        print(f"   - Number of clusters: {clusters_df.iloc[:, 1].nunique()}")
        cluster_sizes = clusters_df.iloc[:, 1].value_counts()
        print(f"   - Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")
    
    # Check for visualizations
    vis_files = list((output_dir / "C1e").glob("*.png"))
    if vis_files:
        print(f"\n6. Visualizations created:")
        for f in vis_files:
            print(f"   - {f.name}")
    
    print("\n" + "="*70)
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Key file: {splits_file}")

if __name__ == "__main__":
    main()
