"""
Test DataSAIL on BACE dataset - Cluster-Based Split with ECFP
This is what we'll actually use for the thesis
"""
import pandas as pd
import time
from pathlib import Path
from datasail.sail import sail

def main():
    print("="*70)
    print("Testing DataSAIL - Cluster-Based Split (C1e with ECFP)")
    print("="*70)
    
    # 1. Load and prepare data
    data_path = Path("data/raw/moleculenet/bace.csv")
    df = pd.read_csv(data_path)
    
    smiles_col = 'mol' if 'mol' in df.columns else 'smiles'
    
    # Create CSV with ID and SMILES
    processed_file = Path("data/processed/bace_smiles.csv")
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    
    smiles_df = pd.DataFrame({
        'id': [f"mol_{i}" for i in range(len(df))],
        'smiles': df[smiles_col].values
    })
    smiles_df.to_csv(processed_file, index=False)
    
    print(f"\n1. Prepared {len(smiles_df)} molecules")
    print(f"   - File: {processed_file}")
    
    # 2. Output directory
    output_dir = Path("results/splits/bace_cluster_ecfp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Run DataSAIL with cluster-based split
    print(f"\n2. Running DataSAIL Cluster Split...")
    print(f"   - Technique: C1e (Cluster-based, 1D, e-entity)")
    print(f"   - Similarity: ECFP fingerprints")
    print(f"   - Number of clusters: 10")
    print(f"   - Splits: 70% train, 20% val, 10% test")
    
    start_time = time.time()
    
    try:
        sail(
            e_type='M',              # Molecule
            e_data=str(processed_file),
            techniques=['C1e'],      # C1e = Cluster-based 1D on e-entity
            e_sim='ecfp',           # Use ECFP for similarity
            e_clusters=10,          # Number of clusters
            splits=[0.7, 0.2, 0.1],
            names=['train', 'val', 'test'],
            output=str(output_dir),
            verbosity='W',          # Warning level (less verbose)
            runs=1,
            threads=1,
            max_sec=60              # Max 60 seconds for optimization
        )
        
        runtime = time.time() - start_time
        print(f"\n✓ DataSAIL completed!")
        print(f"   - Runtime: {runtime:.2f} seconds")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Analyze results
    print(f"\n3. Analyzing output...")
    output_files = list(output_dir.glob("*.tsv")) + list(output_dir.glob("*.txt"))
    print(f"   - Found {len(output_files)} files:")
    for f in sorted(output_files):
        print(f"     • {f.name}")
    
    # Find split assignments file
    possible_files = [
        output_dir / "e_mol.tsv",
        output_dir / "splits.tsv",
        output_dir / "e_data.tsv"
    ]
    
    split_file = None
    for f in possible_files:
        if f.exists():
            split_file = f
            break
    
    if split_file:
        print(f"\n4. Split distribution (from {split_file.name}):")
        assignments = {}
        with open(split_file) as f:
            header = f.readline()
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        assignments[parts[0]] = parts[1]
        
        from collections import Counter
        counts = Counter(assignments.values())
        total = len(assignments)
        for split_name in ['train', 'val', 'test']:
            if split_name in counts:
                count = counts[split_name]
                pct = 100 * count / total
                print(f"   - {split_name:5s}: {count:4d} ({pct:5.1f}%)")
    else:
        print("\n✗ Could not find split assignments file")
        print(f"   Available files: {[f.name for f in output_files]}")
    
    print("\n" + "="*70)
    print("✓ Test Complete!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Examine the output files")
    print("  2. Check the cluster assignments")
    print("  3. This is your baseline for comparison!")

if __name__ == "__main__":
    main()
