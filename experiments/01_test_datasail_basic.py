"""
Test DataSAIL on BACE dataset - Basic Random Split
Goal: Understand DataSAIL's workflow and measure baseline performance
"""
import pandas as pd
import time
from pathlib import Path
from datasail.sail import sail

def main():
    print("="*70)
    print("Testing DataSAIL on BACE Dataset - Random Split")
    print("="*70)
    
    # 1. Load data
    data_path = Path("data/raw/moleculenet/bace.csv")
    df = pd.read_csv(data_path)
    print(f"\n1. Loaded BACE dataset")
    print(f"   - Shape: {df.shape}")
    
    # Check what column contains SMILES
    smiles_col = 'mol' if 'mol' in df.columns else 'smiles'
    print(f"   - SMILES column: '{smiles_col}'")
    
    # 2. Prepare data in CSV format (DataSAIL expects CSV or TSV)
    # Create a simple CSV with ID and SMILES
    processed_file = Path("data/processed/bace_smiles.csv")
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with just ID and SMILES
    smiles_df = pd.DataFrame({
        'id': [f"mol_{i}" for i in range(len(df))],
        'smiles': df[smiles_col].values
    })
    smiles_df.to_csv(processed_file, index=False)
    
    print(f"\n2. Prepared input file: {processed_file}")
    print(f"   - Format: CSV")
    print(f"   - Columns: id, smiles")
    print(f"   - {len(smiles_df)} molecules")
    
    # 3. Set up output directory
    output_dir = Path("results/splits/bace_random")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Run DataSAIL with Random split (simplest)
    print(f"\n3. Running DataSAIL Random Split...")
    print(f"   - Technique: Random (R)")
    print(f"   - Splits: 70% train, 20% val, 10% test")
    
    start_time = time.time()
    
    try:
        sail(
            e_type='M',           # M = Molecule
            e_data=str(processed_file),
            techniques=['R'],      # R = Random split
            splits=[0.7, 0.2, 0.1],
            names=['train', 'val', 'test'],
            output=str(output_dir),
            verbosity='I',        # I = Info level logging
            runs=1,
            threads=1
        )
        
        runtime = time.time() - start_time
        print(f"\n✓ DataSAIL completed successfully!")
        print(f"   - Runtime: {runtime:.2f} seconds")
        
    except Exception as e:
        print(f"\n✗ Error running DataSAIL: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Check outputs
    print(f"\n4. Checking output files...")
    output_files = list(output_dir.glob("*.tsv")) + list(output_dir.glob("*.txt"))
    print(f"   - Found {len(output_files)} output files:")
    for f in sorted(output_files):
        lines = len(open(f).readlines())
        print(f"     • {f.name}: {lines} lines")
    
    # 6. Parse split assignments
    print(f"\n5. Analyzing splits...")
    # Look for the main splits file
    possible_files = [
        output_dir / "splits.tsv",
        output_dir / "splits.txt",
        output_dir / "e_mol.tsv"
    ]
    
    split_file = None
    for f in possible_files:
        if f.exists():
            split_file = f
            break
    
    if split_file:
        print(f"   - Reading: {split_file.name}")
        assignments = {}
        with open(split_file) as f:
            header = f.readline()  # Skip header if present
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        mol_id = parts[0]
                        split_name = parts[1]
                        assignments[mol_id] = split_name
        
        from collections import Counter
        split_counts = Counter(assignments.values())
        print(f"   - Split distribution:")
        for split_name, count in sorted(split_counts.items()):
            pct = 100 * count / len(assignments)
            print(f"     • {split_name}: {count} ({pct:.1f}%)")
    else:
        print("   - Could not find splits file")
        print(f"   - Available files: {[f.name for f in output_files]}")
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print(f"\nOutput saved to: {output_dir}")

if __name__ == "__main__":
    main()
