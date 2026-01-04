"""Debug DataSAIL output"""
import pandas as pd
from pathlib import Path
from datasail.sail import sail
import os

# Prepare data
data_path = Path("data/raw/moleculenet/bace.csv")
df = pd.read_csv(data_path)
smiles_col = 'mol' if 'mol' in df.columns else 'smiles'

processed_file = Path("data/processed/bace_smiles.csv")
smiles_df = pd.DataFrame({
    'id': [f"mol_{i}" for i in range(len(df))],
    'smiles': df[smiles_col].values
})
smiles_df.to_csv(processed_file, index=False)

output_dir = Path("results/splits/bace_test_debug")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {output_dir}")
print(f"Output directory exists: {output_dir.exists()}")
print(f"Output directory absolute path: {output_dir.absolute()}")

# Run DataSAIL
print("\nRunning DataSAIL...")
result = sail(
    e_type='M',
    e_data=str(processed_file),
    techniques=['C1e'],
    e_sim='ecfp',
    e_clusters=10,
    splits=[0.7, 0.2, 0.1],
    names=['train', 'val', 'test'],
    output=str(output_dir.absolute()),  # Use absolute path
    verbosity='I',  # More verbose
    runs=1,
    threads=1,
    max_sec=60
)

print(f"\nResult from sail(): {result}")
print(f"Result type: {type(result)}")

# Check output directory after run
print(f"\n\nChecking output directory after run:")
print(f"Directory exists: {output_dir.exists()}")
print(f"Contents:")
for item in output_dir.iterdir():
    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
    if item.is_file():
        print(f"    Size: {item.stat().st_size} bytes")

# Also check if there's a run subdirectory
if (output_dir / "run_0").exists():
    print(f"\nFound run_0 subdirectory!")
    for item in (output_dir / "run_0").iterdir():
        print(f"  - {item.name}")
