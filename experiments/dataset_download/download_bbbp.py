"""Download BBBP dataset."""
import urllib.request
from pathlib import Path
import pandas as pd

data_dir = Path("data/raw/moleculenet")
data_dir.mkdir(parents=True, exist_ok=True)

url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
output_file = data_dir / "bbbp.csv"

print("Downloading BBBP dataset...")
try:
    urllib.request.urlretrieve(url, output_file)
    print(f"✓ Downloaded to {output_file}")
    
    # Verify
    df = pd.read_csv(output_file)
    print(f"\nDataset info:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    
    # Check for SMILES column
    smiles_cols = [col for col in df.columns if 'smiles' in col.lower() or 'mol' in col.lower()]
    print(f"  - SMILES column: {smiles_cols}")
    
    # Check for label column
    if 'p_np' in df.columns:
        print(f"  - Label distribution: {df['p_np'].value_counts().to_dict()}")
        
except Exception as e:
    print(f"✗ Error: {e}")
