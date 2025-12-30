"""Download BACE dataset from MoleculeNet."""
import urllib.request
import os
from pathlib import Path

# Create directory
data_dir = Path("data/raw/moleculenet")
data_dir.mkdir(parents=True, exist_ok=True)

# Download BACE dataset
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
output_file = data_dir / "bace.csv"

print(f"Downloading BACE dataset from MoleculeNet...")
print(f"URL: {url}")
print(f"Saving to: {output_file}")

try:
    urllib.request.urlretrieve(url, output_file)
    print(f"✓ Downloaded successfully!")
    
    # Quick check
    import pandas as pd
    df = pd.read_csv(output_file)
    print(f"\nDataset info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    if 'Class' in df.columns:
        print(f"  Label distribution:\n{df['Class'].value_counts()}")
    
except Exception as e:
    print(f"✗ Error downloading: {e}")
