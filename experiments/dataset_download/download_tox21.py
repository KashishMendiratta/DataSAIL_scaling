"""
Download Tox21 dataset from MoleculeNet.
"""

import pandas as pd
from pathlib import Path
import urllib.request


def download_tox21():

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

    raw_dir = Path("data/raw/moleculenet")
    raw_dir.mkdir(parents=True, exist_ok=True)

    gz_path = raw_dir / "tox21.csv.gz"
    csv_path = raw_dir / "tox21.csv"

    print("Downloading Tox21 dataset...")

    urllib.request.urlretrieve(url, gz_path)

    print("Download complete. Extracting...")

    df = pd.read_csv(gz_path, compression="gzip")
    df.to_csv(csv_path, index=False)

    gz_path.unlink()  # remove gzip

    print(f"Tox21 saved to: {csv_path}")
    print(f"Dataset size: {len(df)} molecules")


if __name__ == "__main__":
    download_tox21()