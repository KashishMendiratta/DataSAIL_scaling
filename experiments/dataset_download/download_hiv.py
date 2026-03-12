"""Download HIV dataset from MoleculeNet."""
import pandas as pd
from pathlib import Path
import urllib.request


def download_hiv():

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"

    raw_dir = Path("data/raw/moleculenet")
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / "hiv.csv"

    print("Downloading HIV dataset...")

    urllib.request.urlretrieve(url, output_path)

    df = pd.read_csv(output_path)

    print(f"HIV dataset saved to {output_path}")
    print(f"Dataset size: {len(df)} molecules")


if __name__ == "__main__":
    download_hiv()