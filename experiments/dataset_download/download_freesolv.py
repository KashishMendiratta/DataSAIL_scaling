"""Download and validate the FREESOLV MoleculeNet CSV."""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
OUTPUT = Path("data/raw/moleculenet/SAMPL.csv")
COMPRESSION = None
REQUIRED = ['iupac', 'smiles', 'expt']

def download_freesolv(force: bool = False) -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if force or not OUTPUT.exists():
        print(f"Downloading FREESOLV from {URL}")
        urlretrieve(URL, OUTPUT)
    header = pd.read_csv(OUTPUT, compression=COMPRESSION, nrows=0)
    missing = set(REQUIRED) - set(header.columns)
    if missing:
        raise ValueError(f"Unexpected FREESOLV schema; missing {sorted(missing)}")
    print(f"FREESOLV ready: {OUTPUT} ({len(header.columns)} columns)")
    return OUTPUT

if __name__ == "__main__":
    download_freesolv()
