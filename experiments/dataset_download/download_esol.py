"""Download and validate the ESOL MoleculeNet CSV."""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
OUTPUT = Path("data/raw/moleculenet/delaney-processed.csv")
COMPRESSION = None
REQUIRED = ['Compound ID', 'smiles', 'measured log solubility in mols per litre']

def download_esol(force: bool = False) -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if force or not OUTPUT.exists():
        print(f"Downloading ESOL from {URL}")
        urlretrieve(URL, OUTPUT)
    header = pd.read_csv(OUTPUT, compression=COMPRESSION, nrows=0)
    missing = set(REQUIRED) - set(header.columns)
    if missing:
        raise ValueError(f"Unexpected ESOL schema; missing {sorted(missing)}")
    print(f"ESOL ready: {OUTPUT} ({len(header.columns)} columns)")
    return OUTPUT

if __name__ == "__main__":
    download_esol()
