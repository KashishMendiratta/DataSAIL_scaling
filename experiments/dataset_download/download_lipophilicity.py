"""Download and validate the LIPOPHILICITY MoleculeNet CSV."""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
OUTPUT = Path("data/raw/moleculenet/Lipophilicity.csv")
COMPRESSION = None
REQUIRED = ['CMPD_CHEMBLID', 'smiles', 'exp']

def download_lipophilicity(force: bool = False) -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if force or not OUTPUT.exists():
        print(f"Downloading LIPOPHILICITY from {URL}")
        urlretrieve(URL, OUTPUT)
    header = pd.read_csv(OUTPUT, compression=COMPRESSION, nrows=0)
    missing = set(REQUIRED) - set(header.columns)
    if missing:
        raise ValueError(f"Unexpected LIPOPHILICITY schema; missing {sorted(missing)}")
    print(f"LIPOPHILICITY ready: {OUTPUT} ({len(header.columns)} columns)")
    return OUTPUT

if __name__ == "__main__":
    download_lipophilicity()
