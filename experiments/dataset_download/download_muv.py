"""Download and validate the MUV MoleculeNet CSV."""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz"
OUTPUT = Path("data/raw/moleculenet/muv.csv.gz")
COMPRESSION = "gzip"
REQUIRED = ['mol_id', 'smiles']

def download_muv(force: bool = False) -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if force or not OUTPUT.exists():
        print(f"Downloading MUV from {URL}")
        urlretrieve(URL, OUTPUT)
    header = pd.read_csv(OUTPUT, compression=COMPRESSION, nrows=0)
    missing = set(REQUIRED) - set(header.columns)
    if missing:
        raise ValueError(f"Unexpected MUV schema; missing {sorted(missing)}")
    tasks = [c for c in header.columns if c.startswith("MUV-")]
    if len(tasks) != 17:
        raise ValueError(f"Expected 17 MUV assays, found {len(tasks)}")
    print(f"MUV ready: {OUTPUT} ({len(header.columns)} columns)")
    return OUTPUT

if __name__ == "__main__":
    download_muv()
