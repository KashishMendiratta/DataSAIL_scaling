"""Download the QM9 CSV used by the extended DataSAIL experiments."""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
OUTPUT = Path("data/raw/moleculenet/qm9.csv")


def download_qm9(force: bool = False) -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if force or not OUTPUT.exists():
        print(f"Downloading QM9 from {URL}")
        urlretrieve(URL, OUTPUT)
    header = pd.read_csv(OUTPUT, nrows=0)
    required = {"smiles", "mol_id", "gap"}
    if not required.issubset(header.columns):
        raise ValueError(f"Unexpected QM9 columns in {OUTPUT}: missing {required - set(header.columns)}")
    print(f"QM9 ready: {OUTPUT} ({len(header.columns)} columns)")
    return OUTPUT


if __name__ == "__main__":
    download_qm9()
