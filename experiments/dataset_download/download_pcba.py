"""Download the compressed PCBA multi-assay CSV used by the experiments."""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz"
OUTPUT = Path("data/raw/moleculenet/pcba.csv.gz")


def download_pcba(force: bool = False) -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if force or not OUTPUT.exists():
        print(f"Downloading PCBA from {URL}")
        urlretrieve(URL, OUTPUT)
    header = pd.read_csv(OUTPUT, compression="gzip", nrows=0)
    required = {"smiles", "mol_id"}
    assay_columns = [c for c in header.columns if c.startswith("PCBA-")]
    if not required.issubset(header.columns) or len(assay_columns) != 128:
        raise ValueError(f"Unexpected PCBA schema in {OUTPUT}: found {len(assay_columns)} assay columns")
    print(f"PCBA ready: {OUTPUT} ({len(assay_columns)} assays; compressed CSV retained)")
    return OUTPUT


if __name__ == "__main__":
    download_pcba()
