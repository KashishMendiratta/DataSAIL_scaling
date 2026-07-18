"""Utility functions for molecular data processing."""

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import RDLogger

from typing import Dict
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Tuple

import os

def save_splits(train_df, val_df, test_df, dataset, split_type):
    base_dir = f"results/experiments/splits/{dataset}/{split_type}"
    os.makedirs(base_dir, exist_ok=True)

    train_df.to_csv(f"{base_dir}/train.csv", index=False)
    val_df.to_csv(f"{base_dir}/val.csv", index=False)
    test_df.to_csv(f"{base_dir}/test.csv", index=False)

RDLogger.DisableLog('rdApp.*')

def smiles_to_fingerprint(
    smiles: str,
    generator=None
) -> np.ndarray | None:
    """
    Convert SMILES string to Morgan (ECFP) fingerprint.

    Args:
        smiles: SMILES string
        radius: Radius for Morgan fingerprint (default: 2 = ECFP4)
        n_bits: Number of bits in fingerprint

    Returns:
        Numpy array of fingerprint bits, or None if invalid SMILES
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        fp = generator.GetFingerprint(mol)
        return np.asarray(fp, dtype=np.int8)
    except Exception:
        return None

def compute_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute fingerprints for list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius
        n_bits: Number of bits

    Returns:
        Tuple of (fingerprints array, valid_indices list)
    """
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    fingerprints = []
    valid_indices = []

    for i, smiles in enumerate(smiles_list):
        fp = smiles_to_fingerprint(smiles, generator)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(i)

    return np.vstack(fingerprints), valid_indices


def load_bace_dataset(
    data_path: str = "data/raw/moleculenet/bace.csv"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load BACE dataset and compute fingerprints.

    Args:
        data_path: Path to BACE CSV file

    Returns:
        Tuple of (dataframe, fingerprints array)
    """
    df = pd.read_csv(data_path)

    # Identify SMILES column
    smiles_col = "mol" if "mol" in df.columns else "smiles"

    # Compute fingerprints
    fps, valid_idx = compute_fingerprints(df[smiles_col].tolist())

    # Keep only valid molecules
    df = df.iloc[valid_idx].reset_index(drop=True)

    print(f"Loaded {len(df)} molecules with valid SMILES")

    return df, fps




def log_experiment(dataset, n_samples, runtime, results):
    """
    Save experiment summary for thesis tables.
    """

    log_file = Path("results/analysis/experiment_summary.csv")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "dataset": dataset,
        "n_samples": n_samples,
        "runtime_seconds": runtime,
        "train_pct": results["train"]["pct"],
        "val_pct": results["val"]["pct"],
        "test_pct": results["test"]["pct"],
        "train_diff": results["train"]["diff"],
        "val_diff": results["val"]["diff"],
        "test_diff": results["test"]["diff"],
    }

    if log_file.exists():
        df = pd.read_csv(log_file)

        # remove previous runs of same dataset
        df = df[df["dataset"] != dataset]

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(log_file, index=False)

    print(f"\nExperiment logged → {log_file}")

def compute_max_deviation(counts: Dict[str, int], total: int, target=(0.7, 0.2, 0.1)) -> float:
    """
    Compute max deviation in percentage points.
    """
    target_map = {
        "train": target[0] * 100,
        "val": target[1] * 100,
        "test": target[2] * 100
    }

    deviations = []

    for split in ["train", "val", "test"]:
        actual_pct = 100 * counts.get(split, 0) / total if total > 0 else 0
        target_pct = target_map[split]
        deviations.append(abs(actual_pct - target_pct))

    return max(deviations)

def compute_deviation_details(counts, total, target=(0.7, 0.2, 0.1)):
    target_map = {
        "train": target[0] * 100,
        "val": target[1] * 100,
        "test": target[2] * 100
    }

    details = {}

    for split in ["train", "val", "test"]:
        actual_pct = 100 * counts.get(split, 0) / total
        target_pct = target_map[split]
        details[split] = actual_pct - target_pct  # signed deviation

    return details