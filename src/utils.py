"""Utility functions for molecular data processing."""

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np
import pandas as pd
from typing import List, Tuple


def smiles_to_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048
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
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
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
    fingerprints = []
    valid_indices = []

    for i, smiles in enumerate(smiles_list):
        fp = smiles_to_fingerprint(smiles, radius, n_bits)
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
