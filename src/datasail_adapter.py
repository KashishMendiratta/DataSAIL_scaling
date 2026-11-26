"""Stable adapters around DataSAIL's package-level return values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DataSAILAssignments:
    technique: str
    run: int
    e_splits: dict[str, str]
    f_splits: dict[str, str]
    interaction_splits: dict[tuple[str, str], str]


def assignments_from_result(result, technique: str, run: int = 0) -> DataSAILAssignments:
    """Validate and normalize the three dictionaries returned by ``datasail``."""
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("DataSAIL must return (e_splits, f_splits, interaction_splits)")
    normalized = []
    for label, part in zip(("e", "f", "interaction"), result):
        if part is None:
            required = (label == "e" and technique.endswith("e")) or (
                label == "f" and technique.endswith("f")
            )
            if required:
                raise ValueError(f"DataSAIL returned no {label}-assignment dictionary")
            normalized.append({})
            continue
        runs = part.get(technique, [])
        if not runs:
            normalized.append({})
            continue
        if run >= len(runs):
            raise ValueError(f"DataSAIL {technique} has {len(runs)} runs, not run {run}")
        if not isinstance(runs[run], Mapping):
            raise ValueError(f"DataSAIL {technique} run {run} is not a mapping")
        normalized.append(dict(runs[run]))
    if not normalized[0] and technique.endswith("e"):
        raise ValueError(f"DataSAIL returned no e-assignments for {technique}")
    if not normalized[1] and technique.endswith("f"):
        raise ValueError(f"DataSAIL returned no f-assignments for {technique}")
    return DataSAILAssignments(technique, run, *normalized)


def run_c1e(
    ligand_smiles: Mapping[str, str],
    ligand_weights: Mapping[str, float],
    *,
    splits: Mapping[str, float],
    clusters: int,
    max_sec: int,
    threads: int,
) -> DataSAILAssignments:
    """Run C1e through DataSAIL's Python API and consume its returned mapping."""
    from datasail.sail import datasail

    names = list(splits)
    result = datasail(
        techniques=["C1e"],
        e_type="M",
        e_data=dict(ligand_smiles),
        e_weights=dict(ligand_weights),
        e_sim="ecfp",
        e_clusters=clusters,
        splits=[splits[name] for name in names],
        names=names,
        runs=1,
        max_sec=max_sec,
        threads=threads,
        verbose="W",
    )
    return assignments_from_result(result, "C1e")


def run_c1f(
    target_sequences: Mapping[str, str],
    target_weights: Mapping[str, float],
    similarity_names: Sequence[str],
    similarity_matrix: np.ndarray,
    *,
    splits: Mapping[str, float],
    clusters: int,
    max_sec: int,
    threads: int,
) -> DataSAILAssignments:
    """Run C1f with a precomputed protein-similarity matrix."""
    from datasail.sail import datasail

    names = list(splits)
    result = datasail(
        techniques=["C1f"],
        f_type="P",
        f_data=dict(target_sequences),
        f_weights=dict(target_weights),
        f_sim=(list(similarity_names), np.asarray(similarity_matrix, dtype=float)),
        f_clusters=clusters,
        splits=[splits[name] for name in names],
        names=names,
        runs=1,
        max_sec=max_sec,
        threads=threads,
        verbose="W",
    )
    return assignments_from_result(result, "C1f")


def run_c2(
    ligand_smiles: Mapping[str, str], ligand_weights: Mapping[str, float],
    ligand_names: Sequence[str], ligand_similarity: np.ndarray,
    target_sequences: Mapping[str, str], target_weights: Mapping[str, float],
    target_names: Sequence[str], target_similarity: np.ndarray,
    interactions: list[tuple[str, str]], *, splits: Mapping[str, float],
    ligand_clusters: int, target_clusters: int, max_sec: int, threads: int,
) -> DataSAILAssignments:
    """Run C2 with precomputed similarities and consume in-memory assignments."""
    from datasail.sail import datasail

    names = list(splits)
    result = datasail(
        techniques=["C2"], inter=interactions,
        e_type="M", e_data=dict(ligand_smiles), e_weights=dict(ligand_weights),
        e_sim=(list(ligand_names), np.asarray(ligand_similarity, dtype=float)),
        e_clusters=ligand_clusters,
        f_type="P", f_data=dict(target_sequences), f_weights=dict(target_weights),
        f_sim=(list(target_names), np.asarray(target_similarity, dtype=float)),
        f_clusters=target_clusters,
        splits=[splits[name] for name in names], names=names,
        runs=1, max_sec=max_sec, threads=threads, verbose="W",
    )
    assignments = assignments_from_result(result, "C2")
    if not assignments.e_splits or not assignments.f_splits:
        raise ValueError("DataSAIL returned incomplete C2 entity assignments")
    return assignments
