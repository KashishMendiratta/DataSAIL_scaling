"""Scaled cold-ligand (C1e) splitting for normalized BindingDB data."""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.datasail_adapter import run_c1e


RDLogger.DisableLog("rdApp.*")
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def load_ligand_stats(database: Path) -> pd.DataFrame:
    """Load one canonical chemical entity with aggregated observation degree."""
    uri = f"file:{Path(database).resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        frame = pd.read_sql_query(
            """SELECT l.canonical_smiles,
                      COUNT(DISTINCT l.ligand_id) AS ligand_id_count,
                      COUNT(i.observation_id) AS degree
               FROM ligands l
               JOIN interactions i ON i.ligand_id = l.ligand_id
               GROUP BY l.canonical_smiles
               ORDER BY l.canonical_smiles""",
            connection,
        )
    frame.insert(0, "chemical_id", frame["canonical_smiles"].map(
        lambda value: "chem:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    ))
    frame["degree"] = frame["degree"].astype(np.int64)
    frame["ligand_id_count"] = frame["ligand_id_count"].astype(np.int64)
    if frame.empty or frame["chemical_id"].duplicated().any() or (frame["degree"] <= 0).any():
        raise ValueError("Invalid or empty canonical-chemical degree table")
    return frame


def seeded_subset(frame: pd.DataFrame, max_ligands: int | None, seed: int) -> pd.DataFrame:
    if max_ligands is None or max_ligands >= len(frame):
        return frame.reset_index(drop=True)
    if max_ligands < 30:
        raise ValueError("C1e pilot requires at least 30 ligands")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(frame), size=max_ligands, replace=False))
    return frame.iloc[selected].reset_index(drop=True)


def choose_degree_anchors(
    frame: pd.DataFrame, ratio: float, seed: int, bins: int = 10,
) -> np.ndarray:
    """Select exact-sized anchors while preserving ligand-degree quantiles."""
    if not 0 < ratio < 1:
        raise ValueError("anchor ratio must be between zero and one")
    count = max(30, min(len(frame) - 1, round(len(frame) * ratio)))
    degree_bins = pd.qcut(
        frame["degree"].rank(method="first"), q=min(bins, len(frame)),
        labels=False, duplicates="drop",
    )
    from sklearn.model_selection import train_test_split
    positions = np.arange(len(frame))
    anchors, _ = train_test_split(
        positions, train_size=count, random_state=seed, stratify=degree_bins,
    )
    anchors = np.sort(anchors)
    if len(anchors) != count or len(np.unique(anchors)) != count:
        raise RuntimeError("Degree-aware anchor selection cardinality failed")
    return anchors


def _fingerprint(generator, smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Prepared ligand contains invalid SMILES: {smiles!r}")
    return generator.GetFingerprint(molecule)


def assign_remaining_ligands(
    frame: pd.DataFrame,
    anchor_positions: np.ndarray,
    anchor_splits: np.ndarray,
    *,
    k: int,
    balance_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign non-anchors by exact Tanimoto neighbours and degree-weighted balance."""
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    anchor_fps = [
        _fingerprint(generator, value)
        for value in frame.iloc[anchor_positions]["canonical_smiles"]
    ]
    k = min(k, len(anchor_fps))
    labels = np.empty(len(frame), dtype=object)
    confidence = np.full(len(frame), np.nan, dtype=float)
    labels[anchor_positions] = anchor_splits
    confidence[anchor_positions] = 1.0
    current_weight = Counter()
    for split, degree in zip(anchor_splits, frame.iloc[anchor_positions]["degree"]):
        current_weight[str(split)] += int(degree)
    total_weight = sum(current_weight.values())
    remaining = np.setdiff1d(np.arange(len(frame)), anchor_positions, assume_unique=True)
    rng = np.random.default_rng(seed)
    remaining = remaining[rng.permutation(len(remaining))]

    for position in remaining:
        query = _fingerprint(generator, frame.iloc[position]["canonical_smiles"])
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query, anchor_fps))
        neighbors = np.argpartition(-similarities, k - 1)[:k]
        scores = {}
        for split, target in SPLIT_RATIOS.items():
            mask = anchor_splits[neighbors] == split
            similarity = float(similarities[neighbors][mask].mean()) if mask.any() else 0.0
            current_ratio = current_weight[split] / max(total_weight, 1)
            balance = target / max(current_ratio, 1e-12)
            # A tiny baseline lets balance resolve zero-similarity ties instead of
            # sending every chemically remote ligand to the first split.
            scores[split] = (similarity + 1e-9) * balance**balance_weight
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best = ordered[0][0]
        labels[position] = best
        if ordered[0][1] > 0:
            confidence[position] = (ordered[0][1] - ordered[1][1]) / ordered[0][1]
        else:
            confidence[position] = 0.0
        degree = int(frame.iloc[position]["degree"])
        current_weight[best] += degree
        total_weight += degree
    return labels, confidence


def _write_assignments(
    output_database: Path,
    prepared_database: Path,
    frame: pd.DataFrame,
    labels: np.ndarray,
    confidence: np.ndarray,
    anchor_positions: np.ndarray,
) -> None:
    output_database.parent.mkdir(parents=True, exist_ok=True)
    if output_database.exists():
        raise FileExistsError(f"C1e output already exists: {output_database}")
    anchors = np.zeros(len(frame), dtype=np.int8)
    anchors[anchor_positions] = 1
    with sqlite3.connect(output_database) as connection:
        connection.execute(
            """CREATE TABLE chemical_assignments (
                   chemical_id TEXT PRIMARY KEY, canonical_smiles TEXT NOT NULL UNIQUE,
                   split TEXT NOT NULL,
                   is_anchor INTEGER NOT NULL, degree INTEGER NOT NULL,
                   confidence REAL NOT NULL
               )"""
        )
        connection.executemany(
            "INSERT INTO chemical_assignments VALUES (?, ?, ?, ?, ?, ?)",
            zip(
                frame["chemical_id"].astype(str), frame["canonical_smiles"].astype(str),
                labels.astype(str), anchors,
                frame["degree"].astype(int), confidence.astype(float),
            ),
        )
        connection.execute(
            "ATTACH DATABASE ? AS prepared", (str(Path(prepared_database).resolve()),)
        )
        connection.execute(
            """CREATE TABLE assignments AS
               SELECT l.ligand_id, c.chemical_id, c.split, c.is_anchor,
                      c.degree, c.confidence
               FROM prepared.ligands l
               JOIN chemical_assignments c USING(canonical_smiles)"""
        )
        connection.execute("CREATE UNIQUE INDEX assignments_ligand ON assignments(ligand_id)")
        connection.execute("CREATE INDEX assignments_split ON assignments(split)")


def audit_c1e(prepared_database: Path, assignment_database: Path) -> dict:
    """Validate C1e entity/observation accounting and summarize split balance."""
    with sqlite3.connect(assignment_database) as connection:
        connection.execute(
            "ATTACH DATABASE ? AS prepared", (str(Path(prepared_database).resolve()),)
        )
        ligand_count = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
        chemical_count = connection.execute("SELECT COUNT(*) FROM chemical_assignments").fetchone()[0]
        invalid = connection.execute(
            "SELECT COUNT(*) FROM assignments WHERE split NOT IN ('train','val','test')"
        ).fetchone()[0]
        interaction_rows = connection.execute(
            """SELECT a.split, COUNT(*)
               FROM prepared.interactions i JOIN assignments a USING(ligand_id)
               GROUP BY a.split"""
        ).fetchall()
        counts = dict(interaction_rows)
        total = sum(counts.values())
        duplicate_conflicts = connection.execute(
            """SELECT COUNT(*) FROM (
                   SELECT i.duplicate_group_id
                   FROM prepared.interactions i JOIN assignments a USING(ligand_id)
                   GROUP BY i.duplicate_group_id HAVING COUNT(DISTINCT a.split) > 1
               )"""
        ).fetchone()[0]
        chemical_conflicts = connection.execute(
            """SELECT COUNT(*) FROM (
                   SELECT chemical_id FROM assignments
                   GROUP BY chemical_id HAVING COUNT(DISTINCT split) > 1
               )"""
        ).fetchone()[0]
        affinity_rows = connection.execute(
            """SELECT a.split, i.affinity_type, COUNT(*), AVG(i.p_affinity)
               FROM prepared.interactions i JOIN assignments a USING(ligand_id)
               GROUP BY a.split, i.affinity_type ORDER BY a.split, i.affinity_type"""
        ).fetchall()
        target_rows = connection.execute(
            """SELECT split_count, COUNT(*) FROM (
                   SELECT i.target_id, COUNT(DISTINCT a.split) AS split_count
                   FROM prepared.interactions i JOIN assignments a USING(ligand_id)
                   GROUP BY i.target_id
               ) GROUP BY split_count ORDER BY split_count"""
        ).fetchall()
    if invalid or set(counts) != set(SPLIT_RATIOS) or duplicate_conflicts or chemical_conflicts:
        raise RuntimeError(
            f"C1e validation failed: invalid={invalid}, splits={counts}, "
            f"duplicate_conflicts={duplicate_conflicts}, chemical_conflicts={chemical_conflicts}"
        )
    return {
        "assigned_ligands": ligand_count,
        "assigned_canonical_chemicals": chemical_count,
        "retained_observations": total,
        "dropped_observations": 0,
        "interaction_counts": counts,
        "interaction_ratios": {name: counts[name] / total for name in SPLIT_RATIOS},
        "requested_interaction_ratios": SPLIT_RATIOS,
        "duplicate_split_conflicts": duplicate_conflicts,
        "canonical_chemical_split_conflicts": chemical_conflicts,
        "affinity_distribution": [
            {"split": s, "affinity_type": t, "count": n, "mean_p_affinity": mean}
            for s, t, n, mean in affinity_rows
        ],
        "targets_by_number_of_observed_splits": dict(target_rows),
        "cold_ligand_test_rate": 1.0,
        "target_assignment_policy": "not_applicable_for_C1e_targets_may_span_splits",
    }


def run_scaled_c1e(
    prepared_database: Path,
    output_dir: Path,
    *,
    max_ligands: int | None,
    anchor_ratio: float,
    seed: int,
    k: int,
    balance_weight: float,
    clusters: int,
    max_sec: int,
    threads: int,
) -> dict:
    started = time.perf_counter()
    stage = started
    print("C1e stage: loading ligand statistics", flush=True)
    stats = seeded_subset(load_ligand_stats(prepared_database), max_ligands, seed)
    load_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    print(f"C1e stage: selecting anchors from {len(stats):,} canonical chemicals", flush=True)
    anchor_positions = choose_degree_anchors(stats, anchor_ratio, seed)
    anchors = stats.iloc[anchor_positions]
    anchor_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    print(f"C1e stage: running DataSAIL on {len(anchors):,} anchors", flush=True)
    result = run_c1e(
        dict(zip(anchors.chemical_id, anchors.canonical_smiles)),
        dict(zip(anchors.chemical_id, anchors.degree.astype(float))),
        splits=SPLIT_RATIOS,
        clusters=min(clusters, max(3, len(anchors) // 5)),
        max_sec=max_sec,
        threads=threads,
    )
    datasail_seconds = time.perf_counter() - stage
    expected = set(anchors.chemical_id)
    if set(result.e_splits) != expected:
        raise RuntimeError("DataSAIL anchor assignments do not match anchor ligand IDs")
    anchor_splits = np.asarray([result.e_splits[value] for value in anchors.chemical_id])
    if set(anchor_splits) != set(SPLIT_RATIOS):
        raise RuntimeError(f"DataSAIL anchor split labels are incomplete: {set(anchor_splits)}")
    stage = time.perf_counter()
    print("C1e stage: assigning remaining ligands", flush=True)
    labels, confidence = assign_remaining_ligands(
        stats, anchor_positions, anchor_splits, k=k,
        balance_weight=balance_weight, seed=seed,
    )
    assignment_seconds = time.perf_counter() - stage
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "c1e_assignments.sqlite"
    _write_assignments(database, prepared_database, stats, labels, confidence, anchor_positions)
    stage = time.perf_counter()
    print("C1e stage: auditing assignments and observations", flush=True)
    audit = audit_c1e(prepared_database, database)
    audit_seconds = time.perf_counter() - stage
    manifest = {
        "format_version": 1,
        "method": "bindingdb_scaled_c1e",
        "prepared_database": str(prepared_database),
        "seed": seed,
        "max_ligands": max_ligands,
        "n_canonical_chemicals": len(stats),
        "n_ligand_ids": int(stats.ligand_id_count.sum()),
        "anchor_ratio": anchor_ratio,
        "anchor_count": len(anchors),
        "anchor_degree_sum": int(anchors.degree.sum()),
        "k": k,
        "balance_weight": balance_weight,
        "clusters": clusters,
        "max_sec": max_sec,
        "threads": threads,
        "runtime_seconds": {
            "load_ligand_stats": load_seconds,
            "anchor_selection": anchor_seconds,
            "datasail": datasail_seconds,
            "remaining_assignment": assignment_seconds,
            "audit": audit_seconds,
            "total": time.perf_counter() - started,
        },
        "peak_memory_mb": peak_rss_mb(),
        "validation_status": "core_assignments_passed_pending_similarity_and_graph_audits",
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": (
            None if os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY") is None
            else os.environ["DATASAIL_GIT_WORKTREE_DIRTY"].lower() == "true"
        ),
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"),
        "python_version": platform.python_version(),
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("datasail", "numpy", "pandas", "scikit-learn", "rdkit")
        },
        "audit": audit,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with gzip.open(output_dir / "ligand_assignments.tsv.gz", "wt") as handle:
        with sqlite3.connect(database) as connection:
            handle.write("ligand_id\tchemical_id\tsplit\tis_anchor\tdegree\tconfidence\n")
            for row in connection.execute("SELECT * FROM assignments ORDER BY ligand_id"):
                handle.write("\t".join(map(str, row)) + "\n")
    return manifest
