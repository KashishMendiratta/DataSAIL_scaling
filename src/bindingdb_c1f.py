"""Scaled cold-target (C1f) splitting for normalized BindingDB data."""
from __future__ import annotations

import gzip
import json
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
from sklearn.feature_extraction.text import HashingVectorizer

from src.bindingdb_c1e import choose_degree_anchors, seeded_subset
from src.datasail_adapter import run_c1f


SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
PROTEIN_FEATURES = 32768


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def load_target_stats(database: Path) -> pd.DataFrame:
    uri = f"file:{Path(database).resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        frame = pd.read_sql_query(
            """SELECT t.target_id, t.sequence, COUNT(i.observation_id) degree
               FROM targets t JOIN interactions i USING(target_id)
               GROUP BY t.target_id, t.sequence ORDER BY t.target_id""",
            connection,
        )
    frame["degree"] = frame["degree"].astype(np.int64)
    if frame.empty or frame.target_id.duplicated().any() or (frame.degree <= 0).any():
        raise ValueError("Invalid or empty target degree table")
    return frame


def protein_features(sequences: list[str]):
    """Deterministic normalized 3-mer frequency vectors without external binaries."""
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=(3, 3), n_features=PROTEIN_FEATURES,
        alternate_sign=False, binary=False, norm="l2", lowercase=False,
    )
    return vectorizer.transform(sequences)


def assign_remaining_targets(
    frame: pd.DataFrame,
    features,
    anchor_positions: np.ndarray,
    anchor_splits: np.ndarray,
    *,
    k: int,
    balance_weight: float,
    seed: int,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.empty(len(frame), dtype=object)
    confidence = np.full(len(frame), np.nan, dtype=float)
    labels[anchor_positions] = anchor_splits
    confidence[anchor_positions] = 1.0
    anchor_features = features[anchor_positions]
    current_weight = Counter()
    for split, degree in zip(anchor_splits, frame.iloc[anchor_positions].degree):
        current_weight[str(split)] += int(degree)
    total_weight = sum(current_weight.values())
    remaining = np.setdiff1d(np.arange(len(frame)), anchor_positions, assume_unique=True)
    remaining = remaining[np.random.default_rng(seed).permutation(len(remaining))]

    for start in range(0, len(remaining), chunk_size):
        positions = remaining[start:start + chunk_size]
        similarities = (features[positions] @ anchor_features.T).toarray()
        for row, position in enumerate(positions):
            scores = {}
            for split, target in SPLIT_RATIOS.items():
                split_positions = np.flatnonzero(anchor_splits == split)
                split_k = min(k, len(split_positions))
                if split_k:
                    local = similarities[row, split_positions]
                    nearest = np.argpartition(-local, split_k - 1)[:split_k]
                    similarity = float(local[nearest].mean())
                else:
                    similarity = 0.0
                current_ratio = current_weight[split] / max(total_weight, 1)
                scores[split] = (similarity + 1e-9) * (target / max(current_ratio, 1e-12)) ** balance_weight
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            best = ordered[0][0]
            labels[position] = best
            confidence[position] = (
                (ordered[0][1] - ordered[1][1]) / ordered[0][1]
                if ordered[0][1] > 0 else 0.0
            )
            degree = int(frame.iloc[position].degree)
            current_weight[best] += degree
            total_weight += degree
    return labels, confidence


def write_assignments(
    output_database: Path,
    frame: pd.DataFrame,
    labels: np.ndarray,
    confidence: np.ndarray,
    anchor_positions: np.ndarray,
) -> None:
    if output_database.exists():
        raise FileExistsError(f"C1f output already exists: {output_database}")
    anchors = np.zeros(len(frame), dtype=np.int8)
    anchors[anchor_positions] = 1
    with sqlite3.connect(output_database) as connection:
        connection.execute(
            """CREATE TABLE assignments (
                   target_id TEXT PRIMARY KEY, split TEXT NOT NULL,
                   is_anchor INTEGER NOT NULL, degree INTEGER NOT NULL,
                   confidence REAL NOT NULL)"""
        )
        connection.executemany(
            "INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
            zip(frame.target_id.astype(str), labels.astype(str), anchors,
                frame.degree.astype(int), confidence.astype(float)),
        )
        connection.execute("CREATE INDEX assignments_split ON assignments(split)")


def audit_c1f(
    prepared_database: Path, assignment_database: Path, *, expected_target_count: int,
) -> dict:
    with sqlite3.connect(assignment_database) as connection:
        connection.execute("ATTACH DATABASE ? AS prepared", (str(Path(prepared_database).resolve()),))
        prepared_targets = connection.execute("SELECT COUNT(*) FROM prepared.targets").fetchone()[0]
        assigned_targets = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
        missing = connection.execute(
            """SELECT COUNT(*) FROM prepared.targets t LEFT JOIN assignments a USING(target_id)
               WHERE a.target_id IS NULL"""
        ).fetchone()[0]
        invalid = connection.execute(
            "SELECT COUNT(*) FROM assignments WHERE split NOT IN ('train','val','test')"
        ).fetchone()[0]
        rows = connection.execute(
            """SELECT a.split, COUNT(*) FROM prepared.interactions i
               JOIN assignments a USING(target_id) GROUP BY a.split"""
        ).fetchall()
        counts = dict(rows)
        total = sum(counts.values())
        duplicate_conflicts = connection.execute(
            """SELECT COUNT(*) FROM (
                   SELECT i.duplicate_group_id FROM prepared.interactions i
                   JOIN assignments a USING(target_id) GROUP BY i.duplicate_group_id
                   HAVING COUNT(DISTINCT a.split)>1)"""
        ).fetchone()[0]
        ligand_overlap = dict(connection.execute(
            """SELECT n_splits, COUNT(*) FROM (
                   SELECT i.ligand_id, COUNT(DISTINCT a.split) n_splits
                   FROM prepared.interactions i JOIN assignments a USING(target_id)
                   GROUP BY i.ligand_id) GROUP BY n_splits"""
        ).fetchall())
        affinity = connection.execute(
            """SELECT a.split, i.affinity_type, COUNT(*), AVG(i.p_affinity)
               FROM prepared.interactions i JOIN assignments a USING(target_id)
               GROUP BY a.split, i.affinity_type ORDER BY a.split, i.affinity_type"""
        ).fetchall()
    ratios = {split: counts.get(split, 0) / total for split in SPLIT_RATIOS}
    ratio_deviation = {
        split: abs(ratios[split] - SPLIT_RATIOS[split]) for split in SPLIT_RATIOS
    }
    selected_is_full_release = expected_target_count == prepared_targets
    if (assigned_targets != expected_target_count or
            (selected_is_full_release and missing) or invalid or
            set(counts) != set(SPLIT_RATIOS) or duplicate_conflicts or
            max(ratio_deviation.values()) > 0.05):
        raise RuntimeError("C1f core validation failed")
    return {
        "prepared_targets": prepared_targets,
        "selected_target_universe": expected_target_count,
        "selected_is_full_release": selected_is_full_release,
        "assigned_targets": assigned_targets,
        "missing_target_assignments": missing,
        "retained_observations": total,
        "dropped_observations": 0,
        "interaction_counts": counts,
        "interaction_ratios": ratios,
        "absolute_ratio_deviation": ratio_deviation,
        "maximum_allowed_ratio_deviation": 0.05,
        "requested_interaction_ratios": SPLIT_RATIOS,
        "duplicate_split_conflicts": duplicate_conflicts,
        "ligands_by_number_of_observed_splits": ligand_overlap,
        "cold_target_test_rate": 1.0,
        "ligand_assignment_policy": "not_applicable_for_C1f_ligands_may_span_splits",
        "affinity_distribution": [
            {"split": s, "affinity_type": t, "count": n, "mean_p_affinity": mean}
            for s, t, n, mean in affinity
        ],
    }


def run_scaled_c1f(
    prepared_database: Path,
    output_dir: Path,
    *,
    max_targets: int | None,
    anchor_ratio: float,
    seed: int,
    k: int,
    balance_weight: float,
    clusters: int,
    max_sec: int,
    threads: int,
) -> dict:
    started = time.perf_counter()
    timings = {}
    stage = time.perf_counter()
    print("C1f stage: loading target statistics", flush=True)
    stats = seeded_subset(load_target_stats(prepared_database), max_targets, seed)
    timings["load_target_stats"] = time.perf_counter() - stage
    stage = time.perf_counter()
    print(f"C1f stage: encoding {len(stats):,} target sequences", flush=True)
    features = protein_features(stats.sequence.astype(str).tolist())
    timings["protein_features"] = time.perf_counter() - stage
    stage = time.perf_counter()
    anchors = choose_degree_anchors(stats, anchor_ratio, seed)
    anchor_stats = stats.iloc[anchors]
    anchor_similarity = (features[anchors] @ features[anchors].T).toarray()
    timings["anchor_selection_and_similarity"] = time.perf_counter() - stage
    stage = time.perf_counter()
    print(f"C1f stage: running DataSAIL on {len(anchors):,} anchors", flush=True)
    result = run_c1f(
        dict(zip(anchor_stats.target_id, anchor_stats.sequence)),
        dict(zip(anchor_stats.target_id, np.log1p(anchor_stats.degree.astype(float)))),
        anchor_stats.target_id.tolist(), anchor_similarity,
        splits=SPLIT_RATIOS, clusters=min(clusters, max(3, len(anchors) // 5)),
        max_sec=max_sec, threads=threads,
    )
    timings["datasail"] = time.perf_counter() - stage
    if set(result.f_splits) != set(anchor_stats.target_id):
        raise RuntimeError("DataSAIL C1f anchor assignments do not match target IDs")
    anchor_splits = np.asarray([result.f_splits[value] for value in anchor_stats.target_id])
    if set(anchor_splits) != set(SPLIT_RATIOS):
        raise RuntimeError("DataSAIL C1f did not populate all split labels")
    stage = time.perf_counter()
    print("C1f stage: assigning remaining targets", flush=True)
    labels, confidence = assign_remaining_targets(
        stats, features, anchors, anchor_splits, k=k,
        balance_weight=balance_weight, seed=seed,
    )
    timings["remaining_assignment"] = time.perf_counter() - stage
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "c1f_assignments.sqlite"
    write_assignments(database, stats, labels, confidence, anchors)
    stage = time.perf_counter()
    print("C1f stage: auditing assignments and observations", flush=True)
    audit = audit_c1f(
        prepared_database, database, expected_target_count=len(stats),
    )
    timings["audit"] = time.perf_counter() - stage
    manifest = {
        "format_version": 1,
        "method": "bindingdb_scaled_c1f",
        "protein_similarity": "normalized hashed 3-mer frequency cosine",
        "protein_feature_count": PROTEIN_FEATURES,
        "prepared_database": str(prepared_database),
        "seed": seed, "max_targets": max_targets,
        "n_targets": len(stats), "anchor_ratio": anchor_ratio,
        "anchor_count": len(anchors), "anchor_degree_sum": int(anchor_stats.degree.sum()),
        "datasail_anchor_weight": "log1p_observation_degree",
        "k": k, "balance_weight": balance_weight, "clusters": clusters,
        "neighbor_policy": "top_k_within_each_split",
        "max_sec": max_sec, "threads": threads,
        "runtime_seconds": {**timings, "total": time.perf_counter() - started},
        "peak_memory_mb": peak_rss_mb(),
        "validation_status": "core_assignments_passed_pending_similarity_and_graph_audits",
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": None if os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY") is None else os.environ["DATASAIL_GIT_WORKTREE_DIRTY"].lower() == "true",
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"),
        "python_version": platform.python_version(),
        "audit": audit,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with gzip.open(output_dir / "target_assignments.tsv.gz", "wt") as handle:
        handle.write("target_id\tsplit\tis_anchor\tdegree\tconfidence\n")
        with sqlite3.connect(database) as connection:
            for row in connection.execute("SELECT * FROM assignments ORDER BY target_id"):
                handle.write("\t".join(map(str, row)) + "\n")
    return manifest
