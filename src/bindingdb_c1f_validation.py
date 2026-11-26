"""Hard validation gates for a completed BindingDB scaled C1f candidate."""
from __future__ import annotations

import json
import os
import platform
import resource
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from src.bindingdb_c1e_validation import UnionFind, summarize
from src.bindingdb_c1f import SPLIT_RATIOS, protein_features


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def accounting(connection: sqlite3.Connection) -> dict:
    targets = connection.execute("SELECT COUNT(*) FROM prepared.targets").fetchone()[0]
    assigned = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    distinct = connection.execute("SELECT COUNT(DISTINCT target_id) FROM assignments").fetchone()[0]
    missing = connection.execute(
        "SELECT COUNT(*) FROM prepared.targets t LEFT JOIN assignments a USING(target_id) WHERE a.target_id IS NULL"
    ).fetchone()[0]
    observations = connection.execute("SELECT COUNT(*) FROM prepared.interactions").fetchone()[0]
    joined = connection.execute(
        "SELECT COUNT(*) FROM prepared.interactions i JOIN assignments a USING(target_id)"
    ).fetchone()[0]
    counts = dict(connection.execute(
        "SELECT a.split, COUNT(*) FROM prepared.interactions i JOIN assignments a USING(target_id) GROUP BY a.split"
    ).fetchall())
    ratios = {split: counts[split] / joined for split in SPLIT_RATIOS}
    deviations = {split: abs(ratios[split] - SPLIT_RATIOS[split]) for split in SPLIT_RATIOS}
    measurement_status = dict(connection.execute(
        "SELECT status, COUNT(*) FROM prepared.measurements GROUP BY status"
    ).fetchall())
    unexplained = connection.execute(
        """SELECT COUNT(*) FROM prepared.measurements WHERE status NOT IN ('retained','excluded')
           OR (status='excluded' AND exclusion_reason IS NULL)"""
    ).fetchone()[0]
    duplicates = connection.execute(
        """SELECT COUNT(*) FROM (SELECT i.duplicate_group_id FROM prepared.interactions i
           JOIN assignments a USING(target_id) GROUP BY i.duplicate_group_id
           HAVING COUNT(DISTINCT a.split)>1)"""
    ).fetchone()[0]
    if not (targets == assigned == distinct and missing == unexplained == duplicates == 0
            and observations == joined and max(deviations.values()) <= 0.05):
        raise RuntimeError("C1f accounting or ratio gate failed")
    return {
        "prepared_targets": targets, "assigned_targets": assigned,
        "missing_target_assignments": missing, "retained_observations": joined,
        "dropped_cross_split_observations": 0,
        "requested_interaction_ratios": SPLIT_RATIOS,
        "actual_interaction_counts": counts, "actual_interaction_ratios": ratios,
        "absolute_ratio_deviation": deviations, "maximum_allowed_ratio_deviation": 0.05,
        "source_measurements_by_status": measurement_status,
        "measurements_without_status_or_exclusion_reason": unexplained,
        "duplicate_split_conflicts": duplicates,
    }


def exact_protein_similarity(connection: sqlite3.Connection, chunk_size: int = 256) -> dict:
    rows = connection.execute(
        """SELECT t.target_id, t.sequence, a.split FROM prepared.targets t
           JOIN assignments a USING(target_id) ORDER BY t.target_id"""
    ).fetchall()
    ids = np.asarray([row[0] for row in rows], dtype=object)
    splits = np.asarray([row[2] for row in rows], dtype=object)
    features = protein_features([row[1] for row in rows])
    pair_results = []
    global_best = (-1.0, None, None)
    split_names = list(SPLIT_RATIOS)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1:]:
            left = np.flatnonzero(splits == left_name)
            right = np.flatnonzero(splits == right_name)
            best = (-1.0, None, None)
            for start in range(0, len(left), chunk_size):
                block_indices = left[start:start + chunk_size]
                block = (features[block_indices] @ features[right].T).toarray()
                flat = int(np.argmax(block))
                row, column = np.unravel_index(flat, block.shape)
                value = float(block[row, column])
                if value > best[0]:
                    best = (value, ids[block_indices[row]], ids[right[column]])
            pair_results.append({
                "splits": [left_name, right_name], "maximum_similarity": best[0],
                "target_ids": [best[1], best[2]],
            })
            if best[0] > global_best[0]:
                global_best = best
    return {
        "metric": "normalized hashed 3-mer frequency cosine",
        "calculation": "exact exhaustive cross-split target pairs",
        "maximum_cross_split_protein_similarity": global_best[0],
        "certificate_target_ids": [global_best[1], global_best[2]],
        "pairwise_split_maxima": pair_results,
    }


def regime(connection: sqlite3.Connection) -> dict:
    target_conflicts = connection.execute(
        "SELECT COUNT(*) FROM (SELECT target_id FROM assignments GROUP BY target_id HAVING COUNT(DISTINCT split)>1)"
    ).fetchone()[0]
    test_targets = connection.execute("SELECT COUNT(*) FROM assignments WHERE split='test'").fetchone()[0]
    ligand_counts = dict(connection.execute(
        """SELECT n_splits, COUNT(*) FROM (SELECT i.ligand_id, COUNT(DISTINCT a.split) n_splits
           FROM prepared.interactions i JOIN assignments a USING(target_id) GROUP BY i.ligand_id)
           GROUP BY n_splits"""
    ).fetchall())
    exact_ligand = connection.execute(
        """SELECT i.ligand_id FROM prepared.interactions i JOIN assignments a USING(target_id)
           GROUP BY i.ligand_id HAVING COUNT(DISTINCT a.split)>1 LIMIT 1"""
    ).fetchone()
    if target_conflicts or not test_targets:
        raise RuntimeError("C1f cold-target identity gate failed")
    return {
        "unseen_test_target_rate": 1.0,
        "intended_target_regime": "cold; exact target identity must be unseen",
        "target_split_conflicts": target_conflicts,
        "ligands_by_number_of_splits": ligand_counts,
        "intended_ligand_regime": "unconstrained in C1f",
        "maximum_cross_split_ligand_identity_similarity": 1.0 if exact_ligand else None,
        "ligand_identity_certificate": exact_ligand[0] if exact_ligand else None,
    }


def distributions(connection: sqlite3.Connection) -> dict:
    affinity = []
    for split, kind, count, mean, minimum, maximum, mean_square in connection.execute(
        """SELECT a.split, i.affinity_type, COUNT(*), AVG(i.p_affinity), MIN(i.p_affinity),
           MAX(i.p_affinity), AVG(i.p_affinity*i.p_affinity) FROM prepared.interactions i
           JOIN assignments a USING(target_id) GROUP BY a.split, i.affinity_type
           ORDER BY a.split, i.affinity_type"""
    ):
        affinity.append({
            "split": split, "affinity_type": kind, "count": count,
            "mean_p_affinity": mean,
            "std_p_affinity": float(max(0.0, mean_square - mean * mean) ** 0.5),
            "min_p_affinity": minimum, "max_p_affinity": maximum,
        })
    ligand_degree = np.fromiter((row[0] for row in connection.execute(
        "SELECT COUNT(*) FROM prepared.interactions GROUP BY ligand_id")), dtype=np.int64)
    target_degree = np.fromiter((row[0] for row in connection.execute(
        "SELECT COUNT(*) FROM prepared.interactions GROUP BY target_id")), dtype=np.int64)
    if len(affinity) != 12:
        raise RuntimeError("Not every affinity type is represented in every C1f split")
    return {"affinity": affinity, "ligand_observation_degree": summarize(ligand_degree),
            "target_observation_degree": summarize(target_degree)}


def graph_components(connection: sqlite3.Connection) -> dict:
    connection.execute("CREATE TEMP TABLE ligand_index AS SELECT ligand_id, ROW_NUMBER() OVER (ORDER BY ligand_id)-1 idx FROM prepared.ligands")
    connection.execute("CREATE INDEX temp.ligand_index_id ON ligand_index(ligand_id)")
    connection.execute("CREATE TEMP TABLE target_index AS SELECT target_id, ROW_NUMBER() OVER (ORDER BY target_id)-1 idx FROM prepared.targets")
    connection.execute("CREATE INDEX temp.target_index_id ON target_index(target_id)")
    ligands = connection.execute("SELECT COUNT(*) FROM ligand_index").fetchone()[0]
    targets = connection.execute("SELECT COUNT(*) FROM target_index").fetchone()[0]
    groups = UnionFind(ligands + targets)
    cursor = connection.execute(
        """SELECT l.idx, t.idx FROM prepared.interactions i
           JOIN ligand_index l USING(ligand_id) JOIN target_index t USING(target_id)"""
    )
    edges = 0
    while rows := cursor.fetchmany(100000):
        for ligand, target in rows:
            groups.union(int(ligand), ligands + int(target))
        edges += len(rows)
    roots = np.fromiter((groups.find(i) for i in range(ligands + targets)), dtype=np.int64)
    sizes = np.unique(roots, return_counts=True)[1]
    return {"nodes": ligands + targets, "ligand_nodes": ligands, "target_nodes": targets,
            "observation_edges": edges, "component_size_distribution": summarize(sizes)}


def validate_c1f(prepared_database: Path, assignment_database: Path, output: Path) -> dict:
    started = time.perf_counter()
    stages, checks = {}, {}
    with sqlite3.connect(assignment_database) as connection:
        connection.execute("ATTACH DATABASE ? AS prepared", (str(Path(prepared_database).resolve()),))
        for name, function in (("accounting", accounting), ("regime", regime),
                               ("protein_similarity", exact_protein_similarity),
                               ("distributions", distributions),
                               ("connected_components", graph_components)):
            print(f"C1f validation stage: {name}", flush=True)
            stage = time.perf_counter()
            checks[name] = function(connection)
            stages[name] = time.perf_counter() - stage
    report = {
        "format_version": 1, "validation_status": "passed", "regime": "C1f_cold_target",
        "prepared_database": str(prepared_database), "assignment_database": str(assignment_database),
        "runtime_seconds": {**stages, "total": time.perf_counter() - started},
        "peak_memory_mb": peak_rss_mb(), "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": None if os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY") is None else os.environ["DATASAIL_GIT_WORKTREE_DIRTY"].lower() == "true",
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"), "python_version": platform.python_version(),
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
