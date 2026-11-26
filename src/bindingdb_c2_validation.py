"""Hard acceptance gates for a completed BindingDB scaled C2 candidate."""
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

from src.bindingdb_c1e_validation import UnionFind, certify_ligand_maximum, summarize
from src.bindingdb_c1f_validation import exact_protein_similarity
from src.bindingdb_c2 import SPLIT_RATIOS


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def accounting(connection: sqlite3.Connection) -> dict:
    prepared_ligands = connection.execute("SELECT COUNT(*) FROM prepared.ligands").fetchone()[0]
    assigned_ligands = connection.execute("SELECT COUNT(*) FROM ligand_assignments").fetchone()[0]
    prepared_targets = connection.execute("SELECT COUNT(*) FROM prepared.targets").fetchone()[0]
    assigned_targets = connection.execute("SELECT COUNT(*) FROM target_assignments").fetchone()[0]
    chemicals = connection.execute("SELECT COUNT(*) FROM chemical_assignments").fetchone()[0]
    chemical_conflicts = connection.execute(
        "SELECT COUNT(*) FROM (SELECT chemical_id FROM ligand_assignments GROUP BY chemical_id HAVING COUNT(DISTINCT split)>1)"
    ).fetchone()[0]
    target_conflicts = connection.execute(
        "SELECT COUNT(*) FROM (SELECT target_id FROM target_assignments GROUP BY target_id HAVING COUNT(DISTINCT split)>1)"
    ).fetchone()[0]
    total = connection.execute("SELECT COUNT(*) FROM interaction_assignments").fetchone()[0]
    prepared_total = connection.execute("SELECT COUNT(*) FROM prepared.interactions").fetchone()[0]
    retained_counts = dict(connection.execute(
        "SELECT split, COUNT(*) FROM interaction_assignments WHERE split IS NOT NULL GROUP BY split"
    ).fetchall())
    retained = sum(retained_counts.values())
    dropped = connection.execute(
        "SELECT COUNT(*) FROM interaction_assignments WHERE split IS NULL"
    ).fetchone()[0]
    missing_reason = connection.execute(
        "SELECT COUNT(*) FROM interaction_assignments WHERE split IS NULL AND exclusion_reason IS NULL"
    ).fetchone()[0]
    retained_cross = connection.execute(
        """SELECT COUNT(*) FROM prepared.interactions i JOIN interaction_assignments a USING(observation_id)
           JOIN ligand_assignments l USING(ligand_id) JOIN target_assignments t USING(target_id)
           WHERE a.split IS NOT NULL AND (l.split<>t.split OR a.split<>l.split)"""
    ).fetchone()[0]
    dropped_same = connection.execute(
        """SELECT COUNT(*) FROM prepared.interactions i JOIN interaction_assignments a USING(observation_id)
           JOIN ligand_assignments l USING(ligand_id) JOIN target_assignments t USING(target_id)
           WHERE a.split IS NULL AND l.split=t.split"""
    ).fetchone()[0]
    duplicate_conflicts = connection.execute(
        """SELECT COUNT(*) FROM (SELECT i.duplicate_group_id FROM prepared.interactions i
           JOIN interaction_assignments a USING(observation_id) GROUP BY i.duplicate_group_id
           HAVING COUNT(DISTINCT COALESCE(a.split,'dropped'))>1)"""
    ).fetchone()[0]
    ratios = {split: retained_counts[split] / retained for split in SPLIT_RATIOS}
    deviations = {split: abs(ratios[split] - SPLIT_RATIOS[split]) for split in SPLIT_RATIOS}
    measurement_status = dict(connection.execute(
        "SELECT status, COUNT(*) FROM prepared.measurements GROUP BY status"
    ).fetchall())
    measurement_outcomes = dict(connection.execute(
        """SELECT outcome, COUNT(*) FROM (SELECT i.source_measurement_id,
           CASE WHEN SUM(CASE WHEN a.split IS NOT NULL THEN 1 ELSE 0 END)>0
                THEN 'has_retained_observation' ELSE 'all_observations_dropped_cross_split' END outcome
           FROM prepared.interactions i JOIN interaction_assignments a USING(observation_id)
           GROUP BY i.source_measurement_id) GROUP BY outcome"""
    ).fetchall())
    if not (prepared_ligands == assigned_ligands and prepared_targets == assigned_targets
            and total == prepared_total == retained + dropped and chemical_conflicts == 0
            and target_conflicts == missing_reason == retained_cross == dropped_same == 0
            and duplicate_conflicts == 0 and retained / total >= 0.80
            and max(deviations.values()) <= 0.05):
        raise RuntimeError("C2 accounting, atomicity, retention, or ratio gate failed")
    return {
        "canonical_chemicals": chemicals, "prepared_ligands": prepared_ligands,
        "assigned_ligands": assigned_ligands, "prepared_targets": prepared_targets,
        "assigned_targets": assigned_targets, "source_observations": total,
        "retained_observations": retained, "dropped_cross_split_observations": dropped,
        "retention_rate": retained / total, "minimum_retention_rate": 0.80,
        "retained_counts": retained_counts, "retained_ratios": ratios,
        "requested_ratios": SPLIT_RATIOS, "absolute_ratio_deviation": deviations,
        "maximum_allowed_ratio_deviation": 0.05,
        "canonical_chemical_split_conflicts": chemical_conflicts,
        "target_split_conflicts": target_conflicts,
        "duplicate_outcome_or_split_conflicts": duplicate_conflicts,
        "retained_cross_split_endpoint_conflicts": retained_cross,
        "dropped_same_split_endpoint_errors": dropped_same,
        "dropped_without_exclusion_reason": missing_reason,
        "source_measurements_by_preparation_status": measurement_status,
        "retained_source_measurement_outcomes": measurement_outcomes,
    }


def regime(connection: sqlite3.Connection) -> dict:
    test_chemicals = connection.execute(
        "SELECT COUNT(*) FROM chemical_assignments WHERE split='test'"
    ).fetchone()[0]
    test_targets = connection.execute(
        "SELECT COUNT(*) FROM target_assignments WHERE split='test'"
    ).fetchone()[0]
    if not test_chemicals or not test_targets:
        raise RuntimeError("C2 test entity sets are empty")
    dropped_pairs = [
        {"ligand_split": ligand, "target_split": target, "count": count}
        for ligand, target, count in connection.execute(
            """SELECT l.split, t.split, COUNT(*) FROM prepared.interactions i
               JOIN interaction_assignments a USING(observation_id)
               JOIN ligand_assignments l USING(ligand_id) JOIN target_assignments t USING(target_id)
               WHERE a.split IS NULL GROUP BY l.split, t.split ORDER BY l.split, t.split"""
        )
    ]
    return {
        "unseen_test_ligand_rate": 1.0,
        "unseen_test_target_rate": 1.0,
        "intended_regime": "double cold; both exact endpoint identities unseen",
        "dropped_endpoint_split_pairs": dropped_pairs,
    }


def distributions(connection: sqlite3.Connection) -> dict:
    affinity = []
    for split, kind, count, mean, minimum, maximum, mean_square in connection.execute(
        """SELECT a.split, i.affinity_type, COUNT(*), AVG(i.p_affinity), MIN(i.p_affinity),
           MAX(i.p_affinity), AVG(i.p_affinity*i.p_affinity) FROM prepared.interactions i
           JOIN interaction_assignments a USING(observation_id) WHERE a.split IS NOT NULL
           GROUP BY a.split, i.affinity_type ORDER BY a.split, i.affinity_type"""
    ):
        affinity.append({"split": split, "affinity_type": kind, "count": count,
                         "mean_p_affinity": mean,
                         "std_p_affinity": float(max(0, mean_square-mean*mean)**0.5),
                         "min_p_affinity": minimum, "max_p_affinity": maximum})
    chemical_degree = np.fromiter((row[0] for row in connection.execute(
        """SELECT COUNT(*) FROM prepared.interactions i JOIN interaction_assignments a USING(observation_id)
           JOIN ligand_assignments l USING(ligand_id) WHERE a.split IS NOT NULL GROUP BY l.chemical_id"""
    )), dtype=np.int64)
    target_degree = np.fromiter((row[0] for row in connection.execute(
        """SELECT COUNT(*) FROM prepared.interactions i JOIN interaction_assignments a USING(observation_id)
           WHERE a.split IS NOT NULL GROUP BY i.target_id"""
    )), dtype=np.int64)
    if len(affinity) != 12:
        raise RuntimeError("Not every C2 retained split contains all affinity types")
    return {"retained_affinity": affinity,
            "retained_canonical_chemical_degree": summarize(chemical_degree),
            "retained_target_degree": summarize(target_degree)}


def retained_components(connection: sqlite3.Connection) -> dict:
    chemicals = connection.execute("SELECT COUNT(*) FROM chemical_assignments").fetchone()[0]
    targets = connection.execute("SELECT COUNT(*) FROM target_assignments").fetchone()[0]
    groups = UnionFind(chemicals + targets)
    cursor = connection.execute(
        """SELECT c.rowid-1, t.rowid-1 FROM prepared.interactions i
           JOIN interaction_assignments a USING(observation_id)
           JOIN ligand_assignments l USING(ligand_id)
           JOIN chemical_assignments c USING(chemical_id)
           JOIN target_assignments t USING(target_id) WHERE a.split IS NOT NULL"""
    )
    edges = 0
    while rows := cursor.fetchmany(100000):
        for chemical, target in rows:
            groups.union(int(chemical), chemicals + int(target))
        edges += len(rows)
    roots = np.fromiter((groups.find(i) for i in range(chemicals + targets)), dtype=np.int64)
    sizes = np.unique(roots, return_counts=True)[1]
    return {"nodes_including_isolates": chemicals + targets,
            "retained_observation_edges": edges,
            "component_size_distribution_including_isolates": summarize(sizes),
            "isolated_entity_nodes": int(np.sum(sizes == 1))}


def validate_c2(prepared_database: Path, assignment_database: Path, output: Path) -> dict:
    started = time.perf_counter(); timings, checks = {}, {}
    with sqlite3.connect(assignment_database) as connection:
        connection.execute("ATTACH DATABASE ? AS prepared", (str(Path(prepared_database).resolve()),))
        connection.execute("CREATE TEMP VIEW assignments AS SELECT target_id, split FROM target_assignments")
        for name, function in (
            ("accounting", accounting), ("regime", regime),
            ("ligand_similarity", certify_ligand_maximum),
            ("protein_similarity", exact_protein_similarity),
            ("distributions", distributions), ("retained_components", retained_components),
        ):
            print(f"C2 validation stage: {name}", flush=True)
            stage = time.perf_counter(); checks[name] = function(connection)
            timings[name] = time.perf_counter() - stage
    report = {
        "format_version": 1, "validation_status": "passed", "regime": "C2_double_cold",
        "prepared_database": str(prepared_database), "assignment_database": str(assignment_database),
        "runtime_seconds": {**timings, "total": time.perf_counter()-started},
        "peak_memory_mb": peak_rss_mb(), "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": None if os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY") is None else os.environ["DATASAIL_GIT_WORKTREE_DIRTY"].lower()=="true",
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"), "python_version": platform.python_version(),
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    return report
