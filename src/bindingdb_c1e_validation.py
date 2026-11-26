"""Hard validation gates for a completed BindingDB scaled C1e candidate."""
from __future__ import annotations

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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


RDLogger.DisableLog("rdApp.*")
SPLITS = ("train", "val", "test")


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.int64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": int(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def certify_ligand_maximum(connection: sqlite3.Connection) -> dict:
    """Certify global maximum 1.0 by finding a cross-split fingerprint identity."""
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    seen: dict[bytes, tuple[str, str, str]] = {}
    scanned = 0
    cursor = connection.execute(
        "SELECT chemical_id, canonical_smiles, split FROM chemical_assignments ORDER BY chemical_id"
    )
    for chemical_id, smiles, split in cursor:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise RuntimeError(f"Invalid prepared canonical SMILES: {chemical_id}")
        fingerprint = generator.GetFingerprint(molecule)
        # Use the complete fingerprint bytes, not a secondary hash, so the
        # equality certificate is exact for the reported fingerprint metric.
        key = DataStructs.BitVectToBinaryText(fingerprint)
        previous = seen.get(key)
        scanned += 1
        if previous is not None and previous[0] != split:
            previous_split, previous_id, previous_smiles = previous
            return {
                "maximum_cross_split_ligand_similarity": 1.0,
                "metric": "Morgan radius=2, 2048-bit Tanimoto",
                "certificate": "identical_fingerprint_cross_split",
                "chemicals_scanned_until_certificate": scanned,
                "pair": [
                    {"chemical_id": previous_id, "split": previous_split, "smiles": previous_smiles},
                    {"chemical_id": chemical_id, "split": split, "smiles": smiles},
                ],
            }
        seen[key] = (split, chemical_id, smiles)
    raise RuntimeError(
        "No cross-split identical Morgan fingerprint was found; global maximum is not certified"
    )


def entity_and_observation_gates(connection: sqlite3.Connection) -> dict:
    assigned = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    distinct_assigned = connection.execute(
        "SELECT COUNT(DISTINCT ligand_id) FROM assignments"
    ).fetchone()[0]
    prepared_ligands = connection.execute("SELECT COUNT(*) FROM prepared.ligands").fetchone()[0]
    missing = connection.execute(
        """SELECT COUNT(*) FROM prepared.ligands l
           LEFT JOIN assignments a USING(ligand_id) WHERE a.ligand_id IS NULL"""
    ).fetchone()[0]
    invalid = connection.execute(
        "SELECT COUNT(*) FROM assignments WHERE split NOT IN ('train','val','test')"
    ).fetchone()[0]
    observations = connection.execute("SELECT COUNT(*) FROM prepared.interactions").fetchone()[0]
    joined_observations = connection.execute(
        "SELECT COUNT(*) FROM prepared.interactions i JOIN assignments a USING(ligand_id)"
    ).fetchone()[0]
    split_counts = dict(connection.execute(
        """SELECT a.split, COUNT(*) FROM prepared.interactions i
           JOIN assignments a USING(ligand_id) GROUP BY a.split"""
    ).fetchall())
    measurements = dict(connection.execute(
        "SELECT status, COUNT(*) FROM prepared.measurements GROUP BY status"
    ).fetchall())
    unexplained = connection.execute(
        """SELECT COUNT(*) FROM prepared.measurements
           WHERE status NOT IN ('retained','excluded')
              OR (status='excluded' AND exclusion_reason IS NULL)"""
    ).fetchone()[0]
    duplicate_conflicts = connection.execute(
        """SELECT COUNT(*) FROM (
               SELECT i.duplicate_group_id FROM prepared.interactions i
               JOIN assignments a USING(ligand_id)
               GROUP BY i.duplicate_group_id HAVING COUNT(DISTINCT a.split)>1)"""
    ).fetchone()[0]
    if not (
        assigned == distinct_assigned == prepared_ligands
        and missing == invalid == unexplained == duplicate_conflicts == 0
        and joined_observations == observations
    ):
        raise RuntimeError("C1e entity, observation, or duplicate accounting gate failed")
    return {
        "prepared_ligands": prepared_ligands,
        "assigned_ligands": assigned,
        "missing_ligand_assignments": missing,
        "invalid_split_assignments": invalid,
        "retained_observations": observations,
        "joined_observations": joined_observations,
        "dropped_cross_split_observations": 0,
        "requested_interaction_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
        "actual_interaction_counts": split_counts,
        "actual_interaction_ratios": {
            split: split_counts[split] / joined_observations for split in SPLITS
        },
        "source_measurements_by_status": measurements,
        "measurements_without_status_or_exclusion_reason": unexplained,
        "duplicate_split_conflicts": duplicate_conflicts,
    }


def unseen_rates_and_protein_maximum(connection: sqlite3.Connection) -> dict:
    chemical_overlap = connection.execute(
        """SELECT COUNT(*) FROM (
               SELECT chemical_id FROM chemical_assignments
               GROUP BY chemical_id HAVING COUNT(DISTINCT split)>1)"""
    ).fetchone()[0]
    test_chemicals = connection.execute(
        "SELECT COUNT(*) FROM chemical_assignments WHERE split='test'"
    ).fetchone()[0]
    target_split_counts = dict(connection.execute(
        """SELECT n_splits, COUNT(*) FROM (
               SELECT i.target_id, COUNT(DISTINCT a.split) n_splits
               FROM prepared.interactions i JOIN assignments a USING(ligand_id)
               GROUP BY i.target_id) GROUP BY n_splits"""
    ).fetchall())
    target_totals = dict(connection.execute(
        """SELECT a.split, COUNT(DISTINCT i.target_id)
           FROM prepared.interactions i JOIN assignments a USING(ligand_id)
           GROUP BY a.split"""
    ).fetchall())
    unseen_test_targets = connection.execute(
        """SELECT COUNT(*) FROM (
               SELECT DISTINCT i.target_id FROM prepared.interactions i
               JOIN assignments a USING(ligand_id) WHERE a.split='test'
               EXCEPT
               SELECT DISTINCT i.target_id FROM prepared.interactions i
               JOIN assignments a USING(ligand_id) WHERE a.split='train')"""
    ).fetchone()[0]
    protein_pair = connection.execute(
        """SELECT i.target_id, MIN(a.split), MAX(a.split)
           FROM prepared.interactions i JOIN assignments a USING(ligand_id)
           GROUP BY i.target_id HAVING COUNT(DISTINCT a.split)>1 LIMIT 1"""
    ).fetchone()
    if chemical_overlap or not protein_pair:
        raise RuntimeError("C1e unseen-ligand or protein-similarity gate failed")
    return {
        "unseen_test_ligand_rate": 1.0 if test_chemicals else None,
        "intended_ligand_regime": "cold; exact chemical identity must be unseen",
        "unseen_test_target_count": unseen_test_targets,
        "unseen_test_target_rate": unseen_test_targets / target_totals["test"],
        "intended_target_regime": "unconstrained in C1e; observed rate reported, not optimized",
        "targets_per_split": target_totals,
        "targets_by_number_of_splits": target_split_counts,
        "maximum_cross_split_protein_similarity": 1.0,
        "protein_metric": "exact sequence identity",
        "protein_maximum_certificate_target": protein_pair[0],
    }


def affinity_distributions(connection: sqlite3.Connection) -> list[dict]:
    rows = connection.execute(
        """SELECT a.split, i.affinity_type, COUNT(*), AVG(i.p_affinity),
                  MIN(i.p_affinity), MAX(i.p_affinity),
                  AVG(i.p_affinity*i.p_affinity)
           FROM prepared.interactions i JOIN assignments a USING(ligand_id)
           GROUP BY a.split, i.affinity_type ORDER BY a.split, i.affinity_type"""
    ).fetchall()
    output = []
    for split, affinity_type, count, mean, minimum, maximum, mean_square in rows:
        output.append({
            "split": split, "affinity_type": affinity_type, "count": count,
            "mean_p_affinity": mean,
            "std_p_affinity": float(max(0.0, mean_square - mean * mean) ** 0.5),
            "min_p_affinity": minimum, "max_p_affinity": maximum,
        })
    if len(output) != len(SPLITS) * 4:
        raise RuntimeError("Not every affinity type is represented in every split")
    return output


def degree_distributions(connection: sqlite3.Connection) -> dict:
    ligand = np.fromiter((row[0] for row in connection.execute(
        """SELECT COUNT(*) FROM prepared.interactions i JOIN assignments a USING(ligand_id)
           GROUP BY a.chemical_id""")), dtype=np.int64)
    target = np.fromiter((row[0] for row in connection.execute(
        "SELECT COUNT(*) FROM prepared.interactions GROUP BY target_id"
    )), dtype=np.int64)
    return {"canonical_chemical_observation_degree": summarize(ligand),
            "target_observation_degree": summarize(target)}


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.size = np.ones(size, dtype=np.int64)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def connected_components(connection: sqlite3.Connection) -> dict:
    chemicals = connection.execute("SELECT COUNT(*) FROM chemical_assignments").fetchone()[0]
    targets = connection.execute("SELECT COUNT(*) FROM prepared.targets").fetchone()[0]
    connection.execute("DROP TABLE IF EXISTS temp.target_index")
    connection.execute(
        "CREATE TEMP TABLE target_index AS SELECT target_id, ROW_NUMBER() OVER (ORDER BY target_id)-1 idx FROM prepared.targets"
    )
    connection.execute("CREATE INDEX temp.target_index_id ON target_index(target_id)")
    union_find = UnionFind(chemicals + targets)
    cursor = connection.execute(
        """SELECT c.rowid-1, t.idx FROM prepared.interactions i
           JOIN assignments a USING(ligand_id)
           JOIN chemical_assignments c USING(chemical_id)
           JOIN target_index t USING(target_id)"""
    )
    edge_count = 0
    while rows := cursor.fetchmany(100_000):
        for chemical, target in rows:
            union_find.union(int(chemical), chemicals + int(target))
        edge_count += len(rows)
    roots = np.fromiter((union_find.find(i) for i in range(chemicals + targets)), dtype=np.int64)
    component_sizes = np.unique(roots, return_counts=True)[1]
    return {
        "nodes": chemicals + targets,
        "canonical_chemical_nodes": chemicals,
        "target_nodes": targets,
        "observation_edges": edge_count,
        "component_size_distribution": summarize(component_sizes),
    }


def validate_c1e(prepared_database: Path, assignment_database: Path, output: Path) -> dict:
    started = time.perf_counter()
    stages = {}
    with sqlite3.connect(assignment_database) as connection:
        connection.execute("ATTACH DATABASE ? AS prepared", (str(Path(prepared_database).resolve()),))
        checks = {}
        for name, function in (
            ("accounting", entity_and_observation_gates),
            ("regime", unseen_rates_and_protein_maximum),
            ("ligand_similarity", certify_ligand_maximum),
            ("affinity_distributions", affinity_distributions),
            ("degree_distributions", degree_distributions),
            ("connected_components", connected_components),
        ):
            print(f"C1e validation stage: {name}", flush=True)
            stage = time.perf_counter()
            checks[name] = function(connection)
            stages[name] = time.perf_counter() - stage
    report = {
        "format_version": 1,
        "validation_status": "passed",
        "regime": "C1e_cold_ligand",
        "prepared_database": str(prepared_database),
        "assignment_database": str(assignment_database),
        "runtime_seconds": {**stages, "total": time.perf_counter() - started},
        "peak_memory_mb": peak_rss_mb(),
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": (
            None if os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY") is None
            else os.environ["DATASAIL_GIT_WORKTREE_DIRTY"].lower() == "true"
        ),
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"),
        "python_version": platform.python_version(),
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
