"""Scaled double-cold (C2) splitting for normalized BindingDB data."""
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

from src.bindingdb_c1e import SPLIT_RATIOS, choose_degree_anchors, load_ligand_stats, seeded_subset
from src.bindingdb_c1f import load_target_stats, protein_features
from src.datasail_adapter import run_c2


RDLogger.DisableLog("rdApp.*")
SPLIT_INDEX = {name: index for index, name in enumerate(SPLIT_RATIOS)}


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def molecular_fingerprints(smiles: list[str]):
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    output = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid prepared SMILES: {value}")
        output.append(generator.GetFingerprint(molecule))
    return output


def fingerprint_similarity(fingerprints: list, positions: np.ndarray) -> np.ndarray:
    matrix = np.eye(len(positions), dtype=np.float32)
    for row, position in enumerate(positions):
        if row + 1 == len(positions):
            continue
        values = DataStructs.BulkTanimotoSimilarity(
            fingerprints[position], [fingerprints[p] for p in positions[row + 1:]],
        )
        matrix[row, row + 1:] = values
        matrix[row + 1:, row] = values
    return matrix


def assign_targets(frame, features, anchors, anchor_splits, *, k, balance_weight, seed):
    from src.bindingdb_c1f import assign_remaining_targets
    return assign_remaining_targets(
        frame, features, anchors, anchor_splits, k=k,
        balance_weight=balance_weight, seed=seed,
    )


def graph_votes(
    prepared_database: Path, ligand_frame, target_ids: list[str], target_labels: np.ndarray,
) -> np.ndarray:
    target_map = dict(zip(target_ids, target_labels))
    chemical_position = dict(zip(ligand_frame.canonical_smiles, range(len(ligand_frame))))
    votes = np.zeros((len(ligand_frame), len(SPLIT_RATIOS)), dtype=np.int64)
    uri = f"file:{Path(prepared_database).resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        cursor = connection.execute(
            """SELECT l.canonical_smiles, i.target_id, COUNT(*)
               FROM interactions i JOIN ligands l USING(ligand_id)
               GROUP BY l.canonical_smiles, i.target_id"""
        )
        while rows := cursor.fetchmany(100000):
            for smiles, target_id, count in rows:
                position = chemical_position.get(smiles)
                split = target_map.get(target_id)
                if position is not None and split is not None:
                    votes[position, SPLIT_INDEX[str(split)]] += int(count)
    return votes


def assign_ligands(
    frame, fingerprints, anchors, anchor_splits, votes, *, k, balance_weight,
    graph_weight, similarity_weight,
):
    labels = np.empty(len(frame), dtype=object)
    confidence = np.full(len(frame), np.nan, dtype=float)
    labels[anchors] = anchor_splits
    confidence[anchors] = 1.0
    anchor_fps = [fingerprints[p] for p in anchors]
    split_anchor_positions = {
        split: np.flatnonzero(anchor_splits == split) for split in SPLIT_RATIOS
    }
    current = Counter()
    for split, degree in zip(anchor_splits, frame.iloc[anchors].degree):
        current[str(split)] += int(degree)
    total = sum(current.values())
    remaining = np.setdiff1d(np.arange(len(frame)), anchors, assume_unique=True)
    remaining_degrees = frame.degree.to_numpy(dtype=np.int64)[remaining]
    remaining_ids = frame.chemical_id.astype(str).to_numpy()[remaining]
    remaining = remaining[np.lexsort((remaining_ids, -remaining_degrees))]
    for position in remaining:
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(
            fingerprints[position], anchor_fps,
        ))
        vote_total = int(votes[position].sum())
        scores = {}
        for split, target_ratio in SPLIT_RATIOS.items():
            candidate = split_anchor_positions[split]
            split_k = min(k, len(candidate))
            local = similarities[candidate]
            nearest = np.argpartition(-local, split_k - 1)[:split_k]
            similarity = float(local[nearest].mean())
            graph_fraction = (
                float(votes[position, SPLIT_INDEX[split]]) / vote_total if vote_total else 0.0
            )
            balance = target_ratio / max(current[split] / max(total, 1), 1e-12)
            scores[split] = (
                graph_weight * graph_fraction + similarity_weight * similarity + 1e-9
            ) * balance ** balance_weight
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best = ordered[0][0]
        labels[position] = best
        confidence[position] = (ordered[0][1] - ordered[1][1]) / ordered[0][1]
        degree = int(frame.iloc[position].degree)
        current[best] += degree
        total += degree
    return labels, confidence


def write_and_audit(
    prepared_database: Path, output_database: Path, ligand_frame, target_frame,
    ligand_labels, target_labels, ligand_confidence, target_confidence,
    ligand_anchors, target_anchors,
) -> dict:
    if output_database.exists():
        raise FileExistsError(f"C2 output already exists: {output_database}")
    ligand_anchor_flag = np.zeros(len(ligand_frame), dtype=np.int8)
    target_anchor_flag = np.zeros(len(target_frame), dtype=np.int8)
    ligand_anchor_flag[ligand_anchors] = 1
    target_anchor_flag[target_anchors] = 1
    with sqlite3.connect(output_database) as connection:
        connection.execute(
            """CREATE TABLE chemical_assignments (chemical_id TEXT PRIMARY KEY,
               canonical_smiles TEXT UNIQUE, split TEXT, is_anchor INTEGER,
               degree INTEGER, confidence REAL)"""
        )
        connection.executemany("INSERT INTO chemical_assignments VALUES (?,?,?,?,?,?)", zip(
            ligand_frame.chemical_id.astype(str), ligand_frame.canonical_smiles.astype(str),
            ligand_labels.astype(str), ligand_anchor_flag, ligand_frame.degree.astype(int),
            ligand_confidence.astype(float),
        ))
        connection.execute(
            """CREATE TABLE target_assignments (target_id TEXT PRIMARY KEY, split TEXT,
               is_anchor INTEGER, degree INTEGER, confidence REAL)"""
        )
        connection.executemany("INSERT INTO target_assignments VALUES (?,?,?,?,?)", zip(
            target_frame.target_id.astype(str), target_labels.astype(str), target_anchor_flag,
            target_frame.degree.astype(int), target_confidence.astype(float),
        ))
        connection.execute("ATTACH DATABASE ? AS prepared", (str(Path(prepared_database).resolve()),))
        connection.execute(
            """CREATE TABLE ligand_assignments AS SELECT l.ligand_id, c.chemical_id, c.split,
               c.is_anchor, c.degree, c.confidence FROM prepared.ligands l
               JOIN chemical_assignments c USING(canonical_smiles)"""
        )
        connection.execute("CREATE UNIQUE INDEX ligand_assignment_id ON ligand_assignments(ligand_id)")
        connection.execute(
            """CREATE TABLE interaction_assignments AS
               SELECT i.observation_id,
                      CASE WHEN l.split=t.split THEN l.split ELSE NULL END split,
                      CASE WHEN l.split=t.split THEN NULL ELSE 'cold_double_cross_split_endpoints' END exclusion_reason
               FROM prepared.interactions i JOIN ligand_assignments l USING(ligand_id)
               JOIN target_assignments t USING(target_id)"""
        )
        connection.execute("CREATE UNIQUE INDEX interaction_assignment_id ON interaction_assignments(observation_id)")
        retained = dict(connection.execute(
            "SELECT split, COUNT(*) FROM interaction_assignments WHERE split IS NOT NULL GROUP BY split"
        ).fetchall())
        dropped = connection.execute(
            "SELECT COUNT(*) FROM interaction_assignments WHERE split IS NULL"
        ).fetchone()[0]
        total = connection.execute("SELECT COUNT(*) FROM interaction_assignments").fetchone()[0]
        ligand_conflicts = connection.execute(
            "SELECT COUNT(*) FROM (SELECT chemical_id FROM ligand_assignments GROUP BY chemical_id HAVING COUNT(DISTINCT split)>1)"
        ).fetchone()[0]
        target_conflicts = connection.execute(
            "SELECT COUNT(*) FROM (SELECT target_id FROM target_assignments GROUP BY target_id HAVING COUNT(DISTINCT split)>1)"
        ).fetchone()[0]
        duplicate_conflicts = connection.execute(
            """SELECT COUNT(*) FROM (SELECT i.duplicate_group_id FROM prepared.interactions i
               JOIN interaction_assignments a USING(observation_id) WHERE a.split IS NOT NULL
               GROUP BY i.duplicate_group_id HAVING COUNT(DISTINCT a.split)>1)"""
        ).fetchone()[0]
        cross_retained = connection.execute(
            """SELECT COUNT(*) FROM prepared.interactions i JOIN ligand_assignments l USING(ligand_id)
               JOIN target_assignments t USING(target_id) JOIN interaction_assignments a USING(observation_id)
               WHERE a.split IS NOT NULL AND l.split<>t.split"""
        ).fetchone()[0]
    if total != sum(retained.values()) + dropped or ligand_conflicts or target_conflicts or duplicate_conflicts or cross_retained:
        raise RuntimeError("C2 core accounting failed")
    retained_total = sum(retained.values())
    retention_rate = retained_total / total
    retained_ratios = {s: retained.get(s, 0) / retained_total for s in SPLIT_RATIOS}
    ratio_deviation = {
        split: abs(retained_ratios[split] - SPLIT_RATIOS[split]) for split in SPLIT_RATIOS
    }
    if retention_rate < 0.80 or max(ratio_deviation.values()) > 0.05:
        raise RuntimeError(
            f"C2 retention/ratio gate failed: retention={retention_rate}, deviations={ratio_deviation}"
        )
    return {
        "source_observations": total, "retained_observations": retained_total,
        "dropped_cross_split_observations": dropped,
        "retention_rate": retention_rate,
        "minimum_retention_rate": 0.80,
        "retained_interaction_counts": retained,
        "retained_interaction_ratios": retained_ratios,
        "absolute_ratio_deviation": ratio_deviation,
        "maximum_allowed_ratio_deviation": 0.05,
        "requested_interaction_ratios": SPLIT_RATIOS,
        "canonical_chemical_split_conflicts": ligand_conflicts,
        "target_split_conflicts": target_conflicts,
        "duplicate_split_conflicts": duplicate_conflicts,
        "retained_cross_split_endpoint_conflicts": cross_retained,
        "dropped_interaction_reason": "cold_double_cross_split_endpoints",
    }


def run_scaled_c2(
    prepared_database: Path, output_dir: Path, *, max_ligands: int | None,
    max_targets: int | None, ligand_anchor_ratio: float, target_anchor_ratio: float,
    seed: int, k: int, balance_weight: float, graph_weight: float,
    similarity_weight: float, ligand_clusters: int, target_clusters: int,
    max_sec: int, threads: int,
) -> dict:
    started = time.perf_counter(); timings = {}
    stage = time.perf_counter(); print("C2 stage: loading entity statistics", flush=True)
    ligands = seeded_subset(load_ligand_stats(prepared_database), max_ligands, seed)
    targets = seeded_subset(load_target_stats(prepared_database), max_targets, seed)
    timings["load_entities"] = time.perf_counter() - stage
    stage = time.perf_counter(); print("C2 stage: encoding similarities", flush=True)
    ligand_fps = molecular_fingerprints(ligands.canonical_smiles.astype(str).tolist())
    target_features = protein_features(targets.sequence.astype(str).tolist())
    ligand_anchors = choose_degree_anchors(ligands, ligand_anchor_ratio, seed)
    target_anchors = choose_degree_anchors(targets, target_anchor_ratio, seed)
    ligand_sim = fingerprint_similarity(ligand_fps, ligand_anchors)
    target_sim = (target_features[target_anchors] @ target_features[target_anchors].T).toarray()
    timings["features_and_anchor_similarity"] = time.perf_counter() - stage
    ligand_anchor_frame, target_anchor_frame = ligands.iloc[ligand_anchors], targets.iloc[target_anchors]
    # DataSAIL C2 optimizes the two similarity partitions; its solver does not
    # use the interaction list. Retention is optimized explicitly during the
    # graph-aware ligand extension below and audited from the complete graph.
    interactions: list[tuple[str, str]] = []
    stage = time.perf_counter(); print("C2 stage: running DataSAIL anchors", flush=True)
    result = run_c2(
        dict(zip(ligand_anchor_frame.chemical_id, ligand_anchor_frame.canonical_smiles)),
        dict(zip(ligand_anchor_frame.chemical_id, np.log1p(ligand_anchor_frame.degree.astype(float)))),
        ligand_anchor_frame.chemical_id.tolist(), ligand_sim,
        dict(zip(target_anchor_frame.target_id, target_anchor_frame.sequence)),
        dict(zip(target_anchor_frame.target_id, np.log1p(target_anchor_frame.degree.astype(float)))),
        target_anchor_frame.target_id.tolist(), target_sim, interactions,
        splits=SPLIT_RATIOS, ligand_clusters=ligand_clusters,
        target_clusters=target_clusters, max_sec=max_sec, threads=threads,
    )
    timings["datasail"] = time.perf_counter() - stage
    ligand_anchor_labels = np.asarray([result.e_splits[x] for x in ligand_anchor_frame.chemical_id])
    target_anchor_labels = np.asarray([result.f_splits[x] for x in target_anchor_frame.target_id])
    stage = time.perf_counter(); print("C2 stage: extending targets", flush=True)
    target_labels, target_confidence = assign_targets(
        targets, target_features, target_anchors, target_anchor_labels,
        k=k, balance_weight=balance_weight, seed=seed,
    )
    timings["target_extension"] = time.perf_counter() - stage
    stage = time.perf_counter(); print("C2 stage: collecting graph votes", flush=True)
    votes = graph_votes(prepared_database, ligands, targets.target_id.tolist(), target_labels)
    timings["graph_votes"] = time.perf_counter() - stage
    stage = time.perf_counter(); print("C2 stage: extending ligands", flush=True)
    ligand_labels, ligand_confidence = assign_ligands(
        ligands, ligand_fps, ligand_anchors, ligand_anchor_labels, votes, k=k,
        balance_weight=balance_weight, graph_weight=graph_weight,
        similarity_weight=similarity_weight,
    )
    timings["ligand_extension"] = time.perf_counter() - stage
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = time.perf_counter(); print("C2 stage: writing and auditing", flush=True)
    audit = write_and_audit(
        prepared_database, output_dir / "c2_assignments.sqlite", ligands, targets,
        ligand_labels, target_labels, ligand_confidence, target_confidence,
        ligand_anchors, target_anchors,
    )
    timings["write_and_audit"] = time.perf_counter() - stage
    manifest = {
        "format_version": 1, "method": "bindingdb_scaled_c2", "seed": seed,
        "n_canonical_chemicals": len(ligands), "n_targets": len(targets),
        "ligand_anchor_ratio": ligand_anchor_ratio, "ligand_anchor_count": len(ligand_anchors),
        "target_anchor_ratio": target_anchor_ratio, "target_anchor_count": len(target_anchors),
        "ligand_similarity": "Morgan radius=2 2048-bit Tanimoto",
        "target_similarity": "normalized hashed 3-mer frequency cosine",
        "datasail_anchor_weight": "log1p_observation_degree", "k": k,
        "balance_weight": balance_weight, "graph_weight": graph_weight,
        "similarity_weight": similarity_weight, "ligand_clusters": ligand_clusters,
        "target_clusters": target_clusters, "runtime_seconds": {**timings, "total": time.perf_counter()-started},
        "peak_memory_mb": peak_rss_mb(),
        "validation_status": "core_assignments_passed_pending_similarity_and_graph_audits",
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": None if os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY") is None else os.environ["DATASAIL_GIT_WORKTREE_DIRTY"].lower() == "true",
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"), "python_version": platform.python_version(),
        "audit": audit,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
