"""Compute reproducible leakage distributions for accepted BindingDB splits."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import resource
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.bindingdb_c1f import protein_features


RDLogger.DisableLog("rdApp.*")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def deterministic_sample(rows: list[tuple], cap: int, seed: int) -> list[tuple]:
    if len(rows) <= cap:
        return rows
    positions = np.sort(np.random.default_rng(seed).choice(len(rows), cap, replace=False))
    return [rows[position] for position in positions]


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    result = {
        "count": int(len(values)), "mean": float(values.mean()),
        "std": float(values.std()), "min": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)), "max": float(values.max()),
    }
    result["fraction_at_1"] = float(np.mean(np.isclose(values, 1.0)))
    for threshold in THRESHOLDS:
        result[f"fraction_ge_{threshold:.1f}"] = float(np.mean(values >= threshold))
    return result


def fingerprints(rows: list[tuple], chirality: bool):
    generator = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=chirality)
    output = []
    for _, smiles in rows:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise RuntimeError(f"Invalid prepared SMILES: {smiles}")
        output.append(generator.GetFingerprint(molecule))
    return output


def ligand_nn(
    assignment_database: Path, regime: str, split: str, *, query_cap: int,
    reference_cap: int, seed: int, chirality: bool,
) -> tuple[dict, list[dict]]:
    with sqlite3.connect(assignment_database) as connection:
        train = connection.execute(
            "SELECT chemical_id, canonical_smiles FROM chemical_assignments WHERE split='train' ORDER BY chemical_id"
        ).fetchall()
        query = connection.execute(
            "SELECT chemical_id, canonical_smiles FROM chemical_assignments WHERE split=? ORDER BY chemical_id",
            (split,),
        ).fetchall()
    train = deterministic_sample(train, reference_cap, seed)
    query = deterministic_sample(query, query_cap, seed + (1 if split == "val" else 2))
    train_fp, query_fp = fingerprints(train, chirality), fingerprints(query, chirality)
    raw = []
    values = np.empty(len(query), dtype=float)
    for index, ((query_id, _), fp) in enumerate(zip(query, query_fp)):
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fp))
        best = int(np.argmax(similarities))
        values[index] = similarities[best]
        raw.append({
            "regime": regime, "entity": "ligand", "query_split": split,
            "metric": "Morgan_r2_2048_Tanimoto_chiral" if chirality else "Morgan_r2_2048_Tanimoto",
            "query_id": query_id, "nearest_train_id": train[best][0],
            "maximum_similarity": float(similarities[best]),
        })
    summary = {
        "regime": regime, "entity": "ligand", "query_split": split,
        "metric": raw[0]["metric"], "calculation": "deterministic sampled exact comparisons",
        "seed": seed, "query_population": len(query), "query_cap": query_cap,
        "train_population_used": len(train), "train_reference_cap": reference_cap,
        **summarize(values),
    }
    return summary, raw


def protein_nn(
    prepared_database: Path, assignment_database: Path, regime: str, split: str,
    *, chunk_size: int = 256,
) -> tuple[dict, list[dict]]:
    assignment_table = "assignments" if regime == "C1f" else "target_assignments"
    with sqlite3.connect(assignment_database) as connection:
        connection.execute("ATTACH DATABASE ? AS prepared", (str(prepared_database.resolve()),))
        rows = connection.execute(
            f"""SELECT t.target_id, t.sequence, a.split FROM prepared.targets t
                 JOIN {assignment_table} a USING(target_id) ORDER BY t.target_id"""
        ).fetchall()
    ids = np.asarray([row[0] for row in rows], dtype=object)
    labels = np.asarray([row[2] for row in rows], dtype=object)
    features = protein_features([row[1] for row in rows])
    train, query = np.flatnonzero(labels == "train"), np.flatnonzero(labels == split)
    values = np.empty(len(query), dtype=float)
    nearest = np.empty(len(query), dtype=object)
    for start in range(0, len(query), chunk_size):
        query_positions = query[start:start + chunk_size]
        block = (features[query_positions] @ features[train].T).toarray()
        best = np.argmax(block, axis=1)
        values[start:start + len(query_positions)] = block[np.arange(len(best)), best]
        nearest[start:start + len(query_positions)] = ids[train[best]]
    raw = [{
        "regime": regime, "entity": "protein", "query_split": split,
        "metric": "hashed_3mer_frequency_cosine", "query_id": ids[position],
        "nearest_train_id": nearest[index], "maximum_similarity": float(values[index]),
    } for index, position in enumerate(query)]
    summary = {
        "regime": regime, "entity": "protein", "query_split": split,
        "metric": "hashed_3mer_frequency_cosine",
        "calculation": "exhaustive all query-to-train target pairs",
        "query_population": len(query), "train_population_used": len(train),
        **summarize(values),
    }
    return summary, raw


def write_gzip_tsv(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--c1e", type=Path, required=True)
    parser.add_argument("--c1f", type=Path, required=True)
    parser.add_argument("--c2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ligand-query-cap", type=int, default=5000)
    parser.add_argument("--ligand-reference-cap", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.perf_counter(); summaries, raw = [], []
    for regime, database in (("C1e", args.c1e), ("C2", args.c2)):
        for split in ("val", "test"):
            for chirality in (False, True):
                print(f"Leakage: {regime} ligand {split} chirality={chirality}", flush=True)
                stage = time.perf_counter()
                summary, rows = ligand_nn(
                    database, regime, split, query_cap=args.ligand_query_cap,
                    reference_cap=args.ligand_reference_cap, seed=args.seed,
                    chirality=chirality,
                )
                summary["runtime_seconds"] = time.perf_counter() - stage
                summaries.append(summary); raw.extend(rows)
    for regime, database in (("C1f", args.c1f), ("C2", args.c2)):
        for split in ("val", "test"):
            print(f"Leakage: {regime} protein {split}", flush=True)
            stage = time.perf_counter()
            summary, rows = protein_nn(args.prepared, database, regime, split)
            summary["runtime_seconds"] = time.perf_counter() - stage
            summaries.append(summary); raw.extend(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_gzip_tsv(args.output_dir / "nearest_neighbor_raw.tsv.gz", raw)
    (args.output_dir / "leakage_summary.json").write_text(json.dumps({
        "format_version": 1, "seed": args.seed,
        "ligand_query_cap": args.ligand_query_cap,
        "ligand_reference_cap": args.ligand_reference_cap,
        "runtime_seconds": time.perf_counter()-started,
        "peak_memory_mb": peak_rss_mb(),
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": None if os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY") is None else os.environ["DATASAIL_GIT_WORKTREE_DIRTY"].lower() == "true",
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"),
        "python_version": platform.python_version(), "summaries": summaries,
    }, indent=2) + "\n")
    with (args.output_dir / "leakage_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
