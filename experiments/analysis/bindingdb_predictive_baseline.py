"""Bounded interaction-aware regression baselines for accepted BindingDB splits."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import resource
import sqlite3
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RDLogger.DisableLog("rdApp.*")
AFFINITY_TYPES = ("Ki", "Kd", "IC50", "EC50")
LIGAND_FEATURES = 2048
PROTEIN_FEATURES = 4096


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def query_for(regime: str, seed: int) -> str:
    if regime == "C1e":
        join = "JOIN splitdb.assignments a USING(ligand_id)"
        split = "a.split"
    elif regime == "C1f":
        join = "JOIN splitdb.assignments a USING(target_id)"
        split = "a.split"
    elif regime == "C2":
        join = "JOIN splitdb.interaction_assignments a USING(observation_id)"
        split = "a.split"
    else:
        raise ValueError(regime)
    return f"""SELECT i.observation_id, l.canonical_smiles, t.sequence,
                      i.affinity_type, i.p_affinity
               FROM interactions i JOIN ligands l USING(ligand_id)
               JOIN targets t USING(target_id) {join}
               WHERE {split}=? AND i.affinity_relation='='
               ORDER BY ((i.source_row * 1103515245 +
                          CASE i.affinity_type WHEN 'Ki' THEN 101 WHEN 'Kd' THEN 211
                               WHEN 'IC50' THEN 307 ELSE 401 END + {int(seed)}) % 2147483647),
                        i.observation_id LIMIT ?"""


class FeatureEncoder:
    def __init__(self):
        self.ligand = GetMorganGenerator(radius=2, fpSize=LIGAND_FEATURES, includeChirality=True)
        self.protein = HashingVectorizer(
            analyzer="char", ngram_range=(3, 3), n_features=PROTEIN_FEATURES,
            alternate_sign=False, norm="l2", lowercase=False,
        )

    def transform(self, rows: list[tuple]):
        row_indices, columns, values = [], [], []
        for row_index, (_, smiles, _, _, _) in enumerate(rows):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise RuntimeError(f"Invalid prepared SMILES: {smiles}")
            bits = list(self.ligand.GetFingerprint(molecule).GetOnBits())
            scale = 1.0 / math.sqrt(max(len(bits), 1))
            row_indices.extend([row_index] * len(bits)); columns.extend(bits)
            values.extend([scale] * len(bits))
        ligand = sparse.csr_matrix(
            (values, (row_indices, columns)), shape=(len(rows), LIGAND_FEATURES), dtype=np.float32,
        )
        protein = self.protein.transform([row[2] for row in rows]).astype(np.float32)
        affinity_rows = np.arange(len(rows))
        affinity_columns = np.asarray([AFFINITY_TYPES.index(row[3]) for row in rows])
        affinity = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (affinity_rows, affinity_columns)),
            shape=(len(rows), len(AFFINITY_TYPES)),
        )
        return sparse.hstack((ligand, protein, affinity), format="csr"), np.asarray(
            [row[4] for row in rows], dtype=np.float64,
        )


def rows_for(connection, regime: str, split: str, cap: int, seed: int) -> list[tuple]:
    return connection.execute(query_for(regime, seed), (split, cap)).fetchall()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "n": int(len(truth)), "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else None,
        "pearson": float(pearsonr(truth, prediction).statistic) if len(truth) > 1 else None,
        "spearman": float(spearmanr(truth, prediction).statistic) if len(truth) > 1 else None,
    }


def evaluate(model, encoder, rows: list[tuple], regime: str, split: str):
    matrix, truth = encoder.transform(rows)
    prediction = model.predict(matrix)
    output = [{"regime": regime, "split": split, "affinity_type": "all", **metrics(truth, prediction)}]
    types = np.asarray([row[3] for row in rows])
    for affinity_type in AFFINITY_TYPES:
        mask = types == affinity_type
        if mask.any():
            output.append({"regime": regime, "split": split, "affinity_type": affinity_type,
                           **metrics(truth[mask], prediction[mask])})
    predictions = [{
        "regime": regime, "split": split, "observation_id": row[0],
        "affinity_type": row[3], "truth_p_affinity": float(truth[index]),
        "prediction_p_affinity": float(prediction[index]),
    } for index, row in enumerate(rows)]
    return output, predictions


def run_regime(
    prepared: Path, split_database: Path, regime: str, output_dir: Path,
    *, train_cap: int, val_cap: int, test_cap: int, epochs: int, batch_size: int, seed: int,
):
    started = time.perf_counter(); encoder = FeatureEncoder()
    with sqlite3.connect(prepared) as connection:
        connection.execute("ATTACH DATABASE ? AS splitdb", (str(split_database.resolve()),))
        populations, populations_by_type = {}, {}
        for split in ("train", "val", "test"):
            base_query = query_for(regime, seed).split("ORDER BY", 1)[0]
            populations[split] = connection.execute(
                f"SELECT COUNT(*) FROM ({base_query})",
                (split,),
            ).fetchone()[0]
            populations_by_type[split] = dict(connection.execute(
                f"SELECT affinity_type, COUNT(*) FROM ({base_query}) GROUP BY affinity_type",
                (split,),
            ).fetchall())
        train_rows = rows_for(connection, regime, "train", train_cap, seed)
        val_rows = rows_for(connection, regime, "val", val_cap, seed + 1)
        test_rows = rows_for(connection, regime, "test", test_cap, seed + 2)
    y_train = np.asarray([row[4] for row in train_rows], dtype=float)
    y_mean, y_scale = float(y_train.mean()), float(y_train.std())
    model = SGDRegressor(
        loss="huber", epsilon=1.35, penalty="l2", alpha=1e-5,
        learning_rate="adaptive", eta0=0.01, random_state=seed,
        average=True,
    )
    rng = np.random.default_rng(seed)
    training_started = time.perf_counter()
    for epoch in range(epochs):
        order = rng.permutation(len(train_rows))
        for start in range(0, len(order), batch_size):
            batch = [train_rows[index] for index in order[start:start + batch_size]]
            matrix, truth = encoder.transform(batch)
            model.partial_fit(matrix, (truth - y_mean) / max(y_scale, 1e-12))
    training_seconds = time.perf_counter() - training_started

    class RescaledModel:
        def predict(self, matrix):
            return model.predict(matrix) * y_scale + y_mean

    wrapped = RescaledModel()
    evaluation_started = time.perf_counter()
    val_metrics, val_predictions = evaluate(wrapped, encoder, val_rows, regime, "val")
    test_metrics, test_predictions = evaluate(wrapped, encoder, test_rows, regime, "test")
    evaluation_seconds = time.perf_counter() - evaluation_started
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "y_mean": y_mean, "y_scale": y_scale}, output_dir / f"{regime}_model.joblib")
    return {
        "regime": regime, "population_exact_affinity_rows": populations,
        "population_exact_affinity_rows_by_type": populations_by_type,
        "sample_rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "sample_rows_by_type": {
            split: {kind: sum(row[3] == kind for row in rows) for kind in AFFINITY_TYPES}
            for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows))
        },
        "epochs": epochs, "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds, "total_seconds": time.perf_counter()-started,
        "metrics": val_metrics + test_metrics,
    }, val_predictions + test_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--c1e", type=Path, required=True)
    parser.add_argument("--c1f", type=Path, required=True)
    parser.add_argument("--c2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=100000)
    parser.add_argument("--val-cap", type=int, default=25000)
    parser.add_argument("--test-cap", type=int, default=25000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); started = time.perf_counter()
    results, predictions = [], []
    for regime, database in (("C1e", args.c1e), ("C1f", args.c1f), ("C2", args.c2)):
        print(f"Predictive baseline: {regime}", flush=True)
        result, rows = run_regime(
            args.prepared, database, regime, args.output_dir,
            train_cap=args.train_cap, val_cap=args.val_cap, test_cap=args.test_cap,
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        )
        results.append(result); predictions.extend(rows)
    manifest = {
        "format_version": 1, "model": "SGDRegressor_huber_interaction_features",
        "features": {"ligand": "chiral Morgan radius=2 2048-bit",
                     "protein": "normalized hashed 3-mer frequencies 4096 features",
                     "affinity_type": "one-hot"},
        "label_policy": "exact equality affinity observations only",
        "sampling_policy": "deterministic pseudo-random ordering of source rows with affinity-type salt",
        "seed": args.seed, "caps": {"train": args.train_cap, "val": args.val_cap, "test": args.test_cap},
        "epochs": args.epochs, "batch_size": args.batch_size,
        "runtime_seconds": time.perf_counter()-started, "peak_memory_mb": peak_rss_mb(),
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": None if os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY") is None else os.environ["DATASAIL_GIT_WORKTREE_DIRTY"].lower()=="true",
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"), "python_version": platform.python_version(),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictive_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    metric_rows = [row for result in results for row in result["metrics"]]
    for filename, rows in (("predictive_metrics.tsv", metric_rows), ("predictions.tsv", predictions)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
