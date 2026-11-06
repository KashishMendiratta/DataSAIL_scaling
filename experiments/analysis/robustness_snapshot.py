"""Analyze leakage thresholds, duplicate/scaffold overlap, and performance by similarity.

Nearest-neighbor statistics use a deterministic sample of test queries against
a capped training reference set. Duplicate and scaffold overlap use all rows.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             matthews_corrcoef, r2_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pipeline_engine import CONFIG
from src.analysis_common import (ANALYSIS_DIR, METHODS, fingerprint_matrix,
                                 read_splits, smiles_column, target_columns)

BINS = [(0.0, 0.3, "[0.0,0.3)"), (0.3, 0.5, "[0.3,0.5)"),
        (0.5, 0.7, "[0.5,0.7)"), (0.7, 1.000001, "[0.7,1.0]")]
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)

def _canonical_and_scaffold(smiles):
    canonical, scaffolds = [], []
    for value in smiles:
        mol = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
        canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        scaffolds.append(scaffold or None)
    return canonical, scaffolds

def _overlap(train_values, test_values):
    train = {x for x in train_values if x}
    test = [x for x in test_values if x]
    shared = train.intersection(test)
    unique_test = set(test)
    return {"test_nonempty_n": len(test),
            "test_rows_overlapping": sum(x in train for x in test),
            "test_row_overlap_fraction": float(np.mean([x in train for x in test])) if test else np.nan,
            "test_unique_values": len(unique_test),
            "test_unique_overlapping": len(shared),
            "test_unique_overlap_fraction": len(shared) / len(unique_test) if unique_test else np.nan}

def _query_similarity(dataset, train, test, query_n, train_cap, seed, n_jobs):
    tc, qc = smiles_column(dataset, train), smiles_column(dataset, test)
    rng = np.random.default_rng(seed)
    ti = np.sort(rng.choice(len(train), min(train_cap, len(train)), replace=False))
    qi = np.sort(rng.choice(len(test), min(query_n, len(test)), replace=False))
    tr = fingerprint_matrix(train.iloc[ti][tc].fillna("").astype(str).tolist(), n_jobs)
    te = fingerprint_matrix(test.iloc[qi][qc].fillna("").astype(str).tolist(), n_jobs)
    nn = NearestNeighbors(n_neighbors=1, metric="jaccard", n_jobs=n_jobs).fit(tr)
    sims = 1.0 - nn.kneighbors(te, return_distance=True)[0].ravel()
    return qi, sims, len(ti)

def _metric_rows(dataset, method, run_id, train, test, qi, sims, n_jobs, seed):
    spec = CONFIG["datasets"][dataset]
    smiles = smiles_column(dataset, train)
    x_train = fingerprint_matrix(train[smiles].fillna("").astype(str).tolist(), n_jobs)
    x_query = fingerprint_matrix(test.iloc[qi][smiles].fillna("").astype(str).tolist(), n_jobs)
    classification = spec["task_type"] != "regression"
    out = []
    for target in target_columns(dataset, train):
        yt = pd.to_numeric(train[target], errors="coerce")
        yq = pd.to_numeric(test.iloc[qi][target], errors="coerce")
        tm, qm = yt.notna().to_numpy(), yq.notna().to_numpy()
        if classification:
            tm &= yt.isin([0, 1]).to_numpy()
            qm &= yq.isin([0, 1]).to_numpy()
        if not tm.any() or not qm.any():
            continue
        y_train = yt.to_numpy()[tm]
        y_query = yq.to_numpy()[qm].astype(int if classification else float)
        x_fit, x_eval, sim_eval = x_train[tm], x_query[qm], sims[qm]
        if classification:
            if len(np.unique(y_train)) < 2:
                continue
            minority_n = int(np.min(np.bincount(y_train.astype(int))))
            model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5,
                                  class_weight="balanced", max_iter=50, tol=1e-3,
                                  early_stopping=minority_n >= 2, validation_fraction=0.1,
                                  n_iter_no_change=5, random_state=seed, n_jobs=n_jobs)
            model.fit(x_fit, y_train.astype(int))
            pred, prob = model.predict(x_eval), model.predict_proba(x_eval)[:, 1]
        else:
            model = Ridge(alpha=1.0, solver="lsqr")
            model.fit(x_fit, y_train)
            pred, prob = model.predict(x_eval), None
        for lo, hi, label in BINS:
            mask = (sim_eval >= lo) & (sim_eval < hi)
            truth, estimate = y_query[mask], pred[mask]
            row = {"dataset": dataset, "method": method, "run_id": run_id, "seed": seed,
                   "task": target, "similarity_bin": label, "n_test": int(mask.sum()),
                   "n_positive": int(np.sum(truth == 1)) if classification else np.nan}
            if not len(truth):
                row["status"] = "empty_bin"
            elif classification:
                row.update({"status": "ok", "accuracy": float(accuracy_score(truth, estimate)),
                            "f1": float(f1_score(truth, estimate, zero_division=0)),
                            "mcc": float(matthews_corrcoef(truth, estimate)),
                            "roc_auc": float(roc_auc_score(truth, prob[mask])) if len(np.unique(truth)) == 2 else np.nan,
                            "pr_auc": float(average_precision_score(truth, prob[mask])) if np.sum(truth == 1) else np.nan})
            else:
                row.update({"status": "ok", "mae": float(mean_absolute_error(truth, estimate)),
                            "rmse": float(np.sqrt(mean_squared_error(truth, estimate))),
                            "r2": float(r2_score(truth, estimate)) if len(truth) > 1 else np.nan})
            out.append(row)
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=list(CONFIG["datasets"]), default=list(CONFIG["datasets"]))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--run-ids", nargs="+", required=True,
                   help="Use baseline for existing un-nested results, or seed_42 and similar replicate IDs.")
    p.add_argument("--query-n", type=int, default=500)
    p.add_argument("--train-cap", type=int, default=10000)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42, help="Deterministic sample seed for leakage queries.")
    a = p.parse_args()
    threshold_rows, overlap_rows, performance_rows = [], [], []
    for run_id_arg in a.run_ids:
        run_id = None if run_id_arg == "baseline" else run_id_arg
        for dataset in a.datasets:
            for method in a.methods:
                split = read_splits(dataset, method, run_id)
                if split is None:
                    print(f"Skipping missing split: {dataset}/{method}/{run_id_arg}", flush=True)
                    continue
                train, test = split["train"], split["test"]
                tc, qc = smiles_column(dataset, train), smiles_column(dataset, test)
                train_canon, train_scaff = _canonical_and_scaffold(train[tc])
                test_canon, test_scaff = _canonical_and_scaffold(test[qc])
                exact, scaffold = _overlap(train_canon, test_canon), _overlap(train_scaff, test_scaff)
                overlap_rows.append({"dataset": dataset, "method": method, "run_id": run_id_arg,
                                     "train_n": len(train), "test_n": len(test),
                                     "exact_duplicate_test_row_fraction": exact["test_row_overlap_fraction"],
                                     "exact_duplicate_test_rows": exact["test_rows_overlapping"],
                                     "exact_unique_overlap_fraction": exact["test_unique_overlap_fraction"],
                                     "shared_scaffold_test_row_fraction": scaffold["test_row_overlap_fraction"],
                                     "shared_scaffold_test_rows": scaffold["test_rows_overlapping"],
                                     "shared_unique_scaffold_fraction": scaffold["test_unique_overlap_fraction"],
                                     "test_acyclic_or_invalid_n": len(test) - scaffold["test_nonempty_n"]})
                qi, sims, train_n = _query_similarity(dataset, train, test, a.query_n, a.train_cap, a.seed, a.n_jobs)
                common = {"dataset": dataset, "method": method, "run_id": run_id_arg,
                          "query_n": len(sims), "train_reference_n": train_n,
                          "mean_max_similarity": float(np.mean(sims)),
                          "median_max_similarity": float(np.median(sims)),
                          "q90_max_similarity": float(np.quantile(sims, .9)),
                          "sample_seed": a.seed}
                for threshold in THRESHOLDS:
                    threshold_rows.append({**common, "threshold": threshold,
                                           "fraction_above_threshold": float(np.mean(sims > threshold))})
                metadata_path = ROOT / "results/experiments/unified/splits" / dataset / method
                if run_id:
                    metadata_path = metadata_path / run_id
                try:
                    seed = int(json.loads((metadata_path / "run_metadata.json").read_text()).get("seed", 42))
                except (OSError, ValueError, json.JSONDecodeError):
                    seed = 42
                performance_rows.extend(_metric_rows(dataset, method, run_id_arg, train, test,
                                                      qi, sims, a.n_jobs, seed))
    outdir = ANALYSIS_DIR / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)
    th, ov, perf = pd.DataFrame(threshold_rows), pd.DataFrame(overlap_rows), pd.DataFrame(performance_rows)
    th.to_csv(outdir / "leakage_thresholds_raw.csv", index=False)
    ov.to_csv(outdir / "structure_overlap_raw.csv", index=False)
    perf.to_csv(outdir / "performance_by_similarity_raw.csv", index=False)
    if not th.empty:
        th.groupby(["dataset", "method", "threshold"], dropna=False)["fraction_above_threshold"].agg(
            mean="mean", std="std", n="count").reset_index().to_csv(
                outdir / "leakage_thresholds_summary.csv", index=False)
    if not ov.empty:
        numeric = [c for c in ov.select_dtypes(include=[np.number]).columns if c not in {"train_n", "test_n"}]
        summary = ov.groupby(["dataset", "method"], dropna=False)[numeric].agg(["mean", "std", "count"])
        summary.columns = ["_".join(str(x) for x in col) for col in summary.columns]
        summary.reset_index().to_csv(outdir / "structure_overlap_summary.csv", index=False)
    if not perf.empty:
        metrics = [c for c in ("accuracy", "f1", "mcc", "roc_auc", "pr_auc", "mae", "rmse", "r2") if c in perf]
        summary = perf.groupby(["dataset", "method", "task", "similarity_bin"], dropna=False)[metrics].agg(
            ["mean", "std", "count"])
        summary.columns = ["_".join(str(x) for x in col) for col in summary.columns]
        summary.reset_index().to_csv(outdir / "performance_by_similarity_summary.csv", index=False)
    print(f"Saved robustness tables under {outdir}", flush=True)

if __name__ == "__main__":
    main()
