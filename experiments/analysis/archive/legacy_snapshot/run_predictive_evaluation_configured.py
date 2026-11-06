"""Configured predictive evaluation for the extended and original molecular datasets."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, mean_absolute_error, mean_squared_error,
                             matthews_corrcoef, r2_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
SPLIT_TYPES = [
    "random",
    "datasail_scaled_random",
    "datasail_scaled_stratified",
    "datasail_full",
]

import sys
sys.path.insert(0, str(ROOT / "experiments/pipelines"))
from run_scaled_configured import CONFIG  # noqa: E402
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def fast_fingerprints(smiles, chunk_size=2000, n_jobs=1):
    fp_cfg = CONFIG["defaults"]["fingerprint"]
    generator = GetMorganGenerator(radius=fp_cfg["radius"], fpSize=fp_cfg["bits"])
    result = []
    for start in range(0, len(smiles), chunk_size):
        mols = [Chem.MolFromSmiles(str(s)) for s in smiles[start:start+chunk_size]]
        fps = generator.GetFingerprints(mols, numThreads=max(1, n_jobs))
        result.append(np.vstack([np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8)-ord("0") for fp in fps]))
    return np.vstack(result)
DATASET_SCHEMA = {
    name: {"smiles": spec["smiles_column"], "id": "mol_id", "task": "regression" if spec["task_type"] == "regression" else "classification",
           "targets": spec.get("target_columns"), "target_prefix": spec.get("target_prefix")}
    for name, spec in CONFIG["datasets"].items()
}


def evaluate_qm9(split_type: str, train: pd.DataFrame, test: pd.DataFrame,
                 n_jobs: int, seed: int) -> list[dict]:
    id_and_smiles = {"mol_id", "smiles", "_datasail_id"}
    targets = [c for c in train.columns if c not in id_and_smiles]
    train_x = fast_fingerprints(train["smiles"].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    test_x = fast_fingerprints(test["smiles"].fillna("").astype(str).tolist(), n_jobs=n_jobs)

    train_y = train[targets].apply(pd.to_numeric, errors="coerce")
    test_y = test[targets].apply(pd.to_numeric, errors="coerce")
    usable = [c for c in targets if train_y[c].notna().any() and test_y[c].notna().any()]
    if not usable:
        raise ValueError(f"No numeric QM9 target columns found for {split_type}")

    # Ridge is a fast, reproducible fingerprint baseline suitable for all 19
    # continuous QM9 properties; fit one multi-output model per split.
    train_rows = train_y[usable].notna().all(axis=1).to_numpy()
    test_rows = test_y[usable].notna().all(axis=1).to_numpy()
    model = Ridge(alpha=1.0, solver="lsqr")
    model.fit(train_x[train_rows], train_y.loc[train_rows, usable].to_numpy())
    pred = model.predict(test_x[test_rows])
    truth = test_y.loc[test_rows, usable].to_numpy()
    rows = []
    for j, target in enumerate(usable):
        y = truth[:, j]
        p = pred[:, j]
        rows.append({
            "dataset": "qm9", "split_type": split_type, "task": target,
            "model": "Ridge(alpha=1.0, solver=lsqr)",
            "n_train": int(train_rows.sum()), "n_test": int(test_rows.sum()),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
            "status": "ok",
        })
    return rows


def evaluate_pcba(split_type: str, train: pd.DataFrame, test: pd.DataFrame,
                  n_jobs: int, seed: int, dataset: str = "pcba") -> list[dict]:
    spec = DATASET_SCHEMA[dataset]
    prefix = spec.get("target_prefix")
    configured = spec.get("targets") or []
    assays = [c for c in train.columns if (c.startswith(prefix) if prefix else c in configured)]
    if not assays:
        raise ValueError(f"No configured classification targets found for {dataset}/{split_type}")
    train_x = fast_fingerprints(train["smiles"].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    test_x = fast_fingerprints(test["smiles"].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    rows = []
    for index, assay in enumerate(assays, start=1):
        y_train_all = pd.to_numeric(train[assay], errors="coerce")
        y_test_all = pd.to_numeric(test[assay], errors="coerce")
        train_mask = y_train_all.notna().to_numpy()
        test_mask = y_test_all.notna().to_numpy()
        y_train = y_train_all.loc[train_mask].astype(int).to_numpy()
        y_test = y_test_all.loc[test_mask].astype(int).to_numpy()
        record = {
            "dataset": dataset, "split_type": split_type, "task": assay,
            "model": "SGDClassifier(log_loss, balanced, early_stopping)",
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "train_positive_fraction": float(y_train.mean()) if len(y_train) else np.nan,
            "test_positive_fraction": float(y_test.mean()) if len(y_test) else np.nan,
        }
        if len(y_train) == 0 or len(np.unique(y_train)) < 2:
            record.update({"status": "skipped_train_has_one_class", "roc_auc": np.nan,
                           "pr_auc": np.nan, "accuracy": np.nan, "f1": np.nan, "mcc": np.nan})
            rows.append(record)
            continue
        if len(y_test) == 0:
            record.update({"status": "skipped_no_observed_test_labels", "roc_auc": np.nan,
                           "pr_auc": np.nan, "accuracy": np.nan, "f1": np.nan, "mcc": np.nan})
            rows.append(record)
            continue
        model = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-5, class_weight="balanced",
            max_iter=50, tol=1e-3,
            early_stopping=bool(np.min(np.bincount(y_train)) >= 2), validation_fraction=0.1,
            n_iter_no_change=5, random_state=seed, n_jobs=n_jobs,
        )
        model.fit(train_x[train_mask], y_train)
        pred = model.predict(test_x[test_mask])
        prob = model.predict_proba(test_x[test_mask])[:, 1]
        both_test_classes = len(np.unique(y_test)) == 2
        record.update({
            "status": "ok" if both_test_classes else "test_has_one_class_auc_unavailable",
            "roc_auc": float(roc_auc_score(y_test, prob)) if both_test_classes else np.nan,
            "pr_auc": float(average_precision_score(y_test, prob)) if both_test_classes else np.nan,
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, pred)),
        })
        rows.append(record)
        if index % 16 == 0:
            print(f"{split_type}: evaluated {index}/{len(assays)} PCBA assays", flush=True)
    return rows



def evaluate_regression(dataset: str, split_type: str, train: pd.DataFrame, test: pd.DataFrame,
                        n_jobs: int) -> list[dict]:
    schema = DATASET_SCHEMA[dataset]
    targets = schema["targets"] or [c for c in train.columns if c not in {"mol_id", "smiles", "_datasail_id"}]
    targets = [c for c in targets if c in train.columns and c in test.columns]
    train_x = fast_fingerprints(train["smiles"].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    test_x = fast_fingerprints(test["smiles"].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    rows = []
    for target in targets:
        train_y = pd.to_numeric(train[target], errors="coerce")
        test_y = pd.to_numeric(test[target], errors="coerce")
        train_mask, test_mask = train_y.notna().to_numpy(), test_y.notna().to_numpy()
        if not train_mask.any() or not test_mask.any():
            rows.append({"dataset": dataset, "split_type": split_type, "task": target, "status": "skipped_no_labels"})
            continue
        model = Ridge(alpha=1.0, solver="lsqr")
        model.fit(train_x[train_mask], train_y[train_mask].to_numpy())
        pred = model.predict(test_x[test_mask]); truth = test_y[test_mask].to_numpy()
        rows.append({"dataset": dataset, "split_type": split_type, "task": target,
            "model": "Ridge(alpha=1.0, solver=lsqr)", "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "mae": float(mean_absolute_error(truth, pred)),
            "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
            "r2": float(r2_score(truth, pred)) if len(truth)>1 else np.nan, "status": "ok"})
    return rows

def evaluate_muv(split_type: str, train: pd.DataFrame, test: pd.DataFrame,
                 n_jobs: int, seed: int) -> list[dict]:
    train_x = fast_fingerprints(train["smiles"].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    test_x = fast_fingerprints(test["smiles"].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    spec = DATASET_SCHEMA["muv"]
    tasks = [c for c in train.columns if (c.startswith(spec["target_prefix"]) if spec.get("target_prefix") else c in (spec.get("targets") or []))]
    rows = []
    for task in tasks:
        yt, yv = pd.to_numeric(train[task], errors="coerce"), pd.to_numeric(test[task], errors="coerce")
        tm, vm = yt.notna().to_numpy(), yv.notna().to_numpy()
        y_train, y_test = yt[tm].astype(int).to_numpy(), yv[vm].astype(int).to_numpy()
        base = {"dataset":"muv", "split_type":split_type, "task":task, "model":"SGDClassifier(log_loss, balanced)",
                "n_train":int(tm.sum()), "n_test":int(vm.sum())}
        if len(np.unique(y_train))<2 or len(y_test)==0:
            rows.append({**base,"status":"skipped_insufficient_classes"}); continue
        model=SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, class_weight="balanced",
            max_iter=50, tol=1e-3, early_stopping=bool(np.min(np.bincount(y_train))>=2),
            validation_fraction=0.1, n_iter_no_change=5, random_state=seed, n_jobs=n_jobs)
        model.fit(train_x[tm],y_train); pred=model.predict(test_x[vm]); prob=model.predict_proba(test_x[vm])[:,1]
        two=len(np.unique(y_test))==2
        rows.append({**base,"status":"ok" if two else "test_has_one_class_auc_unavailable",
            "roc_auc":float(roc_auc_score(y_test,prob)) if two else np.nan,
            "pr_auc":float(average_precision_score(y_test,prob)),"accuracy":float(accuracy_score(y_test,pred)),
            "f1":float(f1_score(y_test,pred,zero_division=0)),"mcc":float(matthews_corrcoef(y_test,pred))})
    return rows

def split_directory(dataset: str, split_type: str) -> Path:
    """Resolve configured scaled splits and shared legacy baseline/full splits."""
    candidates = []
    if split_type in {"datasail_scaled_random", "datasail_scaled_stratified"}:
        candidates.append(ROOT / "results/configured/splits" / dataset / split_type)
    candidates.append(ROOT / "results/experiments/splits" / dataset / split_type)
    for path in candidates:
        if (path / "train.csv").exists() and (path / "test.csv").exists():
            return path
    return candidates[0]


def run(dataset: str, split_types: list[str], n_jobs: int, seed: int) -> pd.DataFrame:
    records = []
    for split_type in split_types:
        split_dir = split_directory(dataset, split_type)
        train_path, test_path = split_dir / "train.csv", split_dir / "test.csv"
        if not train_path.exists() or not test_path.exists():
            print(f"Skipping {dataset}/{split_type}: train/test split files missing at {split_dir}", flush=True)
            continue
        start = time.time()
        train, test = pd.read_csv(train_path), pd.read_csv(test_path)
        # Legacy original-dataset splits sometimes retain the source SMILES
        # column name (for example BACE's `mol`) instead of standardized `smiles`.
        smiles_col = CONFIG["datasets"][dataset]["smiles_column"]
        for frame in (train, test):
            if "smiles" not in frame and smiles_col in frame:
                frame["smiles"] = frame[smiles_col]
        print(f"Evaluating {dataset}/{split_type}: train={len(train)}, test={len(test)}", flush=True)
        if DATASET_SCHEMA[dataset]["task"] == "regression":
            split_rows = evaluate_qm9(split_type, train, test, n_jobs, seed) if dataset == "qm9" else evaluate_regression(dataset, split_type, train, test, n_jobs)
        elif dataset == "muv":
            split_rows = evaluate_muv(split_type, train, test, n_jobs, seed)
        else:
            split_rows = evaluate_pcba(split_type, train, test, n_jobs, seed, dataset=dataset)
        for row in split_rows:
            row["elapsed_seconds"] = time.time() - start
        records.extend(split_rows)

    if not records:
        raise FileNotFoundError(f"No usable split files found for {dataset}")
    results = pd.DataFrame(records)
    outdir = ROOT / "results/analysis/configured"
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{dataset}_predictive_metrics.csv"
    results.to_csv(output, index=False)
    metadata = {
        "dataset": dataset, "split_types": split_types, "seed": seed,
        "fingerprint": CONFIG["defaults"]["fingerprint"],
        "models": {"qm9": "multi-output Ridge(alpha=1.0, solver=lsqr)",
                   "pcba/muv": "per-assay SGDClassifier(log_loss, class_weight=balanced, early_stopping=True)",
                   "molecular regression": "per-target Ridge(alpha=1.0, solver=lsqr)"},
        "output": str(output.relative_to(ROOT)), "n_result_rows": len(results),
    }
    (outdir / f"{dataset}_predictive_metrics_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(results.groupby(["split_type", "status"], dropna=False).size().to_string(), flush=True)
    print(f"Saved predictive metrics to {output}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=list(DATASET_SCHEMA))
    parser.add_argument("--split-types", nargs="+", default=SPLIT_TYPES)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.dataset, args.split_types, args.n_jobs, args.seed)


if __name__ == "__main__":
    main()
