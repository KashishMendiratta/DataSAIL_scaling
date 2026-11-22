"""Evaluate Chemprop's directed message-passing neural network on fixed k=5 splits.

The script uses the repository's existing train/validation/test files. It does
not create or modify splits. Validation is used by Chemprop for checkpoint
selection; final metrics are computed from predictions on the held-out test set.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG = json.loads((ROOT / "experiments/config/datasets.json").read_text())
DATASETS = DATASET_CONFIG["datasets"]
CHEMPROP = str(Path(sys.executable).with_name("chemprop"))

SEEDS = (42, 101, 202, 303, 404)
DEFAULT_DATASETS = ("bace", "hiv", "tox21", "muv", "esol", "qm9")
DEFAULT_METHODS = ("random", "datasail_scaled_random", "datasail_scaled_stratified", "datasail_full")
OUT = ROOT / "results/analysis/unified/dmpnn_k5"


def target_columns(dataset: str, frames: list[pd.DataFrame]) -> list[str]:
    spec = DATASETS[dataset]
    configured = spec.get("target_columns")
    if configured:
        return [c for c in configured if all(c in f.columns for f in frames)]
    prefix = spec.get("target_prefix")
    if prefix:
        return [c for c in frames[0].columns if c.startswith(prefix) and all(c in f.columns for f in frames)]
    excluded = {"mol_id", "smiles", "_datasail_id", spec.get("smiles_column"), spec.get("id_column")}
    return [c for c in frames[0].columns if c not in excluded and all(c in f.columns for f in frames)]


def resolve_split(dataset: str, method: str, seed: int) -> Path:
    root = ROOT / "results/experiments/unified/splits" / dataset / method
    if method == "datasail_full":
        return root
    seeded = root / f"seed_{seed}"
    return seeded if (seeded / "train.csv").exists() else root


def metric_rows(dataset: str, method: str, seed: int | str, task_type: str,
                targets: list[str], test: pd.DataFrame, pred: pd.DataFrame,
                elapsed: float) -> list[dict]:
    prediction_cols = [c for c in pred.columns if c.startswith("pred_")]
    prediction_cols = sorted(prediction_cols, key=lambda c: int(c.rsplit("_", 1)[1]))
    if len(prediction_cols) != len(targets):
        prediction_cols = [task for task in targets if task in pred.columns]
    if len(prediction_cols) != len(targets):
        raise ValueError(f"Expected {len(targets)} prediction columns; found {prediction_cols}")
    if len(test) != len(pred):
        raise ValueError(f"Test/prediction row mismatch: {len(test)} vs {len(pred)}")

    rows: list[dict] = []
    for index, task in enumerate(targets):
        y = pd.to_numeric(test[task], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(pred[prediction_cols[index]], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(p)
        y, p = y[mask], p[mask]
        row = {"dataset": dataset, "method": method, "split_seed": seed,
               "task": task, "model": "Chemprop D-MPNN (2D)", "n_test": len(y),
               "elapsed_seconds": elapsed, "status": "ok"}
        if not len(y):
            row.update(status="no_observed_test_labels")
        elif task_type == "regression":
            row.update(mae=float(mean_absolute_error(y, p)),
                       rmse=float(math.sqrt(mean_squared_error(y, p))),
                       r2=float(r2_score(y, p)) if len(y) > 1 else np.nan)
        else:
            labels = y.astype(int)
            predicted = (p >= 0.5).astype(int)
            two_classes = len(np.unique(labels)) == 2
            row.update(roc_auc=float(roc_auc_score(labels, p)) if two_classes else np.nan,
                       pr_auc=float(average_precision_score(labels, p)) if two_classes else np.nan,
                       accuracy=float(accuracy_score(labels, predicted)),
                       f1=float(f1_score(labels, predicted, zero_division=0)),
                       mcc=float(matthews_corrcoef(labels, predicted)),
                       status="ok" if two_classes else "test_has_one_class_auc_unavailable")
        rows.append(row)
    return rows


def run_one(dataset: str, method: str, seed: int | str, epochs: int,
            patience: int, batch_size: int, output_dir: Path,
            dry_run: bool = False) -> list[dict] | None:
    split_dir = resolve_split(dataset, method, int(seed) if seed != "full" else 42)
    paths = [split_dir / f"{part}.csv" for part in ("train", "val", "test")]
    if not all(p.exists() for p in paths):
        print(f"SKIP {dataset}/{method}/{seed}: missing split files in {split_dir}", flush=True)
        return None
    frames = [pd.read_csv(path) for path in paths]
    smiles_col = DATASETS[dataset]["smiles_column"]
    for frame in frames:
        if "smiles" not in frame and smiles_col in frame:
            frame["smiles"] = frame[smiles_col]
    targets = target_columns(dataset, frames)
    if not targets:
        raise ValueError(f"No target columns found for {dataset} in {split_dir}")
    task_type = "classification" if DATASETS[dataset]["task_type"] in ("classification", "classification_multitask") else "regression"
    run_name = f"{dataset}__{method}__seed_{seed}"
    run_dir = output_dir / "runs" / run_name
    metric_path = run_dir / "metrics.csv"
    if metric_path.exists():
        print(f"DONE {run_name}: metrics already exist; skipping", flush=True)
        return pd.read_csv(metric_path).to_dict("records")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Chemprop's separate-file interface preserves the original split exactly.
    with tempfile.TemporaryDirectory(prefix="dmpnn_k5_") as temp_dir:
        prepared: list[Path] = []
        for part, frame in zip(("train", "val", "test"), frames):
            standardized = frame[["smiles", *targets]].copy()
            standardized["smiles"] = standardized["smiles"].fillna("").astype(str)
            for task in targets:
                standardized[task] = pd.to_numeric(standardized[task], errors="coerce")
            out_csv = Path(temp_dir) / f"{part}.csv"
            standardized.to_csv(out_csv, index=False)
            prepared.append(out_csv)

        command = [CHEMPROP, "train", "--data-path", *map(str, prepared),
                   "--smiles-columns", "smiles", "--target-columns", *targets,
                   "--task-type", task_type, "--output-dir", str(run_dir),
                   "--accelerator", "cpu", "--devices", "1", "--num-workers", "0",
                   "--batch-size", str(batch_size), "--epochs", str(epochs),
                   "--patience", str(patience), "--pytorch-seed", "42"]
        if task_type == "classification":
            command.extend(["--metrics", "roc", "prc"])
            # Chemprop's class-balance sampler is defined for single-task data.
            # On multitask assays it balances rows with any positive label, not
            # each task's positives, so retain the natural labeled rows there.
            if len(targets) == 1:
                command.append("--class-balance")
        else:
            command.extend(["--metrics", "rmse", "mae"])
        if dry_run:
            print(" ".join(command), flush=True)
            return None

        started = time.time()
        print(f"TRAIN {run_name} ({len(frames[0])} train / {len(frames[1])} val / {len(frames[2])} test, {len(targets)} targets)", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        checkpoints = sorted(run_dir.rglob("best.pt"))
        if not checkpoints:
            checkpoints = sorted(run_dir.rglob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No best checkpoint found below {run_dir}")
        pred_path = run_dir / "test_predictions.csv"
        subprocess.run([CHEMPROP, "predict", "--test-path", str(prepared[2]),
                        "--model-paths", str(checkpoints[0]), "--preds-path", str(pred_path),
                        "--smiles-columns", "smiles", "--accelerator", "cpu", "--devices", "1",
                        "--num-workers", "0"], check=True, cwd=ROOT)
        elapsed = time.time() - started
        test = pd.read_csv(prepared[2])
        pred = pd.read_csv(pred_path)
        rows = metric_rows(dataset, method, seed, task_type, targets, test, pred, elapsed)
        pd.DataFrame(rows).to_csv(metric_path, index=False)
        metadata = {
            "dataset": dataset,
            "method": method,
            "split_seed": seed,
            "split_directory": str(split_dir.relative_to(ROOT)),
            "split_sha256": {
                part: hashlib.sha256(path.read_bytes()).hexdigest()
                for part, path in zip(("train", "val", "test"), paths)
            },
            "targets": targets,
            "task_type": task_type,
            "elapsed_seconds": elapsed,
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "pytorch_seed": 42,
            "chemprop_version": package_version("chemprop"),
            "torch_version": package_version("torch"),
            "python_version": sys.version.split()[0],
            "analysis_git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
            "analysis_git_worktree_dirty": os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY"),
            "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
            "condor_job_id": os.environ.get("CONDOR_JOB_ID"),
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"SAVED {run_name}: {metric_path}", flush=True)
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DEFAULT_DATASETS))
    parser.add_argument("--methods", nargs="+", choices=list(DEFAULT_METHODS), default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "Chemprop 2D directed message-passing neural network",
        "datasets": args.datasets,
        "methods": args.methods,
        "split_seeds": args.seeds,
        "full_datasail_replicates": 1,
        "training": {"epochs": args.epochs, "patience": args.patience,
                     "batch_size": args.batch_size, "accelerator": "cpu",
                     "pytorch_seed": 42,
                     "class_balance": "single-task classification only",
                     "checkpoint_selection": "validation only"},
        "protocol": "Use existing split train/val/test CSVs; no split regeneration or k tuning. Test predictions are used only for final metrics.",
        "chemprop_version": None,
        "torch_version": None,
        "python_version": sys.version.split()[0],
        "completed_runs": [],
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous.get("datasets"), previous.get("methods"), previous.get("split_seeds")) == (args.datasets, args.methods, args.seeds):
            manifest = previous
    manifest["training"] = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "accelerator": "cpu",
        "pytorch_seed": 42,
        "class_balance": "single-task classification only",
        "checkpoint_selection": "validation only",
    }
    manifest["python_version"] = sys.version.split()[0]
    for package, key in (("chemprop", "chemprop_version"), ("torch", "torch_version")):
        try:
            manifest[key] = package_version(package)
        except PackageNotFoundError:
            manifest[key] = None
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    records: list[dict] = []
    failures: list[dict] = []
    for dataset in args.datasets:
        for method in args.methods:
            seeds: list[int | str] = ["full"] if method == "datasail_full" else args.seeds
            for seed in seeds:
                try:
                    rows = run_one(dataset, method, seed, args.epochs, args.patience,
                                   args.batch_size, output_dir, dry_run=args.dry_run)
                    if rows:
                        records.extend(rows)
                        completed_name = f"{dataset}/{method}/{seed}"
                        if completed_name not in manifest["completed_runs"]:
                            manifest["completed_runs"].append(completed_name)
                        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                        failures_path = output_dir / "failures.jsonl"
                        if failures_path.exists():
                            remaining = []
                            for line in failures_path.read_text().splitlines():
                                failure = json.loads(line)
                                if (failure.get("dataset"), failure.get("method"), str(failure.get("seed"))) != (dataset, method, str(seed)):
                                    remaining.append(line)
                            failures_path.write_text("\n".join(remaining) + ("\n" if remaining else ""))
                except Exception as exc:
                    print(f"FAILED {dataset}/{method}/{seed}: {exc}", flush=True)
                    failure = {"dataset": dataset, "method": method,
                               "seed": seed, "error": str(exc)}
                    failures.append(failure)
                    with (output_dir / "failures.jsonl").open("a") as f:
                        f.write(json.dumps(failure) + "\n")
    metric_files = sorted((output_dir / "runs").glob("*/metrics.csv"))
    if metric_files:
        combined = pd.concat([pd.read_csv(p) for p in metric_files], ignore_index=True)
        combined.to_csv(output_dir / "metrics.csv", index=False)
    if failures:
        raise SystemExit(f"{len(failures)} D-MPNN run(s) failed; inspect {output_dir / 'failures.jsonl'}")
    if not args.dry_run and len(manifest["completed_runs"]) == 0:
        raise SystemExit("No D-MPNN runs completed; inspect failures.jsonl")


if __name__ == "__main__":
    main()
