"""Analyze all configured datasets and available random/scaled/full splits.

This is the shared leakage/runtime and split-summary implementation for the
original and added datasets. Results from legacy split files are normalized to
the same output schema as configured scaled splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/pipelines"))
from run_scaled_configured import CONFIG, fingerprint_matrix, load_dataset  # noqa: E402

METHODS = ["random", "datasail_scaled_random", "datasail_scaled_stratified", "datasail_full"]


def split_directory(dataset: str, method: str) -> Path:
    candidates = []
    if method in {"datasail_scaled_random", "datasail_scaled_stratified"}:
        candidates.append(ROOT / "results/configured/splits" / dataset / method)
    legacy_names = {
        "datasail_full": ("datasail_full", "full_datasail", "datasail"),
        "random": ("random",),
        "datasail_scaled_random": ("datasail_scaled_random",),
        "datasail_scaled_stratified": ("datasail_scaled_stratified",),
    }
    candidates.extend(ROOT / "results/experiments/splits" / dataset / name
                      for name in legacy_names.get(method, (method,)))
    for path in candidates:
        if all((path / f"{split}.csv").exists() for split in ("train", "val", "test")):
            return path
    return candidates[0]


def runtime_from_metadata(dataset: str, method: str, split_dir: Path) -> tuple[float, float, str]:
    """Return end-to-end seconds, DataSAIL-only seconds, and metadata source."""
    metadata_path = split_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    total = metadata.get("total_seconds", metadata.get("runtime_seconds"))
    datasail = metadata.get("datasail_seconds")
    # Ignore corrupted or placeholder timing values instead of plotting them.
    try:
        if total is not None and (not np.isfinite(float(total)) or float(total) <= 0):
            total = None
        if datasail is not None and (not np.isfinite(float(datasail)) or float(datasail) <= 0):
            datasail = None
    except (TypeError, ValueError):
        total, datasail = None, None
    source = str(metadata_path.relative_to(ROOT)) if metadata else ""

    # Historical scripts stored these timings in per-dataset text files.
    runtime_file = ROOT / "results/experiments" / dataset / {
        "random": "runtime_random.txt",
        "datasail_full": "runtime_full.txt",
        "datasail_scaled_random": "runtime_scaled_random.txt",
        "datasail_scaled_stratified": "runtime_scaled_stratified.txt",
    }[method]
    if runtime_file.exists():
        try:
            for line in runtime_file.read_text().splitlines():
                key, separator, raw_value = line.partition(":")
                if not separator:
                    continue
                key = key.strip()
                try:
                    value = float(raw_value.strip())
                except ValueError:
                    continue
                if total is None and key in {"total_runtime_seconds", "runtime_seconds"}:
                    total = value
                if datasail is None and key in {"datasail_time", "datasail_seconds"}:
                    datasail = value
            source = str(runtime_file.relative_to(ROOT)) if source else str(runtime_file.relative_to(ROOT))
        except OSError:
            pass
    # Historical text files are also untrusted: only positive finite durations
    # are meaningful for runtime comparisons.
    try:
        if total is not None and (not np.isfinite(float(total)) or float(total) <= 0):
            total = None
        if datasail is not None and (not np.isfinite(float(datasail)) or float(datasail) <= 0):
            datasail = None
    except (TypeError, ValueError):
        total, datasail = None, None
    if method == "random":
        datasail = np.nan  # DataSAIL is not part of the random baseline.
    return (float(total) if total is not None else np.nan,
            float(datasail) if datasail is not None else np.nan, source)


def analyze(dataset: str, split_types: list[str] | None = None,
            leakage_query_size: int = 500, n_jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_types = split_types or METHODS
    cfg = CONFIG["datasets"][dataset]
    split_rows, task_rows, leakage_rows = [], [], []

    for method in split_types:
        folder = split_directory(dataset, method)
        paths = {split: folder / f"{split}.csv" for split in ("train", "val", "test")}
        total_seconds, datasail_seconds, runtime_source = runtime_from_metadata(dataset, method, folder)
        base = {"dataset": dataset, "method": method,
                "end_to_end_runtime_seconds": total_seconds,
                "runtime_seconds": total_seconds,
                "datasail_seconds": datasail_seconds,
                "runtime_status": "recorded" if np.isfinite(total_seconds) else "not_recorded",
                "datasail_runtime_status": ("not_applicable" if method == "random" else
                                            "recorded" if np.isfinite(datasail_seconds) else "not_recorded"),
                "runtime_source": runtime_source}
        if not all(path.exists() for path in paths.values()):
            leakage_rows.append({**base, "split_status": "missing_split_files",
                                 "mean_max_similarity": np.nan,
                                 "leakage_fraction_gt_0_7": np.nan,
                                 "query_n": 0, "train_n": 0})
            continue

        frames = {split: pd.read_csv(path) for split, path in paths.items()}
        cfg_targets = cfg.get("target_columns")
        for split, frame in frames.items():
            split_rows.append({"dataset": dataset, "method": method,
                               "split": split, "n_rows": len(frame)})
            if cfg_targets is None:
                prefix = cfg.get("target_prefix")
                targets = [column for column in frame.columns
                           if prefix and column.startswith(prefix)]
                if not prefix:
                    targets = [column for column in frame.select_dtypes(include=[np.number]).columns
                               if column not in {"mol_id", "_datasail_id"}]
            else:
                targets = cfg_targets
            for target in targets:
                if target not in frame:
                    continue
                values = pd.to_numeric(frame[target], errors="coerce").dropna()
                task_rows.append({
                    "dataset": dataset, "method": method, "split": split,
                    "task": target, "n_observed": len(values),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std()) if len(values) else np.nan,
                    "positive_fraction": float((values == 1).mean())
                    if cfg["task_type"].startswith("classification") and len(values) else np.nan,
                })

        train, test = frames["train"], frames["test"]
        smiles_col = cfg["smiles_column"]
        if smiles_col not in train or smiles_col not in test:
            smiles_col = "smiles"
        if not len(train) or not len(test) or smiles_col not in train or smiles_col not in test:
            leakage_rows.append({**base, "split_status": "missing_smiles_or_empty_split",
                                 "mean_max_similarity": np.nan,
                                 "leakage_fraction_gt_0_7": np.nan,
                                 "query_n": 0, "train_n": 0,
                                 "invalid_train_smiles": 0, "invalid_query_smiles": 0})
            continue

        rng = np.random.default_rng(42)
        train_idx = np.sort(rng.choice(len(train), min(10_000, len(train)), replace=False))
        test_idx = np.sort(rng.choice(len(test), min(leakage_query_size, len(test)), replace=False))
        train_sample = train.iloc[train_idx].copy()
        test_sample = test.iloc[test_idx].copy()
        train_sample["_datasail_smiles"] = train_sample[smiles_col].astype(str)
        test_sample["_datasail_smiles"] = test_sample[smiles_col].astype(str)
        train_valid = train_sample["_datasail_smiles"].fillna("").map(
            lambda value: Chem.MolFromSmiles(str(value)) is not None)
        test_valid = test_sample["_datasail_smiles"].fillna("").map(
            lambda value: Chem.MolFromSmiles(str(value)) is not None)
        invalid_train_smiles = int((~train_valid).sum())
        invalid_query_smiles = int((~test_valid).sum())
        train_sample = train_sample.loc[train_valid].reset_index(drop=True)
        test_sample = test_sample.loc[test_valid].reset_index(drop=True)
        if train_sample.empty or test_sample.empty:
            leakage_rows.append({**base, "split_status": "no_valid_smiles_in_sample",
                                 "mean_max_similarity": np.nan,
                                 "leakage_fraction_gt_0_7": np.nan,
                                 "query_n": len(test_sample), "train_n": len(train_sample),
                                 "invalid_train_smiles": invalid_train_smiles,
                                 "invalid_query_smiles": invalid_query_smiles})
            continue
        train_fp = fingerprint_matrix(train_sample, CONFIG, n_jobs)
        test_fp = fingerprint_matrix(test_sample, CONFIG, n_jobs)
        nearest = NearestNeighbors(n_neighbors=1, metric="jaccard", n_jobs=n_jobs).fit(train_fp)
        distances, _ = nearest.kneighbors(test_fp)
        max_similarity = 1.0 - distances.ravel()
        leakage_rows.append({**base, "split_status": "ok",
                             "query_n": len(test_sample), "train_n": len(train_sample),
                             "invalid_train_smiles": invalid_train_smiles,
                             "invalid_query_smiles": invalid_query_smiles,
                             "mean_max_similarity": float(max_similarity.mean()),
                             "leakage_fraction_gt_0_7": float(np.mean(max_similarity > 0.7)),
                             "similarity_threshold": 0.7, "query_seed": 42})
        print(f"Leakage {dataset}/{method}: {leakage_rows[-1]['leakage_fraction_gt_0_7']:.3f}", flush=True)

    output_dir = ROOT / "results/analysis/configured"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_df, task_df, leakage_df = pd.DataFrame(split_rows), pd.DataFrame(task_rows), pd.DataFrame(leakage_rows)
    split_df.to_csv(output_dir / f"{dataset}_split_summary.csv", index=False)
    task_df.to_csv(output_dir / f"{dataset}_task_summary.csv", index=False)
    leakage_df.to_csv(output_dir / f"{dataset}_runtime_leakage.csv", index=False)
    return split_df, leakage_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CONFIG["datasets"])
    parser.add_argument("--split-types", nargs="+", default=METHODS)
    parser.add_argument("--leakage-query-size", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    analyze(args.dataset, args.split_types, args.leakage_query_size, args.n_jobs)


if __name__ == "__main__":
    main()
