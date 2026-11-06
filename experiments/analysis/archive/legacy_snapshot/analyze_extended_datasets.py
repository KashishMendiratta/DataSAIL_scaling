"""Summarize split sizes, label coverage, runtime and leakage for extended molecular datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "qm9": ("data/raw/moleculenet/qm9.csv", None),
    "pcba": ("data/raw/moleculenet/pcba.csv.gz", "gzip"),
    "muv": ("data/raw/moleculenet/muv.csv.gz", "gzip"),
    "esol": ("data/raw/moleculenet/delaney-processed.csv", None),
    "freesolv": ("data/raw/moleculenet/SAMPL.csv", None),
    "lipophilicity": ("data/raw/moleculenet/Lipophilicity.csv", None),
}
sys.path.insert(0, str(ROOT / "experiments/pipelines"))
from run_extended_datasets import fast_fingerprints, load_valid_dataset, DATASET_SCHEMA


DEFAULT_SPLITS = ["random", "datasail_scaled_naive", "datasail_scaled_naive_confidence",
                  "datasail_scaled_confidence", "datasail_scaled_random", "datasail_full"]


def analyze(dataset: str, split_types: list[str] | None = None,
            leakage_query_size: int = 500, n_jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path, compression = CONFIG[dataset]
    raw = pd.read_csv(ROOT / raw_path, compression=compression)
    split_types = split_types or DEFAULT_SPLITS
    assay_prefix = "PCBA-" if dataset == "pcba" else "MUV-" if dataset == "muv" else None
    assay_cols = [c for c in raw.columns if assay_prefix and c.startswith(assay_prefix)]
    configured = DATASET_SCHEMA[dataset]["targets"]
    task_cols = configured or assay_cols or [c for c in raw.columns if c not in {"mol_id", "smiles", DATASET_SCHEMA[dataset]["smiles"], DATASET_SCHEMA[dataset]["id"]}]
    task_cols = [c for c in task_cols if c in raw.columns]

    records = []
    task_records = []
    for split_type in split_types:
        split_dir = ROOT / "results/experiments/splits" / dataset / split_type
        if not all((split_dir / f"{s}.csv").exists() for s in ("train", "val", "test")):
            print(f"Skipping {dataset}/{split_type}: split files not found")
            continue
        splits = {s: pd.read_csv(split_dir / f"{s}.csv") for s in ("train", "val", "test")}
        for split, frame in splits.items():
            record = {"dataset": dataset, "split_type": split_type, "split": split, "n_molecules": len(frame)}
            if DATASET_SCHEMA[dataset]["task"] == "regression":
                for col in task_cols:
                    vals = pd.to_numeric(frame[col], errors="coerce").dropna()
                    record[f"{col}_n"] = len(vals)
                    record[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
                    record[f"{col}_std"] = float(vals.std()) if len(vals) else np.nan
                record["target_kind"] = f"regression; {len(task_cols)} target column(s) summarized"
            else:
                observed, positives = 0, 0
                prevalence = []
                for col in assay_cols:
                    vals = pd.to_numeric(frame[col], errors="coerce").dropna()
                    pos = int((vals == 1).sum())
                    observed += len(vals)
                    positives += pos
                    prevalence.append(pos / len(vals) if len(vals) else np.nan)
                    record[f"{col}_observed"] = len(vals)
                    record[f"{col}_positive"] = pos
                    record[f"{col}_prevalence"] = prevalence[-1]
                record["n_assays"] = len(assay_cols)
                record["observed_labels_total"] = observed
                record["positive_labels_total"] = positives
                record["mean_assay_prevalence"] = float(np.nanmean(prevalence))
                record["target_kind"] = f"{len(assay_cols)} partially labeled binary assays; NaN values excluded"
            records.append(record)
            for col in task_cols:
                vals = pd.to_numeric(frame[col], errors="coerce").dropna()
                task_records.append({"dataset": dataset, "split_type": split_type, "split": split,
                                     "task": col, "n_observed": len(vals),
                                     "positive_count": int((vals == 1).sum()) if DATASET_SCHEMA[dataset]["task"] == "classification" else np.nan,
                                     "mean_or_prevalence": float((vals == 1).mean()) if DATASET_SCHEMA[dataset]["task"] == "classification" and len(vals) else (float(vals.mean()) if len(vals) else np.nan),
                                     "std": float(vals.std()) if len(vals) else np.nan})

    summary = pd.DataFrame(records)
    tasks = pd.DataFrame(task_records)
    if summary.empty:
        raise FileNotFoundError(f"No split directories found for {dataset}")
    outdir = ROOT / "results/analysis/extended_datasets"
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / f"{dataset}_split_summary.csv", index=False)
    tasks.to_csv(outdir / f"{dataset}_task_summary.csv", index=False)
    leakage = analyze_nearest_neighbor_leakage(dataset, summary["split_type"].drop_duplicates().tolist(),
                                               query_sample_size=leakage_query_size, n_jobs=n_jobs)
    leakage.to_csv(outdir / f"{dataset}_nn_leakage.csv", index=False)
    runtime = summarize_runtime(dataset, summary["split_type"].drop_duplicates().tolist())
    runtime.to_csv(outdir / f"{dataset}_runtime_comparison.csv", index=False)
    print(f"\n{dataset.upper()} — split summaries")
    print(summary[["split_type", "split", "n_molecules"] + (["mean_assay_prevalence", "observed_labels_total"] if dataset in {"pcba", "muv"} else [])].to_string(index=False))
    if dataset in {"pcba", "muv"}:
        print("Assay prevalence range by split:")
        print(tasks.groupby("split")["mean_or_prevalence"].agg(["min", "mean", "max"]).to_string())
    print("\nRuntime comparison (seconds):")
    print(runtime.to_string(index=False))
    return summary, tasks


def summarize_runtime(dataset: str, split_types: list[str]) -> pd.DataFrame:
    """Compare recorded end-to-end and DataSAIL/extension runtimes."""
    rows = []
    for split_type in split_types:
        path = ROOT / "results/experiments/splits" / dataset / split_type / "run_metadata.json"
        if not path.exists():
            continue
        metadata = pd.read_json(path, typ="series")
        rows.append({
            "dataset": dataset,
            "split_type": split_type,
            "n_valid_molecules": metadata.get("n_valid_smiles"),
            "sample_ratio": metadata.get("sample_ratio"),
            "n_datasail_anchor": metadata.get("n_datasail_anchor"),
            "end_to_end_seconds": metadata.get("total_seconds", metadata.get("runtime_seconds")),
            "datasail_seconds": metadata.get("datasail_seconds"),
            "assignment_seconds": metadata.get("assignment_search_and_ablation_seconds"),
        })
    return pd.DataFrame(rows)


def analyze_nearest_neighbor_leakage(dataset: str, split_types: list[str],
                                     query_sample_size: int = 500, threshold: float = 0.7,
                                     n_jobs: int = 1) -> pd.DataFrame:
    """Estimate test-to-train nearest-neighbour leakage on a reproducible test sample."""
    rows = []
    for split_type in split_types:
        split_dir = ROOT / "results/experiments/splits" / dataset / split_type
        train_file, test_file = split_dir / "train.csv", split_dir / "test.csv"
        if not train_file.exists() or not test_file.exists():
            continue
        train, test = pd.read_csv(train_file), pd.read_csv(test_file)
        id_col = "mol_id" if "mol_id" in train.columns else "smiles"
        # Bound this expensive diagnostic on large datasets: compare a fixed
        # sample of test molecules against at most 10k sampled train molecules.
        train = train.sample(n=min(10_000, len(train)), random_state=42)
        test = test.sample(n=min(query_sample_size, len(test)), random_state=42)
        train_smiles = train["smiles"].fillna("").astype(str).tolist()
        test_smiles = test["smiles"].fillna("").astype(str).tolist()
        train_fps = fast_fingerprints(train_smiles, n_jobs=n_jobs)
        test_fps = fast_fingerprints(test_smiles, n_jobs=n_jobs)
        nn = NearestNeighbors(n_neighbors=1, metric="jaccard", n_jobs=n_jobs).fit(train_fps)
        distances, _ = nn.kneighbors(test_fps)
        max_similarity = 1.0 - distances.ravel()
        rows.append({"dataset": dataset, "split_type": split_type,
                     "query_sample_size": len(test), "train_size": len(train),
                     "original_train_size": len(pd.read_csv(train_file)),
                     "comparison_scope": "sampled test vs 10k max sampled training molecules",
                     "mean_max_similarity": float(max_similarity.mean()),
                     "std_max_similarity": float(max_similarity.std()),
                     "leakage_fraction": float(np.mean(max_similarity > threshold)),
                     "similarity_threshold": threshold,
                     "query_sample_seed": 42})
        print(f"NN leakage {dataset}/{split_type}: {rows[-1]['leakage_fraction']:.3f} > {threshold:.1f}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CONFIG.keys())
    parser.add_argument("--split-types", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--leakage-query-size", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    analyze(args.dataset, args.split_types, args.leakage_query_size, args.n_jobs)


if __name__ == "__main__":
    main()
