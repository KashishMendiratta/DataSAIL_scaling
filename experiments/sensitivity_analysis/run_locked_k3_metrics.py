"""Build locked-k final splits and evaluate leakage and predictive metrics."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, vstack as sparse_vstack
from threadpoolctl import threadpool_limits
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments/analysis"))

from knn_order_sensitivity import _assign_in_order, _load_anchor_labels  # noqa: E402
from src.pipeline_engine import CONFIG, fingerprint_matrix, load_dataset  # noqa: E402
from src.analysis_common import fingerprint_matrix as analysis_fingerprints, smiles_column  # noqa: E402
from leakage_analysis import sampled_mean  # noqa: E402
from ml_evaluation_extended import (  # noqa: E402
    evaluate_muv,
    evaluate_pcba,
    evaluate_qm9,
    evaluate_regression,
)

K = 3
STRATEGIES = ("random", "stratified")
METHOD_NAMES = {
    "random": "datasail_scaled_random_locked_k3",
    "stratified": "datasail_scaled_stratified_locked_k3",
}


def sparse_fingerprints(smiles, chunk_size=2000, n_jobs=1):
    """Return the same Morgan bits as the project evaluator in sparse CSR form."""
    cfg = CONFIG["defaults"]["fingerprint"]
    generator = GetMorganGenerator(radius=cfg["radius"], fpSize=cfg["bits"])
    pieces = []
    values = [str(value) for value in smiles]
    for start in range(0, len(values), chunk_size):
        mols = [Chem.MolFromSmiles(value) for value in values[start:start + chunk_size]]
        fps = generator.GetFingerprints(mols, numThreads=max(1, n_jobs))
        dense = np.vstack([
            np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")
            for fp in fps
        ])
        pieces.append(csr_matrix(dense))
    if not pieces:
        return csr_matrix((0, cfg["bits"]), dtype=np.uint8)
    return sparse_vstack(pieces, format="csr")


def _jaccard_neighbors_chunked(anchor_fps: np.ndarray, query_fps: np.ndarray,
                               k: int, n_jobs: int):
    """Compute exact binary Jaccard top-k in bounded chunks using bit intersections."""
    anchors = anchor_fps.astype(np.float32, copy=False)
    anchor_counts = anchors.sum(axis=1, dtype=np.float32)
    indices_out = np.empty((len(query_fps), k), dtype=np.int32)
    distances_out = np.empty((len(query_fps), k), dtype=np.float32)
    batch_size = max(128, min(2048, 2_000_000 // max(1, len(anchor_fps))))
    anchor_transposed = anchors.T.copy()
    with threadpool_limits(limits=max(1, n_jobs)):
        for start in range(0, len(query_fps), batch_size):
            stop = min(start + batch_size, len(query_fps))
            queries = query_fps[start:stop].astype(np.float32, copy=False)
            intersections = queries @ anchor_transposed
            unions = queries.sum(axis=1, dtype=np.float32)[:, None] + anchor_counts[None, :] - intersections
            similarities = np.divide(intersections, unions, out=np.ones_like(intersections), where=unions > 0)
            selected = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
            for row in range(stop - start):
                picks = selected[row]
                order = np.lexsort((picks, -similarities[row, picks]))
                picks = picks[order]
                indices_out[start + row] = picks
                distances_out[start + row] = 1.0 - similarities[row, picks]
    return distances_out, indices_out


def _generate_splits(dataset: str, strategy: str, frame: pd.DataFrame, fps: np.ndarray,
                     split_out: Path, n_jobs: int):
    manifest, label_map, split_path = _load_anchor_labels(dataset, strategy)
    ids = frame["_datasail_id"].astype(str).to_numpy()
    id_to_pos = {value: i for i, value in enumerate(ids)}
    anchor_ids = [str(value) for value in manifest["anchor_ids"]]
    anchor_pos = np.array([id_to_pos[value] for value in anchor_ids], dtype=int)
    anchor_labels = np.array([label_map[value] for value in anchor_ids], dtype=object)
    is_anchor = np.zeros(len(frame), dtype=bool)
    is_anchor[anchor_pos] = True
    rest_pos = np.flatnonzero(~is_anchor)  # canonical source-row processing order

    if len(rest_pos) * len(anchor_pos) > 50_000_000:
        distances, neighbors = _jaccard_neighbors_chunked(fps[anchor_pos], fps[rest_pos], K, n_jobs)
    else:
        nn = NearestNeighbors(n_neighbors=K, metric="jaccard", n_jobs=n_jobs).fit(fps[anchor_pos])
        distances, neighbors = nn.kneighbors(fps[rest_pos])
    rest_labels = _assign_in_order(
        neighbors, distances, anchor_labels, K, np.arange(len(rest_pos)),
        CONFIG["defaults"]["split_ratios"], CONFIG["defaults"]["assignment"]["balance_weight"],
    )[0]
    labels = np.empty(len(frame), dtype=object)
    labels[anchor_pos] = anchor_labels
    labels[rest_pos] = rest_labels
    output = frame.drop(columns=["_datasail_smiles", "_datasail_id"]).copy()
    output["split"] = labels

    split_dir = split_out / dataset / METHOD_NAMES[strategy]
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in CONFIG["defaults"]["split_ratios"]:
        output.loc[output.split == split].drop(columns="split").to_csv(split_dir / f"{split}.csv", index=False)
    metadata = {
        "dataset": dataset,
        "method": METHOD_NAMES[strategy],
        "anchor_strategy": strategy,
        "locked_k": K,
        "processing_order": "source row order",
        "seed": manifest["seed"],
        "n_valid_rows": len(frame),
        "n_anchor_rows": int(is_anchor.sum()),
        "split_counts": dict(Counter(labels)),
        "anchor_manifest_sha256": hashlib.sha256(
            (ROOT / "results/experiments/unified/splits" / dataset /
             f"datasail_scaled_{strategy}" / "anchor_manifest.json").read_bytes()
        ).hexdigest(),
        "datasail_split_file": str(split_path.relative_to(ROOT)),
        "note": "Anchor IDs and labels reused from saved seed-42 DataSAIL results; only non-anchor labels reassigned with locked k=3.",
    }
    (split_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return split_dir


def _leakage_metrics(dataset: str, method: str, frames: dict[str, pd.DataFrame], n_jobs: int):
    train, test = frames["train"], frames["test"]
    train_col, test_col = smiles_column(dataset, train), smiles_column(dataset, test)
    train_smiles = train[train_col].dropna().astype(str).tolist()
    test_smiles = test[test_col].dropna().astype(str).tolist()
    train_fp = analysis_fingerprints(train_smiles, n_jobs=n_jobs)
    test_fp = analysis_fingerprints(test_smiles, n_jobs=n_jobs)
    mean_similarity = sampled_mean(train_fp, test_fp, 100_000, 42)

    rng = np.random.default_rng(42)
    train_rows = np.sort(rng.choice(len(train), min(10_000, len(train)), replace=False))
    test_rows = np.sort(rng.choice(len(test), min(500, len(test)), replace=False))
    capped_train = analysis_fingerprints(
        train.iloc[train_rows][train_col].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    sampled_test = analysis_fingerprints(
        test.iloc[test_rows][test_col].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    nn = NearestNeighbors(n_neighbors=1, metric="jaccard", n_jobs=n_jobs).fit(capped_train)
    similarity = 1.0 - nn.kneighbors(sampled_test, return_distance=True)[0].ravel()
    return {
        "dataset": dataset,
        "method": method,
        "train_n": len(train_fp),
        "test_n": len(test_fp),
        "sampled_pairs": 100_000,
        "mean_train_test_similarity": mean_similarity,
        "nn_query_n": len(similarity),
        "nn_train_n": len(capped_train),
        "nn_mean_max_similarity": float(similarity.mean()),
        "nn_median_max_similarity": float(np.median(similarity)),
        "nn_fraction_above_0_7": float(np.mean(similarity > 0.7)),
        "nn_threshold": 0.7,
        "sample_seed": 42,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(CONFIG["datasets"]),
                        default=list(CONFIG["datasets"]))
    parser.add_argument("--tag", default="k3",
                        help="output suffix for this run (default: k3)")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    datasets = args.datasets
    split_out = ROOT / "results/sensitivity_analysis" / f"final_splits_{args.tag}"
    metric_out = ROOT / "results/sensitivity_analysis" / f"metrics_{args.tag}"
    metric_out.mkdir(parents=True, exist_ok=True)
    leakage_rows = []
    ml_rows = []
    generated = []
    for dataset in datasets:
        frame, _, _ = load_dataset(dataset)
        split_dirs = {}
        missing_strategies = []
        for strategy in STRATEGIES:
            expected_dir = split_out / dataset / METHOD_NAMES[strategy]
            complete_existing = all((expected_dir / f"{name}.csv").exists()
                                    for name in ("train", "val", "test"))
            if complete_existing and (expected_dir / "run_metadata.json").exists():
                split_dirs[strategy] = expected_dir
            else:
                missing_strategies.append(strategy)
        if missing_strategies:
            fps = fingerprint_matrix(frame, CONFIG, n_jobs=args.n_jobs)
            for strategy in missing_strategies:
                split_dirs[strategy] = _generate_splits(
                    dataset, strategy, frame, fps, split_out, args.n_jobs)
            del fps
            gc.collect()
        for strategy in STRATEGIES:
            generated.append(str(split_dirs[strategy].relative_to(ROOT)))
        for strategy in STRATEGIES:
            method = METHOD_NAMES[strategy]
            split_dir = split_dirs[strategy]
            frames = {name: pd.read_csv(split_dir / f"{name}.csv") for name in ("train", "val", "test")}
            leakage_rows.append(_leakage_metrics(dataset, method, frames, args.n_jobs))
            import ml_evaluation_extended as ml_eval
            ml_eval.fast_fingerprints = sparse_fingerprints
            task_type = CONFIG["datasets"][dataset]["task_type"]
            if task_type == "regression":
                rows = (evaluate_qm9(method, frames["train"], frames["test"], args.n_jobs, 42)
                        if dataset == "qm9" else
                        evaluate_regression(dataset, method, frames["train"], frames["test"], args.n_jobs))
            elif dataset == "muv":
                rows = evaluate_muv(method, frames["train"], frames["test"], args.n_jobs, 42)
            else:
                rows = evaluate_pcba(method, frames["train"], frames["test"],
                                     n_jobs=args.n_jobs, seed=42, dataset=dataset)
            ml_rows.extend(rows)
            print(f"Completed k=3 splits and metrics: {dataset}/{strategy}", flush=True)

        # Persist dataset-level results immediately so completed work survives
        # a later large or resource-intensive dataset.
        leak_path = metric_out / "leakage_metrics.csv"
        ml_path = metric_out / "ml_metrics.csv"
        existing_leak = pd.read_csv(leak_path) if leak_path.exists() else pd.DataFrame()
        existing_ml = pd.read_csv(ml_path) if ml_path.exists() else pd.DataFrame()
        replace = set(datasets)
        if not existing_leak.empty:
            existing_leak = existing_leak[~existing_leak.dataset.isin(replace)]
        if not existing_ml.empty:
            existing_ml = existing_ml[~existing_ml.dataset.isin(replace)]
        new_leak = pd.DataFrame(leakage_rows)
        new_ml = pd.DataFrame(ml_rows)
        pd.concat([existing_leak, new_leak], ignore_index=True).to_csv(leak_path, index=False)
        pd.concat([existing_ml, new_ml], ignore_index=True).to_csv(ml_path, index=False)

    leak_path = metric_out / "leakage_metrics.csv"
    ml_path = metric_out / "ml_metrics.csv"
    manifest = {
        "status": "complete",
        "locked_k": K,
        "lock_record": "results/sensitivity_analysis/anchor_holdout_selection/locked_k.json",
        "selection_completed_before_metrics_evaluation": True,
        "datasets": list(datasets),
        "k_selection_validation_scope": list(CONFIG["default_datasets"]),
        "methods": list(METHOD_NAMES.values()),
        "split_processing_order": "source row order",
        "anchor_labels_fixed": True,
        "leakage_metrics": ["mean Tanimoto over 100000 sampled train-test pairs",
                             "test-to-train maximum Tanimoto, 500 test queries, up to 10000 train molecules, threshold 0.7"],
        "ml_metrics_source": "experiments/analysis/ml_evaluation_extended.py; configured classification/regression models",
        "n_jobs": args.n_jobs,
        "ml_seed": 42,
        "split_directories": generated,
        "leakage_output": str(leak_path.relative_to(ROOT)),
        "ml_output": str(ml_path.relative_to(ROOT)),
    }
    (metric_out / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved leakage metrics to {leak_path}", flush=True)
    print(f"Saved ML metrics to {ml_path}", flush=True)


if __name__ == "__main__":
    main()
