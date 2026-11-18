"""Select and lock k by balanced accuracy on held-out DataSAIL anchors."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from knn_order_sensitivity import (  # noqa: E402
    K_VALUES,
    ORDER_SEEDS,
    SPLIT_NAMES,
    _assign_in_order,
    _load_anchor_labels,
    _order_indices,
)
from src.pipeline_engine import CONFIG, fingerprint_matrix, load_dataset  # noqa: E402

DATASETS = CONFIG["default_datasets"]
STRATEGIES = ("random", "stratified")
N_FOLDS = 5
FOLD_SEED = 42
OUT = ROOT / "results/sensitivity_analysis/anchor_holdout_selection"


def evaluate_group(dataset: str, strategy: str, frame: pd.DataFrame, fps: np.ndarray):
    manifest, label_map, split_path = _load_anchor_labels(dataset, strategy)
    ids = frame["_datasail_id"].astype(str).to_numpy()
    id_to_pos = {value: i for i, value in enumerate(ids)}
    anchor_ids = [str(value) for value in manifest["anchor_ids"]]
    anchor_pos = np.array([id_to_pos[value] for value in anchor_ids], dtype=int)
    anchor_labels = np.array([label_map[value] for value in anchor_ids], dtype=object)
    if set(np.unique(anchor_labels)) != set(SPLIT_NAMES):
        raise ValueError(f"{dataset}/{strategy}: anchor set lacks one or more split labels")

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    order_names = ["source", "reverse", *(f"random_{seed}" for seed in ORDER_SEEDS)]
    records = []
    fold_membership = []
    ratios = CONFIG["defaults"]["split_ratios"]
    balance_weight = CONFIG["defaults"]["assignment"]["balance_weight"]

    for fold, (train_idx, heldout_idx) in enumerate(folds.split(anchor_pos, anchor_labels), start=1):
        train_pos = anchor_pos[train_idx]
        heldout_pos = anchor_pos[heldout_idx]
        train_labels = anchor_labels[train_idx]
        heldout_labels = anchor_labels[heldout_idx]
        fold_membership.extend({
            "dataset": dataset,
            "strategy": strategy,
            "fold": fold,
            "mol_id": anchor_ids[index],
            "fixed_datasail_split": str(anchor_labels[index]),
        } for index in heldout_idx)
        nn = NearestNeighbors(n_neighbors=max(K_VALUES), metric="jaccard", n_jobs=-1)
        nn.fit(fps[train_pos])
        distances, neighbor_indices = nn.kneighbors(fps[heldout_pos])

        orders = list(_order_indices(len(heldout_pos)))
        for k in K_VALUES:
            for order_name, order in orders:
                prediction, _ = _assign_in_order(
                    neighbor_indices, distances, train_labels, k, order,
                    ratios, balance_weight,
                )
                balanced_accuracy = float(balanced_accuracy_score(heldout_labels, prediction))
                recalls = recall_score(heldout_labels, prediction, labels=list(SPLIT_NAMES),
                                       average=None, zero_division=0)
                records.append({
                    "dataset": dataset,
                    "strategy": strategy,
                    "fold": fold,
                    "k": k,
                    "processing_order": order_name,
                    "n_training_anchors": len(train_pos),
                    "n_heldout_anchors": len(heldout_pos),
                    "balanced_accuracy": balanced_accuracy,
                    **{f"recall_{split}": float(value) for split, value in zip(SPLIT_NAMES, recalls)},
                    "anchor_manifest_sha256": hashlib.sha256(
                        (ROOT / "results/experiments/unified/splits" / dataset /
                         f"datasail_scaled_{strategy}" / "anchor_manifest.json").read_bytes()
                    ).hexdigest(),
                    "datasail_split_file": str(split_path.relative_to(ROOT)),
                })
    return records, fold_membership


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_records = []
    all_fold_membership = []
    for dataset in DATASETS:
        frame, _, _ = load_dataset(dataset)
        fps = fingerprint_matrix(frame, CONFIG, n_jobs=1)
        for strategy in STRATEGIES:
            records, fold_membership = evaluate_group(dataset, strategy, frame, fps)
            all_records.extend(records)
            all_fold_membership.extend(fold_membership)
            print(f"Completed anchor holdout: {dataset}/{strategy}, {len(records)} evaluations")

    detail = pd.DataFrame(all_records)
    detail_path = OUT / "anchor_holdout_scores.csv"
    detail.to_csv(detail_path, index=False)
    fold_manifest_path = OUT / "anchor_fold_manifest.csv"
    pd.DataFrame(all_fold_membership).to_csv(fold_manifest_path, index=False)

    # Macro-average fold/order balanced accuracy within each dataset/strategy,
    # then macro-average the eight groups so large datasets do not dominate.
    group_scores = (detail.groupby(["dataset", "strategy", "k"], as_index=False)
                    .balanced_accuracy.mean()
                    .rename(columns={"balanced_accuracy": "group_mean_balanced_accuracy"}))
    scores = (group_scores.groupby("k", as_index=False)
              .group_mean_balanced_accuracy.mean()
              .rename(columns={"group_mean_balanced_accuracy": "macro_mean_balanced_accuracy"}))
    scores_path = OUT / "k_selection_scores.csv"
    scores.to_csv(scores_path, index=False)
    best_score = float(scores.macro_mean_balanced_accuracy.max())
    tied = scores[np.isclose(scores.macro_mean_balanced_accuracy, best_score, rtol=0, atol=1e-12)]
    selected_k = int(tied.k.min())

    source_hashes = {}
    for dataset in DATASETS:
        source_hashes[dataset] = hashlib.sha256(
            (ROOT / CONFIG["datasets"][dataset]["path"]).read_bytes()
        ).hexdigest()
    lock = {
        "status": "locked",
        "selected_k": selected_k,
        "selection_metric": "macro mean balanced accuracy across 5 folds, 7 processing orders, and 8 dataset/strategy groups",
        "tie_break": "smallest k among scores equal to the maximum within absolute tolerance 1e-12",
        "candidate_k_values": list(K_VALUES),
        "datasets": list(DATASETS),
        "anchor_strategies": list(STRATEGIES),
        "folds": N_FOLDS,
        "fold_seed": FOLD_SEED,
        "processing_orders": ["source", "reverse", *(f"random_{seed}" for seed in ORDER_SEEDS)],
        "score_by_k": {str(int(row.k)): float(row.macro_mean_balanced_accuracy)
                       for _, row in scores.iterrows()},
        "scores_file": str(scores_path.relative_to(ROOT)),
        "detail_file": str(detail_path.relative_to(ROOT)),
        "fold_manifest_file": str(fold_manifest_path.relative_to(ROOT)),
        "anchor_holdout_only": True,
        "selection_completed_before_metrics_evaluation": True,
        "source_sha256": source_hashes,
    }
    (OUT / "locked_k.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Locked k={selected_k} with macro mean balanced accuracy {best_score:.6f}")


if __name__ == "__main__":
    main()
