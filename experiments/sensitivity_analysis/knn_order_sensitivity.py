"""Measure balanced-kNN sensitivity to k and non-anchor processing order."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pipeline_engine import CONFIG, SPLIT_ROOT, DATASAIL_ROOT, fingerprint_matrix, load_dataset  # noqa: E402

K_VALUES = (1, 3, 5, 10, 20)
ORDER_SEEDS = (11, 23, 42, 101, 202)
SPLIT_NAMES = tuple(CONFIG["defaults"]["split_ratios"])


def _base_dir(dataset: str, strategy: str) -> Path:
    return SPLIT_ROOT / dataset / f"datasail_scaled_{strategy}"


def _load_anchor_labels(dataset: str, strategy: str):
    base = _base_dir(dataset, strategy)
    manifest_path = base / "anchor_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing saved anchor manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("seed") != CONFIG["defaults"]["seed"]:
        raise ValueError(f"Expected base seed {CONFIG['defaults']['seed']} in {manifest_path}")
    split_path = next(iter(sorted((DATASAIL_ROOT / dataset / f"datasail_scaled_{strategy}").glob("C1e/*_splits.tsv"))), None)
    if split_path is None:
        raise FileNotFoundError(f"Missing saved DataSAIL anchor split TSV for {dataset}/{strategy}")
    table = pd.read_csv(split_path, sep="\t")
    if not {"ID", "Split"}.issubset(table.columns):
        raise ValueError(f"Unexpected DataSAIL split schema in {split_path}")
    labels = dict(zip(table["ID"].astype(str), table["Split"].astype(str)))
    anchor_ids = [str(value) for value in manifest["anchor_ids"]]
    if len(anchor_ids) != len(set(anchor_ids)) or set(anchor_ids) != set(labels):
        raise ValueError(f"Anchor manifest and DataSAIL labels do not match for {dataset}/{strategy}")
    if set(labels.values()) != set(SPLIT_NAMES):
        raise ValueError(f"Expected anchor labels {SPLIT_NAMES}, got {set(labels.values())}")
    return manifest, labels, split_path


def _order_indices(n: int):
    yield "source", np.arange(n, dtype=int)
    yield "reverse", np.arange(n - 1, -1, -1, dtype=int)
    for seed in ORDER_SEEDS:
        yield f"random_{seed}", np.random.default_rng(seed).permutation(n)


def _assign_in_order(neighbor_indices, neighbor_distances, anchor_labels, k, order, target_ratios, balance_weight):
    """Apply the shared balanced-kNN scoring rule using cached neighbors."""
    counts = Counter(anchor_labels.tolist())
    assigned_total = len(anchor_labels)
    output = np.empty(len(order), dtype=object)
    confidences = np.empty(len(order), dtype=float)
    for query_pos in order:
        indices = neighbor_indices[query_pos, :k]
        distances = neighbor_distances[query_pos, :k]
        neighbor_splits = anchor_labels[indices]
        scores = {}
        for split_name, target_ratio in target_ratios.items():
            mask = neighbor_splits == split_name
            if mask.any():
                similarity = (1.0 - distances[mask]).mean()
            else:
                similarity = 0.0
            current_ratio = counts[split_name] / assigned_total
            balance_factor = target_ratio / (current_ratio + 1e-10)
            scores[split_name] = similarity * (balance_factor ** balance_weight)
        chosen = max(scores, key=scores.get)
        output[query_pos] = chosen
        counts[chosen] += 1
        assigned_total += 1
        ranked = sorted(scores.values(), reverse=True)
        confidences[query_pos] = ((ranked[0] - ranked[1]) / ranked[0]) if len(ranked) > 1 and ranked[0] > 0 else 0.5
    return output, confidences


def run_dataset(dataset: str, strategy: str, output_root: Path, n_jobs: int):
    manifest, anchor_label_map, split_path = _load_anchor_labels(dataset, strategy)
    frame, source_rows, invalid = load_dataset(dataset)
    ids = frame["_datasail_id"].astype(str).to_numpy()
    id_to_pos = {value: i for i, value in enumerate(ids)}
    try:
        anchor_positions = np.array([id_to_pos[value] for value in manifest["anchor_ids"]], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Anchor ID {exc.args[0]} not present among valid source rows for {dataset}") from exc
    anchor_labels = np.array([anchor_label_map[value] for value in manifest["anchor_ids"]], dtype=object)
    is_anchor = np.zeros(len(frame), dtype=bool)
    is_anchor[anchor_positions] = True
    remaining_positions = np.flatnonzero(~is_anchor)
    if len(anchor_positions) < max(K_VALUES):
        raise ValueError(f"Only {len(anchor_positions)} anchors available for max k={max(K_VALUES)}")

    fps = fingerprint_matrix(frame, CONFIG, n_jobs)
    nn = NearestNeighbors(n_neighbors=max(K_VALUES), metric="jaccard", n_jobs=n_jobs).fit(fps[anchor_positions])
    distances, neighbor_indices = nn.kneighbors(fps[remaining_positions])
    target_ratios = CONFIG["defaults"]["split_ratios"]
    balance_weight = CONFIG["defaults"]["assignment"]["balance_weight"]
    rel_orders = list(_order_indices(len(remaining_positions)))
    out_dir = output_root / dataset / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    source_assignments = {}

    for k in K_VALUES:
        for order_name, order in rel_orders:
            remaining_labels, confidences = _assign_in_order(
                neighbor_indices, distances, anchor_labels, k, order,
                target_ratios, balance_weight,
            )
            all_labels = np.empty(len(frame), dtype=object)
            all_labels[anchor_positions] = anchor_labels
            all_labels[remaining_positions] = remaining_labels
            if order_name == "source":
                source_assignments[k] = remaining_labels.copy()
                changed = 0
            else:
                changed = int(np.count_nonzero(remaining_labels != source_assignments[k]))
            counts = Counter(all_labels.tolist())
            assignments_path = out_dir / f"assignments_k{k}_{order_name}.csv"
            pd.DataFrame({
                "mol_id": ids,
                "split": all_labels,
                "is_anchor": is_anchor,
            }).to_csv(assignments_path, index=False)
            summary = {
                "dataset": dataset,
                "strategy": strategy,
                "seed": manifest["seed"],
                "k": k,
                "processing_order": order_name,
                "n_input_rows": source_rows,
                "n_valid_rows": len(frame),
                "n_invalid_smiles": len(invalid),
                "n_anchors": int(is_anchor.sum()),
                "n_remaining": len(remaining_positions),
                "n_changed_remaining_vs_source_order": changed,
                "fraction_changed_remaining_vs_source_order": changed / len(remaining_positions),
                "mean_assignment_confidence": float(np.mean(confidences)),
                "split_counts": json.dumps({name: counts[name] for name in SPLIT_NAMES}, sort_keys=True),
                "split_fractions": json.dumps({name: counts[name] / len(frame) for name in SPLIT_NAMES}, sort_keys=True),
                "anchor_manifest_sha256": hashlib.sha256((_base_dir(dataset, strategy) / "anchor_manifest.json").read_bytes()).hexdigest(),
                "datasail_split_file": str(split_path.relative_to(ROOT)),
                "assignments_file": str(assignments_path.relative_to(ROOT)),
            }
            summaries.append(summary)

    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                        help="default: primary registry datasets")
    parser.add_argument("--strategies", nargs="+", choices=("random", "stratified"), default=("random", "stratified"))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sensitivity_analysis")
    args = parser.parse_args()
    datasets = args.datasets or CONFIG["default_datasets"]
    unknown = sorted(set(datasets) - set(CONFIG["datasets"]))
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")
    all_summaries = []
    failures = []
    for dataset in datasets:
        for strategy in args.strategies:
            try:
                rows = run_dataset(dataset, strategy, args.output_dir, args.n_jobs)
                all_summaries.extend(rows)
                detail_summary = args.output_dir / dataset / strategy / "summary.csv"
                pd.DataFrame(rows).to_csv(detail_summary, index=False)
                pd.DataFrame(all_summaries).to_csv(args.output_dir / "knn_order_sensitivity_summary.csv", index=False)
                print(f"Completed {dataset}/{strategy}: {len(rows)} k/order combinations")
            except Exception as exc:
                failures.append({"dataset": dataset, "strategy": strategy, "error": str(exc)})
                print(f"FAILED {dataset}/{strategy}: {exc}", file=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "knn_order_sensitivity_summary.csv"
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "k_values": K_VALUES,
        "processing_orders": ["source", "reverse", *(f"random_{seed}" for seed in ORDER_SEEDS)],
        "datasets": datasets,
        "strategies": list(args.strategies),
        "n_jobs": args.n_jobs,
        "failed_runs": failures,
        "summary_file": str(summary_path.relative_to(ROOT)),
        "note": "No leakage or ML metrics computed. Anchor labels are fixed from saved seed-42 DataSAIL outputs.",
    }, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} dataset/strategy run(s) failed; see run_manifest.json")


if __name__ == "__main__":
    main()
