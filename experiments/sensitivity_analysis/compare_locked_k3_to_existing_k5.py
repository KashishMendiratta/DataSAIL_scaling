"""Compare locked-k=3 evaluations with the existing seed-42 k=5 results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pipeline_engine import CONFIG  # noqa: E402

OUT = ROOT / "results/sensitivity_analysis/k3_vs_k5"
DATASETS = list(CONFIG["datasets"])
STRATEGIES = ("random", "stratified")
METHOD_MAP = {
    "datasail_scaled_random": "random",
    "datasail_scaled_stratified": "stratified",
}
K3_METHOD_MAP = {
    "datasail_scaled_random_locked_k3": "random",
    "datasail_scaled_stratified_locked_k3": "stratified",
}
ML_METRICS = ("roc_auc", "pr_auc", "accuracy", "f1", "mcc", "mae", "rmse", "r2")


def _k5_predictive_path(dataset: str) -> Path:
    regular = ROOT / "results/analysis/unified" / dataset / f"{dataset}_predictive_metrics.csv"
    if regular.exists():
        return regular
    legacy_space = ROOT / "results/analysis/unified" / f"{dataset} " / f"{dataset}_predictive_metrics.csv"
    if legacy_space.exists():
        return legacy_space
    raise FileNotFoundError(f"No existing k=5 predictive metrics found for {dataset}")


def _load_ml_k5(dataset: str) -> tuple[pd.DataFrame, str]:
    path = _k5_predictive_path(dataset)
    frame = pd.read_csv(path)
    frame = frame[frame.split_type.isin(METHOD_MAP)].copy()
    frame["strategy"] = frame.split_type.map(METHOD_MAP)
    return frame, str(path.relative_to(ROOT))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    leak_k3 = pd.read_csv(ROOT / "results/sensitivity_analysis/metrics_k3/leakage_metrics.csv")
    leak_k5 = pd.read_csv(ROOT / "results/analysis/unified/leakage_analysis.csv")
    nn_k3 = pd.read_csv(ROOT / "results/sensitivity_analysis/metrics_k3/leakage_metrics.csv")
    nn_k5 = pd.read_csv(ROOT / "results/analysis/unified/nn_leakage.csv")

    leak_k3["strategy"] = leak_k3.method.map(K3_METHOD_MAP)
    leak_k5 = leak_k5[leak_k5.method.isin(METHOD_MAP)].copy()
    leak_k5["strategy"] = leak_k5.method.map(METHOD_MAP)
    leak_cols = ["mean_train_test_similarity"]
    leak_compare = leak_k3[["dataset", "strategy", *leak_cols]].merge(
        leak_k5[["dataset", "strategy", *leak_cols]],
        on=["dataset", "strategy"], suffixes=("_k3", "_k5"), validate="one_to_one")
    leak_compare["delta_k3_minus_k5"] = leak_compare["mean_train_test_similarity_k3"] - leak_compare["mean_train_test_similarity_k5"]
    leak_compare.to_csv(OUT / "mean_leakage_comparison.csv", index=False)

    nn_k3["strategy"] = nn_k3.method.map(K3_METHOD_MAP)
    nn_k3 = nn_k3.rename(columns={
        "nn_mean_max_similarity": "mean_max_similarity",
        "nn_median_max_similarity": "median_max_similarity",
        "nn_fraction_above_0_7": "fraction_above_threshold",
    })
    nn_k5 = nn_k5[nn_k5.method.isin(METHOD_MAP)].copy()
    nn_k5["strategy"] = nn_k5.method.map(METHOD_MAP)
    nn_cols = ["mean_max_similarity", "median_max_similarity", "fraction_above_threshold"]
    nn_compare = nn_k3[["dataset", "strategy", *nn_cols]].merge(
        nn_k5[["dataset", "strategy", *nn_cols]],
        on=["dataset", "strategy"], suffixes=("_k3", "_k5"), validate="one_to_one")
    for col in nn_cols:
        nn_compare[f"delta_{col}_k3_minus_k5"] = nn_compare[f"{col}_k3"] - nn_compare[f"{col}_k5"]
    nn_compare.to_csv(OUT / "nearest_neighbor_leakage_comparison.csv", index=False)

    k3_ml = pd.read_csv(ROOT / "results/sensitivity_analysis/metrics_k3/ml_metrics.csv")
    k3_ml["strategy"] = k3_ml.split_type.map(K3_METHOD_MAP)
    k5_paths = {}
    k5_frames = []
    for dataset in DATASETS:
        frame, source = _load_ml_k5(dataset)
        frame["k5_source"] = source
        k5_paths[dataset] = source
        k5_frames.append(frame)
    k5_ml = pd.concat(k5_frames, ignore_index=True)
    join_keys = ["dataset", "strategy", "task"]
    metric_cols = [c for c in ML_METRICS if c in k3_ml.columns and c in k5_ml.columns]
    k3_fields = k3_ml[join_keys + metric_cols + ["status"]].rename(
        columns={**{c: f"{c}_k3" for c in metric_cols}, "status": "status_k3"})
    k5_fields = k5_ml[join_keys + metric_cols + ["status"]].rename(
        columns={**{c: f"{c}_k5" for c in metric_cols}, "status": "status_k5"})
    ml_compare = k3_fields.merge(k5_fields, on=join_keys, how="outer", validate="one_to_one", indicator=True)
    for col in metric_cols:
        ml_compare[f"delta_{col}_k3_minus_k5"] = ml_compare[f"{col}_k3"] - ml_compare[f"{col}_k5"]
    ml_compare.to_csv(OUT / "ml_task_comparison.csv", index=False)

    summary_rows = []
    for (dataset, strategy), group in ml_compare[ml_compare["_merge"] == "both"].groupby(join_keys[:2]):
        row = {"dataset": dataset, "strategy": strategy, "matched_tasks": len(group)}
        for col in metric_cols:
            paired = group[[f"{col}_k3", f"{col}_k5"]].dropna()
            row[f"n_{col}"] = len(paired)
            row[f"mean_{col}_k3"] = paired[f"{col}_k3"].mean() if len(paired) else np.nan
            row[f"mean_{col}_k5"] = paired[f"{col}_k5"].mean() if len(paired) else np.nan
            row[f"mean_delta_{col}_k3_minus_k5"] = (paired[f"{col}_k3"] - paired[f"{col}_k5"]).mean() if len(paired) else np.nan
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT / "ml_summary_by_dataset_strategy.csv", index=False)

    split_k5 = {}
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            path = ROOT / "results/experiments/unified/splits" / dataset / f"datasail_scaled_{strategy}" / "run_metadata.json"
            meta = json.loads(path.read_text())
            split_k5[f"{dataset}/{strategy}"] = {"k": meta.get("k"), "seed": meta.get("seed")}
            if meta.get("k") != 5:
                raise ValueError(f"Expected k=5 metadata in {path}")
    manifest = {
        "comparison": "locked k=3 versus existing seed-42 k=5 scaled DataSAIL results",
        "datasets": DATASETS,
        "strategies": list(STRATEGIES),
        "k3_split_root": "results/sensitivity_analysis/final_splits_k3",
        "k3_metrics_root": "results/sensitivity_analysis/metrics_k3",
        "k5_split_metadata": split_k5,
        "k5_leakage_source": "results/analysis/unified/leakage_analysis.csv",
        "k5_nearest_neighbor_leakage_source": "results/analysis/unified/nn_leakage.csv",
        "k5_predictive_metric_sources": k5_paths,
        "matched_ml_task_rows": int((ml_compare["_merge"] == "both").sum()),
        "unmatched_ml_task_rows": int((ml_compare["_merge"] != "both").sum()),
        "ml_comparison_note": "Task-level paired comparisons; regression targets are retained individually because target scales differ.",
        "outputs": [
            "results/sensitivity_analysis/k3_vs_k5/mean_leakage_comparison.csv",
            "results/sensitivity_analysis/k3_vs_k5/nearest_neighbor_leakage_comparison.csv",
            "results/sensitivity_analysis/k3_vs_k5/ml_task_comparison.csv",
            "results/sensitivity_analysis/k3_vs_k5/ml_summary_by_dataset_strategy.csv",
        ],
    }
    (OUT / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lower_mean_leak = int((leak_compare.delta_k3_minus_k5 < 0).sum())
    lower_nn = int((nn_compare.delta_mean_max_similarity_k3_minus_k5 < 0).sum())
    higher_nn = int((nn_compare.delta_mean_max_similarity_k3_minus_k5 > 0).sum())
    equal_nn = len(nn_compare) - lower_nn - higher_nn
    lower_threshold = int((nn_compare.delta_fraction_above_threshold_k3_minus_k5 < 0).sum())
    higher_threshold = int((nn_compare.delta_fraction_above_threshold_k3_minus_k5 > 0).sum())
    equal_threshold = len(nn_compare) - lower_threshold - higher_threshold
    cls = ml_compare[ml_compare.dataset.isin(["bace", "bbbp", "tox21", "hiv", "pcba", "muv"])]
    class_summary = {}
    for metric in ("roc_auc", "pr_auc", "accuracy", "f1", "mcc"):
        group_delta = cls.groupby(["dataset", "strategy"])[f"delta_{metric}_k3_minus_k5"].mean().dropna()
        class_summary[metric] = float(group_delta.mean()) if len(group_delta) else np.nan
    report = [
        "# Locked k=3 vs existing k=5 evaluation",
        "",
        "Compared the locked `k=3` splits with the existing seed-42 `k=5` scaled DataSAIL results across all ten datasets and both anchor strategies. All k=5 split metadata records `k=5`; all 364 predictive task rows matched by dataset, strategy, and task.",
        "",
        "## Leakage",
        "",
        f"Mean sampled train-test Tanimoto was lower for k=3 in {lower_mean_leak}/20 dataset/strategy pairs; the average paired change was {leak_compare.delta_k3_minus_k5.mean():+.5f}.",
        f"Nearest-neighbor mean maximum similarity was lower in {lower_nn}/20 pairs, higher in {higher_nn}, and unchanged in {equal_nn}; its mean paired change was {nn_compare.delta_mean_max_similarity_k3_minus_k5.mean():+.5f}.",
        f"The fraction of test molecules above Tanimoto 0.7 was lower in {lower_threshold}/20 pairs, higher in {higher_threshold}, and unchanged in {equal_threshold}.",
        "",
        "## Predictive metrics",
        "",
        f"Across the 12 classification dataset/strategy groups (assays averaged within each group), mean k=3 minus k=5 changes were ROC AUC {class_summary['roc_auc']:+.4f}, PR AUC {class_summary['pr_auc']:+.4f}, accuracy {class_summary['accuracy']:+.4f}, F1 {class_summary['f1']:+.4f}, and MCC {class_summary['mcc']:+.4f}. Results vary by dataset and metric; these small mixed changes do not show a consistent predictive advantage for either k.",
        "Regression comparisons are retained per target in `ml_task_comparison.csv`. MAE and RMSE are lower-is-better, R² is higher-is-better, and target scales differ, so the per-target rows are more informative than an across-dataset average.",
        "",
        "## Interpretation limits",
        "",
        "The k=3 lock was selected using held-out anchor balanced accuracy before final-split metrics. This comparison is descriptive and does not reopen k selection. The six datasets outside holdout selection scope use the locked k=3 without separate k validation.",
        "For the six additional datasets, k=3 fingerprint matrices were passed to the configured models as sparse matrices to reduce memory use; existing k=5 evaluations used dense matrices. Both contain the same binary fingerprint bits and use the same model settings, but small numerical/training differences are possible. The primary four datasets used dense matrices for both k values.",
        "",
        "## Files",
        "",
        "- `mean_leakage_comparison.csv`",
        "- `nearest_neighbor_leakage_comparison.csv`",
        "- `ml_task_comparison.csv`",
        "- `ml_summary_by_dataset_strategy.csv`",
        "- `comparison_manifest.json`",
    ]
    (OUT / "comparison_summary.md").write_text("\n".join(report) + "\n")
    print(f"Created k=3 versus k=5 comparison in {OUT}")
    print(f"Leakage comparisons: {len(leak_compare)}; nearest-neighbor: {len(nn_compare)}; matched ML tasks: {manifest['matched_ml_task_rows']}")


if __name__ == "__main__":
    main()
