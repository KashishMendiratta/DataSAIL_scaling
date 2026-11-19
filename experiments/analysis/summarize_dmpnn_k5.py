"""Summarize fixed-split Chemprop D-MPNN results with paired seed contrasts."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/analysis/unified/dmpnn_k5"
COMPARISONS = ("datasail_scaled_random", "datasail_scaled_stratified", "datasail_full")
METRICS = ("roc_auc", "pr_auc", "mae", "rmse", "r2")
# Two-sided 95% Student-t critical values for df 1..29. Runs have at most five
# matched split seeds, so only the first four values are normally used.
T95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
       6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281,
       11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
       16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860,
       21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639, 25: 2.0595,
       26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUT / "metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Missing combined metrics file: {args.input}")
    data = pd.read_csv(args.input)
    available = [metric for metric in METRICS if metric in data.columns]
    long = data.melt(id_vars=["dataset", "method", "split_seed", "task", "model", "n_test", "status"],
                     value_vars=available, var_name="metric", value_name="value")
    long = long[np.isfinite(pd.to_numeric(long["value"], errors="coerce"))].copy()
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long.to_csv(args.output_dir / "metrics_long.csv", index=False)

    # Full DataSAIL is one split, so preserve it as a reference and never treat
    # it as a five-seed paired comparison.
    long[long.method == "datasail_full"].to_csv(args.output_dir / "full_reference.csv", index=False)

    random = long[long.method == "random"][["dataset", "split_seed", "task", "metric", "value"]]
    paired_parts = []
    for method in COMPARISONS:
        if method == "datasail_full":
            continue
        alternative = long[long.method == method][["dataset", "split_seed", "task", "metric", "value"]]
        joined = alternative.merge(random, on=["dataset", "split_seed", "task", "metric"],
                                   suffixes=("_method", "_random"), validate="one_to_one")
        if joined.empty:
            continue
        joined["method"] = method
        joined["delta_method_minus_random"] = joined.value_method - joined.value_random
        paired_parts.append(joined)
    paired = (pd.concat(paired_parts, ignore_index=True) if paired_parts else pd.DataFrame(
        columns=["dataset", "split_seed", "task", "metric", "value_method", "value_random", "method",
                 "delta_method_minus_random"]))
    paired.to_csv(args.output_dir / "paired_task_deltas.csv", index=False)

    summaries = []
    for keys, group in paired.groupby(["dataset", "task", "metric", "method"], dropna=False):
        dataset, task, metric, method = keys
        values = group.delta_method_minus_random.dropna().to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        sd = float(values.std(ddof=1)) if n > 1 else np.nan
        half = T95.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else np.nan
        summaries.append({"dataset": dataset, "task": task, "metric": metric, "method": method,
                          "contrast": "method - random", "n_paired_seeds": n,
                          "mean_delta": mean, "sd_delta": sd,
                          "ci95_low": mean - half if n > 1 else np.nan,
                          "ci95_high": mean + half if n > 1 else np.nan,
                          "direction": "higher_is_better" if metric in ("roc_auc", "pr_auc", "r2") else "lower_is_better"})
    pd.DataFrame(summaries).to_csv(args.output_dir / "paired_summary.csv", index=False)

    # For sparse multitask classification, also show a within-seed macro mean
    # over valid assays, then treat the five seeds as the comparison units.
    multitask = paired[paired.dataset.isin(["tox21", "muv"])
                       & paired.metric.isin(["roc_auc", "pr_auc"])].copy()
    macro_by_seed = (multitask.groupby(["dataset", "method", "split_seed", "metric"], as_index=False)
                     .agg(n_assays=("task", "nunique"), mean_task_delta=("delta_method_minus_random", "mean")))
    macro_rows = []
    for keys, group in macro_by_seed.groupby(["dataset", "method", "metric"], dropna=False):
        dataset, method, metric = keys
        values = group.mean_task_delta.dropna().to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        sd = float(values.std(ddof=1)) if n > 1 else np.nan
        half = T95.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else np.nan
        macro_rows.append({"dataset": dataset, "method": method, "metric": metric,
                           "contrast": "macro assay mean: method - random", "n_paired_seeds": n,
                           "mean_assays_per_seed": float(group.n_assays.mean()) if n else np.nan,
                           "mean_delta": mean, "sd_delta": sd,
                           "ci95_low": mean - half if n > 1 else np.nan,
                           "ci95_high": mean + half if n > 1 else np.nan})
    pd.DataFrame(macro_rows).to_csv(args.output_dir / "multitask_macro_paired_summary.csv", index=False)

    # Descriptive means by dataset/method/task; regression targets stay separate.
    method_summary = (long.groupby(["dataset", "method", "task", "metric"], as_index=False)
                      .agg(n_runs=("value", "count"), mean=("value", "mean"), sd=("value", "std")))
    method_summary.to_csv(args.output_dir / "method_task_summary.csv", index=False)


if __name__ == "__main__":
    main()
