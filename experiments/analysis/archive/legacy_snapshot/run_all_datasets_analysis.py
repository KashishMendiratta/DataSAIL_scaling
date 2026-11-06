"""Run and aggregate runtime, leakage, and predictive analyses across all datasets.

The dataset registry is the single source of truth. Random, scaled-random,
scaled-stratified, and full DataSAIL results use one normalized schema regardless
of whether the split was produced by the legacy or configured pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/pipelines"))
sys.path.insert(0, str(ROOT / "experiments/analysis"))
from run_scaled_configured import CONFIG  # noqa: E402
from run_scaled_configured import run as run_scaled  # noqa: E402
from analyze_scaled_configured import (METHODS, analyze, runtime_from_metadata,
                                       split_directory)  # noqa: E402
from run_predictive_evaluation_configured import run as run_predictive  # noqa: E402


def summarize_predictive(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for (dataset, method), group in metrics.groupby(["dataset", "split_type"], dropna=False):
        record = {"dataset": dataset, "method": method,
                  "ml_task_count": int(group["task"].nunique()) if "task" in group else len(group),
                  "ml_successful_task_count": int(group["status"].eq("ok").sum()) if "status" in group else 0,
                  "ml_status": "available"}
        for column in ("roc_auc", "pr_auc", "accuracy", "f1", "mcc", "r2", "mae", "rmse"):
            if column in group:
                record[f"{column}_macro"] = pd.to_numeric(group[column], errors="coerce").mean()
        rows.append(record)
    return pd.DataFrame(rows)


def aggregate(datasets: list[str]) -> Path:
    analysis_dir = ROOT / "results/analysis/configured"
    output_dir = ROOT / "results/analysis/all_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    leakage_frames, metric_frames = [], []
    for dataset in datasets:
        leakage_path = analysis_dir / f"{dataset}_runtime_leakage.csv"
        metrics_path = analysis_dir / f"{dataset}_predictive_metrics.csv"
        if leakage_path.exists():
            leakage_frames.append(pd.read_csv(leakage_path))
        if metrics_path.exists():
            metric_frames.append(pd.read_csv(metrics_path))
    leakage = pd.concat(leakage_frames, ignore_index=True) if leakage_frames else pd.DataFrame()
    if not leakage.empty:
        # Older per-dataset analyses wrote separate timing and leakage rows for
        # each method. Coalesce them into the one-row-per-method schema.
        leakage = leakage.groupby(["dataset", "method"], as_index=False).first()
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    if not metrics.empty:
        metrics.to_csv(output_dir / "predictive_metrics_all_tasks.csv", index=False)
    ml_summary = summarize_predictive(metrics)
    if not leakage.empty:
        leakage.to_csv(output_dir / "runtime_leakage_all_datasets.csv", index=False)

    # Keep one row for each dataset/method, including missing or failed methods.
    expected = pd.MultiIndex.from_product([datasets, METHODS], names=["dataset", "method"]).to_frame(index=False)
    summary = expected.merge(leakage, on=["dataset", "method"], how="left") if not leakage.empty else expected
    # Always resolve runtimes from source metadata. Historical `runtime_seconds`
    # in early configured analysis CSVs meant DataSAIL-only time, so never treat
    # that column as an end-to-end measurement.
    end_to_end, datasail_only, runtime_sources, split_statuses = [], [], [], []
    for row in summary.itertuples(index=False):
        folder = split_directory(row.dataset, row.method)
        total, datasail, source = runtime_from_metadata(row.dataset, row.method, folder)
        end_to_end.append(total)
        datasail_only.append(datasail)
        runtime_sources.append(source)
        split_statuses.append("available" if all((folder / f"{part}.csv").exists()
                                                  for part in ("train", "val", "test"))
                              else "missing_split_files")
    summary["end_to_end_runtime_seconds"] = end_to_end
    summary["runtime_seconds"] = end_to_end  # standardized legacy-compatible alias
    summary["datasail_seconds"] = datasail_only
    summary["runtime_source"] = runtime_sources
    if "split_status" not in summary:
        summary["split_status"] = split_statuses
    else:
        summary["split_status"] = [
            value if isinstance(value, str) and value else status
            for value, status in zip(summary["split_status"], split_statuses)
        ]
    if not ml_summary.empty:
        summary = summary.merge(ml_summary, on=["dataset", "method"], how="left")
    if "split_status" in summary:
        summary["split_status"] = summary["split_status"].fillna("not_analyzed")
    if "ml_status" in summary:
        summary["ml_status"] = summary["ml_status"].fillna("no_metrics")
    summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)

    labels = {"random": "Random", "datasail_scaled_random": "Scaled random",
              "datasail_scaled_stratified": "Scaled stratified", "datasail_full": "Full DataSAIL"}
    palette = {"Random": "#4C78A8", "Scaled random": "#F58518",
               "Scaled stratified": "#ECA82C", "Full DataSAIL": "#54A24B"}
    if not summary.empty:
        plot = summary.copy()
        plot["method_label"] = plot["method"].map(labels)
        for metric, filename, ylabel, logscale in [
            ("end_to_end_runtime_seconds", "runtime_end_to_end.png", "End-to-end runtime (seconds)", True),
            ("datasail_seconds", "runtime_datasail_only.png", "DataSAIL-only runtime (seconds)", True),
            ("leakage_fraction_gt_0_7", "leakage_fraction.png", "Fraction of test sample with max similarity > 0.7", False),
        ]:
            if metric not in plot:
                continue
            available = plot.dropna(subset=[metric])
            if available.empty:
                continue
            pivot = available.pivot_table(index="dataset", columns="method_label", values=metric, aggfunc="mean")
            pivot = pivot.reindex(index=datasets, columns=list(labels.values()))
            pivot = pivot.dropna(axis=1, how="all")
            fig, ax = plt.subplots(figsize=(15, 7))
            pivot.plot(kind="bar", ax=ax, color=[palette[column] for column in pivot.columns])
            if logscale:
                ax.set_yscale("log")
            ax.set_xlabel("Dataset")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=35)
            ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.01, 1))
            ax.grid(axis="y", alpha=.25)
            fig.tight_layout(rect=[0, 0, .84, 1])
            fig.savefig(output_dir / filename, dpi=220)
            plt.close(fig)

        for metric, filename, ylabel in [
            ("mcc_macro", "ml_classification_mcc.png", "Macro MCC"),
            ("r2_macro", "ml_regression_r2.png", "Mean R² across targets"),
        ]:
            if metric not in plot:
                continue
            available = plot.dropna(subset=[metric])
            if available.empty:
                continue
            pivot = available.pivot_table(index="dataset", columns="method_label", values=metric, aggfunc="mean")
            pivot = pivot.reindex(index=datasets, columns=list(labels.values()))
            pivot = pivot.dropna(axis=1, how="all")
            fig, ax = plt.subplots(figsize=(15, 7))
            pivot.plot(kind="bar", ax=ax, color=[palette[column] for column in pivot.columns])
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_xlabel("Dataset")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=35)
            ax.legend(title="Method", loc="upper left", bbox_to_anchor=(1.01, 1))
            ax.grid(axis="y", alpha=.25)
            fig.tight_layout(rect=[0, 0, .84, 1])
            fig.savefig(output_dir / filename, dpi=220)
            plt.close(fig)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(CONFIG["datasets"]),
                        default=list(CONFIG["datasets"]))
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--skip-leakage", action="store_true")
    parser.add_argument("--skip-predictive", action="store_true")
    parser.add_argument("--rerun-scaled", action="store_true",
                        help="Regenerate both configured scaled splits first, recording fresh end-to-end runtimes")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Combine existing per-dataset outputs without recomputing analyses")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Run requested dataset analyses but defer the combined aggregation")
    args = parser.parse_args()

    if not args.aggregate_only:
        for dataset in args.datasets:
            if args.rerun_scaled:
                for strategy in ("random", "stratified"):
                    print(f"Running configured scaled pipeline: {dataset}/{strategy}", flush=True)
                    run_scaled(dataset, strategy, n_jobs=args.n_jobs)
            if not args.skip_leakage:
                analyze(dataset, METHODS, n_jobs=args.n_jobs)
            if not args.skip_predictive:
                run_predictive(dataset, METHODS, args.n_jobs, seed=CONFIG["defaults"]["seed"])
    if not args.no_aggregate:
        output_dir = aggregate(args.datasets)
        print(f"Unified all-dataset tables and plots written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
