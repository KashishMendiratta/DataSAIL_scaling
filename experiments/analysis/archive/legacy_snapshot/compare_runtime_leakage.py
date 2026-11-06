"""Combine recorded runtime and nearest-neighbor leakage results by dataset and method.

The comparison keeps end-to-end timings separate from configured DataSAIL anchor-stage
runtimes. It reads existing leakage samples rather than recomputing fingerprints.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "results/analysis"
DATASETS = ["bace", "bbbp", "tox21", "hiv", "qm9", "pcba", "muv", "esol", "freesolv", "lipophilicity"]
METHODS = ["random", "datasail_scaled_random", "datasail_scaled_stratified", "datasail_full"]
LABELS = {
    "random": "Random",
    "datasail_scaled_random": "Scaled random-anchor",
    "datasail_scaled_stratified": "Scaled stratified-anchor",
    "datasail_full": "Full DataSAIL",
}
COLORS = {
    "Random": "#4C78A8",
    "Scaled random-anchor": "#F58518",
    "Scaled stratified-anchor": "#ECA82C",
    "Full DataSAIL": "#54A24B",
}
FULL_STATUS = {
    "qm9": "No successful full run in local results (previous Condor attempt crashed)",
    "pcba": "Not run: standard full DataSAIL considered infeasible",
    "muv": "Failed: segmentation fault after about 331 GB memory use",
}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_runtime_text(path: Path) -> float | None:
    try:
        content = path.read_text()
    except OSError:
        return None
    for line in content.splitlines():
        match = re.match(r"\s*(?:total_runtime_seconds|runtime_seconds)\s*:\s*([0-9.eE+-]+)", line)
        if match:
            return float(match.group(1))
    return None


def runtime_record(dataset: str, method: str) -> tuple[float | None, str, str]:
    """Return end-to-end seconds, source description, and completeness status."""
    dataset_dir = ROOT / "results/experiments" / dataset
    split_meta = ROOT / "results/experiments/splits" / dataset / method / "run_metadata.json"
    if method == "random":
        value = read_runtime_text(dataset_dir / "runtime_random.txt")
        if value is not None:
            return value, "runtime_random.txt (end-to-end)", "recorded_end_to_end"
        meta = read_json(split_meta)
        value = meta.get("runtime_seconds")
        if value is not None:
            return float(value), "random split run_metadata.json (end-to-end)", "recorded_end_to_end"
        return None, "", "not_recorded"
    if method == "datasail_full":
        meta = read_json(split_meta)
        value = meta.get("runtime_seconds")
        if value is not None:
            return float(value), "full split run_metadata.json (end-to-end)", "recorded_end_to_end"
        value = read_runtime_text(dataset_dir / "runtime_full.txt")
        if value is not None:
            return value, "runtime_full.txt (end-to-end)", "recorded_end_to_end"
        return None, "", FULL_STATUS.get(dataset, "not completed or runtime not recorded")

    # Prefer an end-to-end total written by the current configured runner. If this
    # dataset/method has not been rerun since total timing was added, fall back to
    # historical end-to-end timings and label their provenance in the output.
    meta = read_json(ROOT / "results/configured/splits" / dataset / method / "run_metadata.json")
    value = meta.get("total_seconds")
    if value is not None:
        return float(value), "configured run_metadata.json total_seconds (end-to-end)", "recorded_end_to_end_configured"
    suffix = "random" if method == "datasail_scaled_random" else "stratified"
    value = read_runtime_text(dataset_dir / f"runtime_scaled_{suffix}.txt")
    if value is not None:
        return value, f"runtime_scaled_{suffix}.txt (end-to-end; historical selected pipeline)", "recorded_end_to_end_historical"
    meta = read_json(split_meta)
    value = meta.get("total_seconds")
    if value is not None:
        return float(value), "scaled split run_metadata.json total_seconds (end-to-end; earlier extension runner)", "recorded_end_to_end_historical"
    return None, "", "end_to_end_not_recorded_for_configured_run"


def scaled_anchor_runtime(dataset: str, method: str) -> float | None:
    if method not in {"datasail_scaled_random", "datasail_scaled_stratified"}:
        return None
    meta = read_json(ROOT / "results/configured/splits" / dataset / method / "run_metadata.json")
    value = meta.get("datasail_seconds")
    return float(value) if value is not None else None


def load_leakage() -> dict[tuple[str, str], dict]:
    values: dict[tuple[str, str], dict] = {}
    # Original-dataset leakage table, including the recently completed HIV full run.
    original = ANALYSIS / "nn_leakage_results.csv"
    if original.exists():
        df = pd.read_csv(original)
        for _, row in df.iterrows():
            method = str(row["split"])
            if method not in METHODS:
                continue
            values[(str(row["dataset"]), method)] = {
                "mean_max_similarity": row.get("mean_max_similarity"),
                "leakage_fraction_gt_0_7": row.get("leakage_fraction"),
                "query_n": row.get("test_sample_size"),
                "train_n": row.get("train_sample_size"),
                "leakage_source": "nn_leakage_results.csv",
            }
    # Configured random-anchor and stratified-anchor leakage for all ten datasets.
    for dataset in DATASETS:
        path = ANALYSIS / "configured" / f"{dataset}_runtime_leakage.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.dropna(subset=["mean_max_similarity"])
        for _, row in df.iterrows():
            method = str(row["method"])
            if method in {"datasail_scaled_random", "datasail_scaled_stratified"}:
                values[(dataset, method)] = {
                    "mean_max_similarity": row.get("mean_max_similarity"),
                    "leakage_fraction_gt_0_7": row.get("leakage_fraction_gt_0_7"),
                    "query_n": row.get("query_n"),
                    "train_n": row.get("train_n"),
                    "leakage_source": "configured split analysis (seed 42, up to 10k train and 500 test)",
                }
    # Extended-dataset random and completed full DataSAIL measurements.
    for dataset in ["qm9", "pcba", "muv", "esol", "freesolv", "lipophilicity"]:
        path = ANALYSIS / "extended_datasets" / f"{dataset}_nn_leakage.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            method = str(row.get("split_type", ""))
            if method not in {"random", "datasail_full"}:
                continue
            values[(dataset, method)] = {
                "mean_max_similarity": row.get("mean_max_similarity"),
                "leakage_fraction_gt_0_7": row.get("leakage_fraction"),
                "query_n": row.get("query_sample_size"),
                "train_n": row.get("train_size"),
                "leakage_source": f"{dataset}_nn_leakage.csv (seed 42, up to 10k train and 500 test)",
            }
    return values


def main() -> None:
    leakage = load_leakage()
    records = []
    for dataset in DATASETS:
        for method in METHODS:
            leak = leakage.get((dataset, method), {})
            runtime, runtime_source, runtime_status = runtime_record(dataset, method)
            anchor_runtime = scaled_anchor_runtime(dataset, method)
            records.append({
                "dataset": dataset,
                "method": method,
                "method_label": LABELS[method],
                "end_to_end_runtime_seconds": runtime,
                "end_to_end_runtime_source": runtime_source,
                "runtime_status": runtime_status,
                "configured_datasail_anchor_stage_seconds": anchor_runtime,
                "mean_max_train_similarity": leak.get("mean_max_similarity"),
                "fraction_test_gt_0_7_similarity": leak.get("leakage_fraction_gt_0_7"),
                "leakage_query_n": leak.get("query_n"),
                "leakage_train_n": leak.get("train_n"),
                "leakage_source": leak.get("leakage_source", "not available"),
                "full_run_status": FULL_STATUS.get(dataset, "completed" if method != "datasail_full" or (dataset, method) in leakage else "not available"),
            })
    table = pd.DataFrame(records)
    # Quantify changes versus the random baseline, as positive values represent lower leakage.
    random_leak = table[table.method.eq("random")].set_index("dataset")["fraction_test_gt_0_7_similarity"]
    table["leakage_reduction_vs_random"] = table.apply(
        lambda row: random_leak.get(row.dataset, float("nan")) - row.fraction_test_gt_0_7_similarity
        if pd.notna(row.fraction_test_gt_0_7_similarity) else float("nan"), axis=1)
    random_rt = table[table.method.eq("random")].set_index("dataset")["end_to_end_runtime_seconds"]
    table["runtime_ratio_vs_random"] = table.apply(
        lambda row: row.end_to_end_runtime_seconds / random_rt.get(row.dataset)
        if pd.notna(row.end_to_end_runtime_seconds) and pd.notna(random_rt.get(row.dataset)) and random_rt.get(row.dataset) > 0
        else float("nan"), axis=1)

    outdir = ANALYSIS / "runtime_leakage_comparison"
    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / "runtime_leakage_by_dataset_method.csv", index=False)
    wide = table.pivot(index="dataset", columns="method", values="end_to_end_runtime_seconds").reindex(DATASETS)
    wide = wide.rename(columns=LABELS).reset_index()
    status = table[table.method.eq("datasail_full")][["dataset", "full_run_status"]]
    wide = wide.merge(status, on="dataset", how="left")
    wide.to_csv(outdir / "end_to_end_runtime_by_dataset.csv", index=False)
    leakage_wide = table.pivot(index="dataset", columns="method", values="fraction_test_gt_0_7_similarity").reindex(DATASETS)
    leakage_wide = leakage_wide.rename(columns=LABELS).reset_index()
    leakage_wide.to_csv(outdir / "leakage_fraction_by_dataset.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    order = DATASETS
    runtime = table.dropna(subset=["end_to_end_runtime_seconds"])
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(data=runtime, x="dataset", y="end_to_end_runtime_seconds", hue="method_label",
                order=order, hue_order=list(LABELS.values()), palette=COLORS, ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("End-to-end runtime (seconds, log scale)")
    ax.set_xlabel("Dataset")
    ax.set_title("Recorded pipeline runtime (missing measurements omitted)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Pipeline", loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.text(.01, .01, "Configured totals are used when recorded; otherwise historical end-to-end timings are shown and identified in the CSV.", fontsize=9)
    fig.tight_layout(rect=[0, .04, .84, 1])
    fig.savefig(outdir / "end_to_end_runtime_by_dataset.png", dpi=220)
    plt.close(fig)

    leak_plot = table.dropna(subset=["fraction_test_gt_0_7_similarity"])
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(data=leak_plot, x="dataset", y="fraction_test_gt_0_7_similarity", hue="method_label",
                order=order, hue_order=list(LABELS.values()), palette=COLORS, ax=ax)
    ax.set_ylabel("Fraction of test sample with max similarity > 0.7")
    ax.set_xlabel("Dataset")
    ax.set_title("Nearest-neighbor leakage estimate by dataset")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Pipeline", loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.text(.01, .01, "Leakage estimates use up to 10,000 train and 500 test molecules (seed 42); absent full runs are omitted.", fontsize=9)
    fig.tight_layout(rect=[0, .04, .84, 1])
    fig.savefig(outdir / "leakage_fraction_by_dataset.png", dpi=220)
    plt.close(fig)

    reduction = table.dropna(subset=["leakage_reduction_vs_random"])
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(data=reduction[reduction.method.ne("random")], x="dataset", y="leakage_reduction_vs_random",
                hue="method_label", order=order, hue_order=[LABELS[m] for m in METHODS if m != "random"],
                palette={k: COLORS[k] for k in COLORS if k != "Random"}, ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Leakage reduction vs random (fraction; positive is lower leakage)")
    ax.set_xlabel("Dataset")
    ax.set_title("Leakage change relative to random splitting")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Pipeline", loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.text(.01, .01, "Missing bars indicate unavailable full DataSAIL splits or missing leakage measurements.", fontsize=9)
    fig.tight_layout(rect=[0, .04, .84, 1])
    fig.savefig(outdir / "leakage_reduction_vs_random.png", dpi=220)
    plt.close(fig)

    anchor = table.dropna(subset=["configured_datasail_anchor_stage_seconds"])
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(data=anchor, x="dataset", y="configured_datasail_anchor_stage_seconds",
                hue="method_label", order=order,
                hue_order=[LABELS["datasail_scaled_random"], LABELS["datasail_scaled_stratified"]],
                palette={LABELS["datasail_scaled_random"]: COLORS[LABELS["datasail_scaled_random"]],
                         LABELS["datasail_scaled_stratified"]: COLORS[LABELS["datasail_scaled_stratified"]]}, ax=ax)
    ax.set_ylabel("DataSAIL time on anchor subset (seconds)")
    ax.set_xlabel("Dataset")
    ax.set_title("Configured scaled DataSAIL anchor-stage time")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Scaled method", loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(rect=[0, 0, .84, 1])
    fig.savefig(outdir / "scaled_anchor_stage_runtime.png", dpi=220)
    plt.close(fig)

    print(f"Wrote {len(table)} dataset/method rows to {outdir}")
    print("End-to-end runtime records:", int(table.end_to_end_runtime_seconds.notna().sum()))
    print("Leakage records:", int(table.fraction_test_gt_0_7_similarity.notna().sum()))
    print(table[table.method.eq("datasail_full")][["dataset", "end_to_end_runtime_seconds", "runtime_status", "full_run_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
