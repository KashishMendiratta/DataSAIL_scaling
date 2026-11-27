#!/usr/bin/env python3
"""Create thesis-, defence-, and application-note-ready figures and tables."""

from pathlib import Path
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
PROV = ROOT / "results/analysis/provenance_20261120"
QUALITY = PROV / "quality_multiseed/by_dataset"
DMPNN = PROV / "dmpnn_k5/by_dataset"
AUDIT = PROV / "predictive_quality_audit_k5"
OUT = ROOT / "results/presentation/provenance_figure_suite_20261123"
FIG = OUT / "figures"
TAB = OUT / "tables"

DATASETS = ["bace", "bbbp", "tox21", "hiv", "qm9", "pcba", "muv", "esol", "freesolv", "lipophilicity"]
DS_LABEL = {d: d.upper() if d in {"bbbp", "hiv", "qm9", "pcba", "muv"} else d.title() for d in DATASETS}
DS_LABEL["tox21"] = "Tox21"
DS_LABEL["freesolv"] = "FreeSolv"
METHOD_LABEL = {
    "random": "Random",
    "datasail_scaled_random": "Scaled, random anchors",
    "datasail_scaled_stratified": "Scaled, stratified anchors",
    "datasail_full": "Full DataSAIL",
}
COLORS = {
    "random": "#7A7A7A",
    "datasail_scaled_random": "#0072B2",
    "datasail_scaled_stratified": "#D55E00",
    "datasail_full": "#009E73",
}


def read(folder, name):
    return pd.read_csv(folder / name)


def concat_dataset_files(base, relative):
    frames = []
    for dataset in DATASETS:
        p = base / dataset / relative
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def leakage_summaries():
    base = PROV / "leakage/by_dataset"
    repeated_leak = concat_dataset_files(base, "repeated/leakage_thresholds_summary.csv")
    repeated_overlap = concat_dataset_files(base, "repeated/structure_overlap_summary.csv")
    full_leak = concat_dataset_files(base, "full/leakage_thresholds_summary.csv")
    full_overlap = concat_dataset_files(base, "full/structure_overlap_summary.csv")
    return (pd.concat([repeated_leak, full_leak], ignore_index=True),
            pd.concat([repeated_overlap, full_overlap], ignore_index=True))


def quality_summary():
    return concat_dataset_files(QUALITY, "distribution_quality_summary.csv")


def dmpnn_paired():
    raw = concat_dataset_files(DMPNN, "metrics.csv")
    raw = raw[raw.split_seed.astype(str).isin(["42", "101", "202", "303", "404"])]
    rows = []
    long = raw.melt(id_vars=["dataset", "method", "split_seed", "task"],
                    value_vars=["roc_auc", "r2"], var_name="metric", value_name="value").dropna(subset=["value"])
    for (ds, metric, method), g in long[long.method != "random"].groupby(["dataset", "metric", "method"]):
        base = long[(long.dataset == ds) & (long.metric == metric) & (long.method == "random")]
        z = g.merge(base, on=["dataset", "split_seed", "task", "metric"], suffixes=("", "_base"))
        seed_delta = (z.assign(delta=z.value-z.value_base).groupby("split_seed").delta.mean())
        n = len(seed_delta)
        mean = seed_delta.mean()
        half = 2.776 * seed_delta.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append([ds, metric, method, n, mean, mean-half, mean+half])
    return pd.DataFrame(rows, columns=["dataset", "metric", "method", "n", "delta", "low", "high"])


def style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ordered(df):
    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], DATASETS, ordered=True)
    return df.sort_values("dataset")


def figure_runtime():
    d = read(PROV / "runtime_multiseed", "runtime_summary.csv")
    methods = ["random", "datasail_scaled_random", "datasail_scaled_stratified", "datasail_full"]
    end = d.pivot(index="dataset", columns="method", values="mean_end_to_end_seconds")
    sd = d.pivot(index="dataset", columns="method", values="sd_end_to_end_seconds")
    small = ["freesolv", "esol", "bace", "bbbp", "lipophilicity", "tox21"]
    large = ["hiv", "muv", "pcba", "qm9"]
    offsets = dict(zip(methods, [-.27, -.09, .09, .27]))

    def time_text(seconds):
        if seconds >= 3600:
            return f"{seconds/3600:.1f} h"
        if seconds >= 120:
            return f"{seconds/60:.1f} min"
        return f"{seconds:.1f} s"

    def plot_small(ax):
        y = np.arange(len(small))
        for method in methods:
            vals = end.reindex(small)[method]
            errs = sd.reindex(small)[method].fillna(0)
            yy = y + offsets[method]
            ax.errorbar(vals, yy, xerr=errs, fmt="o", ms=6, capsize=2, lw=1.2,
                        color=COLORS[method], label=METHOD_LABEL[method], zorder=3)
            for x, yp in zip(vals, yy):
                if pd.notna(x):
                    ax.annotate(time_text(x), (x, yp), xytext=(5, 0), textcoords="offset points",
                                va="center", fontsize=7, color=COLORS[method])
        ax.set_xscale("log")
        ax.set_xlim(.055, 130)
        ax.set_yticks(y, [DS_LABEL[x] for x in small])
        ax.invert_yaxis()
        ax.set_xlabel("Mean end-to-end runtime (seconds, log scale)")
        ax.set_title("Smaller datasets: absolute runtime")

    def plot_large(ax):
        y = np.arange(len(large))
        ratios = end.reindex(large)[methods].div(end.reindex(large)["random"], axis=0)
        for method in methods:
            vals = ratios[method]
            yy = y + offsets[method]
            ax.scatter(vals, yy, s=48, color=COLORS[method], label=METHOD_LABEL[method], zorder=3)
            for ds, x, yp in zip(large, vals, yy):
                if pd.notna(x):
                    ax.annotate(f"{x:.0f}×  ({time_text(end.loc[ds, method])})", (x, yp),
                                xytext=(5, 0), textcoords="offset points", va="center",
                                fontsize=7, color=COLORS[method])
            if method == "datasail_full" and pd.isna(vals.loc["pcba"]):
                ax.annotate("Full unavailable", (1, y[large.index("pcba")] + offsets[method]),
                            xytext=(5, 0), textcoords="offset points", va="center",
                            fontsize=7, color=COLORS[method], fontstyle="italic")
        ax.axvline(1, color="#777777", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(.75, 2200)
        ax.set_xticks([1, 10, 100, 1000], labels=["1×", "10×", "100×", "1000×"])
        ax.set_yticks(y, [DS_LABEL[x] for x in large])
        ax.invert_yaxis()
        ax.set_xlabel("Runtime relative to Random (Random = 1×; log scale)")
        ax.set_title("Larger datasets: normalized runtime")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), gridspec_kw={"width_ratios": [1, 1.2]})
    plot_small(axes[0])
    plot_large(axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4)
    fig.suptitle("End-to-end split-generation runtime by dataset", y=1.02, fontsize=13, fontweight="bold")
    fig.text(.5, .005, "Points show five-run means for Random and scaled methods; Full DataSAIL is a single-run reference.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, .04, 1, .9])
    save(fig, "fig01_runtime_end_to_end_comparison")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    plot_small(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle("End-to-end runtime on smaller datasets", y=.99, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig01a_runtime_small_datasets_absolute")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    plot_large(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.18), ncol=2)
    fig.suptitle("End-to-end runtime on larger datasets", y=.99, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, .12, 1, .95])
    save(fig, "fig01b_runtime_large_datasets_normalized")


def figure_datasail_stage_runtime():
    d = read(PROV / "runtime_multiseed", "runtime_summary.csv")
    methods = ["datasail_scaled_random", "datasail_scaled_stratified", "datasail_full"]
    stage = d.pivot(index="dataset", columns="method", values="mean_datasail_stage_seconds")
    sd = d.pivot(index="dataset", columns="method", values="sd_datasail_stage_seconds")
    small = ["freesolv", "esol", "bace", "bbbp", "lipophilicity", "tox21"]
    large = ["hiv", "muv", "pcba", "qm9"]
    offsets = dict(zip(methods, [-.18, 0, .18]))

    def time_text(seconds):
        if seconds >= 3600:
            return f"{seconds/3600:.1f} h"
        if seconds >= 120:
            return f"{seconds/60:.1f} min"
        return f"{seconds:.1f} s"

    def plot_small(ax):
        y = np.arange(len(small))
        for method in methods:
            vals = stage.reindex(small)[method]
            errs = sd.reindex(small)[method].fillna(0)
            yy = y + offsets[method]
            ax.errorbar(vals, yy, xerr=errs, fmt="o", ms=6, capsize=2, lw=1.2,
                        color=COLORS[method], label=METHOD_LABEL[method], zorder=3)
            for x, yp in zip(vals, yy):
                if pd.notna(x):
                    ax.annotate(time_text(x), (x, yp), xytext=(5, 0), textcoords="offset points",
                                va="center", fontsize=7, color=COLORS[method])
        ax.set_xscale("log")
        ax.set_xlim(.7, 70)
        ax.set_yticks(y, [DS_LABEL[x] for x in small])
        ax.invert_yaxis()
        ax.set_xlabel("Mean DataSAIL-stage runtime (seconds, log scale)")
        ax.set_title("Smaller datasets: absolute stage runtime")

    def plot_large(ax):
        y = np.arange(len(large))
        ratios = stage.reindex(large)[methods].div(stage.reindex(large)["datasail_scaled_random"], axis=0)
        for method in methods:
            vals = ratios[method]
            yy = y + offsets[method]
            ax.scatter(vals, yy, s=48, color=COLORS[method], label=METHOD_LABEL[method], zorder=3)
            for ds, x, yp in zip(large, vals, yy):
                if pd.notna(x):
                    ax.annotate(f"{x:.0f}×  ({time_text(stage.loc[ds, method])})", (x, yp),
                                xytext=(5, 0), textcoords="offset points", va="center",
                                fontsize=7, color=COLORS[method])
            if method == "datasail_full" and pd.isna(vals.loc["pcba"]):
                ax.annotate("Full unavailable", (1, y[large.index("pcba")] + offsets[method]),
                            xytext=(5, 0), textcoords="offset points", va="center",
                            fontsize=7, color=COLORS[method], fontstyle="italic")
        ax.axvline(1, color="#777777", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(.75, 3000)
        ax.set_xticks([1, 10, 100, 1000], labels=["1×", "10×", "100×", "1000×"])
        ax.set_yticks(y, [DS_LABEL[x] for x in large])
        ax.invert_yaxis()
        ax.set_xlabel("Stage runtime relative to scaled-random anchors (baseline = 1×; log scale)")
        ax.set_title("Larger datasets: normalized stage runtime")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), gridspec_kw={"width_ratios": [1, 1.2]})
    plot_small(axes[0])
    plot_large(axes[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3)
    fig.suptitle("DataSAIL-stage runtime by dataset", y=1.02, fontsize=13, fontweight="bold")
    fig.text(.5, .005, "Points show five-run means for scaled methods; Full DataSAIL is a single-run reference.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, .04, 1, .9])
    save(fig, "fig01c_datasail_stage_runtime_comparison")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    plot_small(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle("DataSAIL-stage runtime on smaller datasets", y=.99, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig01d_datasail_stage_small_datasets_absolute")

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    plot_large(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.18), ncol=2)
    fig.suptitle("DataSAIL-stage runtime on larger datasets", y=.99, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, .12, 1, .95])
    save(fig, "fig01e_datasail_stage_large_datasets_normalized")


def figure_separation():
    leak, overlap = leakage_summaries()
    leak = leak[np.isclose(leak.threshold, .7)]
    methods = ["random", "datasail_scaled_random", "datasail_scaled_stratified", "datasail_full"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), sharey=True)
    y = np.arange(len(DATASETS))
    offsets = dict(zip(methods, [-.27, -.09, .09, .27]))
    for ax, metric, title, xlabel in [
        (axes[0], "mean", "Close test-to-train analogues", "Test queries with nearest neighbor similarity > 0.7"),
        (axes[1], "shared_scaffold_test_row_fraction_mean", "Shared scaffold overlap", "Test rows sharing a training scaffold"),
    ]:
        src = leak if metric == "mean" else overlap
        for method in methods:
            x = src[src.method == method].set_index("dataset").reindex(DATASETS)
            vals = x[metric]
            ax.scatter(vals, y + offsets[method], s=32, color=COLORS[method], label=METHOD_LABEL[method], alpha=.95)
        ax.set_xlim(left=0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        ax.set_yticks(y, [DS_LABEL[x] for x in DATASETS])
        ax.invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Scaled splitting reduces two distinct forms of chemical overlap", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig02_chemical_separation")


def figure_tradeoff():
    rt = read(PROV / "runtime_multiseed", "runtime_summary.csv")
    lc, _ = leakage_summaries()
    lc = lc[np.isclose(lc.threshold, .7) & lc.method.ne("datasail_full")]
    lp = lc.pivot(index="dataset", columns="method", values="mean")
    rp = rt.pivot(index="dataset", columns="method", values="mean_end_to_end_seconds")
    rows = []
    for ds in DATASETS:
        for variant, method in [("random", "datasail_scaled_random"), ("stratified", "datasail_scaled_stratified")]:
            rows.append([ds, variant, lp.loc[ds, "random"]-lp.loc[ds, method], rp.loc[ds, method]/rp.loc[ds, "random"]])
    m = pd.DataFrame(rows, columns=["dataset", "variant", "leakage_reduction", "mean_runtime_ratio"])
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for variant, color, marker in [("random", COLORS["datasail_scaled_random"], "o"), ("stratified", COLORS["datasail_scaled_stratified"], "s")]:
        q = m[m.variant == variant]
        ax.scatter(q.mean_runtime_ratio, q.leakage_reduction, s=65, color=color, marker=marker,
                   label=f"Scaled, {variant} anchors", alpha=.9)
        for _, r in q.iterrows():
            ax.annotate(DS_LABEL[r.dataset], (r.mean_runtime_ratio, r.leakage_reduction),
                        xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Mean end-to-end runtime ratio vs Random (log scale)")
    ax.set_ylabel("Reduction in fraction with similarity > 0.7")
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.axhline(0, color="#555555", lw=.8)
    ax.legend(frameon=False)
    ax.set_title("Chemical separation improves, but the practical cost varies by dataset", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig03_runtime_separation_tradeoff")


def figure_predictive():
    classic = read(AUDIT, "predictive_paired_five_seed_summary_corrected.csv")
    classic = classic[(classic.contrast == "scaled_random - random") & classic.metric.isin(["roc_auc", "r2"])]
    classic["model"] = "Morgan model"
    classic = classic.rename(columns={"paired_delta_mean": "delta", "ci95_low": "low", "ci95_high": "high"})
    classic = classic[["dataset", "metric", "model", "delta", "low", "high"]]

    dm = dmpnn_paired()
    dm = dm[(dm.method == "datasail_scaled_random") & dm.metric.isin(["roc_auc", "r2"])]
    dm["model"] = "D-MPNN"
    allv = pd.concat([classic, dm], ignore_index=True)
    allv = allv[allv.dataset.isin(DATASETS)]
    allv["dataset"] = pd.Categorical(allv.dataset, DATASETS, ordered=True)
    panels = [("roc_auc", "Classification: ROC-AUC change"), ("r2", "Regression: R² change")]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1))
    for ax, (metric, title) in zip(axes, panels):
        q = allv[allv.metric == metric].sort_values("dataset")
        if metric == "r2":
            q = q[q.dataset.astype(str) != "qm9"]
        ds = [d for d in DATASETS if d in q.dataset.astype(str).tolist()]
        pos = {d: i for i, d in enumerate(ds)}
        for model, color, off in [("Morgan model", "#6A3D9A", -.12), ("D-MPNN", "#1B9E77", .12)]:
            z = q[q.model == model]
            yy = np.array([pos[str(d)] for d in z.dataset]) + off
            ax.errorbar(z.delta, yy, xerr=[z.delta-z.low, z.high-z.delta], fmt="o", ms=5, capsize=2,
                        color=color, label=model, lw=1.2)
        ax.axvline(0, color="#333333", lw=.9)
        ax.set_yticks(range(len(ds)), [DS_LABEL[d] for d in ds])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Scaled random anchors minus Random (paired five-seed mean)")
        if metric == "r2":
            ax.text(.02, .02, "QM9 omitted: target-wise effects are highly heterogeneous", transform=ax.transAxes,
                    fontsize=7, color="#555555")
    axes[0].legend(frameon=False)
    fig.suptitle("Reduced overlap often produces a harder predictive test set", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig04_predictive_difficulty_cross_model")


def figure_quality():
    q = quality_summary()
    q = q[(q.split == "test") & q.metric.isin(["DRE", "KL"])]
    methods = ["random", "datasail_scaled_random", "datasail_scaled_stratified"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for ax, metric in zip(axes, ["DRE", "KL"]):
        z = q[q.metric == metric].pivot(index="dataset", columns="method", values="mean_across_seeds").reindex(DATASETS)[methods]
        sns.heatmap(z, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f", cbar_kws={"label": f"Mean test {metric}"}, linewidths=.5)
        ax.set_title(f"Target-distribution error: {metric}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([DS_LABEL[d] for d in DATASETS], rotation=0)
        ax.set_xticklabels(["Random", "Scaled random", "Scaled stratified"], rotation=20, ha="right")
    fig.suptitle("Chemical separation and target-distribution preservation are separate objectives", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig05_target_distribution_quality")


def figure_active_leakage():
    d = read(AUDIT, "active_test_nearest_neighbor_summary_by_task.csv")
    d = d.groupby(["dataset", "method", "seed"], as_index=False).agg(
        fraction=("fraction_nearest_active_ge_0_7", "mean"),
        n_queries=("n_test_active_queries", "sum"))
    q = d.groupby(["dataset", "method"], as_index=False).agg(mean=("fraction", "mean"), sd=("fraction", "std"))
    datasets = [x for x in ["hiv", "muv"] if x in q.dataset.unique()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(10.7, 4.2), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        z = q[q.dataset == ds].copy()
        z["label"] = z.method.map(METHOD_LABEL)
        z = z.sort_values("method")
        ax.bar(z.label, z["mean"], yerr=z.sd.fillna(0), color=z.method.map(COLORS), capsize=3)
        ax.set_title(DS_LABEL[ds])
        ax.tick_params(axis="x", rotation=25)
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        ax.set_ylabel("Active test queries with active training neighbor ≥ 0.7")
    fig.suptitle("Active-conditioned overlap also falls under scaled splitting", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig06_active_conditioned_leakage")


def tables():
    rt = read(PROV / "runtime_multiseed", "runtime_summary.csv")
    leak, overlaps = leakage_summaries()
    leak = leak[np.isclose(leak.threshold, .7)][["dataset", "method", "mean", "std", "n"]]
    ov = overlaps[["dataset", "method", "shared_scaffold_test_row_fraction_mean"]]
    summary = rt.merge(leak, on=["dataset", "method"], how="outer").merge(ov, on=["dataset", "method"], how="outer")
    summary = summary.rename(columns={
        "mean": "nn_fraction_above_0_7_mean", "std": "nn_fraction_above_0_7_sd", "n": "n_leakage_runs",
        "shared_scaffold_test_row_fraction_mean": "shared_scaffold_test_fraction_mean",
    })
    summary.to_csv(TAB / "table01_core_evidence_by_dataset_method.csv", index=False)

    pred = read(AUDIT, "predictive_paired_five_seed_summary_corrected.csv")
    pred = pred[pred.metric.isin(["roc_auc", "pr_auc", "r2", "mae"])].copy()
    pred.to_csv(TAB / "table02_predictive_paired_effects.csv", index=False)

    n_full = int(rt.loc[rt.method.eq("datasail_full"), "dataset"].nunique())
    completeness = pd.DataFrame([
        ["Selected five-seed split runs", 150, 150, "All passed split integrity checks"],
        ["Datasets with scaled five-seed evidence", 10, 10, "Random plus two scaled variants"],
        ["Datasets with Full DataSAIL split", n_full, 10, "Unavailable for PCBA in the provenance runtime summary"],
        ["D-MPNN configured runs completed", 94, 94, "Six representative datasets"],
        ["Single-class labelled split-target cells", 24, np.nan, "18 MUV and 6 PCBA; ROC-AUC unavailable"],
    ], columns=["evidence_item", "available_or_observed", "planned_or_total", "interpretation"])
    completeness.to_csv(TAB / "table03_evidence_completeness.csv", index=False)

    claims = pd.DataFrame([
        ["Scaled splitting reduces close chemical analogues", "Primary", "Five-seed nearest-neighbor fraction at >0.7; scaffold overlap", "Does not prove zero leakage or cover every chemical definition"],
        ["Scaled splits are often predictively harder", "Qualified", "Paired Morgan-model and D-MPNN effects", "Dataset-, task-, and model-dependent; n=5 descriptive intervals"],
        ["Scaled DataSAIL expands practical feasibility", "Primary", "Successful scaled runs on all 10 datasets; Full missing on PCBA", "End-to-end preprocessing can still dominate runtime"],
        ["Stratified anchors are universally superior", "Not supported", "Small/uncertain differences in most paired comparisons", "Report exceptions by dataset"],
        ["Scaled DataSAIL is equivalent to Full DataSAIL", "Not supported", "Full is single-run and unavailable on 3 datasets", "No replicated equivalence analysis"],
    ], columns=["claim", "status", "supporting_evidence", "required_caveat"])
    claims.to_csv(TAB / "table04_claim_evidence_caveat_map.csv", index=False)


def report():
    leak, _ = leakage_summaries()
    z = leak[np.isclose(leak.threshold, .7)].pivot(index="dataset", columns="method", values="mean")
    all_lower = int((z.datasail_scaled_random < z.random).sum())
    text = f"""# Results synthesis and presentation guide

## Executive conclusion

The 20 November provenance evidence supports a qualified central result: scaled DataSAIL produces more chemically separated test sets than Random splitting and supplies repeated evidence across all ten datasets. At the 0.7 nearest-neighbor threshold, scaled-random anchors reduced the close-analogue fraction in {all_lower}/10 datasets. The corresponding predictive problem is often harder, which is expected when close train-test analogues are removed, but the effect is dataset-, task-, and model-dependent. Practical runtime must be reported both end-to-end and for the DataSAIL stage because preprocessing dominates on several large datasets.

## Recommended narrative

1. **Motivation:** Random molecular splits permit close analogues and shared scaffolds across train and test, inflating apparent generalization.
2. **Method:** Run DataSAIL on an anchor subset and assign remaining molecules using local neighbor evidence. Keep parameter-selection evidence separate from this provenance result bundle.
3. **Chemical validity:** Lead with Figure 2. It shows the strongest and most consistent result across the ten datasets.
4. **Computational feasibility:** Use Figures 1 and 3 together. Scaled splitting succeeds broadly, but user-visible cost is not captured by the DataSAIL stage alone.
5. **Predictive consequence:** Use Figure 4. Lower predictive scores are evidence that some scaled test sets are harder, not that the splitting method improves the model.
6. **Representativeness:** Use Figure 5 to show that chemical separation and label-distribution preservation are separate objectives.

## Placement by audience

| Output | Thesis | Defence | Application note |
|---|---|---|---|
| Figure 1 end-to-end runtime | Main results | Main runtime slide | Main practical-performance figure |
| Figure 1c DataSAIL-stage runtime | Main results / appendix | Follow-up runtime slide | Algorithm-stage benchmark |
| Figure 2 chemical separation | Main results | Central result slide | Main quality figure |
| Figure 3 runtime–separation trade-off | Discussion | Central synthesis slide | Executive summary figure |
| Figure 4 predictive difficulty | Main results / discussion | One qualification slide | Optional technical appendix |
| Figure 5 distribution quality | Main results / appendix | Backup slide | Selection/monitoring guidance |
| Figure 6 active-conditioned leakage | Main results / appendix | Backup result slide | Technical quality figure |
| Table 1 core evidence | Appendix | Backup | Detailed benchmark table |
| Table 4 claim map | Writing aid | Speaker notes | Claims review |

## Claims to avoid

- Do not call Full DataSAIL a replicated baseline. The provenance runtime summary contains one run on nine datasets and no Full run for PCBA.
- Do not claim universal predictive improvement, universal stratification benefit, or equivalence to Full DataSAIL.
- Do not present stage-only runtime as total runtime.
- Do not convert unavailable Full results or single-class ROC-AUC values to zero.
- Do not average regression errors across targets with different units.

## Statistical framing

Five-seed intervals are descriptive t intervals with no multiplicity adjustment. They quantify split-seed variation, not model-initialization uncertainty. Show per-dataset effects and intervals rather than pooled significance claims. Treat D-MPNN as a cross-model robustness check with one fixed initialization seed per split.
"""
    (OUT / "RESULTS_SYNTHESIS.md").write_text(text, encoding="utf-8")


def manifest():
    items = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())
    (OUT / "manifest.json").write_text(json.dumps({
        "created": "2026-11-23", "source_root": str(PROV.relative_to(ROOT)),
        "scope": "10 one-dimensional molecular datasets; five split seeds; Full DataSAIL descriptive where available",
        "files": items,
    }, indent=2), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    style()
    figure_runtime()
    figure_datasail_stage_runtime()
    figure_separation()
    figure_tradeoff()
    figure_predictive()
    figure_quality()
    figure_active_leakage()
    tables()
    report()
    manifest()
    print(OUT)


if __name__ == "__main__":
    main()
