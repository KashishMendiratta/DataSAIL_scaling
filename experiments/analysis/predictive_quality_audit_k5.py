"""Build focused label-coverage and regression-distribution audits for k=5 splits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.stats import t
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis_common import read_splits, target_columns, smiles_column  # noqa: E402
from src.pipeline_engine import CONFIG, load_dataset  # noqa: E402

OUT = ROOT / "results/analysis/provenance_20261120/predictive_quality_audit_k5"
PREDICTIVE_ROOT = ROOT / "results/analysis/provenance_20261120/predictive_fingerprint/by_dataset"
SEEDS = (42, 101, 202, 303, 404)
REPEATED_METHODS = ("random", "datasail_scaled_random", "datasail_scaled_stratified")
CLASSIFICATION_METRICS = ("roc_auc", "pr_auc", "accuracy", "f1", "mcc")
REGRESSION_DATASETS = ("esol", "freesolv", "lipophilicity", "qm9")
QM9_REPRESENTATIVE_TARGETS = ("mu", "alpha", "gap", "u0")
N_BINS = 10
PSEUDOCOUNT = 0.5


def _load_predictive_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    repeated_paths = sorted(PREDICTIVE_ROOT.glob("*/seed_*/*_predictive_metrics_seed_*.csv"))
    for path in repeated_paths:
        frame = pd.read_csv(path)
        frame["dataset"] = path.parents[1].name
        frame["seed"] = int(path.parent.name.removeprefix("seed_"))
        parts.append(frame)
    if len(repeated_paths) != 50:
        raise ValueError(f"Expected 50 clean repeated predictive files; found {len(repeated_paths)}")
    raw = pd.concat(parts, ignore_index=True)
    if "method" not in raw and "split_type" in raw:
        raw = raw.rename(columns={"split_type": "method"})
    if raw.duplicated(["dataset", "seed", "method", "task"]).any():
        raise ValueError("Predictive metric source files contain duplicate dataset/seed/method/task rows")

    invalid_muv_pr = ((raw.dataset == "muv") &
                      (raw.status == "test_has_one_class_auc_unavailable"))
    corrected = int((invalid_muv_pr & raw.pr_auc.notna()).sum())
    raw.loc[invalid_muv_pr, "pr_auc"] = np.nan
    audit_out = pd.DataFrame([{
        "source_root": str(PREDICTIVE_ROOT.relative_to(ROOT)),
        "source_files": len(repeated_paths),
        "source_rows": len(raw),
        "one_class_muv_rows_pr_auc_set_unavailable": corrected,
        "note": "MUV PR-AUC is unavailable when the observed test labels contain only one class.",
    }])
    full_parts = []
    full_paths = sorted(PREDICTIVE_ROOT.glob("*/full/*_predictive_metrics.csv"))
    for path in full_paths:
        frame = pd.read_csv(path)
        frame["dataset"] = path.parents[1].name
        frame["seed"] = "full_single_run"
        full_parts.append(frame)
    if len(full_paths) != 9:
        raise ValueError(f"Expected 9 clean Full DataSAIL predictive files; found {len(full_paths)}")
    full = pd.concat(full_parts, ignore_index=True)
    if "method" not in full and "split_type" in full:
        full = full.rename(columns={"split_type": "method"})
    return raw, full, audit_out


def _paired_metric_summary(raw: pd.DataFrame) -> pd.DataFrame:
    contrasts = (
        ("scaled_random - random", "datasail_scaled_random", "random"),
        ("scaled_stratified - random", "datasail_scaled_stratified", "random"),
        ("stratified - scaled_random", "datasail_scaled_stratified", "datasail_scaled_random"),
    )
    records = []
    metric_by_dataset = {}
    for dataset, group in raw.groupby("dataset"):
        metric_by_dataset[dataset] = CLASSIFICATION_METRICS if group.roc_auc.notna().any() else ("mae", "rmse", "r2")

    for dataset, metrics in metric_by_dataset.items():
        data = raw[raw.dataset == dataset]
        for contrast, left_method, right_method in contrasts:
            for metric in metrics:
                seed_deltas: list[tuple[int, float, int]] = []
                for seed in sorted(data.seed.unique()):
                    sample = data[data.seed == seed]
                    left = sample[sample.method == left_method].set_index("task")
                    right = sample[sample.method == right_method].set_index("task")
                    shared = left.index.intersection(right.index)
                    if not len(shared):
                        continue
                    x = pd.to_numeric(left.loc[shared, metric], errors="coerce")
                    y = pd.to_numeric(right.loc[shared, metric], errors="coerce")
                    valid = x.notna() & y.notna()
                    n_tasks = int(valid.sum())
                    if n_tasks:
                        seed_deltas.append((int(seed), float((x[valid] - y[valid]).mean()), n_tasks))
                values = np.array([x[1] for x in seed_deltas], dtype=float)
                n = len(values)
                mean = float(values.mean()) if n else np.nan
                sd = float(values.std(ddof=1)) if n > 1 else np.nan
                half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
                records.append({
                    "dataset": dataset,
                    "contrast": contrast,
                    "metric": metric,
                    "n_paired_seeds": n,
                    "paired_delta_mean": mean,
                    "paired_delta_sd": sd,
                    "ci95_low": mean - half if n > 1 else np.nan,
                    "ci95_high": mean + half if n > 1 else np.nan,
                    "mean_paired_tasks_per_seed": float(np.mean([x[2] for x in seed_deltas])) if n else np.nan,
                    "seed_level_deltas": ";".join(f"{seed}:{value:.10g}" for seed, value, _ in seed_deltas),
                    "paired_task_counts": ";".join(f"{seed}:{count}" for seed, _, count in seed_deltas),
                })
    return pd.DataFrame(records)


def _fingerprints_by_id(dataset: str, frame: pd.DataFrame) -> dict[str, object]:
    cfg = CONFIG["defaults"]["fingerprint"]
    generator = GetMorganGenerator(radius=cfg["radius"], fpSize=cfg["bits"])
    smiles_col = smiles_column(dataset, frame)
    ids = frame["mol_id"].astype(str).to_numpy()
    smiles = frame[smiles_col].fillna("").astype(str).to_numpy()
    result: dict[str, object] = {}
    for start in range(0, len(frame), 2000):
        mols = [Chem.MolFromSmiles(value) for value in smiles[start:start + 2000]]
        fps = generator.GetFingerprints(mols, numThreads=1)
        result.update((mol_id, fp) for mol_id, fp in zip(ids[start:start + 2000], fps))
    return result


def _active_neighbor_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_records = []
    cell_records = []
    for dataset in ("hiv", "muv"):
        source, _, _ = load_dataset(dataset)
        fp_by_id = _fingerprints_by_id(dataset, source)
        targets = target_columns(dataset, source)
        runs = [(method, f"seed_{seed}", seed) for method in REPEATED_METHODS for seed in SEEDS]
        runs.append(("datasail_full", "", "full_single_run"))

        for method, run_id, seed in runs:
            splits = read_splits(dataset, method, run_id=run_id or None)
            if splits is None:
                continue
            train, test = splits["train"], splits["test"]
            train_ids = train["mol_id"].astype(str).to_numpy()
            test_ids = test["mol_id"].astype(str).to_numpy()
            train_fps = [fp_by_id[mol_id] for mol_id in train_ids]
            for target in targets:
                if target not in train or target not in test:
                    continue
                train_y = pd.to_numeric(train[target], errors="coerce")
                test_y = pd.to_numeric(test[target], errors="coerce")
                train_observed = train_y.notna().to_numpy()
                test_observed = test_y.notna().to_numpy()
                test_active_rows = np.flatnonzero(test_y.eq(1).fillna(False).to_numpy())
                train_active_rows = np.flatnonzero(train_y.eq(1).fillna(False).to_numpy())
                train_active_fps = [train_fps[i] for i in train_active_rows]
                train_active_ids = train_ids[train_active_rows]
                cell_records.append({
                    "dataset": dataset, "method": method, "seed": seed, "task": target,
                    "n_test_observed": int(test_observed.sum()),
                    "n_test_active": len(test_active_rows),
                    "n_train_observed": int(train_observed.sum()),
                    "n_train_active": len(train_active_rows),
                })

                for test_row in test_active_rows:
                    query_id = test_ids[test_row]
                    query_fp = fp_by_id[query_id]
                    all_scores = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, train_fps), dtype=float)
                    all_idx = int(np.argmax(all_scores))
                    if train_active_fps:
                        active_scores = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, train_active_fps), dtype=float)
                        active_idx = int(np.argmax(active_scores))
                        active_similarity = float(active_scores[active_idx])
                        active_neighbor_id = str(train_active_ids[active_idx])
                    else:
                        active_similarity = np.nan
                        active_neighbor_id = ""
                    raw_records.append({
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "task": target,
                        "test_active_id": query_id,
                        "n_test_active": len(test_active_rows),
                        "n_train_observed": int(train_observed.sum()),
                        "n_train_active": len(train_active_rows),
                        "nearest_train_any_id": str(train_ids[all_idx]),
                        "nearest_train_any_similarity": float(all_scores[all_idx]),
                        "nearest_train_active_id": active_neighbor_id,
                        "nearest_train_active_similarity": active_similarity,
                    })

        del fp_by_id

    raw = pd.DataFrame(raw_records)
    cells = pd.DataFrame(cell_records)
    if raw.empty:
        return raw, cells
    measured = (raw.groupby(["dataset", "method", "seed", "task"], dropna=False)
               .agg(n_test_active_queries=("test_active_id", "nunique"),
                    mean_nearest_any=("nearest_train_any_similarity", "mean"),
                    median_nearest_any=("nearest_train_any_similarity", "median"),
                    q90_nearest_any=("nearest_train_any_similarity", lambda x: x.quantile(.9)),
                    fraction_nearest_any_ge_0_7=("nearest_train_any_similarity", lambda x: (x >= .7).mean()),
                    fraction_nearest_any_ge_0_8=("nearest_train_any_similarity", lambda x: (x >= .8).mean()),
                    mean_nearest_active=("nearest_train_active_similarity", "mean"),
                    median_nearest_active=("nearest_train_active_similarity", "median"),
                    q90_nearest_active=("nearest_train_active_similarity", lambda x: x.quantile(.9)),
                    fraction_nearest_active_ge_0_7=("nearest_train_active_similarity", lambda x: (x >= .7).mean()),
                    fraction_nearest_active_ge_0_8=("nearest_train_active_similarity", lambda x: (x >= .8).mean()))
               .reset_index())
    summary = cells.merge(measured, on=["dataset", "method", "seed", "task"], how="left")
    summary["n_test_active_queries"] = summary["n_test_active_queries"].fillna(0).astype(int)
    return raw, summary


def _quantile_bins(train_values: np.ndarray, heldout_values: np.ndarray, bins: int = N_BINS) -> tuple[float, int]:
    train_values = np.asarray(train_values, dtype=float)
    heldout_values = np.asarray(heldout_values, dtype=float)
    if not len(train_values) or not len(heldout_values):
        return np.nan, 0
    quantiles = np.unique(np.quantile(train_values, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 2:
        return 0.0, 1
    edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
    n_actual = len(edges) - 1
    train_bins = np.clip(np.searchsorted(edges[1:-1], train_values, side="right"), 0, n_actual - 1)
    heldout_bins = np.clip(np.searchsorted(edges[1:-1], heldout_values, side="right"), 0, n_actual - 1)
    p = np.bincount(train_bins, minlength=n_actual).astype(float) + PSEUDOCOUNT
    q = np.bincount(heldout_bins, minlength=n_actual).astype(float) + PSEUDOCOUNT
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q))), n_actual


def _target_distribution_record(dataset: str, method: str, seed: str | int, target: str,
                                split_name: str, train_values: pd.Series, compare_values: pd.Series,
                                representation: str) -> dict:
    train_values = pd.to_numeric(train_values, errors="coerce").dropna().to_numpy(dtype=float)
    compare_values = pd.to_numeric(compare_values, errors="coerce").dropna().to_numpy(dtype=float)
    kl, n_bins = _quantile_bins(train_values, compare_values)
    if len(train_values):
        q05, q95 = np.quantile(train_values, [.05, .95])
        train_lo, train_hi = int(np.sum(train_values < q05)), int(np.sum(train_values > q95))
        test_lo, test_hi = int(np.sum(compare_values < q05)), int(np.sum(compare_values > q95))
    else:
        q05 = q95 = np.nan
        train_lo = train_hi = test_lo = test_hi = 0
    return {
        "dataset": dataset, "method": method, "seed": seed, "target_or_component": target,
        "split": split_name, "representation": representation,
        "n_train_labelled": len(train_values), "n_split_labelled": len(compare_values),
        "n_unique_train_quantile_bins": n_bins, "dirichlet_pseudocount_per_bin": PSEUDOCOUNT,
        "kl_train_to_split": kl, "train_q05": q05, "train_q95": q95,
        "train_bottom_tail_n": train_lo, "train_top_tail_n": train_hi,
        "split_below_train_q05_n": test_lo, "split_above_train_q95_n": test_hi,
        "split_below_train_q05_fraction": test_lo / len(compare_values) if len(compare_values) else np.nan,
        "split_above_train_q95_fraction": test_hi / len(compare_values) if len(compare_values) else np.nan,
        "bottom_tail_ratio_to_5pct": (test_lo / len(compare_values)) / .05 if len(compare_values) else np.nan,
        "top_tail_ratio_to_5pct": (test_hi / len(compare_values)) / .05 if len(compare_values) else np.nan,
    }


def _regression_distribution_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows = []
    pca_rows = []
    for dataset in REGRESSION_DATASETS:
        source, _, _ = load_dataset(dataset)
        targets = target_columns(dataset, source)
        target_subset = list(QM9_REPRESENTATIVE_TARGETS) if dataset == "qm9" else targets
        runs = [(method, f"seed_{seed}", seed) for method in REPEATED_METHODS for seed in SEEDS]
        runs.append(("datasail_full", "", "full_single_run"))

        for method, run_id, seed in runs:
            splits = read_splits(dataset, method, run_id=run_id or None)
            if splits is None:
                continue
            train = splits["train"]
            for split_name in ("val", "test"):
                compare = splits[split_name]
                for target in target_subset:
                    if target not in train or target not in compare:
                        continue
                    target_rows.append(_target_distribution_record(
                        dataset, method, seed, target, split_name,
                        train[target], compare[target], "train-defined target deciles and tails"))

            if dataset == "qm9":
                train_matrix = train[targets].apply(pd.to_numeric, errors="coerce")
                train_mask = train_matrix.notna().all(axis=1).to_numpy()
                if train_mask.sum() < 4:
                    continue
                scaler = StandardScaler().fit(train_matrix.loc[train_mask].to_numpy())
                pca = PCA(n_components=3, svd_solver="full").fit(scaler.transform(train_matrix.loc[train_mask].to_numpy()))
                for split_name in ("val", "test"):
                    compare_matrix = splits[split_name][targets].apply(pd.to_numeric, errors="coerce")
                    compare_mask = compare_matrix.notna().all(axis=1).to_numpy()
                    train_pc = pca.transform(scaler.transform(train_matrix.loc[train_mask].to_numpy()))
                    compare_pc = pca.transform(scaler.transform(compare_matrix.loc[compare_mask].to_numpy()))
                    for component in range(3):
                        record = _target_distribution_record(
                            dataset, method, seed, f"PC{component + 1}", split_name,
                            pd.Series(train_pc[:, component]), pd.Series(compare_pc[:, component]),
                            "training-standardized PCA; PCA fit on complete training targets")
                        record["explained_variance_ratio"] = float(pca.explained_variance_ratio_[component])
                        record["n_train_complete_target_rows"] = int(train_mask.sum())
                        record["n_split_complete_target_rows"] = int(compare_mask.sum())
                        pca_rows.append(record)

    return pd.DataFrame(target_rows), pd.DataFrame(pca_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, full_reference, correction_audit = _load_predictive_metrics()
    raw.to_csv(OUT / "predictive_metrics_seed_raw_corrected.csv", index=False)
    full_reference.to_csv(OUT / "predictive_full_reference.csv", index=False)
    correction_audit.to_csv(OUT / "muv_pr_auc_correction_audit.csv", index=False)
    _paired_metric_summary(raw).to_csv(OUT / "predictive_paired_five_seed_summary_corrected.csv", index=False)

    active_raw, active_summary = _active_neighbor_audit()
    active_raw.to_csv(OUT / "active_test_nearest_neighbor_queries.csv", index=False)
    active_summary.to_csv(OUT / "active_test_nearest_neighbor_summary_by_task.csv", index=False)

    regression, qm9_pca = _regression_distribution_audit()
    regression.to_csv(OUT / "regression_train_quantile_kl_and_tails.csv", index=False)
    qm9_pca.to_csv(OUT / "qm9_pca_distribution_kl_and_tails.csv", index=False)

    manifest = {
        "scope": "k=5 thesis splits; repeated methods use seeds 42, 101, 202, 303, 404",
        "datasets_active_neighbor": ["hiv", "muv"],
        "active_neighbor_method": "exact Morgan radius-2, 2048-bit Tanimoto against all training molecules and same-assay training actives; every observed test active included",
        "datasets_regression": list(REGRESSION_DATASETS),
        "regression_method": "training-defined decile cutoffs, KL(train || held-out) with 0.5 pseudocount per bin, plus bottom/top 5% tail counts relative to training quantiles",
        "qm9_targets": list(QM9_REPRESENTATIVE_TARGETS),
        "qm9_target_count_in_local_file": 19,
        "qm9_pca_method": "StandardScaler and 3-component PCA fit on complete training targets only; validation/test transformed with training fit",
        "pca_bin_count": N_BINS,
        "pca_pseudocount_per_bin": PSEUDOCOUNT,
        "predictive_source": str(PREDICTIVE_ROOT.relative_to(ROOT)),
        "full_datasail_predictive_reference_datasets": sorted(full_reference.dataset.unique().tolist()),
        "full_datasail_included_for_regression": list(REGRESSION_DATASETS),
        "full_datasail_included_for_hiv": True,
        "full_datasail_included_for_muv": True,
        "outputs": [
            "predictive_metrics_seed_raw_corrected.csv",
            "predictive_full_reference.csv",
            "predictive_paired_five_seed_summary_corrected.csv",
            "muv_pr_auc_correction_audit.csv",
            "active_test_nearest_neighbor_queries.csv",
            "active_test_nearest_neighbor_summary_by_task.csv",
            "regression_train_quantile_kl_and_tails.csv",
            "qm9_pca_distribution_kl_and_tails.csv",
        ],
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved k=5 predictive-quality audit to {OUT.relative_to(ROOT)}")
    print(f"MUV one-class rows corrected: {int(correction_audit.iloc[0]['one_class_muv_rows_pr_auc_set_unavailable'])}")
    print(f"Active query rows: {len(active_raw)}; regression target rows: {len(regression)}; QM9 PCA rows: {len(qm9_pca)}")


if __name__ == "__main__":
    main()
