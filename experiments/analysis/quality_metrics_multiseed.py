"""Cached multi-seed split-quality analyses for DataSAIL scaling experiments.

Computes source-label distribution ratio error (DRE), KL divergence, and
anchor/source pairwise-similarity preservation for the selected split runs.
Source datasets are loaded once and Morgan fingerprints are generated once per
selected molecule ID, then reused across methods and seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def mean_sampled_pair_similarity(fps: np.ndarray, n_pairs: int, seed: int) -> float:
    """Estimate mean Tanimoto similarity from sampled non-self pairs."""
    if len(fps) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(len(fps), size=n_pairs)
    j = rng.integers(len(fps), size=n_pairs)
    same = i == j
    while same.any():
        j[same] = rng.integers(len(fps), size=int(same.sum()))
        same = i == j
    sims: list[float] = []
    for start in range(0, n_pairs, 10_000):
        x = fps[i[start : start + 10_000]].astype(np.uint32)
        y = fps[j[start : start + 10_000]].astype(np.uint32)
        inter = np.sum(x * y, axis=1)
        union = np.sum((x + y) > 0, axis=1)
        vals = np.divide(inter, union, out=np.ones_like(inter, dtype=float), where=union > 0)
        sims.extend(vals.tolist())
    return float(np.mean(sims))


def paired_summary(frame: pd.DataFrame, group_cols: list[str], value_col: str,
                   method_a: str, method_b: str, contrast: str,
                   run_col: str = "run_id") -> list[dict]:
    """Summarize paired method deltas with the run/seed as the replicate."""
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        wide = group.pivot_table(index=run_col, columns="method", values=value_col, aggfunc="mean")
        if method_a not in wide or method_b not in wide:
            continue
        delta = (wide[method_a] - wide[method_b]).dropna()
        n = len(delta)
        mean = float(delta.mean()) if n else float("nan")
        sd = float(delta.std(ddof=1)) if n > 1 else float("nan")
        half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
        row = dict(zip(group_cols, keys))
        row.update({
            "contrast": contrast,
            "n_paired_runs": n,
            "mean_delta_a_minus_b": mean,
            "sd_delta": sd,
            "ci95_low": mean - half if n > 1 else float("nan"),
            "ci95_high": mean + half if n > 1 else float("nan"),
            "run_level_deltas": ";".join(f"{k}:{v:.9g}" for k, v in delta.items()),
        })
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Repository root (defaults to the parent repository).")
    parser.add_argument("--datasets", nargs="+", help="Dataset names; defaults to all configured datasets.")
    parser.add_argument("--methods", nargs="+", default=["random", "datasail_scaled_random", "datasail_scaled_stratified"])
    parser.add_argument("--run-ids", nargs="+", default=["seed_42", "seed_101", "seed_202", "seed_303", "seed_404"])
    parser.add_argument("--similarity-pairs", type=int, default=100_000)
    parser.add_argument("--anchor-sample-max", type=int, default=5_000)
    parser.add_argument("--pair-sample-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--revalidate-source-smiles", action="store_true",
                        help="Run RDKit validation once per source instead of trusting hash-matched run metadata.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory; defaults under results/analysis/unified.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    sys.path.insert(0, str(root))
    from src import pipeline_engine as pe
    from src.analysis_common import _distribution, fingerprint_matrix, target_columns
    from rdkit import Chem

    datasets = args.datasets or list(pe.CONFIG["datasets"])
    unknown = sorted(set(datasets) - set(pe.CONFIG["datasets"]))
    if unknown:
        parser.error(f"Unknown datasets: {unknown}")
    valid_methods = {"random", "datasail_scaled_random", "datasail_scaled_stratified", "datasail_full"}
    if set(args.methods) - valid_methods:
        parser.error(f"Unknown methods: {sorted(set(args.methods)-valid_methods)}")
    if args.similarity_pairs < 1 or args.anchor_sample_max < 2:
        parser.error("similarity-pairs must be positive and anchor-sample-max at least 2")

    output_dir = (args.output_dir or root / "results/analysis/unified/quality_multiseed").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_root = root / "results/experiments/unified/splits"
    quality_rows: list[dict] = []
    anchor_rows: list[dict] = []

    for dataset in datasets:
        print(f"Loading {dataset} source and selected run metadata once", flush=True)
        spec = pe.CONFIG["datasets"][dataset]
        source_path = root / spec["path"]
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        task_type = "regression" if spec["task_type"] == "regression" else "classification"

        # Resolve all split/metadata/anchor files before reading source labels.
        selected: dict[tuple[str, str], dict] = {}
        selected_ids: set[str] = set()
        invalid_id_sets = []
        for run_id in args.run_ids:
            for method in args.methods:
                folder = split_root / dataset / method
                folder = folder / run_id if run_id != "baseline" else folder
                meta_path = folder / "run_metadata.json"
                if not meta_path.is_file():
                    raise FileNotFoundError(f"Missing run metadata: {meta_path}")
                metadata = json.loads(meta_path.read_text())
                if metadata.get("source_sha256") != source_hash:
                    raise ValueError(f"Source hash mismatch for {dataset}/{method}/{run_id}; cannot reuse invalid-ID metadata")
                if "invalid_ids" not in metadata:
                    raise ValueError(f"Run metadata has no invalid_ids list: {meta_path}")
                invalid_id_sets.append(set(map(str, metadata["invalid_ids"])))
                split_frames = {}
                for split in ("train", "val", "test"):
                    path = folder / f"{split}.csv"
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing split file: {path}")
                    split_frames[split] = pd.read_csv(path)
                anchors = []
                anchor_path = folder / "anchor_manifest.json"
                if method.startswith("datasail_scaled_"):
                    if not anchor_path.is_file():
                        raise FileNotFoundError(f"Missing anchor manifest: {anchor_path}")
                    anchors = list(map(str, json.loads(anchor_path.read_text()).get("anchor_ids", [])))
                    selected_ids.update(anchors)
                selected[(run_id, method)] = {"folder": folder, "metadata": metadata,
                                               "frames": split_frames, "anchors": anchors}
        if not invalid_id_sets or any(ids != invalid_id_sets[0] for ids in invalid_id_sets[1:]):
            raise ValueError(f"Inconsistent invalid-ID lists across selected {dataset} runs")
        raw = pd.read_csv(source_path, compression=spec.get("compression"))
        source_rows = len(raw)
        smiles_source_col = spec["smiles_column"]
        raw_id_col = spec.get("id_column")
        if raw_id_col and raw_id_col in raw:
            ids = raw[raw_id_col].astype("string")
        else:
            ids = pd.Series([f"{dataset}:row:{i}" for i in range(source_rows)], dtype="string")
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"{dataset}: source IDs are missing or duplicated")
        raw = raw.copy()
        raw["_datasail_id"] = ids.to_numpy()
        raw["mol_id"] = ids.to_numpy()
        raw["_datasail_smiles"] = raw[smiles_source_col].astype("string")
        invalid_ids = invalid_id_sets[0]
        id_set = set(raw["_datasail_id"].astype(str))
        if not invalid_ids.issubset(id_set):
            raise ValueError(f"{dataset}: metadata invalid IDs are not present in current source")
        if args.revalidate_source_smiles:
            valid_mask = raw["_datasail_smiles"].fillna("").map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
            observed_invalid = set(raw.loc[~valid_mask, "_datasail_id"].astype(str))
            if observed_invalid != invalid_ids:
                raise ValueError(f"{dataset}: RDKit invalid IDs differ from run metadata")
            source = raw.loc[valid_mask].reset_index(drop=True)
        else:
            source = raw.loc[~raw["_datasail_id"].astype(str).isin(invalid_ids)].reset_index(drop=True)
        expected_valid_n = len(raw) - len(invalid_ids)
        if len(source) != expected_valid_n:
            raise ValueError(f"{dataset}: valid row count does not match recorded invalid IDs")
        targets = target_columns(dataset, source)
        id_col = "_datasail_id"
        smiles_col = "_datasail_smiles"
        source_ids = source[id_col].astype(str).to_numpy()
        source_id_set = set(source_ids)
        unknown_split_ids = set(selected_ids) - source_id_set
        if unknown_split_ids:
            raise ValueError(f"{dataset}: {len(unknown_split_ids)} anchor IDs absent from hash-verified valid source")
        for (run_id, method), item in selected.items():
            frames = item["frames"]
            if any("mol_id" not in part for part in frames.values()):
                raise ValueError(f"{dataset}/{method}/{run_id}: split is missing mol_id")
            split_id_lists = [part["mol_id"].astype(str).tolist() for part in frames.values()]
            split_ids = [mol_id for ids in split_id_lists for mol_id in ids]
            if len(split_ids) != len(set(split_ids)) or set(split_ids) != source_id_set:
                raise ValueError(f"{dataset}/{method}/{run_id}: splits do not partition the hash-verified valid source IDs")
        # Sample one source reference per dataset and cache fingerprints only for
        # that reference plus the union of anchors used by selected runs.
        source_rng = np.random.default_rng(args.pair_sample_seed)
        source_idx = np.sort(source_rng.choice(len(source), min(len(source), args.anchor_sample_max), replace=False))
        source_sample_ids = source_ids[source_idx].tolist()
        selected_ids.update(source_sample_ids)
        cache_ids = sorted(selected_ids)
        cache_smiles = source.set_index(id_col)[smiles_col]
        smiles = cache_smiles.loc[cache_ids].fillna("").astype(str).tolist()
        fps = fingerprint_matrix(smiles, args.n_jobs)
        fp_by_id = {mol_id: fps[i] for i, mol_id in enumerate(cache_ids)}
        source_fps = np.vstack([fp_by_id[mol_id] for mol_id in source_sample_ids])
        source_similarity = mean_sampled_pair_similarity(source_fps, args.similarity_pairs, args.pair_sample_seed)

        for (run_id, method), item in selected.items():
            md = item["metadata"]
            run_seed = md.get("seed")
            frames = item["frames"]
            for target in targets:
                source_dist, source_n = _distribution(source[target], task_type, source[target])
                if not source_dist:
                    continue
                for split, part in frames.items():
                    if target not in part:
                        continue
                    part_dist, part_n = _distribution(part[target], task_type, source[target])
                    keys = set(source_dist) | set(part_dist)
                    if not part_dist:
                        dre = kl = float("nan")
                    else:
                        dre_vals = [abs(part_dist.get(k, 0.0) - source_dist.get(k, 0.0)) /
                                    (source_dist.get(k, 0.0) + 1e-12)
                                    for k in keys if source_dist.get(k, 0.0) > 0]
                        dre = float(np.mean(dre_vals)) if dre_vals else float("nan")
                        kl = float(sum(p * np.log(p / max(part_dist.get(k, 0.0), 1e-12))
                                       for k, p in source_dist.items() if p > 0))
                    quality_rows.extend([
                        {"dataset": dataset, "method": method, "run_id": run_id, "seed": run_seed,
                         "split": split, "target": target, "metric": "DRE", "value": dre,
                         "reference_n": source_n, "split_n": part_n, "task_type": task_type},
                        {"dataset": dataset, "method": method, "run_id": run_id, "seed": run_seed,
                         "split": split, "target": target, "metric": "KL", "value": kl,
                         "reference_n": source_n, "split_n": part_n, "task_type": task_type},
                    ])
            if method.startswith("datasail_scaled_"):
                anchor_ids = item["anchors"]
                rng = np.random.default_rng(args.pair_sample_seed + int(run_seed or 0))
                if len(anchor_ids) > args.anchor_sample_max:
                    anchor_ids = sorted(rng.choice(anchor_ids, args.anchor_sample_max, replace=False).tolist())
                anchor_fps = np.vstack([fp_by_id[mol_id] for mol_id in anchor_ids])
                anchor_similarity = mean_sampled_pair_similarity(anchor_fps, args.similarity_pairs,
                                                                   args.pair_sample_seed)
                anchor_rows.append({"dataset": dataset, "method": method, "run_id": run_id,
                                    "seed": run_seed, "source_n": len(source),
                                    "anchor_n": len(item["anchors"]),
                                    "anchor_sample_n": len(anchor_ids),
                                    "anchor_ratio": md.get("sample_ratio"),
                                    "sampled_pairs": args.similarity_pairs,
                                    "source_mean_similarity": source_similarity,
                                    "anchor_mean_similarity": anchor_similarity,
                                    "absolute_difference": abs(source_similarity - anchor_similarity),
                                    "pair_sample_seed": args.pair_sample_seed})
        del fp_by_id, fps, cache_smiles
        print(f"Finished {dataset}: {len(selected)} split runs; {len(source)} valid source rows", flush=True)

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(output_dir / "distribution_quality_raw.csv", index=False)
    per_seed = quality.groupby(["dataset", "method", "run_id", "seed", "split", "metric"], dropna=False).agg(
        mean_across_targets=("value", "mean"), sd_across_targets=("value", "std"),
        n_targets=("target", "nunique")).reset_index()
    per_seed.to_csv(output_dir / "distribution_quality_per_seed_summary.csv", index=False)
    summary = per_seed.groupby(["dataset", "method", "split", "metric"], dropna=False).agg(
        mean_across_seeds=("mean_across_targets", "mean"), sd_across_seeds=("mean_across_targets", "std"),
        n_seeds=("seed", "nunique"), mean_targets_per_seed=("n_targets", "mean")).reset_index()
    summary.to_csv(output_dir / "distribution_quality_summary.csv", index=False)

    contrasts = []
    for metric in ("DRE", "KL"):
        subset = per_seed[per_seed.metric == metric]
        for a, b, label in [
            ("datasail_scaled_random", "random", "scaled_random - random"),
            ("datasail_scaled_stratified", "random", "scaled_stratified - random"),
            ("datasail_scaled_stratified", "datasail_scaled_random", "stratified - scaled_random"),
        ]:
            contrasts.extend(paired_summary(subset, ["dataset", "split", "metric"],
                                            "mean_across_targets", a, b, label))
    pd.DataFrame(contrasts).to_csv(output_dir / "distribution_quality_paired_contrasts.csv", index=False)

    anchors = pd.DataFrame(anchor_rows)
    anchors.to_csv(output_dir / "anchor_similarity_preservation_raw.csv", index=False)
    if anchors.empty:
        pd.DataFrame(columns=[
            "dataset", "method", "mean_absolute_difference", "sd_absolute_difference",
            "n_seeds", "mean_source_similarity", "mean_anchor_similarity",
        ]).to_csv(output_dir / "anchor_similarity_preservation_summary.csv", index=False)
        pd.DataFrame().to_csv(
            output_dir / "anchor_similarity_preservation_paired_contrasts.csv", index=False)
    else:
        anchor_summary = anchors.groupby(["dataset", "method"], dropna=False).agg(
            mean_absolute_difference=("absolute_difference", "mean"),
            sd_absolute_difference=("absolute_difference", "std"),
            n_seeds=("seed", "nunique"), mean_source_similarity=("source_mean_similarity", "mean"),
            mean_anchor_similarity=("anchor_mean_similarity", "mean")).reset_index()
        anchor_summary.to_csv(output_dir / "anchor_similarity_preservation_summary.csv", index=False)
        anchor_contrasts = paired_summary(anchors, ["dataset"], "absolute_difference",
                                          "datasail_scaled_stratified", "datasail_scaled_random",
                                          "stratified - scaled_random")
        pd.DataFrame(anchor_contrasts).to_csv(
            output_dir / "anchor_similarity_preservation_paired_contrasts.csv", index=False)
    print(f"Saved multi-seed quality tables to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
