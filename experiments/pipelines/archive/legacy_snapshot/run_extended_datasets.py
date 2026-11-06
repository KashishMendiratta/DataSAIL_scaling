"""Run baselines and scaled DataSAIL experiments for more datasets"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.neighbors import NearestNeighbors

# Emit Python tracebacks if native code (for example RDKit/DataSAIL dependencies)
# terminates the interpreter with a fatal signal such as SIGSEGV.
faulthandler.enable()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONFIG = {
    "qm9": ("data/raw/moleculenet/qm9.csv", None),
    "hiv": ("data/raw/moleculenet/hiv.csv", None),
    "pcba": ("data/raw/moleculenet/pcba.csv.gz", "gzip"),
    "muv": ("data/raw/moleculenet/muv.csv.gz", "gzip"),
    "esol": ("data/raw/moleculenet/delaney-processed.csv", None),
    "freesolv": ("data/raw/moleculenet/SAMPL.csv", None),
    "lipophilicity": ("data/raw/moleculenet/Lipophilicity.csv", None),
}
SOURCE_URLS = {
    "qm9": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
    "hiv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/hiv.csv",
    "pcba": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz",
    "muv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "freesolv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
    "lipophilicity": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
}
RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
DATASET_SCHEMA = {
    "qm9": {"smiles": "smiles", "id": "mol_id", "task": "regression", "targets": None},
    "hiv": {"smiles": "smiles", "id": "mol_id", "task": "classification", "targets": ["HIV_active"]},
    "pcba": {"smiles": "smiles", "id": "mol_id", "task": "classification", "targets": None},
    "muv": {"smiles": "smiles", "id": "mol_id", "task": "classification", "targets": None},
    "esol": {"smiles": "smiles", "id": "Compound ID", "task": "regression", "targets": ["measured log solubility in mols per litre"]},
    "freesolv": {"smiles": "smiles", "id": "iupac", "task": "regression", "targets": ["expt"]},
    "lipophilicity": {"smiles": "smiles", "id": "CMPD_CHEMBLID", "task": "regression", "targets": ["exp"]},
}
RDLogger.DisableLog("rdApp.*")


def reproducibility_metadata(dataset: str) -> dict:
    relative_path, _ = CONFIG[dataset]
    source_path = ROOT / relative_path
    digest = hashlib.sha256()
    with source_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    versions = {}
    for package in ("datasail", "numpy", "pandas", "scikit-learn", "rdkit"):
        versions[package] = importlib.metadata.version(package)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"source_path": relative_path, "source_url": SOURCE_URLS[dataset],
            "source_sha256": digest.hexdigest(), "git_commit": commit, "git_worktree_dirty": dirty,
            "python_version": platform.python_version(), "package_versions": versions,
            "fingerprint": {"type": "Morgan/ECFP", "radius": 2, "bits": 2048}}


def load_valid_dataset(dataset: str) -> tuple[pd.DataFrame, int]:
    path, compression = CONFIG[dataset]
    df = pd.read_csv(ROOT / path, compression=compression)
    input_rows = len(df)
    schema = DATASET_SCHEMA[dataset]
    smiles_col = schema["smiles"]
    if smiles_col not in df:
        raise ValueError(f"{dataset}: missing SMILES column {smiles_col!r}")
    if smiles_col != "smiles":
        df["smiles"] = df[smiles_col]
    id_col = schema["id"]
    if "mol_id" not in df:
        df["mol_id"] = df[id_col].astype(str) if id_col in df else [f"{dataset}_{i}" for i in range(len(df))]
    valid = [Chem.MolFromSmiles(s) is not None for s in df["smiles"].fillna("").astype(str)]
    return df.loc[valid].reset_index(drop=True), input_rows


def fast_fingerprints(smiles: list[str], chunk_size: int = 2000, n_jobs: int = 1) -> np.ndarray:
    """Compute 2048-bit radius-2 Morgan fingerprints with RDKit's batch API."""
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    chunks = []
    for start in range(0, len(smiles), chunk_size):
        mols = [Chem.MolFromSmiles(s) for s in smiles[start:start + chunk_size]]
        fps = generator.GetFingerprints(mols, numThreads=0 if n_jobs < 1 else n_jobs)
        block = np.vstack([
            # Interpret the fingerprint string as ASCII byte values before
            # converting "0"/"1" to numeric bits. Casting S1 directly to
            # int8 parses the characters as digits, then subtracting 48 turns
            # every bit into -48/-47; sklearn treats both as True and Jaccard
            # distances collapse to zero.
            np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")
            for fp in fps
        ])
        chunks.append(block)
        print(f"Fingerprinting: {min(start + chunk_size, len(smiles))}/{len(smiles)}", flush=True)
    return np.vstack(chunks)


def save_split_outputs(df: pd.DataFrame, labels: np.ndarray, split_type: str,
                       confidence: np.ndarray | None = None) -> dict:
    outdir = ROOT / "results/experiments/splits" / df.attrs["dataset"] / split_type
    outdir.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    data["split"] = labels
    counts = Counter(labels)
    for split in RATIOS:
        data.loc[data["split"] == split].drop(columns="split").to_csv(outdir / f"{split}.csv", index=False)
    if confidence is not None:
        key = "mol_id" if "mol_id" in data.columns else "smiles"
        pd.DataFrame({key: data[key], "split": labels, "confidence": confidence}).to_csv(
            outdir / "confidence_scores.csv", index=False
        )
    return {"split_counts": {s: int(counts.get(s, 0)) for s in RATIOS},
            "split_fractions": {s: float(counts.get(s, 0) / len(df)) for s in RATIOS}}


def run_random(dataset: str, seed: int = 42) -> dict:
    started = time.time()
    df, input_rows = load_valid_dataset(dataset)
    df.attrs["dataset"] = dataset
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_train = int(np.floor(len(df) * RATIOS["train"]))
    n_val = int(np.floor(len(df) * RATIOS["val"]))
    labels = np.empty(len(df), dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:]] = "test"
    output = ROOT / "results/experiments/splits" / dataset / "random"
    result = save_split_outputs(df, labels, "random")
    result.update({"dataset": dataset, "seed": seed, "target_ratios": RATIOS,
                   "n_input_rows": input_rows,
                   "n_valid_smiles": len(df), "method": "random", "runtime_seconds": time.time() - started})
    result.update(reproducibility_metadata(dataset))
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def _nearest_assignment_ablation(anchor_fps: np.ndarray, anchor_splits: np.ndarray,
                                 query_fps: np.ndarray, n_total: int, k: int,
                                 n_jobs: int) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Share one neighbor search across naive, soft-balanced and quota methods."""
    nn = NearestNeighbors(n_neighbors=k, metric="jaccard", n_jobs=n_jobs).fit(anchor_fps)
    distances, neighbors = nn.kneighbors(query_fps)
    names = list(RATIOS)
    wanted = {s: int(np.floor(n_total * RATIOS[s])) for s in names}
    wanted[names[0]] += n_total - sum(wanted.values())
    hard_counts = Counter(anchor_splits)
    slots = {s: wanted[s] - hard_counts[s] for s in names}
    if any(v < 0 for v in slots.values()):
        raise ValueError(f"Anchor split exceeds desired final quotas: {slots}")
    soft_counts = Counter(anchor_splits)
    soft_total = len(anchor_splits)
    outputs = {key: np.empty(len(query_fps), dtype=object)
               for key in ("datasail_scaled_naive", "datasail_scaled_naive_confidence",
                           "datasail_scaled_confidence", "datasail_scaled_random")}
    confidence = {key: np.empty(len(query_fps), dtype=float)
                  for key in ("datasail_scaled_naive_confidence", "datasail_scaled_confidence")}

    for i, (drow, nrow) in enumerate(zip(distances, neighbors)):
        labels = anchor_splits[nrow]
        vote_counts = Counter(labels)
        naive, votes = vote_counts.most_common(1)[0]
        outputs["datasail_scaled_naive"][i] = naive
        outputs["datasail_scaled_naive_confidence"][i] = naive
        confidence["datasail_scaled_naive_confidence"][i] = votes / k

        # Preserve the existing soft balance correction as an ablation.
        soft_scores = {}
        for split in names:
            mask = labels == split
            similarity = float(np.mean(1.0 - drow[mask])) if np.any(mask) else 0.0
            current_ratio = soft_counts[split] / soft_total
            factor = RATIOS[split] / (current_ratio + 1e-10)
            soft_scores[split] = similarity * factor ** 2
        ordered = sorted(soft_scores.values(), reverse=True)
        soft = max(soft_scores, key=soft_scores.get)
        outputs["datasail_scaled_confidence"][i] = soft
        confidence["datasail_scaled_confidence"][i] = (
            (ordered[0] - ordered[1]) / ordered[0] if len(ordered) > 1 and ordered[0] > 0 else 0.5
        )
        soft_counts[soft] += 1
        soft_total += 1

        eligible = [s for s in names if slots[s] > 0]
        if not eligible:
            raise RuntimeError("Quota assignment ran out of split capacity")
        affinity = {s: float(np.mean(1.0 - drow[labels == s])) if np.any(labels == s) else 0.0
                    for s in eligible}
        quota = max(eligible, key=affinity.get)
        outputs["datasail_scaled_random"][i] = quota
        slots[quota] -= 1
        if i and i % 50000 == 0:
            print(f"Assignment ablation: {i}/{len(query_fps)}", flush=True)

    if any(slots.values()):
        raise RuntimeError(f"Quota assignment left slots unfilled: {slots}")
    return {key: (values, confidence.get(key)) for key, values in outputs.items()}


def run_scaled(dataset: str, sample_ratio: float = 0.02, seed: int = 42,
               k: int = 5, n_jobs: int = 1, reuse_anchor: bool = False) -> dict:
    if not 0 < sample_ratio < 1:
        raise ValueError("sample_ratio must be between 0 and 1")
    started = time.time()
    df, input_rows = load_valid_dataset(dataset)
    df.attrs["dataset"] = dataset
    fps = fast_fingerprints(df["smiles"].fillna("").astype(str).tolist(), n_jobs=n_jobs)
    rng = np.random.default_rng(seed)
    sample_n = max(10, min(len(df) - 1, int(len(df) * sample_ratio)))
    sampled_pos = np.sort(rng.choice(len(df), sample_n, replace=False))
    sampled_mask = np.zeros(len(df), dtype=bool)
    sampled_mask[sampled_pos] = True
    remaining_pos = np.flatnonzero(~sampled_mask)
    sampled = df.iloc[sampled_pos]

    work = ROOT / "data/temp"
    work.mkdir(parents=True, exist_ok=True)
    ds_input = work / f"{dataset}_extended_sampled.csv"
    ds_input_df = pd.DataFrame({"id": [f"mol_{i}" for i in sampled_pos], "smiles": sampled.smiles.values})
    ds_input_df.to_csv(ds_input, index=False)
    ds_out = ROOT / "results/experiments" / dataset / "datasail_scaled_random"
    ds_out.mkdir(parents=True, exist_ok=True)
    split_dir = ds_out / "C1e"
    existing = sorted(split_dir.glob("Molecule_*_splits.tsv")) if split_dir.exists() else []
    t0 = time.time()
    if reuse_anchor and existing:
        split_file = existing[-1]
        split_rows = pd.read_csv(split_file, sep="\t")
        expected_ids = set(ds_input_df["id"])
        if set(split_rows["ID"].astype(str)) != expected_ids:
            raise ValueError("Existing DataSAIL anchor IDs do not match this seed/sample; rerun without --reuse-anchor")
        datasail_seconds = 0.0
    else:
        from datasail.sail import sail
        sail(e_type="M", e_data=str(ds_input), techniques=["C1e"], e_sim="ecfp", e_clusters=10,
             splits=list(RATIOS.values()), names=list(RATIOS), output=str(ds_out), verbosity="W",
             runs=1, threads=n_jobs)
        datasail_seconds = time.time() - t0
        split_files = sorted((ds_out / "C1e").glob("Molecule_*_splits.tsv"))
        if not split_files:
            raise FileNotFoundError(f"No DataSAIL split TSV under {ds_out / 'C1e'}")
        split_file = split_files[-1]
        split_rows = pd.read_csv(split_file, sep="\t")
    split_by_id = dict(zip(split_rows["ID"].astype(str), split_rows["Split"].astype(str)))
    anchor_splits = np.array([split_by_id[f"mol_{i}"] for i in sampled_pos])
    if set(anchor_splits) - set(RATIOS):
        raise ValueError(f"Unexpected DataSAIL split values: {set(anchor_splits) - set(RATIOS)}")

    t0 = time.time()
    extensions = _nearest_assignment_ablation(fps[sampled_pos], anchor_splits, fps[remaining_pos],
                                               len(df), k, n_jobs)
    assignment_seconds = time.time() - t0
    all_meta = {}
    for split_type, (extension, conf) in extensions.items():
        labels = np.empty(len(df), dtype=object)
        labels[sampled_pos] = anchor_splits
        labels[remaining_pos] = extension
        confidence_all = None
        if conf is not None:
            confidence_all = np.full(len(df), np.nan, dtype=float)
            confidence_all[remaining_pos] = conf
        meta = save_split_outputs(df, labels, split_type, confidence_all)
        if conf is not None:
            meta["confidence_summary"] = {
                "mean": float(np.mean(conf)), "std": float(np.std(conf)),
                "min": float(np.min(conf)), "fraction_below_0_5": float(np.mean(conf < 0.5)),
                "interpretation": "uncalibrated heuristic; do not treat as probability",
            }
        meta.update({"dataset": dataset, "method": split_type, "sample_ratio": sample_ratio,
                     "n_datasail_anchor": sample_n, "seed": seed, "k": k, "n_jobs": n_jobs,
                     "target_ratios": RATIOS, "downsampling": "uniform random without replacement",
                     "n_valid_smiles": len(df), "n_input_rows": input_rows,
                     "n_invalid_smiles": input_rows - len(df),
                     "datasail_seconds": datasail_seconds,
                     "assignment_search_and_ablation_seconds": assignment_seconds,
                     "total_seconds": time.time() - started})
        meta.update(reproducibility_metadata(dataset))
        outdir = ROOT / "results/experiments/splits" / dataset / split_type
        (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        if conf is not None:
            (outdir / "confidence_metrics.json").write_text(json.dumps(meta["confidence_summary"], indent=2) + "\n")
        all_meta[split_type] = meta
    (ds_out / "anchor_manifest.json").write_text(json.dumps({
        "dataset": dataset, "seed": seed, "sample_ratio": sample_ratio,
        "split_file": str(split_file.relative_to(ROOT)), "n_anchor": sample_n,
        "anchor_ids": [f"mol_{i}" for i in sampled_pos],
        "anchor_split_counts": dict(Counter(anchor_splits)),
    }, indent=2) + "\n")
    print(json.dumps(all_meta, indent=2))
    return all_meta


def run_full(dataset: str, n_jobs: int = 1, max_sec: int = 43200, e_clusters: int = 10) -> dict:
    """Run full DataSAIL; max_sec limits optimization time (default: 12 hours)."""
    started = time.time()
    df, input_rows = load_valid_dataset(dataset)
    df.attrs["dataset"] = dataset
    data_input = ROOT / "data/processed" / f"{dataset}_full_datasail.csv"
    data_input.parent.mkdir(parents=True, exist_ok=True)
    data_input_df = pd.DataFrame({"id": [f"mol_{i}" for i in range(len(df))], "smiles": df.smiles})
    data_input_df.to_csv(data_input, index=False)
    out = ROOT / "results/experiments" / dataset / "datasail_full"
    out.mkdir(parents=True, exist_ok=True)
    from datasail.sail import sail
    datasail_started = time.perf_counter()
    sail(e_type="M", e_data=str(data_input), techniques=["C1e"], e_sim="ecfp", e_clusters=e_clusters,
         splits=list(RATIOS.values()), names=list(RATIOS), output=str(out), verbosity="W", runs=1,
         threads=n_jobs, max_sec=max_sec)
    datasail_seconds = time.perf_counter() - datasail_started
    files = sorted((out / "C1e").glob("Molecule_*_splits.tsv"))
    if not files:
        raise FileNotFoundError(f"No DataSAIL full split TSV under {out / 'C1e'}")
    rows = pd.read_csv(files[-1], sep="\t")
    split_by_id = dict(zip(rows.ID.astype(str), rows.Split.astype(str)))
    labels = np.array([split_by_id[f"mol_{i}"] for i in range(len(df))])
    result = save_split_outputs(df, labels, "datasail_full")
    result.update({"dataset": dataset, "method": "datasail_full", "n_valid_smiles": len(df),
                   "n_input_rows": input_rows, "n_invalid_smiles": input_rows - len(df),
                   "runtime_seconds": time.time() - started, "n_jobs": n_jobs,
                   "datasail_seconds": datasail_seconds,
                   "max_sec": max_sec, "e_clusters": e_clusters, "target_ratios": RATIOS})
    result.update(reproducibility_metadata(dataset))
    outdir = ROOT / "results/experiments/splits" / dataset / "datasail_full"
    (outdir / "run_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def cli(fixed_dataset: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_dataset:
        parser.add_argument("--method", choices=["random", "scaled", "full", "all-local"], default="all-local")
    else:
        parser.add_argument("dataset", choices=CONFIG)
        parser.add_argument("--method", choices=["random", "scaled", "full", "all-local"], default="scaled")
    parser.add_argument("--sample-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-sec", type=int, default=43200,
                        help="DataSAIL optimization time limit for --method full (default: 12 hours)")
    parser.add_argument("--e-clusters", type=int, default=10,
                        help="DataSAIL molecular clustering count for --method full (default: 10)")
    parser.add_argument("--reuse-anchor", action="store_true")
    args = parser.parse_args()
    dataset = fixed_dataset or args.dataset
    if args.method in ("random", "all-local"):
        run_random(dataset, args.seed)
    if args.method in ("scaled", "all-local"):
        run_scaled(dataset, args.sample_ratio, args.seed, args.k, args.n_jobs, args.reuse_anchor)
    if args.method == "full":
        run_full(dataset, args.n_jobs, args.max_sec, args.e_clusters)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
