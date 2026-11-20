"""Collect recorded runtimes and provenance from verified split metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPEATED_METHODS = ("random", "datasail_scaled_random", "datasail_scaled_stratified")
DEFAULT_RUN_IDS = ("seed_42", "seed_101", "seed_202", "seed_303", "seed_404")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--run-ids", nargs="+", default=list(DEFAULT_RUN_IDS))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    config = json.loads((root / "experiments/config/datasets.json").read_text())
    datasets = args.datasets or list(config["datasets"])
    unknown = sorted(set(datasets) - set(config["datasets"]))
    if unknown:
        parser.error(f"Unknown datasets: {unknown}")

    split_root = root / "results/experiments/unified/splits"
    rows: list[dict] = []
    missing: list[str] = []

    selections = [
        (dataset, method, run_id, split_root / dataset / method / run_id)
        for dataset in datasets
        for method in REPEATED_METHODS
        for run_id in args.run_ids
    ]
    selections.extend(
        (dataset, "datasail_full", "baseline", split_root / dataset / "datasail_full")
        for dataset in datasets
        if (split_root / dataset / "datasail_full" / "run_metadata.json").is_file()
    )

    for dataset, method, run_id, folder in selections:
        metadata_path = folder / "run_metadata.json"
        split_paths = [folder / f"{split}.csv" for split in ("train", "val", "test")]
        if not metadata_path.is_file() or not all(path.is_file() for path in split_paths):
            missing.append(f"{dataset}/{method}/{run_id}")
            continue
        metadata = json.loads(metadata_path.read_text())
        rows.append({
            "dataset": dataset,
            "method": method,
            "run_id": run_id,
            "seed": metadata.get("seed"),
            "end_to_end_seconds": metadata.get("total_seconds", metadata.get("runtime_seconds")),
            "datasail_stage_seconds": metadata.get("datasail_seconds"),
            "runtime_scope": metadata.get("runtime_scope"),
            "sample_ratio": metadata.get("sample_ratio"),
            "n_anchor_rows": metadata.get("n_anchor_rows", metadata.get("n_datasail_anchor")),
            "n_valid_rows": metadata.get("n_valid_rows"),
            "git_commit": metadata.get("git_commit"),
            "git_worktree_dirty": metadata.get("git_worktree_dirty"),
            "container_image": metadata.get("container_image"),
            "condor_job_id": metadata.get("condor_job_id"),
            "source_sha256": metadata.get("source_sha256"),
            "metadata_path": str(metadata_path.relative_to(root)),
        })

    if missing:
        raise SystemExit("Missing selected split outputs:\n" + "\n".join(missing))

    frame = pd.DataFrame(rows)
    outdir = (args.output_dir or root / "results/analysis/unified/runtime_multiseed").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "runtime_runs.csv", index=False)
    summary = frame.groupby(["dataset", "method"], dropna=False).agg(
        n_runs=("run_id", "count"),
        mean_end_to_end_seconds=("end_to_end_seconds", "mean"),
        sd_end_to_end_seconds=("end_to_end_seconds", "std"),
        mean_datasail_stage_seconds=("datasail_stage_seconds", "mean"),
        sd_datasail_stage_seconds=("datasail_stage_seconds", "std"),
    ).reset_index()
    summary.to_csv(outdir / "runtime_summary.csv", index=False)
    print(f"Saved {len(frame)} runtime rows under {outdir}", flush=True)


if __name__ == "__main__":
    main()
