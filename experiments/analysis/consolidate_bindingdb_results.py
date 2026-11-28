"""Consolidate accepted BindingDB manifests into thesis-ready CSV and JSON tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


AFFINITIES = ("all", "Ki", "Kd", "IC50", "EC50")
REGIMES = ("C1e", "C1f", "C2")
MODEL_FILES = {
    "linear_100k": "predictive_manifest.json",
    "linear_500k": "predictive_manifest.json",
    "hashed_dual_encoder_100k": "dual_encoder_manifest.json",
    "sequence_cnn_100k": "sequence_dual_encoder_manifest.json",
    "esm2_8m_100k": "esm_dual_encoder_manifest.json",
}


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def total_runtime(value) -> float | None:
    if isinstance(value, dict):
        return value.get("total")
    return value


def discover_one(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename} below {root}; found {matches}")
    return matches[0]


def split_rows(root: Path) -> tuple[list[dict], list[dict]]:
    rows, affinity_rows = [], []
    for regime in REGIMES:
        directory = root / "splits" / regime.lower()
        manifest_path = discover_one(directory, "run_manifest.json")
        validation_path = discover_one(directory, "validation_report.json")
        manifest, validation = read_json(manifest_path), read_json(validation_path)
        if validation.get("validation_status") != "passed":
            raise RuntimeError(f"Unaccepted {regime} validation: {validation_path}")
        audit = manifest["audit"]
        accounting = validation["checks"]["accounting"]
        retained = accounting["retained_observations"]
        dropped = accounting["dropped_cross_split_observations"]
        counts = accounting.get("actual_interaction_counts", accounting.get("retained_counts"))
        ratios = accounting.get("actual_interaction_ratios", accounting.get("retained_ratios"))
        requested = accounting.get("requested_interaction_ratios", accounting.get("requested_ratios"))
        if not counts or not ratios or not requested:
            raise RuntimeError(f"Incomplete accepted accounting in {validation_path}")
        row = {
            "regime": regime,
            "validation_status": "passed",
            "canonical_chemicals": manifest.get("n_canonical_chemicals"),
            "ligand_ids": manifest.get("n_ligand_ids"),
            "targets": manifest.get("n_targets"),
            "retained_observations": retained,
            "dropped_observations": dropped,
            "retention_fraction": retained / (retained + dropped),
            "train_count": counts["train"],
            "val_count": counts["val"],
            "test_count": counts["test"],
            "requested_train_ratio": requested["train"],
            "requested_val_ratio": requested["val"],
            "requested_test_ratio": requested["test"],
            "train_ratio": ratios["train"],
            "val_ratio": ratios["val"],
            "test_ratio": ratios["test"],
            "runtime_seconds": total_runtime(manifest.get("runtime_seconds")),
            "peak_memory_mb": manifest.get("peak_memory_mb"),
            "artifact_path": str(manifest_path),
            "validation_artifact_path": str(validation_path),
            "job_id": manifest.get("condor_job_id"),
            "validation_job_id": validation.get("condor_job_id"),
            "commit": manifest.get("git_commit"),
            "container_digest": manifest.get("container_image"),
        }
        rows.append(row)
        checks = validation["checks"]
        distributions = checks.get("affinity_distributions")
        if distributions is None:
            distribution_group = checks.get("distributions", {})
            distributions = distribution_group.get(
                "affinity", distribution_group.get("retained_affinity", [])
            )
        for item in distributions:
            affinity_rows.append({"regime": regime, **item})
    return rows, affinity_rows


def model_rows(root: Path) -> tuple[list[dict], list[dict]]:
    overall, per_affinity = [], []
    for model, filename in MODEL_FILES.items():
        path = discover_one(root / "models" / model, filename)
        manifest = read_json(path)
        for result in manifest["results"]:
            regime = result["regime"]
            samples = result["sample_rows"]
            base = {
                "regime": regime, "model": model,
                "train_n": samples["train"], "val_n": samples["val"], "test_n": samples["test"],
                "regime_runtime_seconds": result.get("total_seconds"),
                "run_runtime_seconds": total_runtime(manifest.get("runtime_seconds")),
                "peak_memory_mb": manifest.get("peak_memory_mb"),
                "artifact_path": str(path), "job_id": manifest.get("condor_job_id"),
                "commit": manifest.get("git_commit"),
                "container_digest": manifest.get("container_image"),
            }
            for metric in result["metrics"]:
                row = {**base, **metric}
                if metric["affinity_type"] == "all" and metric["split"] == "test":
                    overall.append(row)
                per_affinity.append(row)
    return overall, per_affinity


def leakage_rows(root: Path) -> list[dict]:
    path = discover_one(root / "leakage", "leakage_summary.json")
    manifest = read_json(path)
    return [{
        **row, "run_runtime_seconds": manifest.get("runtime_seconds"),
        "peak_memory_mb": manifest.get("peak_memory_mb"), "artifact_path": str(path),
        "job_id": manifest.get("condor_job_id"), "commit": manifest.get("git_commit"),
        "container_digest": manifest.get("container_image"),
    } for row in manifest["summaries"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    split, split_affinity = split_rows(args.artifact_root)
    models, model_affinity = model_rows(args.artifact_root)
    leakage = leakage_rows(args.artifact_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "split_summary": split,
        "split_affinity_summary": split_affinity,
        "model_test_summary": models,
        "model_metrics_long": model_affinity,
        "leakage_summary": leakage,
    }
    for name, rows in tables.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
    (args.output_dir / "bindingdb_results.json").write_text(
        json.dumps({"format_version": 1, **tables}, indent=2) + "\n"
    )
    print(json.dumps({name: len(rows) for name, rows in tables.items()}, indent=2))


if __name__ == "__main__":
    main()
