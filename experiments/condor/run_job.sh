#!/bin/bash
set -euo pipefail

repo_dir="$1"
dataset="$2"
method="$3"
sample_ratio="$4"
seed="$5"
n_jobs="$6"

cd "$repo_dir"
provenance_dir="results/provenance/condor_jobs"
mkdir -p "$provenance_dir"
job_id="${CONDOR_JOB_ID:-manual-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
provenance_file="$provenance_dir/${job_id//[^A-Za-z0-9_.-]/_}.json"

write_provenance() {
  local job_status="$1"
  local exit_code="$2"
  python - "$repo_dir" "$dataset" "$method" "$sample_ratio" "$seed" "$n_jobs" \
    "$job_id" "$provenance_file" "$job_status" "$exit_code" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

repo = Path(sys.argv[1])
dataset, method, sample_ratio, seed, n_jobs, job_id, out, status, exit_code = sys.argv[2:]
now = datetime.now(timezone.utc).isoformat()
try:
    previous = json.loads(Path(out).read_text())
except (OSError, json.JSONDecodeError):
    previous = {}
manifest = {
    **previous,
    "job_id": job_id,
    "status": status,
    "exit_code": int(exit_code),
    "dataset": dataset,
    "method": method,
    "sample_ratio": sample_ratio,
    "seed": int(seed),
    "n_jobs": int(n_jobs),
    "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
    "hostname": socket.gethostname(),
    "python_version": platform.python_version(),
    "package_versions": {},
}
if status == "started":
    manifest["started_utc"] = now
else:
    manifest["finished_utc"] = now
for package in ("datasail", "numpy", "pandas", "scikit-learn", "rdkit"):
    try:
        manifest["package_versions"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        manifest["package_versions"][package] = None
commit_from_submit = os.environ.get("DATASAIL_GIT_COMMIT")
dirty_from_submit = os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY")
if commit_from_submit:
    manifest["git_commit"] = commit_from_submit
    manifest["git_worktree_dirty"] = (
        None if dirty_from_submit is None else dirty_from_submit.strip().lower() == "true"
    )
else:
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        manifest["git_worktree_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip())
    except Exception:
        manifest["git_commit"] = None
        manifest["git_worktree_dirty"] = None
try:
    config = json.loads((repo / "experiments/config/datasets.json").read_text())
    names = list(config["datasets"]) if dataset in ("all", "") else [dataset]
    source_hashes = {}
    for name in names:
        spec = config["datasets"].get(name)
        if not spec:
            continue
        source = repo / spec["path"]
        if source.is_file():
            source_hashes[name] = {
                "path": spec["path"],
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
    manifest["dataset_sources"] = source_hashes
except Exception as exc:
    manifest["dataset_hash_error"] = str(exc)
Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
PY
}

write_provenance started 0
finish_job() {
  local exit_code=$?
  trap - EXIT
  write_provenance finished "$exit_code" || true
  exit "$exit_code"
}
trap finish_job EXIT

if [[ "$method" == "analysis" ]]; then
  python -u experiments/analysis/analyze_extended_datasets.py "$dataset" --n-jobs "$n_jobs"
elif [[ "$method" == "predictive-evaluation" ]]; then
  python -u experiments/analysis/run_predictive_evaluation_extended.py "$dataset" \
    --n-jobs "$n_jobs" --seed "$seed"
elif [[ "$method" == "unified-analysis" ]]; then
  python -u experiments/analysis/run_all_datasets_analysis.py \
    --datasets "$dataset" --rerun-scaled --n-jobs "$n_jobs" \
    --skip-predictive --no-aggregate
elif [[ "$method" == "unified-aggregate" ]]; then
  python -u experiments/analysis/run_all_datasets_analysis.py --aggregate-only
elif [[ "$method" == "seed-seed" ]]; then
  python -u experiments/analysis/run_seed_sweep.py \
    --datasets "$dataset" --seeds 42 101 202 303 404 \
    --methods random datasail_scaled_random datasail_scaled_stratified --n-jobs "$n_jobs"
elif [[ "$method" == "seed-robustness" ]]; then
  python -u experiments/analysis/robustness_snapshot.py \
    --datasets hiv muv qm9 pcba \
    --methods random datasail_scaled_random datasail_scaled_stratified \
    --run-ids seed_42 seed_101 seed_202 seed_303 seed_404 \
    --query-n 500 --train-cap 10000 --n-jobs "$n_jobs"
elif [[ "$method" == "seed-ml" ]]; then
  for seed_value in 42 101 202 303 404; do
    DATASAIL_RUN_ID="seed_${seed_value}" \
      python -u experiments/analysis/ml_evaluation_extended.py \
        --datasets hiv muv qm9 pcba \
        --methods random datasail_scaled_random datasail_scaled_stratified \
        --seed "$seed_value" --n-jobs "$n_jobs"
  done
else
  python -u experiments/pipelines/run_extended_datasets.py "$dataset" \
    --method "$method" --sample-ratio "$sample_ratio" --seed "$seed" --n-jobs "$n_jobs"
fi
