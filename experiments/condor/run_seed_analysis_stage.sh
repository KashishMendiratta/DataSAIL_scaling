#!/usr/bin/env bash
set -euo pipefail
repo_dir="$1"
stage="$2"
dataset="${3:-all}"
exec "$repo_dir/experiments/condor/run_job.sh" \
  "$repo_dir" "$dataset" "seed-${stage}" 0 42 1
