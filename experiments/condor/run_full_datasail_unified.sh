#!/usr/bin/env bash
set -euo pipefail

repo_dir="$1"
dataset="$2"
seed="$3"
n_jobs="$4"
max_sec="$5"
e_clusters="$6"

cd "$repo_dir"
python -u experiments/pipelines/full_dataSAIL.py "$dataset" \
  --seed "$seed" \
  --n-jobs "$n_jobs" \
  --max-sec "$max_sec" \
  --e-clusters "$e_clusters"
