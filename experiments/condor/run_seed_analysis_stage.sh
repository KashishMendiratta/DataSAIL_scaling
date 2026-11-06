#!/usr/bin/env bash
set -euo pipefail

repo_dir="$1"
stage="$2"
dataset="${3:-all}"
cd "$repo_dir"

case "$stage" in
  seed)
    python -u experiments/analysis/run_seed_sweep.py \
      --datasets "$dataset" \
      --seeds 42 101 202 303 404 \
      --methods random datasail_scaled_random datasail_scaled_stratified \
      --n-jobs 1
    ;;
  robustness)
    python -u experiments/analysis/robustness_snapshot.py \
      --datasets hiv muv qm9 pcba \
      --methods random datasail_scaled_random datasail_scaled_stratified \
      --run-ids seed_42 seed_101 seed_202 seed_303 seed_404 \
      --query-n 500 --train-cap 10000 --n-jobs 1
    ;;
  ml)
    for seed in 42 101 202 303 404; do
      DATASAIL_RUN_ID="seed_${seed}" \
        python -u experiments/analysis/ml_evaluation_extended.py \
          --datasets hiv muv qm9 pcba \
          --methods random datasail_scaled_random datasail_scaled_stratified \
          --seed "$seed" --n-jobs 1
    done
    ;;
  *)
    echo "Unknown workflow stage: $stage" >&2
    exit 2
    ;;
esac
