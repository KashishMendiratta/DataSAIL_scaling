#!/bin/bash
set -euo pipefail

repo_dir="$1"
dataset="$2"
method="$3"
sample_ratio="$4"
seed="$5"
n_jobs="$6"

cd "$repo_dir"
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
else
  python -u experiments/pipelines/run_extended_datasets.py "$dataset" \
    --method "$method" --sample-ratio "$sample_ratio" --seed "$seed" --n-jobs "$n_jobs"
fi
