#!/bin/bash
set -euo pipefail

repo_dir="$1"
dataset="$2"
epochs="$3"
patience="$4"
batch_size="$5"

cd "$repo_dir"
output_dir="results/analysis/provenance_20261120/dmpnn_k5/by_dataset/$dataset"

python -u experiments/analysis/run_dmpnn_k5.py \
  --datasets "$dataset" \
  --methods random datasail_scaled_random datasail_scaled_stratified datasail_full \
  --seeds 42 101 202 303 404 \
  --epochs "$epochs" \
  --patience "$patience" \
  --batch-size "$batch_size" \
  --output-dir "$output_dir"
