#!/usr/bin/env bash
set -euo pipefail

repo_dir="$1"
train_cap="$2"
val_cap="$3"
test_cap="$4"
max_epochs="$5"
run_label="$6"
cd "$repo_dir"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
base="results/experiments/bindingdb"
python -u -m experiments.analysis.bindingdb_sequence_dual_encoder \
  --prepared data/processed/bindingdb/release_202611_all_affinities/bindingdb_staging.sqlite \
  --c1e "$base/c1e/seed_42/c1e_assignments.sqlite" \
  --c1f "$base/c1f/seed_42/c1f_assignments.sqlite" \
  --c2 "$base/c2/seed_42/c2_assignments.sqlite" \
  --output-dir "results/analysis/bindingdb/seed_42/sequence_dual_encoder_${run_label}" \
  --train-cap "$train_cap" --val-cap "$val_cap" --test-cap "$test_cap" \
  --batch-size 128 --max-epochs "$max_epochs" --patience 2 --learning-rate 0.0005 --seed 42
