#!/usr/bin/env bash
set -euo pipefail
repo_dir="$1"
cd "$repo_dir"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
base="results/experiments/bindingdb"
python -u -m experiments.analysis.bindingdb_esm_dual_encoder \
  --prepared data/processed/bindingdb/release_202611_all_affinities/bindingdb_staging.sqlite \
  --esm-cache data/processed/bindingdb/release_202611_all_affinities/esm2_t6_8m_targets.npz \
  --c1e "$base/c1e/seed_42/c1e_assignments.sqlite" \
  --c1f "$base/c1f/seed_42/c1f_assignments.sqlite" \
  --c2 "$base/c2/seed_42/c2_assignments.sqlite" \
  --output-dir results/analysis/bindingdb/seed_42/esm_dual_encoder_100k \
  --train-cap 100000 --val-cap 25000 --test-cap 25000 \
  --batch-size 512 --max-epochs 8 --patience 2 --learning-rate 0.0005 --seed 42
