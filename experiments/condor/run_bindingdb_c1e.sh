#!/usr/bin/env bash
set -euo pipefail

repo_dir="$1"
seed="${2:-42}"
anchor_ratio="${3:-0.001}"
threads="${4:-1}"

cd "$repo_dir"
prepared="data/processed/bindingdb/release_202611_all_affinities/bindingdb_staging.sqlite"
output="results/experiments/bindingdb/c1e/seed_${seed}"

if [[ ! -f "$prepared" ]]; then
  echo "Missing prepared BindingDB staging database: $prepared" >&2
  exit 2
fi

python -u -m experiments.pipelines.bindingdb_scaled_c1e \
  "$prepared" "$output" \
  --max-ligands 2000000000 \
  --anchor-ratio "$anchor_ratio" \
  --seed "$seed" \
  --k 5 \
  --balance-weight 2.0 \
  --clusters 10 \
  --max-sec 3600 \
  --threads "$threads"

