#!/usr/bin/env bash
set -euo pipefail

repo_dir="$1"
seed="${2:-42}"
threads="${3:-1}"
cd "$repo_dir"

prepared="data/processed/bindingdb/release_202611_all_affinities/bindingdb_staging.sqlite"
output="results/experiments/bindingdb/c2/seed_${seed}"
if [[ ! -f "$prepared" ]]; then
  echo "Missing prepared BindingDB staging database: $prepared" >&2
  exit 2
fi

python -u -m experiments.pipelines.bindingdb_scaled_c2 \
  "$prepared" "$output" \
  --max-ligands 2000000000 --max-targets 2000000000 \
  --ligand-anchor-ratio 0.001 --target-anchor-ratio 0.05 \
  --seed "$seed" --k 5 --balance-weight 3.0 \
  --graph-weight 4.0 --similarity-weight 1.0 \
  --ligand-clusters 10 --target-clusters 20 \
  --max-sec 3600 --threads "$threads"
