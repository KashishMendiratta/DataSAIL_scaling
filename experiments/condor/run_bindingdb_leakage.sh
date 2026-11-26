#!/usr/bin/env bash
set -euo pipefail

repo_dir="$1"
cd "$repo_dir"
base="results/experiments/bindingdb"
python -u -m experiments.analysis.bindingdb_leakage_distributions \
  --prepared data/processed/bindingdb/release_202611_all_affinities/bindingdb_staging.sqlite \
  --c1e "$base/c1e/seed_42/c1e_assignments.sqlite" \
  --c1f "$base/c1f/seed_42/c1f_assignments.sqlite" \
  --c2 "$base/c2/seed_42/c2_assignments.sqlite" \
  --output-dir results/analysis/bindingdb/seed_42/leakage \
  --ligand-query-cap 5000 --ligand-reference-cap 50000 --seed 42
