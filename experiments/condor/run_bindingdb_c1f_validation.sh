#!/usr/bin/env bash
set -euo pipefail

repo_dir="$1"
cd "$repo_dir"

prepared="data/processed/bindingdb/release_202611_all_affinities/bindingdb_staging.sqlite"
candidate="results/experiments/bindingdb/c1f/seed_42/c1f_assignments.sqlite"
output="results/experiments/bindingdb/c1f/seed_42/validation_report.json"

for required in "$prepared" "$candidate"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required C1f validation input: $required" >&2
    exit 2
  fi
done

python -u -m experiments.analysis.validate_bindingdb_c1f \
  "$prepared" "$candidate" "$output"
