#!/usr/bin/env bash
set -euo pipefail

repo_dir="${1:?usage: collect_bindingdb_artifacts.sh REPO_DIR OUTPUT_DIR}"
output_dir="${2:?usage: collect_bindingdb_artifacts.sh REPO_DIR OUTPUT_DIR}"

copy_required() {
  local source="$1" destination="$2"
  if [[ ! -f "$repo_dir/$source" ]]; then
    echo "Missing accepted artifact: $repo_dir/$source" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$output_dir/$destination")"
  cp "$repo_dir/$source" "$output_dir/$destination"
}

for regime in c1e c1f c2; do
  copy_required "results/experiments/bindingdb/$regime/seed_42/run_manifest.json" \
    "splits/$regime/run_manifest.json"
  copy_required "results/experiments/bindingdb/$regime/seed_42/validation_report.json" \
    "splits/$regime/validation_report.json"
done

copy_required "results/analysis/bindingdb/seed_42/leakage/leakage_summary.json" \
  "leakage/leakage_summary.json"
copy_required "results/analysis/bindingdb/seed_42/predictive_baseline/predictive_manifest.json" \
  "models/linear_100k/predictive_manifest.json"
copy_required "results/analysis/bindingdb/seed_42/predictive_train500k/predictive_manifest.json" \
  "models/linear_500k/predictive_manifest.json"
copy_required "results/analysis/bindingdb/seed_42/dual_encoder_100k/dual_encoder_manifest.json" \
  "models/hashed_dual_encoder_100k/dual_encoder_manifest.json"
copy_required "results/analysis/bindingdb/seed_42/sequence_dual_encoder_100k/sequence_dual_encoder_manifest.json" \
  "models/sequence_cnn_100k/sequence_dual_encoder_manifest.json"
copy_required "results/analysis/bindingdb/seed_42/esm_dual_encoder_100k/esm_dual_encoder_manifest.json" \
  "models/esm2_8m_100k/esm_dual_encoder_manifest.json"

find "$output_dir" -type f -exec shasum -a 256 {} + | sort > "$output_dir/SHA256SUMS"
echo "Collected accepted BindingDB manifests in $output_dir"
