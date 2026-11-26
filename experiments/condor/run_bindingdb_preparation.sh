#!/usr/bin/env bash
set -euo pipefail

repo_dir="$1"
release="${2:-202611}"
chunk_size="${3:-10000}"

cd "$repo_dir"
archive="data/raw/bindingdb/BindingDB_All_${release}_tsv.zip"
output_dir="data/processed/bindingdb/release_${release}_all_affinities"

if [[ ! -f "$archive" ]]; then
  echo "Missing frozen BindingDB archive: $archive" >&2
  exit 2
fi

python -u experiments/dataset_download/prepare_bindingdb.py "$archive" \
  --member BindingDB_All.tsv \
  --affinity-types Ki Kd IC50 EC50 \
  --chunk-size "$chunk_size" \
  --output-dir "$output_dir"

