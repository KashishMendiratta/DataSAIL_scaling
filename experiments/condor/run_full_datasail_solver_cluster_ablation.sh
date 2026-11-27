#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 REPO_DIR DATASET SOLVER E_CLUSTERS" >&2
  exit 2
fi

repo_dir="$1"
dataset="$2"
solver="$3"
e_clusters="$4"

case "$dataset" in
  qm9|muv) ;;
  *) echo "Unsupported ablation dataset: $dataset" >&2; exit 2 ;;
esac

case "${solver}:${e_clusters}" in
  arpack:10|lobpcg:50) ;;
  *) echo "Unsupported ablation combination: solver=$solver e_clusters=$e_clusters" >&2; exit 2 ;;
esac

cd "$repo_dir"

python -u experiments/pipelines/full_dataSAIL.py "$dataset" \
  --seed 42 \
  --n-jobs 1 \
  --max-sec 43200 \
  --e-clusters "$e_clusters" \
  --spectral-eigen-solver "$solver"
