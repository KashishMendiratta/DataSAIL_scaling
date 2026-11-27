#!/usr/bin/env bash
set -euo pipefail
repo_dir="$1"
cd "$repo_dir"
python -c "import esm, torch; print('torch', torch.__version__, 'esm', esm.__version__)"
python -u -m experiments.analysis.prepare_bindingdb_esm_embeddings \
  --prepared data/processed/bindingdb/release_202611_all_affinities/bindingdb_staging.sqlite \
  --output data/processed/bindingdb/release_202611_all_affinities/esm2_t6_8m_targets.npz \
  --batch-size 4
