# Repeated-seed and leakage robustness workflow

The current seed-42 outputs remain in their existing folders. New runs use a seed-specific folder component such as seed_101, keeping splits, metadata, DataSAIL intermediates, and predictive metric files separate. Existing replicate folders are not overwritten.

## 1. Generate matched random and scaled replicates

Use the same seeds for random splitting, random-anchor scaled DataSAIL, and
stratified-anchor scaled DataSAIL:

    python experiments/analysis/run_seed_sweep.py \
      --datasets bace bbbp tox21 hiv qm9 pcba muv esol freesolv lipophilicity \
      --seeds 42 101 202 303 404 --n-jobs 1


For a replicate's predictive metrics, run the existing evaluation under its run ID. Example:

    DATASAIL_RUN_ID=seed_101 python experiments/analysis/ml_evaluation_extended.py \
      --datasets bace --methods random datasail_scaled_random datasail_scaled_stratified \
      --seed 101 --n-jobs 1

## 2. Analyze leakage, structures, and metrics by similarity

    python experiments/analysis/robustness_snapshot.py \
      --datasets bace bbbp tox21 hiv qm9 pcba muv esol freesolv lipophilicity \
      --methods random datasail_scaled_random datasail_scaled_stratified \
      --run-ids seed_42 seed_101 seed_202 seed_303 seed_404 \
      --query-n 500 --train-cap 10000 --n-jobs 1

The script writes raw per-run and summary tables under
results/analysis/unified/robustness/:

- Leakage sensitivity: maximum train-neighbor Tanimoto similarity and
  fractions above 0.5, 0.6, 0.7, 0.8, and 0.9.
- Structure overlap: canonical-SMILES train/test duplicates and
  Bemis-Murcko scaffold overlap. Empty scaffolds, common for acyclic molecules, are counted separately.
- Similarity-stratified performance: per-task regression and classification metrics for sampled test queries grouped by maximum similarity to training molecules. Per-bin row counts and per-task results remain visible because sparse assays may not support every metric.

Nearest-neighbor values estimate leakage using sampled test queries and a capped training reference set. Report sample sizes, thresholds, and bin counts.
Treat performance-by-similarity as descriptive; it does not establish a causal effect of leakage.

Summarize random and scaled methods over seed-level rows using mean, standard deviation, and replicate count. Check that source hashes, split proportions, fingerprint settings, anchor fractions, and model definitions match before pooling. Keep Full DataSAIL's single-run status explicit.
