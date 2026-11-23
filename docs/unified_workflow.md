# Unified DataSAIL workflow

This copy uses one dataset registry at `experiments/config/datasets.json` for the original MoleculeNet datasets (BACE, BBBP, Tox21, HIV) and the extended datasets (QM9, PCBA, MUV, ESOL, FreeSolv, Lipophilicity). Add a dataset by adding its source path, identifier and SMILES columns, task type, target columns or target prefix, stratification target, and optional anchor ratio to that registry.

The four public pipeline entry points share `src/pipeline_engine.py`. They use the same valid-SMILES policy, stable `mol_id`, 70/20/10 split ratios, seed default, fingerprint settings, output schema, and run metadata. Results from this workflow are isolated at `results/experiments/unified/`; historical outputs remain in their original directories. Analysis tables are written to `results/analysis/unified/`.

## Pipelines

Run one or more registered dataset names per invocation:

```bash
python experiments/pipelines/random_pipeline.py bace qm9
python experiments/pipelines/dataSAIL_scaled_random.py bace qm9
python experiments/pipelines/dataSAIL_scaled_stratified.py bace qm9
python experiments/pipelines/full_dataSAIL.py esol freesolv --max-sec 43200
```

The full DataSAIL pipeline can be resource intensive. Run large datasets individually on a suitable compute slot and use `--max-sec` to bound optimization time. Failed datasets are reported and do not prevent later datasets in the same command from being attempted, except for process-level failures such as an operating-system kill.

Each run writes `train.csv`, `val.csv`, `test.csv`, and `run_metadata.json` below `results/experiments/unified/splits/<dataset>/<method>/`. Scaled runs also write `anchor_manifest.json`, which records the selected anchor IDs. DataSAIL intermediate files are below `results/experiments/unified/datasail/`.

## Analyses

Each script accepts optional `--datasets` and `--methods` arguments. With no dataset list, it processes every registered dataset. It records missing split sets explicitly rather than substituting outputs from the historical pipeline.

- `runtime_comparison.py`: end-to-end and DataSAIL-stage timing from run metadata. The recorded scope is reported with each row.
- `leakage_analysis.py`: sampled mean train-test Tanimoto similarity. This is a global cross-partition similarity summary, not the nearest-neighbour threshold statistic.
- `nn_leakage.py`: sampled test-to-train maximum Tanimoto similarity and fraction above the selected threshold.
- `ml_evaluation_extended.py`: regression and classification evaluation, including multitask assays with missing labels. It reports per-target/per-assay results.
- `compute_distribution_ratio_error.py`: per-target class-frequency error; continuous regression targets are compared through source-wide quantile bins.
- `compute_kl_divergence.py`: per-target KL divergence using observed classes or source-wide quantile bins, as appropriate.
- `compute_pairwise_similarity_preservation.py`: compares sampled source-wide and actual anchor-set pairwise similarity estimates for the two scaled methods. Requires the saved anchor manifest.

Example analysis commands:

```bash
python experiments/analysis/ml_evaluation_extended.py
python experiments/analysis/runtime_comparison.py
python experiments/analysis/leakage_analysis.py --pairs 100000
python experiments/analysis/nn_leakage.py --query-n 500 --train-cap 10000 --threshold 0.7
python experiments/analysis/compute_distribution_ratio_error.py
python experiments/analysis/compute_kl_divergence.py
python experiments/analysis/compute_pairwise_similarity_preservation.py
```

The leakage analysis estimates mean similarity by sampling molecule pairs to bound memory and runtime. The nearest-neighbour analysis bounds work by sampling test molecules and at most the requested number of training molecules. These are different metrics and must remain separately labeled in reports.

Distribution summaries exclude missing assay labels. For regression, quantile bins make distribution comparisons possible but discard within-bin detail; the bin count should be reported. Metrics are written at per-target/per-assay granularity before aggregation so sparse or failed tasks remain visible.

## Reproducible comparisons

For the main comparison, rerun random, scaled-random, scaled-stratified, and full splits with the same source files, seed, split proportions, model definitions, and resource context where feasible. Use multiple seeds for thesis claims. Full DataSAIL fails on the largest datasets; the pipeline records successful outputs and reports failures but cannot guarantee completion under a resource limit.
