# Scaling DataSAIL for molecular datasets

This repository contains the implementation and experiments for scaling DataSAIL to larger molecular datasets. The scaled method runs DataSAIL on an anchor subset, then assigns the remaining molecules using Morgan fingerprints and balanced k-nearest-neighbor assignment. The current study compares random splitting, scaled DataSAIL with random anchors, scaled DataSAIL with stratified anchors, and full DataSAIL where a successful full run is available.

The provenance-controlled November 2026 study covers ten molecular datasets, from 642 to 437,927 valid molecules. Random and both scaled methods were evaluated over five split seeds. Full DataSAIL has one reference run on nine datasets; PCBA is the only dataset without a Full DataSAIL reference in the selected snapshot. On QM9, the observed mean end-to-end runtime was 1,493.7 seconds for scaled-random DataSAIL versus 12,979.8 seconds for the single full-DataSAIL run, an 8.7× comparison. These are observed runs, not a hardware-independent speed guarantee.

The original thesis report is [`reports/MASTER REPORT.pdf`](reports/MASTER%20REPORT.pdf). It is a historical snapshot and retains its original scope and claims. This README describes the newer November 2026 evidence, which supersedes the report for current repository and portfolio summaries.

## Current workflow

The active pipelines are:

| Script | Purpose |
|---|---|
| `experiments/pipelines/random_pipeline.py` | Shuffle valid molecules and divide them into train, validation, and test sets. |
| `experiments/pipelines/dataSAIL_scaled_random.py` | Select anchors randomly, split anchors with DataSAIL, then assign the remaining molecules with balanced kNN. |
| `experiments/pipelines/dataSAIL_scaled_stratified.py` | Select anchors while preserving the configured target strata, then use the same DataSAIL and balanced-kNN steps. |
| `experiments/pipelines/full_dataSAIL.py` | Run DataSAIL on all valid molecules. This requires HPC resources for larger datasets. |

All four pipelines use the shared dataset registry and defaults in `experiments/config/datasets.json`. The default split proportions are 70% train, 20% validation, and 10% test. The command-line default seed is 42; the main repeated-seed study uses 42, 101, 202, 303, and 404; scaled anchor fractions are specified per dataset; the default fingerprint is Morgan radius 2 with 2048 bits; and the default balanced-kNN uses `k=5` and balance weight 2.0. These are configurable, not universal constants.

For classification, stratified anchor selection uses the configured stratification label. For regression, it bins the configured target into quantiles. Stratification applies to anchor selection; it does not guarantee that every final split exactly matches the source target distribution.

## Environment and data

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate datasail-scale
```

The registry currently includes BACE, BBBP, Tox21, HIV, QM9, PCBA, MUV, ESOL, FreeSolv, and Lipophilicity. Dataset paths, columns, task types, and anchor fractions are defined in `experiments/config/datasets.json`.

The core splitting and fingerprint-analysis workflow uses the `datasail-scale` Conda environment. The D-MPNN extension uses a separate `.venv-dmpnn` environment with Chemprop, PyTorch, and the dependencies in `requirements-dmpnn.txt`. Do not describe the D-MPNN runs as having been executed in the core Conda environment.

Downloaders are in `experiments/dataset_download/`. For example:

```bash
python experiments/dataset_download/download_qm9.py
python experiments/dataset_download/download_pcba.py
python experiments/dataset_download/download_muv.py
python experiments/dataset_download/download_esol.py
python experiments/dataset_download/download_freesolv.py
python experiments/dataset_download/download_lipophilicity.py
```

The original MoleculeNet CSVs belong under `data/raw/moleculenet/`. They are local inputs and are not included in Git. Check that every configured input file exists before starting a pipeline.

## Run pipelines

Run commands from the repository root. Each pipeline accepts one or more dataset names. To keep execution sequential and limit parallel CPU work on a modest machine, run one dataset at a time and set `--n-jobs 1` for the scaled and full pipelines:

```bash
set -e
DATASETS="bace bbbp tox21 hiv qm9 pcba muv esol freesolv lipophilicity"

for dataset in $DATASETS; do
  python experiments/pipelines/random_pipeline.py "$dataset"
done

for dataset in $DATASETS; do
  python experiments/pipelines/dataSAIL_scaled_random.py "$dataset" --n-jobs 1
done

for dataset in $DATASETS; do
  python experiments/pipelines/dataSAIL_scaled_stratified.py "$dataset" --n-jobs 1
done
```

`set -e` stops the shell sequence on the first failed command. It does not alter pipeline behavior. `--n-jobs` limits the number of worker threads used by supported fingerprinting, DataSAIL, and nearest-neighbor operations; one job reduces concurrent work but does not remove the memory cost of holding dataset and fingerprint arrays. PCBA is especially demanding. Start with smaller datasets and monitor memory before attempting it locally.

Run full DataSAIL only for datasets and machines with suitable resources. A single invocation can process multiple datasets sequentially:

```bash
python experiments/pipelines/full_dataSAIL.py bace --n-jobs 1
```

The full pipeline has a default 12-hour DataSAIL time limit; adjust `--max-sec` only when appropriate for the machine or scheduler. A timeout or resource failure is a run status to record.

### Reproduce the five-seed comparison

The main repeated-seed study uses seeds 42, 101, 202, 303, and 404 for Random, scaled-random-anchor, and scaled-stratified-anchor splitting. Run the matched sweep with:

```bash
python experiments/analysis/run_seed_sweep.py \
  --seeds 42 101 202 303 404 \
  --n-jobs 1
```

The sweep writes each replicate to a seed-specific directory and refuses to overwrite an existing replicate. Full DataSAIL is excluded from this sweep because it is represented by one reference run per available dataset rather than five independent replicates.

## Run analyses

The original analysis scripts generate working tables under results/analysis/unified/. The current thesis evidence additionally uses multi-seed runtime and quality summaries, robustness analyses, a corrected predictive-quality audit, D-MPNN evaluation, table reconciliation, and unified figure generation.

The core analysis commands are:

```bash
python experiments/analysis/runtime_comparison.py
python experiments/analysis/leakage_analysis.py --n-jobs 1
python experiments/analysis/nn_leakage.py --n-jobs 1
python experiments/analysis/ml_evaluation_extended.py --n-jobs 1
python experiments/analysis/compute_distribution_ratio_error.py
python experiments/analysis/compute_kl_divergence.py
python experiments/analysis/compute_pairwise_similarity_preservation.py --n-jobs 1
```

By default, analyses consider all configured datasets and methods and record missing split files as missing. Options such as `--datasets`, `--methods`, `--pairs`, `--query-n`, `--train-cap`, and `--threshold` allow narrower or sensitivity analyses; check `--help` for each script.

The principal extended analysis entry points are:

- runtime_multiseed.py for repeated-seed and Full DataSAIL runtime summaries
- quality_metrics_multiseed.py for distribution and anchor-quality metrics
- robustness_snapshot.py for leakage-threshold and structural-overlap analyses
- predictive_quality_audit_k5.py for corrected five-seed predictive comparisons
- run_dmpnn_k5.py and summarize_dmpnn_k5.py for the molecular graph-model extension
- reconcile_selected_tables.py for cross-table consistency checks

The analysis scripts write working outputs under `results/analysis/unified/`. These generated results are excluded from Git and should be regenerated locally from the recorded split outputs and metadata.

### Runtime definitions

For scaled and full DataSAIL runs, `total_seconds` is end-to-end wall time, including data loading and validation, fingerprint generation where applicable, DataSAIL, assignment, and output preparation. `datasail_seconds` measures only the DataSAIL call. The random baseline has end-to-end time but no DataSAIL stage. `runtime_comparison.py` reads these values into `runtime_comparison.csv`.

Report end-to-end time when comparing user-visible computational cost. Runtime depends on hardware and software versions; record the machine, environment, code revision, seed, and settings.

### Similarity and leakage interpretation

`leakage_analysis.py` estimates the mean Tanimoto similarity across sampled train-test molecule pairs. Its default is 100,000 sampled pairs. `nn_leakage.py` estimates each sampled test molecule's maximum similarity to training molecules; by default it samples up to 500 test molecules and caps the training set at 10,000 molecules, using a 0.7 threshold for its high-similarity fraction.

These are measures of structural proximity across splits, not direct evidence of target-label leakage. For larger datasets they are sampled estimates, not exhaustive comparisons.

`compute_distribution_ratio_error.py` and `compute_kl_divergence.py` compare target distributions in each split with the source distribution. `compute_pairwise_similarity_preservation.py` estimates how well each scaled method’s anchor set preserves the source dataset’s within-dataset pairwise similarity; it does not measure train-test leakage. These metrics answer different questions and should not be collapsed into a single “leakage” score.

### Predictive evaluation

The current predictive evidence has two model families:

1. `ml_evaluation_extended.py` provides the fixed Morgan-fingerprint baseline: SGD classification for classification tasks and Ridge regression for regression tasks. The provenance snapshot contains five-seed comparisons across all ten datasets.
2. The D-MPNN extension provides a molecular graph-model cross-check on BACE, HIV, Tox21, MUV, ESOL, and QM9. It uses the same five saved split seeds for Random and both scaled strategies, validation-based checkpoint selection, and one fixed model-initialization seed. It is not a tuned model benchmark. Full-DataSAIL D-MPNN results are single-run references for BACE, HIV, Tox21, and ESOL; MUV and QM9 do not have a D-MPNN Full baseline, and PCBA was not run with D-MPNN.

MCC is applicable to classification, not regression; regression results use MAE, RMSE, and R². For sparse multi-task datasets such as PCBA and MUV, interpret accuracy alongside PR-AUC, MCC, label prevalence, and the number of tasks with both classes represented.

A predictive score is affected by the difficulty and composition of each test split. Lower scores on a structurally separated split do not by themselves indicate a defective splitter. Compare predictive metrics with split sizes, target distributions, structural-similarity estimates, and uncertainty across split seeds. The current provenance results do not contain five-seed Random Forest or Logistic Regression comparisons; public summaries should name the evaluated SGD/Ridge and D-MPNN models instead.


## Current result coverage and limitations

The selected November 2026 evidence contains five-seed results for Random, scaled-random-anchor, and scaled-stratified-anchor splitting across all ten configured datasets. It also contains integrity-checked, single-run Full DataSAIL references for BACE, BBBP, Tox21, HIV, QM9, MUV, ESOL, FreeSolv, and Lipophilicity. PCBA remains the only dataset without a Full DataSAIL result.

The largest observed end-to-end Full-versus-scaled runtime difference occurs on QM9. The single Full DataSAIL run took 12,979.8 seconds, compared with a five-seed mean of 1,493.7 seconds for scaled-random DataSAIL—an 8.7× comparison. The corresponding DataSAIL-stage difference is considerably larger because the scaled pipeline still performs full-dataset fingerprinting, nearest-neighbour assignment, validation, and output preparation outside the anchor-only DataSAIL call. Runtime summaries must therefore distinguish DataSAIL-stage time from end-to-end pipeline time and identify comparisons between single Full runs and five-seed scaled means.

Scaled-random reduced the sampled fraction of test molecules with a training neighbour above the selected Tanimoto threshold across all ten datasets relative to Random. Scaffold overlap provides a separate diagnostic: on BACE, the mean fraction of test molecules sharing a scaffold with training fell from 65.6% under Random to 2.45% under scaled-random. These results apply to the stored sampling and overlap definitions and should not be interpreted as proof of zero leakage under every criterion.

QM9 contains 19 heterogeneous regression targets. Target `A` is unstable on some splits and can dominate pooled averages, so QM9 predictive performance should be reported per target rather than as a single unqualified aggregate. More generally, predictive results measure how the evaluated models respond to each split; they are complementary to, but not substitutes for, direct structural-similarity and scaffold-overlap diagnostics.

## Results organization and provenance


`results/analysis/unified/` contains generated working outputs and is excluded from Git. The November 2026 evidence selection is used for the current thesis, CV, portfolio, and repository summaries. Historical and exploratory outputs must not be combined with that selection without checking their methods, seeds, schemas, source hashes, and runtime definitions.

Run metadata records the source-data hash, seed, parameters, package versions, split counts, runtime components, Git revision and worktree status where available. Full DataSAIL results are single-run references; uncertainty across five split seeds applies only to Random and the two scaled methods.
