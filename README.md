# Scaling DataSAIL for molecular datasets

This repository contains the implementation and experiments for scaling DataSAIL to larger molecular datasets. The scaled method runs DataSAIL on an anchor subset, then assigns the remaining molecules using Morgan fingerprints and balanced k-nearest-neighbor assignment. The experiments compare this approach with random splitting and full DataSAIL.

The thesis report is [`reports/MASTER REPORT.pdf`](reports/MASTER%20REPORT.pdf). Results produced after that report are extensions; they do not retroactively change the report's original claims.

## Current workflow

The active pipelines are:

| Script | Purpose |
|---|---|
| `experiments/pipelines/random_pipeline.py` | Shuffle valid molecules and divide them into train, validation, and test sets. |
| `experiments/pipelines/dataSAIL_scaled_random.py` | Select anchors randomly, split anchors with DataSAIL, then assign the remaining molecules with balanced kNN. |
| `experiments/pipelines/dataSAIL_scaled_stratified.py` | Select anchors while preserving the configured target strata, then use the same DataSAIL and balanced-kNN steps. |
| `experiments/pipelines/full_dataSAIL.py` | Run DataSAIL on all valid molecules. This requires HPC resources for larger datasets. |

All four pipelines use the shared dataset registry and defaults in `experiments/config/datasets.json`. The default split proportions are 70% train, 20% validation, and 10% test. The default seed is 42; scaled anchor fractions are specified per dataset; the default fingerprint is Morgan radius 2 with 2048 bits; and the default balanced-kNN uses `k=5` and balance weight 2.0. These are configurable, not universal constants.

For classification, stratified anchor selection uses the configured stratification label. For regression, it bins the configured target into quantiles. Stratification applies to anchor selection; it does not guarantee that every final split exactly matches the source target distribution.

## Environment and data

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate datasail-scale
```

The registry currently includes BACE, BBBP, Tox21, HIV, QM9, PCBA, MUV, ESOL, FreeSolv, and Lipophilicity. Dataset paths, columns, task types, and anchor fractions are defined in `experiments/config/datasets.json`.

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

## Run analyses

After the split files exist, run the seven current analyses from the repository root:

```bash
python experiments/analysis/runtime_comparision.py
python experiments/analysis/leakage_analaysis.py --n-jobs 1
python experiments/analysis/nn_leakage.py --n-jobs 1
python experiments/analysis/ml_evaluation_extended.py --n-jobs 1
python experiments/analysis/compute_distribution_ratio_error.py
python experiments/analysis/compute_kl_divergence.py
python experiments/analysis/compute_pairwise_similarity_preservation.py --n-jobs 1
```

By default, analyses consider all configured datasets and methods and record missing split files as missing. Options such as `--datasets`, `--methods`, `--pairs`, `--query-n`, `--train-cap`, and `--threshold` allow narrower or sensitivity analyses; check `--help` for each script.

The analyses write tables under `results/analysis/unified/`. Per-dataset ML metrics are stored in files such as `bace_predictive_metrics.csv`; metadata files record the evaluated splits and settings. Split files and run metadata are under `results/experiments/unified/splits/<dataset>/<method>/`.

### Runtime definitions

For scaled and full DataSAIL runs, `total_seconds` is end-to-end wall time, including data loading and validation, fingerprint generation where applicable, DataSAIL, assignment, and output preparation. `datasail_seconds` measures only the DataSAIL call. The random baseline has end-to-end time but no DataSAIL stage. `runtime_comparision.py` reads these values into `runtime_comparison.csv`.

Report end-to-end time when comparing user-visible computational cost. Runtime depends on hardware and software versions; record the machine, environment, code revision, seed, and settings.

### Similarity and leakage interpretation

`leakage_analaysis.py` estimates the mean Tanimoto similarity across sampled train-test molecule pairs. Its default is 100,000 sampled pairs. `nn_leakage.py` estimates each sampled test molecule's maximum similarity to training molecules; by default it samples up to 500 test molecules and caps the training set at 10,000 molecules, using a 0.7 threshold for its high-similarity fraction.

These are measures of structural proximity across splits, not direct evidence of target-label leakage. For larger datasets they are sampled estimates, not exhaustive comparisons.

`compute_distribution_ratio_error.py` and `compute_kl_divergence.py` compare target distributions in each split with the source distribution. `compute_pairwise_similarity_preservation.py` estimates how well each scaled method’s anchor set preserves the source dataset’s within-dataset pairwise similarity; it does not measure train-test leakage. These metrics answer different questions and should not be collapsed into a single “leakage” score.

### Predictive evaluation

`ml_evaluation_extended.py` evaluates a fixed fingerprint-based baseline: SGD classification for classification tasks and Ridge regression for regression tasks. It writes per-task metrics and status fields. MCC is applicable to classification, not regression; regression results use MAE, RMSE, and R². For sparse multi-task datasets such as PCBA and MUV, interpret accuracy alongside PR-AUC, MCC, and label prevalence.

A predictive score is affected by the difficulty and composition of each test split. Lower scores on a structurally separated split do not by themselves indicate a defective splitter. Compare predictive metrics with split sizes, target distributions, and structural-similarity estimates. Current single-seed results are exploratory; repeat seeds and report variability before drawing general conclusions.

## Current result coverage and limitations

The unified results currently contain full DataSAIL splits for BACE, BBBP, Tox21, ESOL, FreeSolv, and Lipophilicity. Unified full splits for HIV, QM9, MUV, and PCBA are not currently present. A prior HIV full-DataSAIL runtime log records 1,899 seconds (about 32 minutes), but its DataSAIL output is in the legacy layout and has not been normalized into the unified split directory. A roughly half-day QM9 full-run duration has been reported, but no corresponding unified split and run metadata are present here. Treat both as context until their logs, input versions, and outputs are verified and recorded in the unified format. PCBA full DataSAIL is considered infeasible in the current project plan.

The current scaled runs show that the DataSAIL anchor stage can be a small fraction of end-to-end time on larger datasets; fingerprint generation and nearest-neighbor assignment also contribute substantially. Therefore, do not present DataSAIL-only runtime as the total cost of scaled splitting. Full DataSAIL comparisons on the largest datasets remain incomplete until the planned HPC jobs are completed or documented as infeasible.

QM9's current regression results include extremely poor R² for target `A` on some splits. That target dominates a macro average across QM9 properties and makes a single aggregate R² plot hard to interpret.
## Results organization and provenance

`results/analysis/unified/` is the current analysis output. `results/analysis/Preliminary/` contains older exploratory outputs. Other archived scripts and historical result layouts may use different schemas, paths, seeds, and timing definitions; do not combine them with unified outputs without checking provenance.

Each new run writes metadata including its source-data hash, seed, parameters, package versions, split counts, runtimes, and Git revision/dirty status. 
