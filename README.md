# Scaling DataSAIL for molecular and interaction datasets

This repository contains the implementation and experiments for scaling DataSAIL to larger molecular and ligand–target interaction datasets. The scaled method runs DataSAIL on a representative anchor subset, then extends the anchor assignments to the remaining entities using similarity, split-balance, and, where applicable, interaction-graph information. The original study compares random splitting, scaled DataSAIL with random or stratified molecular anchors, and Full DataSAIL where a verified full run is available. The BindingDB extension adds cold-ligand (C1e), cold-target (C1f), and double-cold (C2) splitting at interaction-database scale.

The molecular study covers ten datasets, from 642 to 437,927 valid molecules. Random and both scaled methods were evaluated over five split seeds. Nine datasets have verified single-run Full DataSAIL references; PCBA is the only dataset without one. The accepted QM9 and MUV Full runs used LOBPCG with `e_clusters=10`. Later failed runs under different cluster/eigensolver configurations are retained as diagnostic ablations and do not invalidate those accepted references.

The BindingDB study uses the frozen 202611 release: 3,243,660 source measurements, 3,036,683 typed affinity observations, 1,357,780 ligand IDs representing 1,348,653 canonical chemicals, and 10,518 single-sequence targets. Full scaled C1e, C1f, and C2 candidates and their independent hard validation stages passed. C2 retained 2,679,363 observations (88.23%) and recorded an explicit exclusion outcome for every dropped cross-split interaction.

The original thesis report is [`reports/MASTER REPORT.pdf`](reports/MASTER%20REPORT.pdf). It is a historical snapshot and retains its original scope and claims. This README describes the newer November 2026 evidence, which supersedes the report for current repository and portfolio summaries.

## Current workflow

The active pipelines are:

| Script | Purpose |
|---|---|
| `experiments/pipelines/random_pipeline.py` | Shuffle valid molecules and divide them into train, validation, and test sets. |
| `experiments/pipelines/dataSAIL_scaled_random.py` | Select anchors randomly, split anchors with DataSAIL, then assign the remaining molecules with balanced kNN. |
| `experiments/pipelines/dataSAIL_scaled_stratified.py` | Select anchors while preserving the configured target strata, then use the same DataSAIL and balanced-kNN steps. |
| `experiments/pipelines/full_dataSAIL.py` | Run DataSAIL on all valid molecules. This requires HPC resources for larger datasets. |
| `experiments/pipelines/bindingdb_scaled_c1e.py` | Split canonical chemicals, then expand assignments to BindingDB ligand IDs. |
| `experiments/pipelines/bindingdb_scaled_c1f.py` | Split protein targets using hashed 3-mer similarity and degree-aware extension. |
| `experiments/pipelines/bindingdb_scaled_c2.py` | Jointly seed both entity sides and extend them with similarity, graph votes, and explicit edge retention. |

The molecular pipelines use the shared dataset registry and defaults in `experiments/config/datasets.json`. The default split proportions are 70% train, 20% validation, and 10% test. The command-line default seed is 42; the main repeated-seed study uses 42, 101, 202, 303, and 404; scaled anchor fractions are specified per dataset; and the default fingerprint is Morgan radius 2 with 2048 bits. The provenance-controlled molecular study uses balanced kNN with `k=5` and balance weight 2.0. Other values of `k` are evaluated only as a separate robustness analysis and do not replace the primary `k=5` results.

The BindingDB pipelines use the same anchor-and-extension principle but currently have regime-specific implementations because C1e balances ligand interactions, C1f operates on protein representations, and C2 must account for both endpoint assignments and retained interactions. Consolidating these paths with the original molecular engine into one reusable, adapter-based splitter is the next software-engineering step; the current accepted scientific outputs remain tied to the recorded implementations and manifests.

For classification, stratified anchor selection uses the configured stratification label. For regression, it bins the configured target into quantiles. Stratification applies to anchor selection; it does not guarantee that every final split exactly matches the source target distribution.

## Environment and data

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate datasail-scale
```

The registry currently includes BACE, BBBP, Tox21, HIV, QM9, PCBA, MUV, ESOL, FreeSolv, and Lipophilicity. Dataset paths, columns, task types, and anchor fractions are defined in `experiments/config/datasets.json`.

The core splitting and fingerprint-analysis workflow uses the `datasail-scale` Conda environment. The D-MPNN extension uses a separate `.venv-dmpnn` environment with Chemprop, PyTorch, and the dependencies in `requirements-dmpnn.txt`. BindingDB neural predictors use separately pinned CPU PyTorch and ESM-2 container images; the split-generation image intentionally does not contain PyTorch. One interchangeable environment will be added later.

Downloaders are in `experiments/dataset_download/`. For example:

```bash
python experiments/dataset_download/download_qm9.py
python experiments/dataset_download/download_pcba.py
python experiments/dataset_download/download_muv.py
python experiments/dataset_download/download_esol.py
python experiments/dataset_download/download_freesolv.py
python experiments/dataset_download/download_lipophilicity.py
```

The original MoleculeNet CSVs belong under `data/raw/moleculenet/`. They are local inputs and are not included in Git.

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

### Reproduce the BindingDB workflow

BindingDB preparation streams `BindingDB_All.tsv` directly from the frozen
202611 zip archive and retains Ki, Kd, IC50, and EC50 as distinct typed
observations. Multi-chain targets are excluded from the first implementation
because C1f and C2 require one unambiguous protein sequence per target. Exact
affinity relations and censored relations are preserved; predictive point
regression uses equality observations only.

The accepted execution order is:

1. prepare and validate the frozen release;
2. generate and independently validate C1e;
3. generate and independently validate C1f;
4. generate and independently validate C2;
5. measure ligand and protein nearest-training-neighbour leakage;
6. run the corrected predictive baselines and representation ablations.

Full runs are submitted through the provenance-enforcing helper rather than by
calling `condor_submit` directly. For example:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_c1e.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro" seed=42 anchor_ratio=0.001 threads=1

experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_c1e_validation.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

The corresponding C1f, C2, leakage, baseline, sequence-CNN, and ESM-2 commands
are documented in `experiments/condor/README.md`. Python entry points retain
bounded pilot defaults; the Condor shell runners contain the locked full-run
values.

Accepted split assignments are immutable. Compact manifests can be copied from
the submit host and consolidated with:

```bash
experiments/analysis/sync_bindingdb_artifacts.sh

python -m experiments.analysis.consolidate_bindingdb_results \
  artifacts/bindingdb/accepted results/analysis/bindingdb/consolidated

python -m experiments.analysis.plot_bindingdb_thesis \
  results/analysis/bindingdb/consolidated reports/figures/bindingdb
```

The detailed append-only experimental record is in
`docs/bindingdb_worklog.md`. The compact accepted-run matrix and artifact layout
are in `docs/BINDINGDB_REPRODUCIBILITY.md`.

### Reproduce the five-seed comparison

The main repeated-seed study uses seeds 42, 101, 202, 303, and 404 for Random, scaled-random-anchor, and scaled-stratified-anchor splitting. Run the matched sweep with:

```bash
python experiments/analysis/run_seed_sweep.py \
  --seeds 42 101 202 303 404 \
  --n-jobs 1
```

The sweep writes each replicate to a seed-specific directory and refuses to overwrite an existing replicate. Full DataSAIL is excluded from this sweep because it is represented by one reference run per available dataset rather than five independent replicates.

### kNN assignment sensitivity analysis

The thesis sensitivity analysis is an appendix-level robustness check. The
primary provenance-controlled molecular results remain the `k=5` runs described
above; the sensitivity outputs do not retrospectively change that configuration.

The non-anchor assignment stage is sequential: every assignment updates the
current split balance, so both neighbourhood size and processing order can
affect later decisions. The sensitivity workflow keeps the saved seed-42
DataSAIL anchor IDs and split labels fixed, then varies `k` over 1, 3, 5, 10,
and 20 and compares source, reverse, and five deterministic random processing
orders.

```bash
python experiments/sensitivity_analysis/knn_order_sensitivity.py
python experiments/sensitivity_analysis/anchor_holdout_selection.py
python experiments/sensitivity_analysis/run_locked_k3_metrics.py
python experiments/sensitivity_analysis/compare_locked_k3_to_existing_k5.py
```

The first audit covers BACE, BBBP, Tox21, and HIV under both random and
stratified anchor selection: 280 dataset/strategy/`k`/order combinations. Across
non-source orders, a mean 3.15% of non-anchor assignments changed relative to
source order; the largest observed change was 11.70%. This is sufficient
evidence that the greedy extension is order-dependent, but it does not identify
one scientifically optimal order.

Within the sensitivity study, `k` was selected without using final-split
leakage or predictive metrics.
Five-fold anchor holdout, averaged across seven processing orders and eight
dataset/strategy groups, produced macro balanced accuracies of 0.8469, 0.8511,
0.8328, 0.7958, and 0.7548 for `k=1,3,5,10,20`, respectively. The resulting
sensitivity value was `k=3`. It was then applied unchanged to all ten configured
datasets for the robustness comparison; the six datasets outside the
holdout-selection scope are transfer applications, not independent validation
of that value. This does not supersede the primary `k=5` provenance runs.

The descriptive comparison with the existing seed-42 `k=5` results found lower
mean sampled train–test Tanimoto similarity for `k=3` in all 20
dataset/strategy pairs and lower mean nearest-training-neighbour similarity in
17 of 20. Predictive changes were small and mixed, so the comparison does not
claim a universal predictive advantage for either `k`. The selection decision,
detailed assignments, evaluation manifests, and interpretation limits are
documented in `experiments/sensitivity_analysis/README.md` and
`results/sensitivity_analysis/audit.md`.

The planned reusable scaled-DataSAIL algorithm will treat assignment order as
an explicit, versioned policy rather than inheriting input-file order. It will
use deterministic ordering and stable entity-ID tie-breaking, record the policy
and seed in every manifest, and test that permuting input rows does not change
the output. A batch or iterative assignment method could reduce order dependence
more fundamentally, but that would be a new methodological variant requiring
its own validation rather than a silent modification of the runs.

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

A predictive score is affected by the difficulty and composition of each test split. Lower scores on a structurally separated split do not by themselves indicate a defective splitter. Compare predictive metrics with split sizes, target distributions, structural-similarity estimates, and uncertainty across split seeds. The current provenance results do not contain five-seed Random Forest or Logistic Regression comparisons; public summaries name the evaluated SGD/Ridge and D-MPNN models instead.

BindingDB has a separate interaction-aware regression suite evaluated on the
accepted C1e, C1f, and retained C2 interactions:

1. a Huber-SGD model using chiral Morgan, hashed protein 3-mer, and affinity-type features;
2. a compact nonlinear ligand–protein dual encoder using the same engineered representations;
3. a sequence-CNN target encoder over the first 512 residues;
4. a cached pretrained ESM-2 8M target representation covering complete sequences through overlapping chunks.

All BindingDB point-regression comparisons use exact-equality affinity records
and a corrected deterministic pseudo-random sampler. The 500k linear sensitivity
showed that additional rows improved C1e but did not materially improve C1f or
C2. ESM-2 provided the strongest overall biological representation trade-off:
it produced the best observed C1e MAE and correlations and the best C2 MAE,
RMSE, and R², while the sequence CNN retained the best C2 correlations. None of
the nonlinear target representations surpassed the linear baseline on all C1f
metrics, so the current results do not claim that cold-target generalization is
solved.


## Current result coverage and limitations

The selected November 2026 molecular evidence contains five-seed results for Random, scaled-random-anchor, and scaled-stratified-anchor splitting across all ten configured datasets. Verified single-run Full DataSAIL references are available for BACE, BBBP, Tox21, HIV, QM9, MUV, ESOL, FreeSolv, and Lipophilicity. PCBA remains the only dataset without a Full reference. QM9 and MUV succeeded with LOBPCG and `e_clusters=10`; unsuccessful alternative configurations remain diagnostic failures.

Runtime summaries distinguish DataSAIL-stage time from end-to-end time. The scaled DataSAIL call can be short while full-dataset validation, fingerprinting, non-anchor assignment, and output generation remain substantial. Full results are single-run references; repeated-seed uncertainty applies only to Random and the two scaled methods.

Scaled-random reduced the sampled fraction of test molecules with a training neighbour above the selected Tanimoto threshold across all ten datasets relative to Random. Scaffold overlap provides a separate diagnostic: on BACE, the mean fraction of test molecules sharing a scaffold with training fell from 65.6% under Random to 2.45% under scaled-random. These results apply to the stored sampling and overlap definitions and should not be interpreted as proof of zero leakage under every criterion.

QM9 contains 19 heterogeneous regression targets. Target `A` is unstable on some splits and can dominate pooled averages, so QM9 predictive performance should be reported per target rather than as a single unqualified aggregate. More generally, predictive results measure how the evaluated models respond to each split; they are complementary to, but not substitutes for, direct structural-similarity and scaffold-overlap diagnostics.

### BindingDB split and leakage results

All three seed-42 scaled BindingDB regimes passed their independent hard
validation gates:

| Regime | Atomic cold entity | Retained observations | Train / validation / test |
|---|---|---:|---|
| C1e | canonical chemical | 3,036,683 (100%) | 69.81% / 20.03% / 10.16% |
| C1f | target sequence | 3,036,683 (100%) | 73.07% / 18.67% / 8.26% |
| C2 | canonical chemical and target sequence | 2,679,363 (88.23%) | 73.76% / 17.98% / 8.26% |

C2 records all 357,320 removed observations with the reason
`cold_double_cross_split_endpoints`. Exact unseen-test-entity rates are 1.0 for
the entity types constrained by each regime, and no accepted split contains an
atomic-identity or duplicate conflict.

Identity-cold does not mean globally similarity-free. The exhaustive maximum
cross-split hashed protein 3-mer cosine similarity is approximately 0.9993 for
C1f and 0.9998 for C2. In the C1f test set, 21.51% of targets have a nearest
training target similarity of at least 0.7 and 10.48% are at least 0.9. C2
reduces this high-similarity tail but does not eliminate near homologues.
Distinct canonical chemicals can also have Morgan similarity 1.0 because of
fingerprint collisions or stereochemical distinctions. Chirality-aware
fingerprints reduce, but do not universally eliminate, these cases.

## Results organization and provenance

`results/analysis/unified/` and `results/analysis/bindingdb/` contain generated working outputs and are excluded from Git. The November 2026 evidence selection is used for the current results and summaries. Historical and exploratory outputs are not combined with the results and summaries.
