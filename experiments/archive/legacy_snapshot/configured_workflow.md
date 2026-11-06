# Configured scaled DataSAIL workflow

The selected random and stratified entry points now use the shared dataset settings in `config/datasets.json`. The previous hard-coded implementations and the non-selected naive pipeline are preserved under `pipelines/archive/legacy/`; configured outputs are kept separate from historical results.

The active scaled strategies are:

- `datasail_scaled_random`: random anchor sampling, DataSAIL anchor splits, then balanced kNN assignment.
- `datasail_scaled_stratified`: label-stratified anchor sampling, DataSAIL anchor splits, then balanced kNN assignment. For regression datasets, stratification uses quantile bins of the configured target.

Both variants preserve a standardized `mol_id` in output CSVs, pass those IDs into DataSAIL, write invalid SMILES IDs into run metadata, and fail if valid IDs are missing, duplicated, or assigned outside the configured splits.

## Run

From the repository root, use the project environment:

```bash
python experiments/pipelines/run_datasail_scaled_random.py qm9
python experiments/pipelines/run_datasail_scaled_stratified.py qm9
python experiments/analysis/analyze_scaled_configured.py qm9 --split-types datasail_scaled_random datasail_scaled_stratified
python experiments/analysis/run_predictive_evaluation_configured.py qm9 --split-types datasail_scaled_random datasail_scaled_stratified
```

For the unified original-plus-extended workflow, use `experiments/analysis/run_all_datasets_analysis.py`. It uses the shared registry for all ten datasets, reads configured scaled splits and available legacy random/full splits, and produces the same runtime, leakage, and predictive outputs in one combined results directory:

```bash
python experiments/analysis/run_all_datasets_analysis.py --rerun-scaled --n-jobs 4
```

This refreshes both scaled strategies for every dataset and then runs leakage and predictive analyses for every available random, scaled, and full split. Use `--aggregate-only` after per-dataset Condor jobs have completed to create the combined tables and plots.

Validate inputs and settings without running DataSAIL:

```bash
python experiments/pipelines/run_datasail_scaled_stratified.py qm9 --dry-run
```

Results from these copies are written under `results/configured/splits/<dataset>/datasail_scaled_<strategy>/` and DataSAIL intermediates under `results/configured/experiments/` and `results/analysis/configured/` to keep the existing experiment outputs separate.

## Confidence variants

Confidence variants are not part of the active configured workflow. The original confidence pipeline scripts have been moved into `pipelines/archive/confidence/` and are no longer in the active pipelines folder. The legacy extended pipeline remains untouched.

## Integration note

These files are an experiment-level proposal for a shared configuration and ID-preserving interface. Review the schema and behavior with the DataSAIL maintainers before moving this interface into the standard package.
