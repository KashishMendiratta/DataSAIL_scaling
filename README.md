# Scaling DataSAIL to Large Datasets

Master thesis project extending DataSAIL to handle datasets with 100k+ samples.

## Project Structure
```
├── data/                  # Datasets (not tracked in git)
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed data
│   └── splits/           # Generated splits
├── src/                   # Source code
│   ├── downsampling/     # Down-sampling strategies
│   ├── assignment/       # Assignment methods (kNN, etc)
│   └── evaluation/       # Metrics and evaluation
├── experiments/           # Experiment scripts and configs
├── results/              # Outputs (figures, tables, models)
└── notebooks/            # Jupyter notebooks for exploration
```

## Setup
```bash
conda env create -f environment.yml
conda activate datasail-scale
```

## Important Note on Pipeline Versions

- `run_full_pipeline.py` contains an earlier implementation with
  a known split-distribution bug and is kept for documentation
  and failure analysis purposes.

- `run_full_pipeline_FIXED.py` is the corrected pipeline used
  for all reported experiments.



