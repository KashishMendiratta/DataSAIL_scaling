# Scaling DataSAIL to Large Datasets

Master's thesis project investigating how to scale the DataSAIL dataset splitting algorithm to large molecular datasets (10k–100k+ samples).
The project explores a pipeline combining:
- dataset down-sampling
- DataSAIL cluster-based splitting
- k-nearest neighbour (kNN) assignment for remaining samples
- balanced assignment to maintain target split distributions

## Project Structure
```
Data-SAIL_scaling/

├── data/
│   ├── raw/
│   │   └── moleculenet/
│   │       ├── bace.csv
│   │       ├── bbbp.csv
│   │       ├── tox21.csv
│   │       └── hiv.csv
│   │
│   ├── sampled/
│   │   ├── bace_sampled_25pct.csv
│   │   └── bace_remaining_75pct.csv
│   │
│   ├── processed/
│   │   └── intermediate files
│   │
│   └── temp/
│       └── temporary pipeline files
│
├── src/
│   ├── downsampling/
│   │   └── strategies.py
│   │
│   ├── assignment/
│   │   ├── knn_assignment.py
│   │   └── balanced_knn.py
│   │
│   └── utils.py
│
├── experiments/
│   ├── pipelines/
│   │   ├── run_balanced_pipeline.py
│   │   ├── run_full_pipeline.py
│   │   └── run_datasail_baseline.py
│   │
│   ├── dataset_tests/
│   │   └── test_pipeline_bbbp.py
│   │
│   ├── dataset_download/
│   │   ├── download_bace.py
│   │   ├── download_bbbp.py
│   │   ├── download_more_datasets.py
│   │   └── download_tox21_fix.py
│   │
│   ├── algorithm_tests/
│   │   ├── test_downsampling.py
│   │   ├── test_knn_assignment.py
│   │   └── test_balanced_assignment.py
│   │
│   └── utils/
│       └── verify_split_distributions.py
│
├── results/
│   ├── experiments/        # final experiment outputs
│   ├── datasail_outputs/   # raw DataSAIL outputs
│   ├── pipeline_debug/     # development experiments
│   ├── analysis/           # figures, tables, models
│   └── archived/
│
├── reports/
│   └── progress reports and experiment summaries
│
├── notebooks/
│   └── exploratory analysis notebooks
│
├── tests/
│   ├── test_setup.py
│   └── test_datasail_import.py
│
├── environment.yml
└── README.md
```

## Setup
Create the conda environment:

conda env create -f environment.yml
conda activate datasail-scale

Verify that the setup works:

python tests/test_setup.py
python tests/test_datasail_import.py

## Pipeline Overview

The scalable pipeline works as follows:
1. Down-sample the dataset (e.g. 10–25%)
2. Run DataSAIL on the sampled subset
3. Compute molecular fingerprints
4. Assign remaining molecules using kNN
5. Apply balanced assignment to enforce target split ratios

- Target split distribution:
  - Train: 70%
  - Validation: 20%
  - Test: 10%

## Running the Pipeline

python experiments/pipelines/run_balanced_pipeline.py

Outputs are saved to:
results/experiments/

## Datasets Used
Datasets come from MoleculeNet.
Dataset | Size | Task
BACE | ~1.5k | binary classification
BBBP | ~2k | binary classification
Tox21 | ~8k | multi-task toxicity
HIV | ~41k | activity prediction

## Pipeline Versions
Two pipeline implementations are kept for reproducibility.
- Original Pipeline
run_full_pipeline.py

Contains an early implementation that produced incorrect split distributions due to an index handling issue.
This version is preserved for debugging and documentation.

- Balanced Pipeline (Current)
run_balanced_pipeline.py

Implements:
- index-safe downsampling
- kNN assignment
- split balancing

This is the version used for experiments.

## Results

- Experiment outputs are stored in:
results/experiments/

- Raw DataSAIL outputs are stored in:
results/datasail_outputs/

- Development runs are archived in:
results/pipeline_debug/



