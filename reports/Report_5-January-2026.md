# Progress Report: Scaling DataSAIL to Large Datasets
## 5th January 2025


## Executive Summary

Successfully implemented and tested the complete end-to-end pipeline for scaling DataSAIL through down-sampling and kNN assignment. The pipeline runs successfully in 2-5 seconds on BACE dataset (1,513 molecules), demonstrating significant speedup potential. However, a critical issue was discovered: final split distributions deviate significantly from targets, with the test set severely underrepresented (0.7% vs 10% target).


### Primary Tasks completed:
- Implement down-sampling strategies (random, stratified, diversity-based)
- Implement kNN assignment methodology
- Create end-to-end pipeline integrating all components
- Test pipeline on BACE dataset with multiple configurations


### Down-sampling Strategies

Implemented three down-sampling approaches in `src/downsampling/strategies.py`:

**A. Random Sampling**
- Simple baseline approach
- Randomly selects specified fraction of data
- Fast execution (<0.01s)
- No guarantee of maintaining class balance

**B. Stratified Sampling**
- Maintains class distribution proportions
- Samples proportionally from each class
- Original proportions: 54.3% / 45.7%
- Sampled proportions: 54.2% / 45.8% (nearly identical)

**C. Diversity-based Sampling (k-means)**
- Selects chemically diverse representatives
- Uses k-means clustering on molecular fingerprints
- Chooses samples closest to cluster centers
- Ensures coverage of chemical space

All methods tested at 10%, 25%, and 50% down-sampling ratios.

### kNN Assignment

Implemented in `src/assignment/knn_assignment.py`:

**Features:**
- Multiple voting strategies: majority, weighted, closest neighbor
- Confidence scoring for each assignment
- Configurable k values (tested: 1, 3, 5, 10)
- Uses Jaccard distance on molecular fingerprints

**Observations:**
- k=1: 100% confidence (always assigns to nearest neighbor's split)
- k=5: ~72% mean confidence, more balanced voting
- Higher k values lead to bias toward majority class

### Full Pipeline

Implemented in `experiments/run_full_pipeline.py`:

**Workflow:**
1. Down-sample dataset to specified ratio
2. Run DataSAIL on down-sampled data (cluster-based split with ECFP)
3. Assign remaining samples using kNN
4. Combine results into final split assignments

**Key Components:**
- Automatic file I/O handling
- Progress tracking and timing
- Results validation and reporting
- TSV output with split assignments and source tracking


### Test Configurations

Tested three configurations on BACE dataset (1,513 molecules):

| Config | Down-sample Ratio | k | Purpose |
|--------|------------------|---|---------|
| 1 | 25% | 3 | Moderate sampling, lower k |
| 2 | 25% | 5 | Moderate sampling, higher k |
| 3 | 10% | 5 | Aggressive sampling, higher k |

### Performance Results

| Config | DataSAIL Time | kNN Time | Total Time | Speedup Potential |
|--------|---------------|----------|------------|-------------------|
| 1 (25%) | 4.20s | 0.75s | 4.96s | ~2x vs baseline |
| 2 (25%) | 3.75s | 0.38s | 4.15s | ~2.3x vs baseline |
| 3 (10%) | 1.63s | 0.27s | 1.91s | **~5x vs baseline** |

**Baseline**: Full DataSAIL on 1,513 molecules = ~2.6s (from Day 1)

**Key Finding**: With 10% down-sampling, total pipeline time is actually comparable to full DataSAIL on this small dataset, but the approach scales better for larger datasets.

### Split Distribution Results

**Configuration 1 (25% sampling, k=3):**
```
Down-sampled splits: Train=67.5%, Val=21.2%, Test=11.4%
Final splits:        Train=58.1%, Val=15.1%, Test=1.9%
Deviation:           -11.9%,      -4.9%,     -8.1%
```

**Configuration 2 (25% sampling, k=5):**
```
Down-sampled splits: Train=67.5%, Val=21.2%, Test=11.4%
Final splits:        Train=60.3%, Val=13.9%, Test=0.7%
Deviation:           -9.7%,       -6.1%,     -9.3%
```

**Configuration 3 (10% sampling, k=5):**
```
Down-sampled splits: Train=69.5%, Val=21.2%, Test=9.3%
Final splits:        Train=78.6%, Val=10.3%, Test=1.1%
Deviation:           +8.6%,       -9.7%,     -8.9%
```

**Target splits**: 70% train, 20% val, 10% test



### Split Distribution Bias

**Problem**: Final split distributions deviate significantly from targets, particularly for the test set.

**Evidence**:
- Test set receives only 0.7-1.9% of samples (target: 10%)
- Train set receives 58-79% of samples (target: 70%)
- Worse with higher k values

**Root Cause**: kNN assignment exhibits majority class bias
- Most neighbors in down-sampled data belong to train set (~70%)
- Majority voting naturally assigns most samples to train
- Minority classes (val, test) become underrepresented

**Impact**: 
- Severely imbalanced splits unsuitable for model evaluation
- Test set too small for reliable performance assessment
- Violates intended 70/20/10 distribution


### 5. Next Steps

**Fix Split Distribution Bias** - Three potential approaches:

**Option A: Stratified kNN Assignment**
- Enforce target proportions during assignment
- Assign samples until each split reaches its quota
- Advantages: Guarantees correct distribution
- Disadvantages: May assign low-confidence samples to meet quotas

**Option B: Distance-Weighted with Class Balancing**
- Weight assignments by both similarity and current split sizes
- Penalize over-represented splits
- Advantages: More principled, considers both factors
- Disadvantages: More complex, requires tuning

**Option C: Cluster-based Assignment**
- Group remaining samples into clusters
- Assign entire clusters to maintain diversity
- Advantages: Better preserves chemical diversity
- Disadvantages: Less granular control


### Further Plans:

2. **Baseline Comparison**
   - Compare splits from pipeline vs full DataSAIL
   - Compute leakage metrics as defined in DataSAIL paper
   - Visualize similarity distributions

3. **Multi-dataset Testing**
   - Test on BBBP (~2,000 molecules)
   - Test on Tox21 (~8,000 molecules)
   - Test on HIV (~41,000 molecules) - first true scalability test

4. **Model Training**
   - Train Random Forest and XGBoost models
   - Compare performance on different splits
   - Validate that splits maintain predictive difficulty


## References

- DataSAIL paper: https://doi.org/10.1038/s41467-025-58606-8
- MoleculeNet datasets: https://moleculenet.org/
- Code repository: https://github.com/kashishm14/Data-SAIL_scaling

