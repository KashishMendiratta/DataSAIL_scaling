# Progress Report: Scaling DataSAIL to Large Datasets
## 5th January 2025

---

## Executive Summary

Successfully implemented and tested the complete end-to-end pipeline for scaling DataSAIL through down-sampling and kNN assignment. A critical bug in index handling was identified and resolved, improving split distributions significantly. The pipeline now achieves 2.5x speedup on BACE dataset (1,513 molecules) with moderate deviations (±2-9%) from target 70/20/10 split distributions. The test set is now adequately represented at 10-12% (target: 10%), compared to severely underrepresented (<2%) in the buggy version.

---

## 1. Tasks Completed

### Core Implementation
- Down-sampling strategies (random, stratified, diversity-based)
- kNN assignment methodology with confidence scoring
- End-to-end pipeline integration
- Testing on BACE dataset with multiple configurations

### Bug Fixes & Improvements
- Fixed critical index preservation bug in down-sampling
- Corrected final split combination logic
- Added split distribution validation and reporting
- Verified all 1,513 samples correctly assigned

---

## 2. Implementation Details

### 2.1 Down-sampling Strategies

Implemented in `src/downsampling/strategies.py`:

**A. Random Sampling**
- Randomly selects specified fraction of data
- Fast execution (<0.02s)
- **Key feature**: Preserves original DataFrame indices using `.loc[]`

**B. Stratified Sampling**
- Maintains class distribution proportions
- Original: 54.3% / 45.7%
- Sampled: 54.2% / 45.8% (nearly identical)

**C. Diversity-based Sampling (k-means)**
- Selects chemically diverse representatives
- Uses k-means on molecular fingerprints
- Ensures coverage of chemical space

All methods tested at 10%, 25%, and 50% ratios.

### 2.2 kNN Assignment

Implemented in `src/assignment/knn_assignment.py`:

**Features:**
- Multiple voting strategies: majority, weighted, closest neighbor
- Confidence scoring for each assignment
- Configurable k values (tested: 1, 3, 5, 10)
- Uses Jaccard distance on molecular fingerprints

**Observations:**
- High confidence maintained (92-97%) across configurations
- k=5 provides good balance between confidence and diversity

### 2.3 Full Pipeline

Implemented in `experiments/run_full_pipeline.py`:

**Workflow:**
1. Down-sample dataset (preserving original indices)
2. Run DataSAIL on down-sampled data (cluster-based split with ECFP)
3. Assign remaining samples using kNN
4. Combine into final split assignments

---

## 3. Experimental Results

### 3.1 Test Configurations

| Config | Down-sample Ratio | k | Purpose |
|--------|------------------|---|---------|
| 1 | 25% | 3 | Moderate sampling, lower k |
| 2 | 25% | 5 | Moderate sampling, higher k |
| 3 | 10% | 5 | Aggressive sampling, higher k |

### 3.2 Performance Results

| Config | DataSAIL | kNN | Total | vs Baseline |
|--------|----------|-----|-------|-------------|
| 1 (25%, k=3) | 2.14s | 0.32s | 2.46s | 0.95x |
| 2 (25%, k=5) | 1.94s | 0.26s | 2.22s | 0.85x |
| 3 (10%, k=5) | 0.87s | 0.16s | 1.04s | **0.40x (2.5x speedup)** |

**Baseline**: Full DataSAIL = 2.6s

### 3.3 Split Distribution Results

#### Before Bug Fix (Initial Implementation):
```
Config 1 (25%, k=3): Train=58.1%, Val=15.1%, Test=1.9%  Test severely underrepresented
Config 2 (25%, k=5): Train=60.3%, Val=13.9%, Test=0.7%  Test critically low
Config 3 (10%, k=5): Train=78.6%, Val=10.3%, Test=1.1%  Major imbalance
```

**Problem**: 378-1,362 samples unassigned due to index handling bug

#### After Bug Fix (Current):

**Configuration 1: 25% down-sampling, k=3**
- Down-sampled: Train=255 (67.5%), Val=80 (21.2%), Test=43 (11.4%)
- kNN-assigned: Train=729 (64.2%), Val=265 (23.3%), Test=141 (12.4%)
- **Combined (all 1,513)**: Train=984 (65.0%), Val=345 (22.8%), Test=184 (12.2%)
- **Deviations**: -5.0%, +2.8%, +2.2% 

**Configuration 2: 25% down-sampling, k=5**
- Down-sampled: Train=255 (67.5%), Val=80 (21.2%), Test=43 (11.4%)
- kNN-assigned: Train=728 (64.1%), Val=265 (23.3%), Test=142 (12.5%)
- **Combined (all 1,513)**: Train=983 (65.0%), Val=345 (22.8%), Test=185 (12.2%)
- **Deviations**: -5.0%, +2.8%, +2.2% 

**Configuration 3: 10% down-sampling, k=5**
- Down-sampled: Train=105 (69.5%), Val=32 (21.2%), Test=14 (9.3%)
- kNN-assigned: Train=810 (59.5%), Val=410 (30.1%), Test=142 (10.4%)
- **Combined (all 1,513)**: Train=915 (60.5%), Val=442 (29.2%), Test=156 (10.3%)
- **Deviations**: -9.5%, +9.2%, +0.3% 

**Target**: 70% train, 20% val, 10% test

---

## 4. Bug Analysis & Resolution

### 4.1 The Bug

**Problem**: Original down-sampling functions used `reset_index(drop=True)`, losing original DataFrame indices.

**Impact**: 
- 25% sampling: 378 samples unassigned (25%)
- 10% sampling: 151 samples unassigned (10%)
- Final splits only contained kNN-assigned samples
- Test set collapsed to <2%

### 4.2 The Fix

**Solution**: Modified all down-sampling functions to:
1. Use `.loc[]` to preserve original indices
2. Track original indices explicitly
3. Map assignments back to original positions

**Code change**:
```python
# Before (WRONG):
sampled_data = data.iloc[sampled_indices].reset_index(drop=True)

# After (CORRECT):
sampled_data = data.loc[sampled_indices].copy()
```

### 4.3 Validation

After fix:
- All 1,513 samples assigned
- No NaN values in final splits
- Splits sum to 100%
- Test set properly represented (10-12%)

---

## 5. Current Issues & Next Steps

### 5.1 Remaining Issue: Moderate Split Deviations

**Status**: Deviations of ±2-9% from targets

**Analysis**:
- Positive: Test set adequately represented (10-12%)
- Issue: Val set over-represented (23-29% vs target 20%)
- Issue: Train set under-represented (60-65% vs target 70%)

**Root Cause**: kNN assignment doesn't account for current split sizes

**Priority**: Moderate (splits are usable but suboptimal)

### 5.2 Next Implementation

**Stratified + Distance-Weighted kNN Assignment**:
```python
for each remaining sample:
    for each split in [train, val, test]:
        # Similarity to split
        similarity = avg_distance_to_k_nearest_in_split
        
        # Balance factor
        current_ratio = current_split_size / total
        target_ratio = target_split_sizes[split]
        balance_weight = target_ratio / (current_ratio + epsilon)
        
        # Combined score
        score = similarity * balance_weight
    
    assign_to_highest_score()
```

**Expected**: Reduce deviations from ±2-9% to ±1-3%

---

## 6. Deliverables

### Code Modules:
- `src/downsampling/strategies.py` - Down-sampling (index-safe)
- `src/assignment/knn_assignment.py` - kNN assignment
- `src/utils.py` - Molecular fingerprints

### Scripts:
- `experiments/run_full_pipeline.py` - Full pipeline
- `experiments/test_downsampling.py` - Down-sampling tests
- `experiments/test_knn_assignment.py` - kNN tests
- `experiments/verify_split_distributions.py` - Validation

---

## 7. References

- DataSAIL paper: https://doi.org/10.1038/s41467-025-58606-8
- MoleculeNet: https://moleculenet.org/
- Repository: https://github.com/kashishm14/Data-SAIL_scaling

---
