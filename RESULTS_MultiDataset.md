# Multi-Dataset Testing Results
## Date: 6th January 2025

## Summary

Tested the down-sampling pipeline on two MoleculeNet datasets to verify generalizability.

### Datasets Tested

| Dataset | Size | Task | Source |
|---------|------|------|--------|
| BACE | 1,513 molecules | Binary classification | Blood-brain barrier permeability |
| BBBP | 2,039 molecules | Binary classification | Blood-brain barrier penetration |

---

## Results: Current Pipeline (25% down-sampling, k=5)

### BACE Dataset

**Performance:**
- DataSAIL time: 1.94s
- kNN time: 0.26s
- **Total: 2.22s** (vs 2.6s baseline = **0.85x, ~15% speedup**)

**Split Distribution:**
- Train: 983 (65.0%) — Target: 70%, **Deviation: -5.0%**
- Val: 345 (22.8%) — Target: 20%, **Deviation: +2.8%**
- Test: 185 (12.2%) — Target: 10%, **Deviation: +2.2%**

**Quality:** ✅ Moderate deviations, acceptable

---

### BBBP Dataset

**Performance:**
- DataSAIL time: 2.39s
- kNN time: 0.54s
- **Total: 2.93s** (estimated baseline ~3.5s = **0.84x, ~16% speedup**)

**Split Distribution:**
- Train: 1,723 (84.5%) — Target: 70%, **Deviation: +14.5%** ❌
- Val: 233 (11.4%) — Target: 20%, **Deviation: -8.6%** ❌
- Test: 83 (4.1%) — Target: 10%, **Deviation: -5.9%** ❌

**Quality:** ❌ Large deviations, NOT acceptable

---

## Key Findings

### ✅ What Works:
1. **Pipeline generalizes** - Works on both datasets
2. **Speedup consistent** - 15-16% speedup on small datasets
3. **All samples assigned** - No missing data bugs
4. **High confidence** - Mean confidence ~74-96%

### ❌ Critical Issue:
**Split distributions are inconsistent and often far from targets**

- BACE: Moderate deviations (±2-5%)
- BBBP: Large deviations (+14.5% on train!)

**Root Cause:** kNN assignment doesn't balance splits
- Assigns based purely on similarity
- No awareness of current split sizes
- Results in majority class (train) accumulating samples

---

## Impact on Thesis

### For Master Seminar:

**Can show:**
- ✅ Working prototype on multiple datasets
- ✅ Proof of concept for speedup
- ✅ Identified the key challenge

**Cannot show yet:**
- ❌ Reliable split distributions
- ❌ Comparison with full DataSAIL baseline
- ❌ Model performance validation

### Next Steps (URGENT):

1. **Implement balanced kNN assignment** (Priority 1)
   - Add split-size weighting to assignment
   - Test on both BACE and BBBP
   - Target: Reduce deviations to ±2-3%

2. **Baseline comparison** (Priority 2)
   - Run full DataSAIL on BACE and BBBP
   - Compare split distributions sample-by-sample
   - Compute leakage metrics

3. **Model validation** (Priority 3)
   - Train Random Forest on different splits
   - Show that splits maintain difficulty
   - Validate approach empirically

---

## Timeline

**This week (Jan 6-12):**
- Implement balanced assignment
- Re-run on BACE and BBBP
- Show improved results

**Next week (Jan 13-19):**
- Add HIV dataset (~41k molecules)
- Baseline comparison
- Start ML model training

**By end of January:**
- Have stable, validated pipeline
- Results on 3-4 datasets
- Ready for seminar prep in February
