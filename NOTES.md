
## Day 1 Progress - COMPLETED ✓

### DataSAIL Baseline Established
- **Dataset:** BACE (1,513 molecules)
- **Method:** Cluster-based split (C1e) with ECFP fingerprints
- **Runtime:** 2.61 seconds
- **Clusters:** 10 (sizes: 67-327, mean=151)
- **Split distribution:**
  - Train: 1,040 (68.7%) - target 70%
  - Val: 323 (21.3%) - target 20%
  - Test: 150 (9.9%) - target 10%

### Key Files
- Splits: `results/splits/bace_baseline/C1e/Molecule_bace_smiles_splits.tsv`
- Clusters: `results/splits/bace_baseline/C1e/Molecule_bace_smiles_clusters.tsv`
- Visualizations: PNG files in same directory

### DataSAIL API Understanding
- Input: CSV with columns `id` and `smiles`
- Output: Subdirectory named after technique (e.g., `C1e/`)
- Files format: TSV with ID and assignment
- Function returns: None (writes to disk)

### Next Steps (Day 2)
1. Implement random down-sampling
2. Run DataSAIL on down-sampled data
3. Implement kNN assignment for remaining samples
4. Compare leakage metrics

