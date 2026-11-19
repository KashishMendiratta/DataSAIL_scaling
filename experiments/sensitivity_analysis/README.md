# kNN assignment sensitivity analysis

This experiment reuses the saved seed-42 DataSAIL anchor memberships and labels
for the configured random-anchor and stratified-anchor methods. It does not rerun
DataSAIL or change anchor labels. It varies balanced-kNN `k` over 1, 3, 5, 10,
and 20, and varies the processing order of non-anchor molecules.

Run from the repository root:

```bash
python experiments/sensitivity_analysis/knn_order_sensitivity.py
```

The default dataset scope is the four primary datasets in the registry
(`bace`, `bbbp`, `tox21`, `hiv`). Results go to `results/sensitivity_analysis/`.
For a smaller or different scope, pass dataset names positionally and optionally
`--strategies random stratified` or `--n-jobs N`.

Processing orders are original source order, reverse source order, and five
deterministic random permutations (seeds 11, 23, 42, 101, 202). For each
dataset/strategy, nearest neighbors are computed once at the largest requested
`k`, then reused across `k` and order runs. The saved detailed assignments allow
the summaries to be audited without recomputing the experiment.

The summary reports final split counts and assignment changes relative to
original source order at the same `k`. It does not calculate leakage or ML
metrics. The separate anchor holdout procedure should be run after these
sensitivity results have been audited.

After the sensitivity audit, select `k` without using leakage or ML metrics:

```bash
python experiments/sensitivity_analysis/anchor_holdout_selection.py
```

This uses fixed five-fold stratified holdouts of the saved anchors and averages
balanced accuracy over the same seven assignment orders. It also saves the
held-out molecule IDs and fixed DataSAIL reference labels. It macro-averages the
fold/order scores within each dataset/strategy, then macro-averages the eight
dataset/strategy groups so larger datasets do not dominate. The selected value
and score table are saved under
`results/sensitivity_analysis/anchor_holdout_selection/`; `locked_k.json` is
the record to use before evaluating final-split leakage or ML metrics.

With the selected `k` locked, generate the source-order final assignments and
run leakage plus predictive evaluation:

```bash
python experiments/sensitivity_analysis/run_locked_k3_metrics.py
```

The final split files for all configured datasets are under
`results/sensitivity_analysis/final_splits_k3/`; consolidated leakage and ML
outputs are under `results/sensitivity_analysis/metrics_k3/`. Running the
command with no dataset arguments processes all configured datasets. The
locked `k=3` was selected by anchor holdout on the four primary datasets and
applied unchanged to the other configured datasets.

We conducted a sensitivity analysis to assess how the balanced k-nearest-neighbor assignment stage depends on its neighborhood size and on the order in which non-anchor molecules are processed. This check was motivated by the greedy assignment procedure: each assignment updates the current split counts, which can affect subsequent decisions. We compared candidate values of *k* and several processing orders while keeping the saved DataSAIL anchor assignments fixed. We then selected *k* using five-fold holdout prediction of the anchor split labels, without using downstream leakage or machine-learning metrics to choose the parameter. This analysis therefore evaluates assignment stability and provides a pre-specified basis for choosing *k*; it does not establish which value gives the best downstream predictive performance.