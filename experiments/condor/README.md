# Condor container pinning and provenance

The selected runtime image is recorded in `container-image.ref` as an amd64 Linux digest reference. The `py310` tag is mutable, so new jobs must be submitted through `submit_pinned.sh`; the helper validates that the reference contains a SHA-256 digest and supplies it to the submit description as the `image` macro.

Example from the repository root on the authenticated Condor submit host:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/unified_all_datasets.sub \
  repo_dir="$HOME/DataSAIL_scaling" dataset=bace n_jobs=4 memory=16GB disk=20GB
```

For submit descriptions with additional required macros, pass them after the submit file, for example `repo_dir=...`, `dataset=...`, `n_jobs=...`, or `memory=...`. The helper refuses a dirty Git worktree and passes the committed Git ID and clean status into each job. Do not submit these files directly with `condor_submit`: their image and Git provenance macros are supplied by the helper.

Every Condor submit description exports `DATASAIL_CONTAINER_REF` and the Condor job ID. The split pipeline stores the image reference in `run_metadata.json`. Jobs launched through `run_job.sh` also write a per-job sidecar under `results/provenance/condor_jobs/`, including the Git commit and dirty state, input dataset hashes, package versions, worker hostname, parameters, and exit status. Preserve those sidecars together with the split/analysis outputs and Condor event, stdout, and stderr logs when transferring results locally.

A digest pins software image contents. It does not pin the Condor worker hardware or restore provenance for historical runs.

## BindingDB preparation

The full frozen BindingDB release is prepared on Condor after the 10k, 100k,
and 500k local gates pass. The archive must already exist at
`data/raw/bindingdb/BindingDB_All_202611_tsv.zip` on the mounted repository.

From a clean, committed worktree on the submit host:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_prepare.sub \
  repo_dir="$HOME/DataSAIL_scaling" release=202611 chunk_size=10000
```

The job requests 1 CPU, 4 GB memory, and 8 GB disk. Its preparation manifest
records source accounting, typed affinity counts, stage runtimes, and peak
memory. Do not start split generation unless that manifest reports
`source_accounting_valid: true`.

After full preparation passes, submit the first full scaled cold-ligand split:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_c1e.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro" seed=42 anchor_ratio=0.001 threads=1
```

This uses canonical chemical structures as atomic splitting units, approximately
1,350 degree-stratified anchors, interaction-degree weights, k=5 exact
Tanimoto-to-anchor assignment, and 10 DataSAIL clusters. Its core assignment
manifest remains pending until the separate similarity and graph audits pass.

After the candidate finishes successfully, run its hard validation gate:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_c1e_validation.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

This stage fails on incomplete entity or source-measurement accounting, dropped
observations, duplicate leakage, incorrect cold-ligand identity, uncertified
maximum cross-split similarities, missing affinity strata, or incomplete graph
statistics. It writes `validation_report.json` beside the C1e candidate.

## BindingDB scaled C1f

After C1e is accepted, submit the full cold-target candidate:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_c1f.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro" seed=42 anchor_ratio=0.05 threads=1
```

The protein path supplies a precomputed hashed 3-mer cosine matrix through the
DataSAIL Python API, so it does not depend on MMseqs or DIAMOND being installed.
The candidate uses approximately 526 anchors, 20 protein clusters, and fails if
any interaction-ratio deviation exceeds five percentage points. It remains
pending until the separate C1f similarity and graph validator passes.

After the C1f candidate passes, submit its acceptance validator:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_c1f_validation.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

The validator exhaustively computes the hashed 3-mer cosine maximum across all
cross-split target pairs and audits source accounting, regime identity, affinity
and degree distributions, and the complete interaction graph.

## BindingDB scaled C2

After C1f is accepted, submit the full graph-aware double-cold candidate:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_c2.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro" seed=42 threads=1
```

The full run uses approximately 1,349 canonical-chemical anchors and 526 target
anchors. Every mismatched-endpoint observation is retained in the accounting
table with an explicit exclusion reason. The candidate hard-fails below 80%
retention, beyond a five-point retained-ratio deviation, or on any atomicity,
duplicate, or retained cross-endpoint conflict.

After the C2 candidate passes, submit its acceptance validator:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_c2_validation.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

This exhaustively validates observation and measurement outcomes, endpoint
atomicity, exact cross-split maxima for the selected ligand/protein metrics,
retained affinity and degree distributions, and retained-graph components.

## BindingDB leakage distributions

After C1e, C1f, and C2 are accepted, compute their nearest-neighbour leakage
distributions:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_leakage.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

Protein query-to-training comparisons are exhaustive. Ligand comparisons use a
deterministic seed-42 sample of 5,000 queries against 50,000 training chemicals
and are repeated with non-chiral and chirality-aware Morgan fingerprints. Raw
nearest-neighbour records and summary tables are stored under
`results/analysis/bindingdb/seed_42/leakage/`.

## BindingDB bounded predictive baseline

After leakage distributions pass, submit the interaction-aware CPU baseline:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_predictive_baseline.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

The gate trains separate C1e/C1f/C2 Huber-SGD regressors on frozen caps of
100,000 exact-affinity training observations and evaluates 25,000 validation and
25,000 test observations. Features combine chiral Morgan fingerprints, hashed
protein 3-mers, and affinity-type indicators. Metrics are reported overall and
per Ki/Kd/IC50/EC50. Inequality-censored measurements are excluded from this
point-regression baseline by design.

If the corrected 100k baseline passes, run the controlled 500k training-data
sensitivity before selecting a heavier model:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_predictive_500k.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

This preserves the feature/model/seed/epoch configuration, increases training
to 500,000 rows and validation/test to 50,000 rows, writes to a separate output
directory, and requests 16 GB because sampled row materialization is in memory.
