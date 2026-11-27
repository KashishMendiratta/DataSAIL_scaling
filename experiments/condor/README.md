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

## BindingDB nonlinear dual encoder

After the 500k linear sensitivity establishes saturation, run the controlled
100k nonlinear ligand-protein model:

```bash
experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_dual_encoder.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

The model has separate nonlinear ligand and protein towers, a learned affinity-
type embedding, and a fusion regressor. It uses the corrected 100k/25k/25k
samples for direct comparison with the accepted linear baseline. The runner
checks for PyTorch before loading data. The pinned image is CPU-only, so the
gate requests four CPUs and 16 GB rather than an unusable GPU.

The standard scaling image does not contain PyTorch. Build and push the
dedicated CPU image once (from a Docker-authenticated machine), then inspect
the pushed manifest to obtain its immutable `linux/amd64` digest:

```bash
docker buildx build --platform linux/amd64 \
  -f experiments/condor/Dockerfile.dmpnn \
  -t docker.io/kashishhm/datasail-dmpnn:cpu-2.5.1 \
  --push .
docker buildx imagetools inspect \
  docker.io/kashishhm/datasail-dmpnn:cpu-2.5.1
```

Supply the resulting repository digest to the same provenance-enforcing helper;
the value must end in `@sha256:` followed by 64 hexadecimal characters:

```bash
DATASAIL_IMAGE_REF='docker.io/kashishhm/datasail-dmpnn@sha256:REPLACE_WITH_DIGEST' \
  experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_dual_encoder.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

This override affects only that invocation. The helper still rejects mutable
tags and dirty worktrees, and records the selected digest in the model manifest.

## BindingDB sequence-aware dual encoder

The next biological-representation gate keeps the accepted splits, corrected
samples, exact-affinity labels, and chiral Morgan ligand tower fixed, but
replaces hashed protein 3-mers with a trainable amino-acid embedding and
multi-kernel sequence CNN. Submit it with the same pinned CPU PyTorch image:

```bash
DATASAIL_IMAGE_REF='docker.io/kashishhm/datasail-dmpnn@sha256:4dbb6cda9267583df3a499647d1f8c437ba26a47d0a57fa0e3ae4497fcdb67ce' \
  experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_sequence_dual_encoder.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro"
```

The controlled gate uses 100k/25k/25k rows per regime, eight CPUs, 24 GB, and
early stopping. Proteins are encoded up to 512 residues; the manifest reports
the observed maximum and the count/fraction of truncated rows. Do not scale the
model unless it improves the aggregate C1f and C2 test metrics over both the
linear and hashed-3-mer nonlinear baselines.

Before that comparison, run a two-epoch 10k/5k/5k feasibility gate:

```bash
DATASAIL_IMAGE_REF='docker.io/kashishhm/datasail-dmpnn@sha256:4dbb6cda9267583df3a499647d1f8c437ba26a47d0a57fa0e3ae4497fcdb67ce' \
  experiments/condor/submit_pinned.sh \
  experiments/condor/bindingdb_sequence_dual_encoder.sub \
  repo_dir="$HOME/DataSAIL_scaling_repro" train_cap=10000 val_cap=5000 \
  test_cap=5000 max_epochs=2 run_label=10k_smoke
```

Use this run only to validate execution, peak memory, per-stage runtime, and
protein-truncation accounting. If it completes safely, submit the file without
the cap overrides to run the comparable 100k gate.
