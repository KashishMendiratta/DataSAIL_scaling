# BindingDB scaling worklog

This is the append-only implementation and experiment record for the BindingDB
extension. Every failed run, material code correction, and change in scientific
technique is recorded here before the next scale is attempted. Generated run
manifests remain the authoritative source for exact inputs, hashes, counts,
runtime, and peak memory.

## 2026-11-26 — Plan locked

- Execution order: prepare data; scaled C1e; scaled C1f; scaled C2; leakage and
  predictive evaluation; full DataSAIL C1e/C1f/C2 on a frozen feasible subset.
- Every split scale must record stage runtimes, peak memory, anchor counts,
  retained interactions, and dropped cross-split interactions.
- Every applicable validation invariant in the interactive roadmap is a hard
  gate, not a post-hoc warning.

## 2026-11-26 — Frozen release downloaded

- Release: BindingDB 202611.
- Archive: `data/raw/bindingdb/BindingDB_All_202611_tsv.zip`.
- Compressed size: 594,284,548 bytes; uncompressed member size: 9,004,582,822
  bytes.
- SHA-256: `470a3c2e4aebe3c493de3e9e46e739592eb8e29b3adafb5189c875442268a0b2`.
- The archive was retained rather than extracting the 9 GB TSV. Preparation
  streams `BindingDB_All.tsv` directly from the zip.

## 2026-11-26 — First real pilot failed on target-sequence schema

- Scale: first 10,000 source rows; Ki-only prototype.
- Failure: `Unexpected BindingDB schema; missing ['BindingDB Target Chain Sequence']`.
- Cause: the published/older format used an unnumbered or duplicate column name,
  while release 202611 contains explicit columns `BindingDB Target Chain
  Sequence 1` through `BindingDB Target Chain Sequence 50`.
- Change: schema discovery now accepts both legacy duplicate-column suffixes and
  numbered 2026 chain headers. A regression test covers numbered headers.
- Scientific technique did not change: multi-chain targets remain excluded from
  the first implementation because a single protein sequence is required for
  the planned C1f/C2 representation.

## 2026-11-26 — Ki-only 10k pilot passed

- Retained 1,155 of 10,000 source measurements; 8,845 were explicitly excluded.
- Exclusions: 8,841 missing/invalid Ki and 4 multi-chain targets.
- Entities: 932 ligands and 46 targets; 10 duplicate observations detected.
- Source accounting passed exactly.
- Initial peak memory was about 425 MiB because all 640 source columns were
  loaded. The reader was changed to select only identity, affinity, sequence,
  and accession columns. The repeated pilot used about 304 MiB.
- Stage timing and peak-memory fields were added to the preparation manifest
  before any larger run.

## 2026-11-26 — Technique changed from Ki-only to typed multi-affinity ingestion

- Reason: splitting depends on the ligand–target graph, so discarding Kd, IC50,
  and EC50 rows would unnecessarily remove graph structure. However, the four
  numerical endpoint types are not scientifically interchangeable regression
  labels.
- New model: one source-measurement record can produce zero or more typed
  affinity observations. Ki, Kd, IC50, and EC50 remain explicit fields in the
  observation identity and are evaluated separately unless a later model
  explicitly conditions on affinity type.
- New accounting gate: retained source measurements plus excluded source
  measurements must equal all source measurements. Observation count is tracked
  separately and may exceed the number of retained source rows.
- Status: implementation in progress; no larger preparation run is authorized
  until synthetic contracts and the real 10k pilot pass under this model.

## 2026-11-26 — Multi-affinity 10k pilot passed

- Synthetic contracts passed for zero, one, and multiple typed observations per
  source measurement.
- Real pilot: 10,000 source measurements produced 6,993 observations from 6,913
  retained source rows, 4,056 ligands, and 103 single-sequence targets.
- Observation types in this leading slice: 1,155 Ki and 5,838 IC50; no Kd or EC50.
  The release is not assumed to be randomly ordered, so this is a parser check,
  not a database-wide prevalence estimate.
- Exclusions: 3,028 multi-chain targets and 59 rows without a usable selected
  affinity. Multi-chain exclusion increased substantially relative to the
  Ki-only pilot because many newly retained IC50 rows describe complexes.
- Technique decision: keep multi-chain targets excluded for the first C1f/C2
  implementation. Flattening a protein complex to one chain would corrupt the
  cold-target identity. A complex-aware representation may be a later extension.
- Source accounting passed. Runtime was 23.78 seconds and peak memory was 304.61
  MiB. This authorizes a 100,000-row local preparation pilot; it does not yet
  authorize the full release.

## 2026-11-26 — Multi-affinity 100k pilot passed

- 100,000 source measurements produced 93,130 typed observations from 90,171
  retained source measurements.
- Observation counts: 18,753 Ki; 3,694 Kd; 54,274 IC50; 16,409 EC50.
- Entities: 47,619 ligands and 1,260 single-sequence targets.
- Exclusions: 8,217 multi-chain targets, 1,596 rows without a valid selected
  affinity, and 16 invalid ligand SMILES.
- Source accounting passed. Runtime was 57.22 seconds and peak memory was 267.08
  MiB. Chunked memory remained below the 10k pilot peak.
- This result authorizes the 500,000-row local pilot. Condor is not necessary
  for preparation at the observed resource level; splitting stages will be
  reassessed separately.

## 2026-11-26 — Multi-affinity 500k pilot passed; full run moved to Condor

- 500,000 source measurements produced 463,948 observations from 458,717
  retained source measurements.
- Observation counts: 97,101 Ki; 7,764 Kd; 307,163 IC50; 51,920 EC50.
- Entities: 252,259 ligands and 3,346 single-sequence targets.
- Exclusions: 22,706 multi-chain targets, 17,682 rows without a valid selected
  affinity, and 895 invalid ligand SMILES.
- Source accounting passed. Runtime was 211.94 seconds and peak memory was
  317.19 MiB.
- Local scale-up stopped despite acceptable CPU/RAM because the workstation
  volume had only 5.7 GiB free and was 98% full. Full preparation needs staging,
  WAL, compressed tables, and later fingerprint headroom; running locally would
  create avoidable disk-pressure risk.
- Added `experiments/condor/run_bindingdb_preparation.sh` and
  `experiments/condor/bindingdb_prepare.sub`, requesting 1 CPU, 4 GB memory, and
  8 GB disk. The job consumes the already frozen 202611 archive and retains all
  four typed affinities.

## 2026-11-26 — Full preparation job 77191.0 failed at Python startup

- Condor status: normal termination with exit code 1 after 66 seconds of remote
  wall time; the Python process itself ran for only a few seconds.
- Error: `ModuleNotFoundError: No module named 'experiments'` while importing
  `experiments.dataset_download.download_bindingdb` from
  `prepare_bindingdb.py`.
- Cause: the Condor runner launched the preparer by filesystem path. In the
  clean container, Python placed `experiments/dataset_download` rather than the
  repository root on its import path. The local Conda shell had masked this
  packaging assumption.
- Change: the runner now uses module execution:
  `python -u -m experiments.dataset_download.prepare_bindingdb`. Module
  execution places the repository root on the import path and preserves package
  imports.
- No output directory or manifest was created. This was not an OOM, disk,
  parsing, or scientific-method failure; resource requests and the preparation
  technique remain unchanged for the retry.

## 2026-11-26 — Full preparation retry 77194.0 passed

- Condor status: completed (`JobStatus=4`) with exit code 0; no signal.
- Condor remote wall time: 5,291 seconds. Manifest total runtime: 5,283.14
  seconds, comprising 4,950.47 seconds normalization/staging, 327.94 seconds
  export, and 1.03 seconds source hashing.
- Peak process memory: 365.93 MiB. Condor reported 260 MiB memory usage against
  a 4,096 MiB request. Preparation was not memory constrained.
- Frozen source: 3,243,660 measurements from BindingDB 202611, SHA-256
  `470a3c2e4aebe3c493de3e9e46e739592eb8e29b3adafb5189c875442268a0b2`.
- Retained 3,031,381 source measurements and excluded 212,279; the exact source
  accounting invariant passed.
- Emitted 3,036,683 typed observations: 588,828 Ki; 127,299 Kd; 2,060,773
  IC50; and 259,783 EC50.
- Normalized 1,357,780 unique ligands and 10,518 unique single-sequence targets.
- Detected 353,342 duplicate observations for group-preserving split validation.
- Exclusions: 186,771 multi-chain targets, 19,613 rows without a valid selected
  affinity, and 5,895 invalid ligand SMILES.
- Output tables occupy about 179 MiB compressed plus a 1.5 GiB SQLite staging
  database (apparent size). The manifest is at
  `data/processed/bindingdb/release_202611_all_affinities/preparation_manifest.json`.
- Result: the interaction-specific loader phase is complete. No preparation
  technique change is required before scaled C1e implementation.

## 2026-11-26 — Scaled C1e architecture and 1k pilot passed

- Added a package-return adapter that consumes the three dictionaries returned
  by `datasail(...)`. The BindingDB path reads `e_splits["C1e"][run]` directly;
  it does not discover or parse generated DataSAIL TSV files.
- C1e anchors are sampled across ligand observation-degree deciles. Anchor
  weights passed to DataSAIL are ligand observation degrees, so the optimizer
  targets interaction balance rather than equal ligand counts.
- Remaining ligands use exact Morgan/Tanimoto neighbours of the anchors and a
  degree-weighted balance update. A `1e-9` similarity baseline was added to
  resolve all-zero-similarity ties by balance; the original molecular method
  multiplied zero similarity by its balance term and therefore defaulted every
  such tie to the first split.
- C1e target policy is explicit: targets may occur in multiple splits. Requiring
  one target assignment in C1e would change the regime into a double-cold split.
- Real pilot input: a seeded 1,000-ligand subset of the prepared 500k source
  pilot; 100 degree-stratified ligand anchors; seed 42; k=5.
- DataSAIL solved the weighted anchor problem through its Python API. Total
  runtime was 11.03 seconds, including 3.99 seconds in DataSAIL; peak memory was
  315.22 MiB.
- All 1,000 ligands were assigned. All 2,013 corresponding observations were
  retained; none were dropped. Interaction ratios were 70.34% train, 20.47%
  validation, and 9.19% test. Duplicate split conflicts: zero.
- This proves algorithm feasibility only. C1e is not declared complete until
  cross-split chemical similarity, protein-similarity applicability, and graph
  connected-component audits are attached to the final validation gate.

## 2026-11-26 — C1e 10k/500-anchor pilot interrupted for runtime

- Configuration: 10,000 seeded ligands, 5% anchor ratio (500 anchors), 50
  DataSAIL clusters, exact Tanimoto-to-anchor assignment, one thread.
- The run produced no stage output or result after approximately nine minutes
  and was interrupted manually. No manifest was produced, so it did not pass a
  scale gate.
- This was a bounded diagnostic interruption, not an OOM or Condor failure.
- Change: explicit flushed stage markers were added around statistics loading,
  anchor selection, DataSAIL, remaining-ligand assignment, and audit. The next
  diagnostic uses 100 anchors to localize the bottleneck before any algorithmic
  substitution or larger run.

## 2026-11-26 — C1e bottleneck isolated; 10 clusters locked

- The 10,000-ligand diagnostic with 100 anchors and 10 clusters completed in
  15.35 seconds. Stage times were 3.95 seconds DataSAIL, 3.61 seconds remaining
  assignment, and 3.52 seconds audit.
- A controlled repeat restored 500 anchors but retained 10 clusters. It completed
  in 15.16 seconds, including 4.04 seconds DataSAIL and 4.14 seconds assignment.
- Therefore, the interrupted runtime was caused by increasing the cluster count
  from 10 to 50, not by the 500-anchor count or exact neighbour assignment.
- Technique change: the BindingDB C1e default is locked to 10 DataSAIL clusters,
  matching the molecular thesis workflow. Cluster-count sensitivity can be a
  later experiment; it is not allowed to silently vary during the scale ladder.
- The 500-anchor repeat assigned all 10,000 ligands and retained all 18,479
  observations. Ratios were 69.90% train, 20.01% validation, and 10.10% test;
  duplicate split conflicts and dropped observations were both zero.

## 2026-11-26 — C1e 100k pilot exposed canonical-structure leakage

- The 100,000-ligand-ID pilot completed in 57.11 seconds with 312.34 MiB peak
  memory. It retained 183,468 observations with ratios 69.89% train, 20.03%
  validation, and 10.08% test; observation-duplicate conflicts were zero.
- A stricter post-run query found four canonical SMILES assigned to more than
  one split. BindingDB can map multiple MonomerIDs to the same canonical
  structure, so MonomerID is not a sufficient cold-ligand splitting unit.
- The 100k result is diagnostic and fails the canonical chemical-identity gate.
- Technique change: C1e now aggregates observation degree by canonical SMILES,
  assigns one stable hash-based chemical ID per canonical structure, splits and
  extends assignments at that chemical level, then expands the result back to
  every BindingDB ligand ID. Canonical chemical groups are atomic and a new hard
  validation rejects any group observed in multiple splits.

## 2026-11-26 — Corrected canonical-chemical C1e gates passed

- The corrected 10,000-chemical run expanded to 10,059 BindingDB ligand IDs and
  retained 18,329 observations. Runtime was 19.86 seconds; peak memory was
  309.91 MiB. Ratios were 69.95% train, 19.88% validation, and 10.17% test.
- The corrected 100,000-chemical run expanded to 100,563 ligand IDs and retained
  184,664 observations. Runtime was 58.15 seconds; peak memory was 330.94 MiB.
  Ratios were 69.66% train, 20.06% validation, and 10.28% test.
- Both corrected runs had zero dropped observations, zero duplicate-observation
  conflicts, and zero canonical-chemical conflicts.
- Full-run technique locked: canonical chemical unit; seed 42; 0.1% anchors
  (approximately 1,350 on the full release); degree-decile anchor sampling;
  observation-degree DataSAIL weights; k=5; balance weight 2.0; 10 clusters.
- Added a Condor runner requesting 1 CPU, 8 GB memory, and 8 GB disk. The split
  manifest is explicitly marked as pending until chemical-similarity, protein-
  similarity applicability, and connected-component audits pass.

## 2026-11-26 — Full scaled C1e candidate 77206.0 passed core gates

- Condor completed normally (`JobStatus=4`, exit code 0, no signal) after 1,073
  seconds of remote wall time. The pipeline recorded 1,058.00 seconds total:
  113.06 seconds loading chemical statistics, 0.75 seconds selecting anchors,
  4.36 seconds in DataSAIL, 721.36 seconds extending assignments, and 183.93
  seconds auditing the result.
- Peak process memory was 751.16 MiB; Condor reported 571 MiB against the 8 GiB
  request. The full C1e candidate was therefore neither memory nor runtime
  constrained.
- Frozen provenance: Git commit
  `fb04bac7869c888c78cabce82b24b65f26592ae9`, clean worktree, pinned container
  digest `sha256:9ac1d19ae19c707d1a6df021248b52db9f15a082fbaface397b27132d2c90abd`,
  and Condor job `77206.0`.
- Assigned all 1,348,653 canonical chemicals, expanding to all 1,357,780
  BindingDB ligand IDs. DataSAIL split 1,349 degree-stratified anchors through
  its Python return value; no generated assignment TSV was used as input.
- Retained all 3,036,683 affinity observations and dropped zero. Interaction
  ratios were 69.8106% train, 20.0283% validation, and 10.1611% test versus the
  requested 70/20/10.
- Duplicate-observation split conflicts: zero. Canonical-chemical split
  conflicts: zero. All four affinity types occur in every split and their
  counts and mean p-affinities are stored in the run manifest.
- The stderr messages for `cd-hit`, `diamond`, `mmseqs`, `mash`, `foldseek`, and
  `TMalign` are DataSAIL optional-tool discovery probes. C1e used RDKit molecular
  fingerprints, completed successfully, and did not require those executables;
  this is not a technique change or failed validation.
- Status remains `core_assignments_passed_pending_similarity_and_graph_audits`.
  This candidate is not yet accepted as the final C1e split. The next gate must
  report maximum cross-split ligand similarity, protein-similarity
  applicability, unseen-entity rates, connected components, and entity-degree
  distributions before changing the status to passed.

## 2026-11-26 — Full C1e acceptance validator implemented

- Added a separate, resumable validation stage so the successful assignment
  database from job 77206.0 remains immutable. The stage writes a new
  `validation_report.json` and exits nonzero if a hard gate cannot be proven.
- Exact entity/accounting gates compare unique assignments against every
  prepared ligand, join every retained observation, require a status or
  exclusion reason for every source measurement, and reject duplicate groups
  spanning splits. Requested and actual interaction ratios and retained/dropped
  counts are repeated in the validation report.
- Ligand maximum-similarity technique: scan complete Morgan radius-2, 2048-bit
  fingerprints until an identical fingerprint occurs in different splits.
  Fingerprint bytes are compared directly. Such a pair certifies the global
  Tanimoto maximum as exactly 1.0 because no value can exceed 1. If no exact
  certificate is found, the validator fails instead of substituting an
  unlabelled sampled maximum.
- Protein maximum-similarity technique: C1e leaves targets unconstrained. An
  identical target sequence observed through interactions in multiple splits
  certifies the exact cross-split protein-similarity maximum as 1.0. The
  observed unseen-target rate is reported but is not optimized in C1e; the
  unseen-test-ligand identity rate must be 1.0.
- Affinity summaries now include count, mean, standard deviation, minimum, and
  maximum for every affinity type in every split. Full ligand/target degree
  distributions and bipartite connected-component size distributions are
  computed over all prepared nodes and all observation edges.
- The validator requests 1 CPU and 8 GiB RAM. It records per-stage runtime,
  peak memory, Git/container/Condor provenance, and fails before writing a
  `passed` status if any required gate is incomplete.

