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

