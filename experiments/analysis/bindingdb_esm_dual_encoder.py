"""BindingDB dual encoder using cached full-sequence ESM-2 target embeddings."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch import nn

from experiments.analysis.bindingdb_dual_encoder import (
    AFFINITY_TYPES, LIGAND_FEATURES, peak_rss_mb, run_regime,
)


class CachedESMEncoder:
    def __init__(self, cache_path):
        cache = np.load(cache_path)
        self.proteins = {key: value for key, value in
                         zip(cache["sequence_sha256"].tolist(), cache["embedding"])}
        self.ligand = GetMorganGenerator(radius=2, fpSize=LIGAND_FEATURES,
                                         includeChirality=True)
        self.rows = 0

    def transform(self, rows):
        ligand = np.zeros((len(rows), LIGAND_FEATURES), dtype=np.float32)
        protein = np.zeros((len(rows), 320), dtype=np.float32)
        for index, (_, smiles, sequence, _, _) in enumerate(rows):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise RuntimeError(f"Invalid prepared SMILES: {smiles}")
            ligand[index, list(self.ligand.GetFingerprint(molecule).GetOnBits())] = 1.0
            key = hashlib.sha256(sequence.encode()).hexdigest()
            if key not in self.proteins:
                raise RuntimeError(f"Missing cached target sequence: {key}")
            protein[index] = self.proteins[key]
        ligand /= np.maximum(np.linalg.norm(ligand, axis=1, keepdims=True), 1.0)
        self.rows += len(rows)
        affinity = np.asarray([AFFINITY_TYPES.index(row[3]) for row in rows], dtype=np.int64)
        labels = np.asarray([row[4] for row in rows], dtype=np.float32)
        return ligand, protein, affinity, labels

    def diagnostics(self):
        return {"protein_rows": self.rows, "cached_target_count": len(self.proteins),
                "missing_cached_targets": 0, "complete_sequence_embeddings": True}


class ESMDualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ligand_tower = nn.Sequential(
            nn.Linear(LIGAND_FEATURES, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.protein_tower = nn.Sequential(
            nn.LayerNorm(320), nn.Linear(320, 160), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(160, 64), nn.ReLU(),
        )
        self.affinity = nn.Embedding(len(AFFINITY_TYPES), 8)
        self.fusion = nn.Sequential(
            nn.Linear(136, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 48), nn.ReLU(), nn.Linear(48, 1),
        )

    def forward(self, ligand, protein, affinity):
        return self.fusion(torch.cat((self.ligand_tower(ligand),
                                      self.protein_tower(protein),
                                      self.affinity(affinity)), dim=1)).squeeze(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "c1e", "c1f", "c2", "esm-cache", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=100000)
    parser.add_argument("--val-cap", type=int, default=25000)
    parser.add_argument("--test-cap", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, predictions = [], []
    factory = lambda: CachedESMEncoder(args.esm_cache)
    for regime, database in (("C1e", args.c1e), ("C1f", args.c1f), ("C2", args.c2)):
        result, rows = run_regime(args.prepared, database, regime, args.output_dir, args,
                                  device, factory, ESMDualEncoder, "esm_dual_encoder")
        results.append(result); predictions.extend(rows)
    manifest = {
        "format_version": 1, "model": "cached_esm2_8m_morgan_dual_encoder",
        "device": str(device), "torch_version": torch.__version__,
        "esm_cache": str(args.esm_cache),
        "sampling_policy": "same corrected deterministic pseudo-random 100k/25k/25k samples",
        "label_policy": "exact equality affinity observations only", "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started, "peak_memory_mb": peak_rss_mb(),
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY"),
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"),
        "python_version": platform.python_version(), "results": results,
    }
    (args.output_dir / "esm_dual_encoder_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metric_rows = [row for result in results for row in result["metrics"]]
    for filename, rows in (("esm_dual_encoder_metrics.tsv", metric_rows),
                           ("esm_dual_encoder_predictions.tsv", predictions)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
