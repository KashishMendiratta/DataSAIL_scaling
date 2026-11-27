"""BindingDB dual encoder with a trainable amino-acid sequence CNN."""
from __future__ import annotations

import argparse
import csv
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
    AFFINITY_TYPES, LIGAND_FEATURES, metric_rows, peak_rss_mb, run_regime,
)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYUXBZJO"
AA_TO_INDEX = {amino_acid: index + 1 for index, amino_acid in enumerate(AMINO_ACIDS)}
MAX_PROTEIN_LENGTH = 512


class SequenceEncoder:
    """Encode ligands densely and proteins as padded amino-acid token sequences."""

    def __init__(self):
        self.ligand = GetMorganGenerator(
            radius=2, fpSize=LIGAND_FEATURES, includeChirality=True,
        )
        self.total_sequences = 0
        self.truncated_sequences = 0
        self.maximum_observed_length = 0

    def transform(self, rows):
        ligand = np.zeros((len(rows), LIGAND_FEATURES), dtype=np.float32)
        protein = np.zeros((len(rows), MAX_PROTEIN_LENGTH), dtype=np.int64)
        for index, (_, smiles, sequence, _, _) in enumerate(rows):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise RuntimeError(f"Invalid prepared SMILES: {smiles}")
            ligand[index, list(self.ligand.GetFingerprint(molecule).GetOnBits())] = 1.0
            sequence = sequence.upper()
            self.total_sequences += 1
            self.maximum_observed_length = max(self.maximum_observed_length, len(sequence))
            if len(sequence) > MAX_PROTEIN_LENGTH:
                self.truncated_sequences += 1
                sequence = sequence[:MAX_PROTEIN_LENGTH]
            protein[index, :len(sequence)] = [AA_TO_INDEX.get(amino_acid, AA_TO_INDEX["X"])
                                               for amino_acid in sequence]
        ligand /= np.maximum(np.linalg.norm(ligand, axis=1, keepdims=True), 1.0)
        affinity = np.asarray([AFFINITY_TYPES.index(row[3]) for row in rows], dtype=np.int64)
        labels = np.asarray([row[4] for row in rows], dtype=np.float32)
        return ligand, protein, affinity, labels

    def diagnostics(self):
        return {
            "protein_rows": self.total_sequences,
            "truncated_protein_rows": self.truncated_sequences,
            "truncated_protein_fraction": (
                self.truncated_sequences / self.total_sequences if self.total_sequences else 0.0
            ),
            "maximum_observed_protein_length": self.maximum_observed_length,
            "maximum_encoded_protein_length": MAX_PROTEIN_LENGTH,
            "truncation_policy": "retain first 512 amino acids",
        }


class SequenceDualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ligand_tower = nn.Sequential(
            nn.Linear(LIGAND_FEATURES, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.amino_acid_embedding = nn.Embedding(len(AMINO_ACIDS) + 1, 16, padding_idx=0)
        self.protein_convolutions = nn.ModuleList([
            nn.Conv1d(16, 24, kernel_size=kernel, padding=kernel // 2)
            for kernel in (3, 5, 9)
        ])
        self.protein_projection = nn.Sequential(
            nn.Linear(144, 96), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(96, 64), nn.ReLU(),
        )
        self.affinity = nn.Embedding(len(AFFINITY_TYPES), 8)
        self.fusion = nn.Sequential(
            nn.Linear(136, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 48), nn.ReLU(), nn.Linear(48, 1),
        )

    def encode_protein(self, tokens):
        mask = tokens.ne(0).unsqueeze(1)
        lengths = mask.sum(dim=2).clamp_min(1)
        embedded = self.amino_acid_embedding(tokens).transpose(1, 2)
        pooled = []
        for convolution in self.protein_convolutions:
            values = torch.relu(convolution(embedded))
            maximum = values.masked_fill(~mask, torch.finfo(values.dtype).min).amax(dim=2)
            mean = values.masked_fill(~mask, 0.0).sum(dim=2) / lengths
            pooled.extend((maximum, mean))
        return self.protein_projection(torch.cat(pooled, dim=1))

    def forward(self, ligand, protein, affinity):
        combined = torch.cat((
            self.ligand_tower(ligand), self.encode_protein(protein),
            self.affinity(affinity),
        ), dim=1)
        return self.fusion(combined).squeeze(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--c1e", type=Path, required=True)
    parser.add_argument("--c1f", type=Path, required=True)
    parser.add_argument("--c2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=100000)
    parser.add_argument("--val-cap", type=int, default=25000)
    parser.add_argument("--test-cap", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, predictions = [], []
    for regime, database in (("C1e", args.c1e), ("C1f", args.c1f), ("C2", args.c2)):
        print(f"Sequence dual encoder: {regime} device={device}", flush=True)
        result, rows = run_regime(
            args.prepared, database, regime, args.output_dir, args, device,
            encoder_class=SequenceEncoder, model_class=SequenceDualEncoder,
            model_slug="sequence_dual_encoder",
        )
        results.append(result)
        predictions.extend(rows)
    manifest = {
        "format_version": 1,
        "model": "morgan_ligand_amino_acid_cnn_protein_dual_encoder",
        "device": str(device), "torch_version": torch.__version__,
        "features": {
            "ligand": "normalized chiral Morgan radius=2 512-bit",
            "protein": "trainable amino-acid embedding plus kernel 3/5/9 CNN over first 512 residues, max and mean pooling",
            "affinity_type": "learned 8-dimensional embedding",
        },
        "sampling_policy": "same corrected deterministic pseudo-random samples as prior baselines",
        "label_policy": "exact equality affinity observations only",
        "seed": args.seed,
        "caps": {"train": args.train_cap, "val": args.val_cap, "test": args.test_cap},
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_mb": peak_rss_mb(),
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY"),
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"),
        "python_version": platform.python_version(), "results": results,
    }
    (args.output_dir / "sequence_dual_encoder_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    metrics_out = [row for result in results for row in result["metrics"]]
    for filename, rows in (("sequence_dual_encoder_metrics.tsv", metrics_out),
                           ("sequence_dual_encoder_predictions.tsv", predictions)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
