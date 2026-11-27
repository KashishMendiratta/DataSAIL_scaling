"""Nonlinear ligand-protein dual-encoder baseline for BindingDB cold splits."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import resource
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.analysis.bindingdb_predictive_baseline import (
    AFFINITY_TYPES, metrics, query_for,
)


RDLogger.DisableLog("rdApp.*")
LIGAND_FEATURES = 512
PROTEIN_FEATURES = 256


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def load_rows(connection, regime: str, split: str, cap: int, seed: int):
    return connection.execute(query_for(regime, seed), (split, cap)).fetchall()


class DenseEncoder:
    def __init__(self):
        self.ligand = GetMorganGenerator(
            radius=2, fpSize=LIGAND_FEATURES, includeChirality=True,
        )
        self.protein = HashingVectorizer(
            analyzer="char", ngram_range=(3, 3), n_features=PROTEIN_FEATURES,
            alternate_sign=False, norm="l2", lowercase=False,
        )

    def transform(self, rows):
        ligand = np.zeros((len(rows), LIGAND_FEATURES), dtype=np.float32)
        for index, (_, smiles, _, _, _) in enumerate(rows):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise RuntimeError(f"Invalid prepared SMILES: {smiles}")
            ligand[index, list(self.ligand.GetFingerprint(molecule).GetOnBits())] = 1.0
        norms = np.linalg.norm(ligand, axis=1, keepdims=True)
        ligand /= np.maximum(norms, 1.0)
        protein = self.protein.transform([row[2] for row in rows]).toarray().astype(np.float32)
        affinity = np.asarray([AFFINITY_TYPES.index(row[3]) for row in rows], dtype=np.int64)
        labels = np.asarray([row[4] for row in rows], dtype=np.float32)
        return ligand, protein, affinity, labels


class DualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ligand_tower = nn.Sequential(
            nn.Linear(LIGAND_FEATURES, 96), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(96, 48), nn.ReLU(),
        )
        self.protein_tower = nn.Sequential(
            nn.Linear(PROTEIN_FEATURES, 96), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(96, 48), nn.ReLU(),
        )
        self.affinity = nn.Embedding(len(AFFINITY_TYPES), 8)
        self.fusion = nn.Sequential(
            nn.Linear(104, 96), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, ligand, protein, affinity):
        combined = torch.cat((
            self.ligand_tower(ligand), self.protein_tower(protein),
            self.affinity(affinity),
        ), dim=1)
        return self.fusion(combined).squeeze(1)


def tensor_dataset(encoded, mean: float, scale: float):
    ligand, protein, affinity, labels = encoded
    return TensorDataset(
        torch.from_numpy(ligand), torch.from_numpy(protein), torch.from_numpy(affinity),
        torch.from_numpy((labels - mean) / max(scale, 1e-12)),
    )


@torch.no_grad()
def predict(model, encoded, device, mean, scale, batch_size):
    dataset = tensor_dataset(encoded, mean, scale)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    values = []
    model.eval()
    for ligand, protein, affinity, _ in loader:
        output = model(ligand.to(device), protein.to(device), affinity.to(device))
        values.append(output.cpu().numpy() * scale + mean)
    return np.concatenate(values)


def metric_rows(regime, split, rows, truth, prediction):
    output = [{"regime": regime, "split": split, "affinity_type": "all",
               **metrics(truth, prediction)}]
    types = np.asarray([row[3] for row in rows])
    for kind in AFFINITY_TYPES:
        mask = types == kind
        if mask.any():
            output.append({"regime": regime, "split": split, "affinity_type": kind,
                           **metrics(truth[mask], prediction[mask])})
    return output


def run_regime(prepared, database, regime, output_dir, args, device,
               encoder_class=DenseEncoder, model_class=DualEncoder,
               model_slug="dual_encoder"):
    started = time.perf_counter()
    with sqlite3.connect(prepared) as connection:
        connection.execute("ATTACH DATABASE ? AS splitdb", (str(database.resolve()),))
        train_rows = load_rows(connection, regime, "train", args.train_cap, args.seed)
        val_rows = load_rows(connection, regime, "val", args.val_cap, args.seed + 1)
        test_rows = load_rows(connection, regime, "test", args.test_cap, args.seed + 2)
    encoding_started = time.perf_counter(); encoder = encoder_class()
    train_encoded = encoder.transform(train_rows)
    val_encoded = encoder.transform(val_rows)
    test_encoded = encoder.transform(test_rows)
    encoding_seconds = time.perf_counter() - encoding_started
    y_mean, y_scale = float(train_encoded[3].mean()), float(train_encoded[3].std())
    train_dataset = tensor_dataset(train_encoded, y_mean, y_scale)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    torch.manual_seed(args.seed)
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss(beta=1.0)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_path = output_dir / f"{regime}_{model_slug}.pt"
    history = []
    training_started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train(); losses = []
        for ligand, protein, affinity, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(ligand.to(device), protein.to(device), affinity.to(device))
            loss = loss_function(prediction, labels.to(device)); loss.backward()
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        val_prediction = predict(model, val_encoded, device, y_mean, y_scale, args.batch_size)
        val_rmse = float(np.sqrt(np.mean((val_encoded[3] - val_prediction) ** 2)))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_rmse": val_rmse})
        print(f"{regime} epoch={epoch} train_loss={np.mean(losses):.6f} val_rmse={val_rmse:.6f}", flush=True)
        if val_rmse < best_loss - 1e-4:
            best_loss, best_epoch, stale = val_rmse, epoch, 0
            torch.save({"state_dict": model.state_dict(), "y_mean": y_mean,
                        "y_scale": y_scale, "epoch": epoch}, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    training_seconds = time.perf_counter() - training_started
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    val_prediction = predict(model, val_encoded, device, y_mean, y_scale, args.batch_size)
    test_prediction = predict(model, test_encoded, device, y_mean, y_scale, args.batch_size)
    metrics_out = metric_rows(regime, "val", val_rows, val_encoded[3], val_prediction)
    metrics_out += metric_rows(regime, "test", test_rows, test_encoded[3], test_prediction)
    predictions = [{
        "regime": regime, "split": split, "observation_id": row[0],
        "affinity_type": row[3], "truth_p_affinity": float(encoded[3][index]),
        "prediction_p_affinity": float(prediction[index]),
    } for split, rows, encoded, prediction in (
        ("val", val_rows, val_encoded, val_prediction),
        ("test", test_rows, test_encoded, test_prediction),
    ) for index, row in enumerate(rows)]
    result = {
        "regime": regime, "sample_rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "sample_rows_by_type": {split: {kind: sum(row[3] == kind for row in rows) for kind in AFFINITY_TYPES}
                                for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows))},
        "encoding_seconds": encoding_seconds, "training_seconds": training_seconds,
        "total_seconds": time.perf_counter()-started, "best_epoch": best_epoch,
        "best_validation_rmse": best_loss, "history": history, "metrics": metrics_out,
    }
    if hasattr(encoder, "diagnostics"):
        result["encoder_diagnostics"] = encoder.diagnostics()
    del train_encoded, val_encoded, test_encoded, train_dataset, loader, model
    gc.collect()
    return result, predictions


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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, predictions = [], []
    for regime, database in (("C1e", args.c1e), ("C1f", args.c1f), ("C2", args.c2)):
        print(f"Dual encoder: {regime} device={device}", flush=True)
        result, rows = run_regime(args.prepared, database, regime, args.output_dir, args, device)
        results.append(result); predictions.extend(rows)
    manifest = {
        "format_version": 1, "model": "nonlinear_ligand_protein_dual_encoder",
        "device": str(device), "torch_version": torch.__version__,
        "features": {"ligand": "normalized chiral Morgan radius=2 512-bit",
                     "protein": "normalized hashed 3-mer frequencies 256 features",
                     "affinity_type": "learned 8-dimensional embedding"},
        "sampling_policy": "same corrected deterministic pseudo-random samples as linear baseline",
        "label_policy": "exact equality affinity observations only",
        "seed": args.seed, "caps": {"train": args.train_cap, "val": args.val_cap, "test": args.test_cap},
        "runtime_seconds": time.perf_counter()-started, "peak_memory_mb": peak_rss_mb(),
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY"),
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"), "python_version": platform.python_version(),
        "results": results,
    }
    (args.output_dir / "dual_encoder_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    metric_rows = [row for result in results for row in result["metrics"]]
    for filename, rows in (("dual_encoder_metrics.tsv", metric_rows), ("dual_encoder_predictions.tsv", predictions)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
