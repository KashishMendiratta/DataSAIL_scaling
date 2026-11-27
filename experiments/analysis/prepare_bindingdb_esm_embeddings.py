"""Cache full-sequence ESM-2 embeddings for every prepared BindingDB target."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sqlite3
import sys
import time
from pathlib import Path

import esm
import numpy as np
import torch


MODEL_NAME = "esm2_t6_8M_UR50D"
REPRESENTATION_LAYER = 6
CHUNK_LENGTH = 1000
STRIDE = 800


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (value if sys.platform == "darwin" else value * 1024) / (1024 * 1024)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunks(sequence):
    if len(sequence) <= CHUNK_LENGTH:
        return [(0, sequence)]
    starts = list(range(0, len(sequence) - CHUNK_LENGTH + 1, STRIDE))
    final = len(sequence) - CHUNK_LENGTH
    if starts[-1] != final:
        starts.append(final)
    return [(start, sequence[start:start + CHUNK_LENGTH]) for start in starts]


@torch.no_grad()
def embed_sequence(sequence, model, alphabet, batch_converter, device, batch_size):
    pieces = chunks(sequence)
    residue_sum = np.zeros((len(sequence), 320), dtype=np.float32)
    residue_count = np.zeros(len(sequence), dtype=np.uint16)
    for offset in range(0, len(pieces), batch_size):
        batch = pieces[offset:offset + batch_size]
        _, _, tokens = batch_converter([(str(start), text) for start, text in batch])
        representations = model(tokens.to(device), repr_layers=[REPRESENTATION_LAYER],
                                return_contacts=False)["representations"][REPRESENTATION_LAYER]
        for row, (start, text) in enumerate(batch):
            values = representations[row, 1:len(text) + 1].float().cpu().numpy()
            residue_sum[start:start + len(text)] += values
            residue_count[start:start + len(text)] += 1
    if not np.all(residue_count):
        raise RuntimeError("Chunking left uncovered residues")
    return (residue_sum / residue_count[:, None]).mean(axis=0), len(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--expected-targets", type=int, default=10518)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)
    converter = alphabet.get_batch_converter()
    with sqlite3.connect(args.prepared) as connection:
        targets = connection.execute(
            "SELECT target_id, sequence FROM targets ORDER BY target_id"
        ).fetchall()
    if len(targets) != args.expected_targets:
        raise RuntimeError(
            f"Expected {args.expected_targets} frozen targets, found {len(targets)}"
        )
    embeddings, hashes, target_ids, lengths, chunk_counts = [], [], [], [], []
    for index, (target_id, sequence) in enumerate(targets, 1):
        embedding, count = embed_sequence(
            sequence, model, alphabet, converter, device, args.batch_size,
        )
        target_ids.append(target_id)
        hashes.append(hashlib.sha256(sequence.encode()).hexdigest())
        embeddings.append(embedding)
        lengths.append(len(sequence))
        chunk_counts.append(count)
        if index % 100 == 0 or index == len(targets):
            print(f"embedded={index}/{len(targets)}", flush=True)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / f"{MODEL_NAME}.pt"
    np.savez(
        args.output,
        target_id=np.asarray(target_ids), sequence_sha256=np.asarray(hashes),
        embedding=np.asarray(embeddings, dtype=np.float32),
        sequence_length=np.asarray(lengths, dtype=np.int32),
        chunk_count=np.asarray(chunk_counts, dtype=np.int16),
    )
    manifest = {
        "format_version": 1, "model": MODEL_NAME,
        "representation_layer": REPRESENTATION_LAYER, "embedding_dimensions": 320,
        "target_count": len(targets), "device": str(device),
        "chunk_length": CHUNK_LENGTH, "chunk_stride": STRIDE,
        "long_target_count": int(sum(length > CHUNK_LENGTH for length in lengths)),
        "maximum_sequence_length": max(lengths), "total_chunks": int(sum(chunk_counts)),
        "complete_sequence_coverage": True,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "output_sha256": sha256_file(args.output),
        "runtime_seconds": time.perf_counter() - started, "peak_memory_mb": peak_rss_mb(),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "git_commit": os.environ.get("DATASAIL_GIT_COMMIT"),
        "git_worktree_dirty": os.environ.get("DATASAIL_GIT_WORKTREE_DIRTY"),
        "container_image": os.environ.get("DATASAIL_CONTAINER_REF"),
        "condor_job_id": os.environ.get("CONDOR_JOB_ID"),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
