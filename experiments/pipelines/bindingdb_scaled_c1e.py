"""Run a bounded or complete scaled BindingDB cold-ligand split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.bindingdb_c1e import run_scaled_c1e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_database", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-ligands", type=int, default=10_000)
    parser.add_argument("--anchor-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--balance-weight", type=float, default=2.0)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--max-sec", type=int, default=3600)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    manifest = run_scaled_c1e(
        args.prepared_database, args.output_dir,
        max_ligands=args.max_ligands, anchor_ratio=args.anchor_ratio,
        seed=args.seed, k=args.k, balance_weight=args.balance_weight,
        clusters=args.clusters, max_sec=args.max_sec, threads=args.threads,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
