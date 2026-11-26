"""Run a bounded or complete scaled BindingDB double-cold split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.bindingdb_c2 import run_scaled_c2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_database", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-ligands", type=int, default=1000)
    parser.add_argument("--max-targets", type=int, default=300)
    parser.add_argument("--ligand-anchor-ratio", type=float, default=0.1)
    parser.add_argument("--target-anchor-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--balance-weight", type=float, default=2.0)
    parser.add_argument("--graph-weight", type=float, default=4.0)
    parser.add_argument("--similarity-weight", type=float, default=1.0)
    parser.add_argument("--ligand-clusters", type=int, default=10)
    parser.add_argument("--target-clusters", type=int, default=20)
    parser.add_argument("--max-sec", type=int, default=3600)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_scaled_c2(
        args.prepared_database, args.output_dir,
        max_ligands=args.max_ligands, max_targets=args.max_targets,
        ligand_anchor_ratio=args.ligand_anchor_ratio,
        target_anchor_ratio=args.target_anchor_ratio, seed=args.seed, k=args.k,
        balance_weight=args.balance_weight, graph_weight=args.graph_weight,
        similarity_weight=args.similarity_weight,
        ligand_clusters=args.ligand_clusters, target_clusters=args.target_clusters,
        max_sec=args.max_sec, threads=args.threads,
    ), indent=2))


if __name__ == "__main__":
    main()
