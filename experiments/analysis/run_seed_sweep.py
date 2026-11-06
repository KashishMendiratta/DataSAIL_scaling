"""Run matched random/scaled split replicates into seed-specific folders.

Full DataSAIL is deliberately excluded: the current wrapper records a seed but
does not pass a verified random seed into DataSAIL's optimizer.
"""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pipeline_engine import CONFIG, SPLIT_ROOT, run, run_random

METHOD_TO_STRATEGY = {
    "random": None,
    "datasail_scaled_random": "random",
    "datasail_scaled_stratified": "stratified",
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=list(CONFIG["datasets"]), default=list(CONFIG["datasets"]))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202, 303, 404])
    p.add_argument("--methods", nargs="+", choices=list(METHOD_TO_STRATEGY), default=list(METHOD_TO_STRATEGY))
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--sample-ratio", type=float)
    p.add_argument("--k", type=int)
    a = p.parse_args()
    run_ids = [f"seed_{seed}" for seed in a.seeds]
    collisions = [str(SPLIT_ROOT / d / m / rid)
                  for d in a.datasets for rid in run_ids for m in a.methods
                  if (SPLIT_ROOT / d / m / rid).exists()]
    if collisions:
        raise SystemExit("Refusing to overwrite existing replicate folders:\n" + "\n".join(collisions))
    records, failures = [], []
    old_run_id = os.environ.get("DATASAIL_RUN_ID")
    try:
        for dataset in a.datasets:
            for seed, run_id in zip(a.seeds, run_ids):
                os.environ["DATASAIL_RUN_ID"] = run_id
                for method in a.methods:
                    try:
                        if method == "random":
                            meta = run_random(dataset, seed)
                        else:
                            meta = run(dataset, METHOD_TO_STRATEGY[method],
                                       sample_ratio=a.sample_ratio, seed=seed,
                                       k=a.k, n_jobs=a.n_jobs)
                        records.append({"dataset": dataset, "method": method, "seed": seed,
                                        "run_id": run_id, "status": "ok",
                                        "total_seconds": meta.get("total_seconds"),
                                        "datasail_seconds": meta.get("datasail_seconds"),
                                        "source_sha256": meta.get("source_sha256")})
                    except Exception as exc:
                        failures.append((dataset, method, seed, str(exc)))
                        records.append({"dataset": dataset, "method": method, "seed": seed,
                                        "run_id": run_id, "status": "failed", "error": str(exc)})
                        print(f"FAILED {dataset}/{method}/{run_id}: {exc}", file=sys.stderr, flush=True)
    finally:
        if old_run_id is None:
            os.environ.pop("DATASAIL_RUN_ID", None)
        else:
            os.environ["DATASAIL_RUN_ID"] = old_run_id
    outdir = ROOT / "results/analysis/unified"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = outdir / f"seed_sweep_manifest_{stamp}.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"Saved run manifest: {out}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} replicate(s) failed; see manifest and messages")

if __name__ == "__main__":
    main()
