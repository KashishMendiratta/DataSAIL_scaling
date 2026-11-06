"""Run full-data DataSAIL for configured datasets; resource failures are reported per dataset."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.pipeline_engine import CONFIG, run_full

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets',nargs='+',choices=list(CONFIG['datasets']))
    parser.add_argument('--seed',type=int,default=CONFIG['defaults']['seed'])
    parser.add_argument('--n-jobs',type=int,default=1)
    parser.add_argument('--max-sec',type=int,default=43200)
    parser.add_argument('--e-clusters',type=int,default=10)
    args=parser.parse_args(); failed=[]
    for dataset in args.datasets:
        try: run_full(dataset,args.seed,args.n_jobs,args.max_sec,args.e_clusters)
        except Exception as exc: failed.append((dataset,str(exc))); print(f'FAILED {dataset}: {exc}',file=sys.stderr)
    if failed: raise SystemExit(f'{len(failed)} dataset(s) failed; see messages above')
if __name__=='__main__': main()
