"""Run target-stratified-anchor scaled DataSAIL for configured datasets."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.pipeline_engine import CONFIG, run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets',nargs='+',choices=list(CONFIG['datasets']))
    parser.add_argument('--sample-ratio',type=float)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--k',type=int)
    parser.add_argument('--n-jobs',type=int,default=1)
    args=parser.parse_args(); failed=[]
    for dataset in args.datasets:
        try: run(dataset,'stratified',args.sample_ratio,args.seed,args.k,args.n_jobs)
        except Exception as exc: failed.append((dataset,str(exc))); print(f'FAILED {dataset}: {exc}',file=sys.stderr)
    if failed: raise SystemExit(f'{len(failed)} dataset(s) failed')
if __name__=='__main__': main()
