"""Generate deterministic random splits for configured datasets."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.pipeline_engine import CONFIG, run_random

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets',nargs='+',choices=list(CONFIG['datasets']))
    parser.add_argument('--seed',type=int,default=CONFIG['defaults']['seed'])
    args=parser.parse_args(); failed=[]
    for dataset in args.datasets:
        try: run_random(dataset,args.seed)
        except Exception as exc: failed.append((dataset,str(exc))); print(f'FAILED {dataset}: {exc}',file=sys.stderr)
    if failed: raise SystemExit(f'{len(failed)} dataset(s) failed')
if __name__=='__main__': main()
