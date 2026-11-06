"""Compatibility entry point for config-driven stratified scaled DataSAIL."""
from __future__ import annotations
import argparse
from run_scaled_configured import CONFIG, run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',nargs='?',choices=CONFIG['datasets'])
    parser.add_argument('--sample-ratio',type=float)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--k',type=int)
    parser.add_argument('--n-jobs',type=int,default=1)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    datasets=[args.dataset] if args.dataset else CONFIG['default_datasets']
    for name in datasets:
        run(name,'stratified',args.sample_ratio,args.seed,args.k,args.n_jobs,args.dry_run)

if __name__=='__main__': main()
