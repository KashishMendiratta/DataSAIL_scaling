"""Compute per-target KL divergence for all available split methods."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.pipeline_engine import CONFIG
from src.analysis_common import METHODS,ANALYSIS_DIR,distribution_metrics,summarize_or_empty

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',choices=list(CONFIG['datasets']),default=list(CONFIG['datasets']));p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);a=p.parse_args();frames=[]
 for d in a.datasets:
  x=distribution_metrics(d,'KL',a.methods)
  if not x.empty:frames.append(x)
 detail=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame();outdir=ANALYSIS_DIR;outdir.mkdir(parents=True,exist_ok=True)
 detail.to_csv(outdir/'kl_divergence_detail.csv',index=False)
 summary=summarize_or_empty(detail,['dataset','method','split']);summary.to_csv(outdir/'kl_divergence.csv',index=False)
 print(f'Saved {outdir}/kl_divergence.csv')
if __name__=='__main__':main()
