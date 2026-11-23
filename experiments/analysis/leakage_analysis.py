"""Estimate mean train-test molecular similarity using sampled pair comparisons.

This is a global cross-partition similarity summary. It is distinct from the
nearest-neighbour leakage statistic produced by nn_leakage.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.pipeline_engine import CONFIG
from src.analysis_common import METHODS,ANALYSIS_DIR,read_splits,smiles_column,fingerprint_matrix

def sampled_mean(a,b,n_pairs,seed):
 if not len(a) or not len(b):return np.nan
 rng=np.random.default_rng(seed);i=rng.integers(len(a),size=n_pairs);j=rng.integers(len(b),size=n_pairs);parts=[]
 for k in range(0,n_pairs,10000):
  x=a[i[k:k+10000]].astype(np.uint32);y=b[j[k:k+10000]].astype(np.uint32)
  inter=np.sum(x*y,axis=1);union=np.sum((x+y)>0,axis=1);parts.extend(np.divide(inter,union,out=np.ones_like(inter,dtype=float),where=union>0))
 return float(np.mean(parts))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',choices=list(CONFIG['datasets']),default=list(CONFIG['datasets']));p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);p.add_argument('--pairs',type=int,default=100000);p.add_argument('--seed',type=int,default=42);p.add_argument('--n-jobs',type=int,default=1);a=p.parse_args();rows=[]
 for d in a.datasets:
  for m in a.methods:
   f=read_splits(d,m)
   if f is None:rows.append({'dataset':d,'method':m,'status':'missing_splits'});continue
   train,test=f['train'],f['test'];tc=smiles_column(d,train);qc=smiles_column(d,test)
   ti=train[tc].dropna().astype(str);qi=test[qc].dropna().astype(str)
   tr=fingerprint_matrix(ti.tolist(),a.n_jobs);te=fingerprint_matrix(qi.tolist(),a.n_jobs)
   rows.append({'dataset':d,'method':m,'status':'ok','train_n':len(tr),'test_n':len(te),'sampled_pairs':a.pairs,
    'mean_train_test_similarity':sampled_mean(tr,te,a.pairs,a.seed),'pair_sample_seed':a.seed,
    'metric':'mean Tanimoto over sampled train-test pairs'})
 out=ANALYSIS_DIR/'leakage_analysis.csv';out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out,index=False);print(f'Saved {out}')
if __name__=='__main__':main()
