"""Estimate test-to-training nearest-neighbour similarity and threshold fraction."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.pipeline_engine import CONFIG
from src.analysis_common import METHODS,ANALYSIS_DIR,read_splits,smiles_column,fingerprint_matrix

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',choices=list(CONFIG['datasets']),default=list(CONFIG['datasets']));p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);p.add_argument('--query-n',type=int,default=500);p.add_argument('--train-cap',type=int,default=10000);p.add_argument('--threshold',type=float,default=.7);p.add_argument('--seed',type=int,default=42);p.add_argument('--n-jobs',type=int,default=1);a=p.parse_args();rows=[]
 for d in a.datasets:
  for m in a.methods:
   f=read_splits(d,m)
   if f is None:rows.append({'dataset':d,'method':m,'status':'missing_splits'});continue
   train,test=f['train'],f['test'];tc=smiles_column(d,train);qc=smiles_column(d,test)
   rng=np.random.default_rng(a.seed);ti=np.sort(rng.choice(len(train),min(a.train_cap,len(train)),replace=False));qi=np.sort(rng.choice(len(test),min(a.query_n,len(test)),replace=False))
   tr=fingerprint_matrix(train.iloc[ti][tc].fillna('').astype(str).tolist(),a.n_jobs);te=fingerprint_matrix(test.iloc[qi][qc].fillna('').astype(str).tolist(),a.n_jobs)
   nn=NearestNeighbors(n_neighbors=1,metric='jaccard',n_jobs=a.n_jobs).fit(tr);dist,_=nn.kneighbors(te);sim=1-dist.ravel()
   rows.append({'dataset':d,'method':m,'status':'ok','query_n':len(sim),'train_n':len(tr),'mean_max_similarity':float(sim.mean()),'median_max_similarity':float(np.median(sim)),
    'fraction_above_threshold':float(np.mean(sim>a.threshold)),'threshold':a.threshold,'sample_seed':a.seed})
 out=ANALYSIS_DIR/'nn_leakage.csv';out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out,index=False);print(f'Saved {out}')
if __name__=='__main__':main()
