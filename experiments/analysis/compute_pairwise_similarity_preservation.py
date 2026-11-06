"""Compare source-wide and DataSAIL anchor-set pairwise similarity estimates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.pipeline_engine import CONFIG,load_dataset
from src.analysis_common import METHODS,ANALYSIS_DIR,fingerprint_matrix

def mean_sampled_pair_similarity(fps,n_pairs,seed):
 if len(fps)<2:return np.nan
 rng=np.random.default_rng(seed);i=rng.integers(len(fps),size=n_pairs);j=rng.integers(len(fps),size=n_pairs)
 same=i==j
 while same.any():j[same]=rng.integers(len(fps),size=int(same.sum()));same=i==j
 vals=[]
 for start in range(0,n_pairs,10000):
  x=fps[i[start:start+10000]].astype(np.uint32);y=fps[j[start:start+10000]].astype(np.uint32)
  inter=np.sum(x*y,axis=1);union=np.sum((x+y)>0,axis=1);vals.extend(np.divide(inter,union,out=np.ones_like(inter,dtype=float),where=union>0))
 return float(np.mean(vals))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',choices=list(CONFIG['datasets']),default=list(CONFIG['datasets']));p.add_argument('--methods',nargs='+',choices=['datasail_scaled_random','datasail_scaled_stratified'],default=['datasail_scaled_random','datasail_scaled_stratified']);p.add_argument('--pairs',type=int,default=100000);p.add_argument('--max-molecules',type=int,default=5000);p.add_argument('--seed',type=int,default=42);p.add_argument('--n-jobs',type=int,default=1);a=p.parse_args();rows=[]
 for d in a.datasets:
  frame,_,_=load_dataset(d);all_ids=frame['_datasail_id'].astype(str).to_numpy();all_smiles=frame['_datasail_smiles'].astype(str).to_numpy()
  rng=np.random.default_rng(a.seed);idx=np.sort(rng.choice(len(frame),min(len(frame),a.max_molecules),replace=False));full_fp=fingerprint_matrix(all_smiles[idx],a.n_jobs)
  full_mean=mean_sampled_pair_similarity(full_fp,a.pairs,a.seed)
  for m in a.methods:
   manifest=ROOT/'results/experiments/unified/splits'/d/m/'anchor_manifest.json'
   if not manifest.exists():rows.append({'dataset':d,'method':m,'status':'missing_anchor_manifest'});continue
   data=json.loads(manifest.read_text());anchor_ids=set(map(str,data.get('anchor_ids',[])));positions=np.flatnonzero(np.isin(all_ids,list(anchor_ids)))
   if not len(positions):rows.append({'dataset':d,'method':m,'status':'no_anchor_ids_matched'});continue
   ai=np.sort(rng.choice(positions,min(len(positions),a.max_molecules),replace=False));anchor_fp=fingerprint_matrix(all_smiles[ai],a.n_jobs)
   anchor_mean=mean_sampled_pair_similarity(anchor_fp,a.pairs,a.seed)
   rows.append({'dataset':d,'method':m,'status':'ok','source_n':len(frame),'anchor_n':len(positions),'anchor_ratio':data.get('sample_ratio'),
    'sampled_pairs':a.pairs,'source_mean_similarity':full_mean,'anchor_mean_similarity':anchor_mean,'absolute_difference':abs(full_mean-anchor_mean),'pair_sample_seed':a.seed})
 out=ANALYSIS_DIR/'pairwise_similarity_preservation.csv';out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out,index=False);print(f'Saved {out}')
if __name__=='__main__':main()
