"""Shared dataset, split, fingerprint, and distribution helpers for analyses."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline_engine import CONFIG, load_dataset
METHODS=['random','datasail_scaled_random','datasail_scaled_stratified','datasail_full']
ANALYSIS_DIR=ROOT/'results/analysis/unified'

def split_dir(dataset, method, run_id=None):
    path = ROOT / 'results/experiments/unified/splits' / dataset / method
    run_id = run_id if run_id is not None else os.environ.get('DATASAIL_RUN_ID', '').strip()
    return path / run_id if run_id else path

def read_splits(dataset, method, run_id=None):
    folder=split_dir(dataset, method, run_id)
    paths={s:folder/f'{s}.csv' for s in ('train','val','test')}
    if not all(p.exists() for p in paths.values()): return None
    return {s:pd.read_csv(p) for s,p in paths.items()}

def fingerprint_matrix(smiles,n_jobs=1):
    generator=GetMorganGenerator(radius=CONFIG['defaults']['fingerprint']['radius'],fpSize=CONFIG['defaults']['fingerprint']['bits'])
    out=[]
    seq=[str(x) if pd.notna(x) else '' for x in smiles]
    for start in range(0,len(seq),2000):
        mols=[Chem.MolFromSmiles(s) for s in seq[start:start+2000]]
        valid=[m if m is not None else Chem.MolFromSmiles('') for m in mols]
        fps=generator.GetFingerprints(valid,numThreads=max(1,n_jobs))
        out.extend(np.frombuffer(fp.ToBitString().encode('ascii'),dtype=np.uint8)-ord('0') for fp in fps)
    return np.vstack(out) if out else np.empty((0,CONFIG['defaults']['fingerprint']['bits']),dtype=np.uint8)

def smiles_column(dataset,frame):
    col=CONFIG['datasets'][dataset]['smiles_column']
    return col if col in frame else 'smiles'

def target_columns(dataset,frame):
    spec=CONFIG['datasets'][dataset]
    if spec.get('target_columns') is not None: return [x for x in spec['target_columns'] if x in frame]
    prefix=spec.get('target_prefix')
    if prefix: return [x for x in frame if x.startswith(prefix)]
    excluded={spec.get('id_column'),'mol_id','smiles','_datasail_id','_datasail_smiles'}
    return [c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded]

def _distribution(values,task_type,reference_values=None,bins=10):
    vals=pd.Series(values); valid=vals.notna(); vals=vals[valid]
    if task_type=='regression':
        ref=pd.Series(reference_values).dropna() if reference_values is not None else vals
        if len(vals)==0 or len(ref)==0: return {},0
        try:
            edges=np.unique(np.quantile(pd.to_numeric(ref,errors='coerce').dropna(),np.linspace(0,1,bins+1)))
            if len(edges)<2: return {'all':1.0},len(vals)
            cat=pd.cut(pd.to_numeric(vals,errors='coerce'),bins=edges,include_lowest=True,duplicates='drop')
        except (ValueError,TypeError): return {},len(vals)
        dist=cat.value_counts(normalize=True,sort=False)
        return {str(k):float(v) for k,v in dist.items()},len(vals)
    dist=vals.astype(str).value_counts(normalize=True)
    return {str(k):float(v) for k,v in dist.items()},len(vals)

def distribution_metrics(dataset,metric,methods=None):
    """Compare each split's per-target distribution with its source distribution.

    Regression targets are binned using source-wide quantiles. Classification
    targets compare observed class proportions; missing assay labels are omitted.
    """
    spec=CONFIG['datasets'][dataset]; task_type=spec['task_type']
    source,_,_=load_dataset(dataset); targets=target_columns(dataset,source)
    rows=[]
    for method in methods or METHODS:
        frames=read_splits(dataset,method)
        if frames is None: continue
        merged=pd.concat([frames[s] for s in ('train','val','test')],ignore_index=True)
        for target in targets:
            ref_dist, n_ref=_distribution(source[target], 'regression' if task_type=='regression' else 'classification',source[target])
            if not ref_dist: continue
            for split,frame in frames.items():
                if target not in frame: continue
                part_dist,n_part=_distribution(frame[target], 'regression' if task_type=='regression' else 'classification',source[target])
                keys=set(ref_dist)|set(part_dist); eps=1e-12
                if metric=='DRE':
                    vals=[abs(part_dist.get(k,0)-ref_dist.get(k,0))/(ref_dist.get(k,0)+eps) for k in keys if ref_dist.get(k,0)>0]
                    score=float(np.mean(vals)) if vals else np.nan
                else:
                    score=float(sum(p*np.log(p/max(part_dist.get(k,0),eps)) for k,p in ref_dist.items() if p>0))
                rows.append({'dataset':dataset,'method':method,'split':split,'target':target,
                             'task_type':task_type,'reference_n':n_ref,'split_n':n_part,
                             'distribution_metric':metric,'value':score,
                             'representation':'source-quantile bins' if task_type=='regression' else 'observed classes'})
    return pd.DataFrame(rows)

def summarize_or_empty(df,group_columns):
    if df.empty: return pd.DataFrame(columns=list(group_columns)+['mean','max'])
    return df.groupby(list(group_columns),dropna=False)['value'].agg(mean='mean',max='max').reset_index()
