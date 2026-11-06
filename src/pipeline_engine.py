"""Config-driven scaled DataSAIL pipeline for the ten configured datasets.

This is a review copy. Existing thesis scripts remain unchanged. The two supported
strategies are random anchor sampling and label-stratified anchor sampling; both
use DataSAIL for anchor splits and balanced kNN for assigning the remaining rows.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, platform, subprocess, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'experiments/config/datasets.json'
SPLIT_ROOT = ROOT / 'results/experiments/unified/splits'
DATASAIL_ROOT = ROOT / 'results/experiments/unified/datasail'
CONFIG = json.loads(CONFIG_PATH.read_text())
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(ROOT))


def load_dataset(name: str):
    spec = CONFIG['datasets'][name]
    path = ROOT / spec['path']
    frame = pd.read_csv(path, compression=spec.get('compression'))
    source_rows = len(frame)
    id_col, smiles_col = spec.get('id_column'), spec['smiles_column']
    if smiles_col not in frame:
        raise ValueError(f'{name}: missing SMILES column {smiles_col!r}')
    if id_col and id_col in frame:
        ids = frame[id_col].astype('string')
    else:
        ids = pd.Series([f'{name}:row:{i}' for i in range(source_rows)], dtype='string')
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f'{name}: source IDs must be present and unique')
    frame = frame.copy()
    frame['_datasail_id'] = ids.to_numpy()
    frame['mol_id'] = ids.to_numpy()
    frame['smiles'] = frame[smiles_col].astype('string')
    frame['_datasail_smiles'] = frame[smiles_col].astype('string')
    valid = frame['_datasail_smiles'].fillna('').map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    invalid_rows = frame.loc[~valid, ['_datasail_id']].copy()
    frame = frame.loc[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f'{name}: no valid SMILES rows')
    return frame, source_rows, invalid_rows


def fingerprint_matrix(frame: pd.DataFrame, cfg: dict, n_jobs: int):
    fp_cfg = cfg['defaults']['fingerprint']
    generator = GetMorganGenerator(radius=fp_cfg['radius'], fpSize=fp_cfg['bits'])
    smiles = frame['_datasail_smiles'].astype(str).tolist()
    output = []
    for start in range(0, len(smiles), 2000):
        mols = [Chem.MolFromSmiles(s) for s in smiles[start:start+2000]]
        fps = generator.GetFingerprints(mols, numThreads=max(1, n_jobs))
        output.append(np.vstack([np.frombuffer(fp.ToBitString().encode('ascii'), dtype=np.uint8)-ord('0') for fp in fps]))
    return np.vstack(output)


def stratification_labels(frame: pd.DataFrame, spec: dict):
    col = spec.get('stratify_column')
    if not col or col not in frame.columns:
        raise ValueError(f"{spec['path']}: stratify_column {col!r} is missing")
    labels = frame[col]
    if spec['task_type'] == 'regression':
        numeric = pd.to_numeric(labels, errors='coerce')
        labels = pd.qcut(numeric.rank(method='first'), q=min(10, max(2, numeric.notna().sum() // 20)), labels=False, duplicates='drop').astype('string')
        labels = labels.fillna('missing')
    else:
        labels = labels.astype('string').fillna('missing')
    counts = labels.value_counts()
    # Collapse very rare strata so sklearn can make a meaningful stratified sample.
    labels = labels.where(labels.map(counts) >= 2, 'rare')
    return labels.astype(str)


def choose_anchors(frame, spec, ratio, strategy, seed):
    n = max(10, min(len(frame)-1, int(len(frame)*ratio)))
    positions = np.arange(len(frame))
    if strategy == 'random':
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(positions, n, replace=False))
    strata = stratification_labels(frame, spec)
    try:
        selected, _ = train_test_split(positions, train_size=n, random_state=seed, stratify=strata)
    except ValueError as exc:
        raise ValueError(f"Cannot stratify {spec['path']} at anchor count {n}: {exc}") from exc
    return np.sort(selected)


def verify_assignments(frame, labels, split_names):
    ids = frame['_datasail_id'].astype(str)
    if len(ids) != len(labels) or ids.duplicated().any() or pd.isna(labels).any():
        raise ValueError('Output ID/assignment cardinality or uniqueness check failed')
    if set(np.unique(labels)) != set(split_names):
        raise ValueError(f'Expected all split labels {split_names}; got {set(np.unique(labels))}')
    return True


def run(name, strategy, sample_ratio=None, seed=None, k=None, n_jobs=1, dry_run=False):
    # Start before data loading so this captures the full pipeline wall time,
    # including validation, fingerprinting, DataSAIL, assignment, and file output.
    run_started = time.perf_counter()
    defaults, spec = CONFIG['defaults'], CONFIG['datasets'][name]
    sample_ratio = sample_ratio if sample_ratio is not None else spec.get('sample_ratio', defaults['sample_ratio'])
    seed = defaults['seed'] if seed is None else seed
    k = defaults['assignment']['k'] if k is None else k
    ratios = defaults['split_ratios']; split_names = list(ratios)
    frame, n_input, invalid = load_dataset(name)
    anchors = choose_anchors(frame, spec, sample_ratio, strategy, seed)
    print(f'{name}: input={n_input}, valid={len(frame)}, invalid={len(invalid)}, anchors={len(anchors)}, strategy={strategy}')
    if dry_run:
        return {'dataset':name,'valid_rows':len(frame),'invalid_rows':len(invalid),'anchor_rows':len(anchors)}
    fps = fingerprint_matrix(frame, CONFIG, n_jobs)
    id_col = '_datasail_id'; smiles_col = '_datasail_smiles'
    sampled = frame.iloc[anchors]
    ds_input = ROOT / 'data/temp' / f'{name}_{strategy}_anchors.csv'
    ds_input.parent.mkdir(parents=True, exist_ok=True)
    sampled[[id_col,smiles_col]].rename(columns={id_col:'id',smiles_col:'smiles'}).to_csv(ds_input,index=False)
    out = DATASAIL_ROOT / name / f'datasail_scaled_{strategy}'
    out.mkdir(parents=True,exist_ok=True)
    from datasail.sail import sail
    start_ds=time.perf_counter()
    sail(e_type='M', e_data=str(ds_input), techniques=['C1e'], e_sim='ecfp', e_clusters=10,
         splits=list(ratios.values()), names=split_names, output=str(out), verbosity='W', runs=1, threads=n_jobs)
    datasail_seconds=time.perf_counter()-start_ds
    split_files=sorted((out/'C1e').glob('Molecule_*_splits.tsv'))
    if not split_files: raise FileNotFoundError(f'No DataSAIL split TSV under {out / "C1e"}')
    split_table=pd.read_csv(split_files[-1],sep='\t')
    if not {'ID','Split'}.issubset(split_table.columns): raise ValueError('DataSAIL split file lacks ID/Split columns')
    split_by_id=dict(zip(split_table.ID.astype(str),split_table.Split.astype(str)))
    expected=set(sampled[id_col].astype(str))
    if set(split_by_id)!=expected or len(split_table)!=len(expected): raise ValueError('DataSAIL anchor IDs do not match input IDs exactly')
    anchor_labels=np.array([split_by_id[x] for x in sampled[id_col].astype(str)])
    rest=np.setdiff1d(np.arange(len(frame)),anchors,assume_unique=True)
    if len(rest):
        from src.assignment.balanced_knn import balanced_knn_assign
        remaining_labels,_=balanced_knn_assign(fps[anchors],pd.Series(anchor_labels),fps[rest],k=min(k,len(anchors)),target_ratios=ratios,balance_weight=defaults['assignment']['balance_weight'],n_jobs=n_jobs)
    else: remaining_labels=np.array([],dtype=object)
    labels=np.empty(len(frame),dtype=object); labels[anchors]=anchor_labels; labels[rest]=remaining_labels
    verify_assignments(frame,labels,split_names)
    output_frame=frame.drop(columns=['_datasail_smiles','_datasail_id']).copy()
    output_frame['split']=labels
    split_root=SPLIT_ROOT/name/f'datasail_scaled_{strategy}'
    split_root.mkdir(parents=True,exist_ok=True)
    for split in split_names: output_frame.loc[output_frame.split==split].drop(columns='split').to_csv(split_root/f'{split}.csv',index=False)
    source=ROOT/spec['path']; digest=hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except Exception: commit,dirty=None,None
    total_seconds=time.perf_counter()-run_started
    meta={'dataset':name,'method':f'datasail_scaled_{strategy}','strategy':strategy,'source_path':spec['path'],'source_sha256':digest,
          'id_column':spec.get('id_column'),'smiles_column':spec['smiles_column'],'target_columns':spec.get('target_columns'),
          'task_type':spec['task_type'],'split_ratios':ratios,'sample_ratio':sample_ratio,'seed':seed,'k':k,
          'fingerprint':defaults['fingerprint'],'assignment':defaults['assignment'],'n_input_rows':n_input,
          'n_valid_rows':len(frame),'n_invalid_smiles':len(invalid),'invalid_ids':invalid['_datasail_id'].astype(str).tolist(),
          'n_anchor_rows':len(anchors),'split_counts':dict(Counter(labels)),'datasail_seconds':datasail_seconds,
          'total_seconds':total_seconds,'runtime_scope':'end-to-end pipeline (load through split and metadata preparation)','n_jobs':n_jobs,
          'git_commit':commit,'git_worktree_dirty':dirty,'python_version':platform.python_version(),
          'package_versions':{p:importlib.metadata.version(p) for p in ('datasail','numpy','pandas','scikit-learn','rdkit')}}
    (split_root/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (split_root/'anchor_manifest.json').write_text(json.dumps({'dataset':name,'method':meta['method'],'seed':seed,'sample_ratio':sample_ratio,'anchor_ids':sampled[id_col].astype(str).tolist()},indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    return meta



def _save_result(name, method, frame, labels, metadata):
    """Write one unified split layout and a provenance record."""
    from collections import Counter
    split_names = list(CONFIG['defaults']['split_ratios'])
    verify_assignments(frame, labels, split_names)
    output = frame.drop(columns=['_datasail_smiles','_datasail_id']).copy()
    output['split'] = labels
    root = SPLIT_ROOT / name / method
    root.mkdir(parents=True, exist_ok=True)
    for split in split_names:
        output.loc[output.split == split].drop(columns='split').to_csv(root / f'{split}.csv', index=False)
    source = ROOT / CONFIG['datasets'][name]['path']
    base = {'dataset':name,'method':method,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'source_path':CONFIG['datasets'][name]['path'],'n_input_rows':metadata.pop('n_input_rows'),
            'n_valid_rows':len(frame),'n_invalid_smiles':metadata.pop('n_invalid_smiles'),
            'invalid_ids':metadata.pop('invalid_ids'),'split_ratios':CONFIG['defaults']['split_ratios'],
            'split_counts':dict(Counter(labels)),'python_version':platform.python_version(),
            'package_versions':{p:importlib.metadata.version(p) for p in ('datasail','numpy','pandas','scikit-learn','rdkit')}}
    try:
        base['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        base['git_worktree_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except Exception:
        base['git_commit']=base['git_worktree_dirty']=None
    started=metadata.pop('_started')
    base.update(metadata)
    base.update({'runtime_scope':'end-to-end split generation','total_seconds':time.perf_counter()-started})
    (root/'run_metadata.json').write_text(json.dumps(base,indent=2)+'\n')
    return base

def run_random(name, seed=42):
    """Create a deterministic random split using the shared valid-row policy."""
    started=time.perf_counter()
    frame,n_input,invalid=load_dataset(name)
    n=len(frame); order=np.random.default_rng(seed).permutation(n)
    ratios=CONFIG['defaults']['split_ratios']; counts={k:int(np.floor(n*v)) for k,v in ratios.items()}
    counts['train'] += n-sum(counts.values())
    labels=np.empty(n,dtype=object); cursor=0
    for split in ratios:
        labels[order[cursor:cursor+counts[split]]]=split; cursor+=counts[split]
    fps_cfg=CONFIG['defaults']['fingerprint']
    result=_save_result(name,'random',frame,labels,{'seed':seed,'sample_ratio':None,'n_input_rows':n_input,
        'n_invalid_smiles':len(invalid),'invalid_ids':invalid['_datasail_id'].astype(str).tolist(),
        'fingerprint':fps_cfg,'_started':started})
    print(json.dumps(result,indent=2)); return result

def run_full(name, seed=42, n_jobs=1, max_sec=43200, e_clusters=10):
    """Run DataSAIL on every valid row; resource failures remain dataset-specific."""
    started=time.perf_counter(); frame,n_input,invalid=load_dataset(name)
    id_col,smiles_col='_datasail_id','_datasail_smiles'
    input_path=ROOT/'data/temp'/f'{name}_full_datasail.csv'; input_path.parent.mkdir(parents=True,exist_ok=True)
    frame[[id_col,smiles_col]].rename(columns={id_col:'id',smiles_col:'smiles'}).to_csv(input_path,index=False)
    output=DATASAIL_ROOT/name/'datasail_full'; output.mkdir(parents=True,exist_ok=True)
    from datasail.sail import sail
    t0=time.perf_counter()
    sail(e_type='M',e_data=str(input_path),techniques=['C1e'],e_sim='ecfp',e_clusters=e_clusters,
         splits=list(CONFIG['defaults']['split_ratios'].values()),names=list(CONFIG['defaults']['split_ratios']),
         output=str(output),verbosity='W',runs=1,threads=n_jobs,max_sec=max_sec)
    datasail_seconds=time.perf_counter()-t0
    files=sorted((output/'C1e').glob('Molecule_*_splits.tsv'),key=lambda p:p.stat().st_mtime)
    if not files: raise FileNotFoundError(f'No full DataSAIL split under {output}/C1e')
    table=pd.read_csv(files[-1],sep='\t')
    if not {'ID','Split'}.issubset(table.columns): raise ValueError('Full DataSAIL output lacks ID/Split columns')
    mapping=dict(zip(table.ID.astype(str),table.Split.astype(str)))
    expected=set(frame[id_col].astype(str))
    if set(mapping)!=expected or len(table)!=len(expected): raise ValueError('Full DataSAIL IDs differ from valid source IDs')
    labels=np.array([mapping[x] for x in frame[id_col].astype(str)])
    result=_save_result(name,'datasail_full',frame,labels,{'seed':seed,'sample_ratio':1.0,'n_input_rows':n_input,
        'n_invalid_smiles':len(invalid),'invalid_ids':invalid['_datasail_id'].astype(str).tolist(),
        'fingerprint':CONFIG['defaults']['fingerprint'],'e_clusters':e_clusters,'n_jobs':n_jobs,
        'datasail_seconds':datasail_seconds,'max_sec':max_sec,'_started':started})
    print(json.dumps(result,indent=2)); return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=CONFIG['datasets'])
    parser.add_argument('--strategy',choices=('random','stratified'),default='random')
    parser.add_argument('--sample-ratio',type=float)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--k',type=int)
    parser.add_argument('--n-jobs',type=int,default=1)
    parser.add_argument('--dry-run',action='store_true',help='Validate config/data/IDs/strata without running DataSAIL')
    args=parser.parse_args()
    run(args.dataset,args.strategy,args.sample_ratio,args.seed,args.k,args.n_jobs,args.dry_run)

if __name__=='__main__': main()
