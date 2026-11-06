"""Compare recorded end-to-end and DataSAIL-stage runtimes across datasets."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.pipeline_engine import CONFIG
from src.analysis_common import METHODS,ANALYSIS_DIR,split_dir

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--datasets',nargs='+',choices=list(CONFIG['datasets']),default=list(CONFIG['datasets']));p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);a=p.parse_args();rows=[]
 for d in a.datasets:
  for m in a.methods:
   folder=split_dir(d,m); meta_path=folder/'run_metadata.json';meta={}
   if meta_path.exists():
    try:meta=json.loads(meta_path.read_text())
    except (OSError,json.JSONDecodeError):pass
   files=all((folder/f'{s}.csv').exists() for s in ('train','val','test'))
   rows.append({'dataset':d,'method':m,'split_status':'available' if files else 'missing',
    'end_to_end_seconds':meta.get('total_seconds',meta.get('runtime_seconds')),
    'datasail_stage_seconds':meta.get('datasail_seconds'),'seed':meta.get('seed'),
    'anchor_ratio':meta.get('sample_ratio'),'anchor_n':meta.get('n_anchor_rows',meta.get('n_datasail_anchor')),
    'runtime_scope':meta.get('runtime_scope'),'source_sha256':meta.get('source_sha256'),
    'git_commit':meta.get('git_commit'),'git_worktree_dirty':meta.get('git_worktree_dirty'),
    'metadata_path':str(meta_path.relative_to(ROOT)) if meta else ''})
 out=ANALYSIS_DIR/'runtime_comparison.csv';out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out,index=False);print(f'Saved {out}')
if __name__=='__main__':main()
