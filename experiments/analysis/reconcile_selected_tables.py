"""Reconcile seed-42 runtime, global leakage, and NN leakage tables.

Scaled/random splits are selected from nested seed_42 runs. Full DataSAIL is
selected from its single successful baseline directory. Existing global
pairwise leakage estimates are retained only after verifying their split row
counts and seed sampling settings; the previously missing HIV Full estimate is
recomputed from the verified split.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd


def sampled_mean(a: np.ndarray, b: np.ndarray, n_pairs: int, seed: int) -> float:
    if not len(a) or not len(b):
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(len(a), size=n_pairs)
    j = rng.integers(len(b), size=n_pairs)
    parts = []
    for start in range(0, n_pairs, 10_000):
        x = a[i[start:start + 10_000]].astype(np.uint32)
        y = b[j[start:start + 10_000]].astype(np.uint32)
        inter = np.sum(x * y, axis=1)
        union = np.sum((x + y) > 0, axis=1)
        parts.extend(np.divide(inter, union, out=np.ones_like(inter, dtype=float), where=union > 0))
    return float(np.mean(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--output-dir', type=Path, help='Defaults to results/analysis/unified.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pairs', type=int, default=100_000)
    p.add_argument('--query-n', type=int, default=500)
    p.add_argument('--train-cap', type=int, default=10_000)
    p.add_argument('--nn-threshold', type=float, default=0.7)
    p.add_argument('--n-jobs', type=int, default=1)
    args = p.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from src.pipeline_engine import CONFIG
    from src.analysis_common import fingerprint_matrix, smiles_column

    methods = ['random', 'datasail_scaled_random', 'datasail_scaled_stratified', 'datasail_full']
    outdir = (args.output_dir or root / 'results/analysis/unified').resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    split_root = root / 'results/experiments/unified/splits'
    analysis_root = root / 'results/analysis/unified'
    runtime_rows = []
    selected_paths = {}
    for dataset in CONFIG['datasets']:
        spec = CONFIG['datasets'][dataset]
        source_path = root / spec['path']
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        for method in methods:
            run_id = 'seed_42' if method != 'datasail_full' else 'baseline'
            folder = split_root / dataset / method
            if run_id != 'baseline':
                folder = folder / run_id
            meta_path = folder / 'run_metadata.json'
            split_paths = [folder / f'{s}.csv' for s in ('train', 'val', 'test')]
            available = meta_path.is_file() and all(x.is_file() for x in split_paths)
            metadata = {}
            if available:
                metadata = json.loads(meta_path.read_text())
                if metadata.get('source_sha256') != source_hash:
                    raise ValueError(f'Source hash mismatch: {meta_path}')
                if method != 'datasail_full' and int(metadata.get('seed', -1)) != args.seed:
                    raise ValueError(f'Unexpected seed in {meta_path}')
                if method == 'datasail_full' and int(metadata.get('seed', args.seed)) != args.seed:
                    raise ValueError(f'Unexpected Full DataSAIL seed metadata in {meta_path}')
            selected_paths[(dataset, method)] = {'folder': folder, 'metadata': metadata,
                                                 'available': available, 'run_id': run_id,
                                                 'source_hash': source_hash}
            runtime_rows.append({
                'dataset': dataset, 'method': method,
                'split_status': 'available' if available else 'missing',
                'end_to_end_seconds': metadata.get('total_seconds', metadata.get('runtime_seconds')),
                'datasail_stage_seconds': metadata.get('datasail_seconds'),
                'seed': metadata.get('seed'),
                'anchor_ratio': metadata.get('sample_ratio'),
                'anchor_n': metadata.get('n_anchor_rows', metadata.get('n_datasail_anchor')),
                'runtime_scope': metadata.get('runtime_scope'),
                'source_sha256': metadata.get('source_sha256', source_hash if available else None),
                'git_commit': metadata.get('git_commit'),
                'git_worktree_dirty': metadata.get('git_worktree_dirty'),
                'metadata_path': str((folder / 'run_metadata.json').relative_to(root)) if available else '',
            })
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(outdir / 'runtime_comparison.csv', index=False)

    # Retain the 39 previously computed global-pairwise values only if the
    # current selected partitions have the same IDs/counts and sampling design.
    old_global = pd.read_csv(analysis_root / 'leakage_analysis.csv')
    expected_keys = {(d, m) for d in CONFIG['datasets'] for m in methods}
    if old_global.duplicated(['dataset', 'method']).any() or set(zip(old_global.dataset, old_global.method)) != expected_keys:
        raise ValueError('Existing leakage_analysis.csv does not contain exactly one row per dataset/method')
    for _, row in old_global.iterrows():
        key = (row.dataset, row.method)
        sel = selected_paths[key]
        if not sel['available']:
            if row.status not in ('missing_splits', 'missing'):
                raise ValueError(f'Unexpected old leakage status for unavailable {key}: {row.status}')
            continue
        frames = {s: pd.read_csv(sel['folder'] / f'{s}.csv') for s in ('train', 'val', 'test')}
        if row.status == 'ok':
            if int(row.train_n) != len(frames['train']) or int(row.test_n) != len(frames['test']):
                raise ValueError(f'Existing leakage counts differ from selected split for {key}')
            if int(row.sampled_pairs) != args.pairs or int(row.pair_sample_seed) != args.seed:
                raise ValueError(f'Existing leakage sample settings differ for {key}')
            if method != 'datasail_full':
                legacy_folder = split_root / dataset / method
                for split in ('train', 'val', 'test'):
                    legacy_file = legacy_folder / f'{split}.csv'
                    if not legacy_file.is_file():
                        raise FileNotFoundError(f'Legacy split used by existing leakage row is missing: {legacy_file}')
                    legacy_ids = set(pd.read_csv(legacy_file, usecols=['mol_id'])['mol_id'].astype(str))
                    selected_ids = set(frames[split]['mol_id'].astype(str))
                    if legacy_ids != selected_ids:
                        raise ValueError(f'Existing leakage row does not match selected IDs for {key}/{split}')

    hiv_full = selected_paths[('hiv', 'datasail_full')]
    if not hiv_full['available']:
        raise FileNotFoundError('Verified HIV Full DataSAIL split is unavailable')
    hf = {s: pd.read_csv(hiv_full['folder'] / f'{s}.csv') for s in ('train', 'val', 'test')}
    smiles_col = smiles_column('hiv', hf['train'])
    train_smiles = hf['train'][smiles_col].fillna('').astype(str).tolist()
    test_smiles = hf['test'][smiles_col].fillna('').astype(str).tolist()
    train_fp = fingerprint_matrix(train_smiles, args.n_jobs)
    test_fp = fingerprint_matrix(test_smiles, args.n_jobs)
    hiv_global_value = sampled_mean(train_fp, test_fp, args.pairs, args.seed)
    computed_hiv_global = {
        'dataset': 'hiv', 'method': 'datasail_full', 'status': 'ok',
        'train_n': len(train_fp), 'test_n': len(test_fp), 'sampled_pairs': args.pairs,
        'mean_train_test_similarity': hiv_global_value, 'pair_sample_seed': args.seed,
        'metric': 'mean Tanimoto over sampled train-test pairs',
    }
    old_global = old_global[~((old_global.dataset == 'hiv') & (old_global.method == 'datasail_full'))]
    global_rows = pd.concat([old_global, pd.DataFrame([computed_hiv_global])], ignore_index=True)
    if global_rows.duplicated(['dataset', 'method']).any() or len(global_rows) != len(expected_keys):
        raise ValueError('Reconciled leakage table has duplicate or missing dataset/method rows')
    global_rows = global_rows.sort_values(['dataset', 'method']).reset_index(drop=True)
    global_rows.to_csv(outdir / 'leakage_analysis.csv', index=False)

    # NN leakage reuses the all-dataset, seed-42 robustness calculation with
    # identical query/train sampling settings, adding the Full HIV row.
    robustness_path = root / 'results/analysis/unified/thesis_snapshot_20261114/leakage_thresholds_raw_all10.csv'
    robust = pd.read_csv(robustness_path)
    if args.query_n != 500 or args.train_cap != 10_000 or args.nn_threshold != 0.7:
        raise ValueError('This reconciler expects the existing robustness sample settings (500, 10000, 0.7)')
    nn_rows = []
    for dataset in CONFIG['datasets']:
        for method in methods:
            run_id = 'baseline' if method == 'datasail_full' else f'seed_{args.seed}'
            hit = robust[(robust.dataset == dataset) & (robust.method == method) &
                         (robust.run_id == run_id) & (robust.threshold == args.nn_threshold)]
            sel = selected_paths[(dataset, method)]
            if not sel['available']:
                nn_rows.append({'dataset': dataset, 'method': method, 'status': 'missing_splits',
                                'query_n': np.nan, 'train_n': np.nan, 'mean_max_similarity': np.nan,
                                'median_max_similarity': np.nan, 'fraction_above_threshold': np.nan,
                                'threshold': args.nn_threshold, 'sample_seed': args.seed})
                continue
            if len(hit) != 1:
                raise ValueError(f'Expected one robust NN result for {dataset}/{method}/{run_id}, found {len(hit)}')
            h = hit.iloc[0]
            nn_rows.append({'dataset': dataset, 'method': method, 'status': 'ok',
                            'query_n': int(h.query_n), 'train_n': int(h.train_reference_n),
                            'mean_max_similarity': h.mean_max_similarity,
                            'median_max_similarity': h.median_max_similarity,
                            'fraction_above_threshold': h.fraction_above_threshold,
                            'threshold': args.nn_threshold, 'sample_seed': args.seed})
    nn = pd.DataFrame(nn_rows).sort_values(['dataset', 'method']).reset_index(drop=True)
    nn.to_csv(outdir / 'nn_leakage.csv', index=False)

    manifest = {
        'seed': args.seed, 'methods': methods, 'selection_rule': 'seed_42 for random and scaled methods; baseline single run for Full DataSAIL',
        'global_leakage_pairs': args.pairs, 'global_pair_sample_seed': args.seed,
        'nn_query_n': args.query_n, 'nn_train_cap': args.train_cap,
        'nn_threshold': args.nn_threshold, 'nn_sample_seed': args.seed,
        'datasets_with_full_splits': sorted(d for d in CONFIG['datasets'] if selected_paths[(d, 'datasail_full')]['available']),
        'datasets_missing_full_splits': sorted(d for d in CONFIG['datasets'] if not selected_paths[(d, 'datasail_full')]['available']),
        'hiv_full_global_mean_train_test_similarity': hiv_global_value,
        'hiv_full_nn_fraction_above_threshold': float(nn[(nn.dataset == 'hiv') & (nn.method == 'datasail_full')].fraction_above_threshold.iloc[0]),
        'hiv_full_metadata_path': str((hiv_full['folder'] / 'run_metadata.json').relative_to(root)),
    }
    (outdir / 'selected_table_reconciliation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Runtime rows: {len(runtime)}; available Full splits: {len(manifest['datasets_with_full_splits'])}")
    print(f"Global leakage rows: {len(global_rows)}; HIV Full mean similarity: {hiv_global_value:.6f}")
    print(f"NN leakage rows: {len(nn)}; HIV Full fraction > {args.nn_threshold}: {manifest['hiv_full_nn_fraction_above_threshold']:.4f}")
    print('Missing Full:', ', '.join(manifest['datasets_missing_full_splits']))

if __name__ == '__main__':
    main()
