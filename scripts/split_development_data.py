import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import sklearn
from sklearn.model_selection import train_test_split


def prepare_frame(path):
    raw = pd.read_csv(path)
    column = 'sequence' if 'sequence' in raw.columns else 'SMILES'
    if column not in raw.columns or 'label' not in raw.columns:
        raise ValueError('Expected sequence/SMILES and label columns.')
    frame = raw.copy()
    frame['source_row_id'] = np.arange(len(raw))
    if 'type' in frame.columns:
        frame = frame.loc[frame['type'].eq('SMILES')].copy()
    frame['sequence'] = frame[column].astype(str)
    labels = pd.to_numeric(frame['label'], errors='raise')
    if not labels.isin([0, 1]).all():
        raise ValueError('Labels must be binary 0/1.')
    frame['label'] = labels.astype(int)
    valid = frame['sequence'].map(lambda value: Chem.MolFromSmiles(value) is not None)
    excluded = frame.loc[~valid, ['source_row_id', 'sequence', 'label']].copy()
    frame = frame.loc[valid, ['source_row_id', 'sequence', 'label']].reset_index(drop=True)
    if frame['sequence'].duplicated().any():
        raise ValueError('Duplicate SMILES in development data.')
    return frame, excluded


def split_indices(n, seed):
    train, rest = train_test_split(np.arange(n), test_size=0.2,
                                   random_state=seed, shuffle=True)
    val, test = train_test_split(rest, test_size=0.5,
                               random_state=seed, shuffle=True)
    return {k: np.sort(v) for k, v in [('train', train), ('val', val), ('test', test)]}


def main():
    p = argparse.ArgumentParser(description='Export fixed development splits.')
    p.add_argument('--data', type=Path, required=True, help='Ordered development CSV; separate benchmark already excluded')
    p.add_argument('--setting', choices=['small', 'mixed'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    seed = 88 if args.setting == 'small' else 42
    frame, excluded = prepare_frame(args.data)
    expected_n = 9116 if args.setting == 'small' else 10018
    if len(frame) != expected_n:
        raise ValueError(f'{args.setting} expects {expected_n} valid development rows, got {len(frame)}. Do not split the full release including held-out records.')
    indices = split_indices(len(frame), seed)
    frame['split'] = ''
    for name, idx in indices.items():
        frame.loc[idx, 'split'] = name
    frame['split_seed'] = seed
    pairs = list(zip(frame.sequence, frame.label.astype(int)))
    ordered_hash = hashlib.sha256(json.dumps(pairs, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    expected_hash = {
        'small': '5a9d240af0b3789afea9c848d79897f241c0acc16457c9c4d2465ed73c48cfc0',
        'mixed': '23c7bc0d5f2cae435e8d438584c6defc72f6aa0b42aac3456ec9fd40204e079e',
    }[args.setting]
    if ordered_hash != expected_hash:
        raise ValueError('Development data or row order differs from the verified input.')
    report = {
        'status': 'reconstructed',
        'setting': args.setting, 'split_seed': seed,
        'method': 'train_test_split(test_size=0.2), then train_test_split(test_size=0.5); shuffle=True, stratify=None',
        'input_file': args.data.name,
        'input_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'ordered_structure_label_sha256': ordered_hash,
        'counts': {k: len(v) for k, v in indices.items()},
        'label_counts': {k: {str(label): int(n) for label, n in frame.iloc[idx].label.value_counts().sort_index().items()} for k, idx in indices.items()},
        'excluded_invalid_n': len(excluded),
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'rdkit': rdBase.rdkitVersion},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / 'split_manifest.csv', index=False)
    for name in indices:
        part = frame.loc[frame['split'].eq(name)].drop(columns='split').copy()
        part.insert(2, 'type', 'SMILES')
        part.to_csv(args.output_dir / f'{name}.csv', index=False)
    excluded.to_csv(args.output_dir / 'excluded_invalid.csv', index=False)
    (args.output_dir / 'split_method.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
