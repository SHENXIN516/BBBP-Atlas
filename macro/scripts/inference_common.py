import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def checksums(package):
    return dict((name, h) for h, name in (
        line.split('  ', 1) for line in (package / 'SHA256SUMS.txt').read_text().splitlines()))


def verified(package, rel, hashes):
    path = (package / rel).resolve()
    require(path.is_relative_to(package), f'Path outside package: {rel}')
    require(rel in hashes and path.is_file() and sha(path) == hashes[rel], f'Package hash mismatch: {rel}')
    return path


def verify_source_match(package, release, hashes):
    def names(root):
        return {'scripts/train_small_molecule_repro.py'} | {
            p.relative_to(root).as_posix() for p in (root / 'plat_model').rglob('*.py')}
    expected = names(package)
    require(expected == names(release), 'Checkpoint package model source file set differs')
    for rel in sorted(expected):
        original = verified(package, rel, hashes)
        target = release / rel
        require(target.is_file() and sha(target) == sha(original), f'Checkpoint package uses different source: {rel}')


def runtime(package, hashes, local=True):
    release = Path(__file__).resolve().parents[1]
    verify_source_match(package, release, hashes)
    source = release if local else package
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'scripts'))
    core = importlib.import_module('train_small_molecule_repro')
    require(Path(core.__file__).resolve() == (source / 'scripts/train_small_molecule_repro.py').resolve(), 'Imported another training core')
    rt = core.require_runtime(source)
    model_module = importlib.import_module('plat_model.model')
    require(Path(model_module.__file__).resolve() == (source / 'plat_model/model.py').resolve(), 'Imported another model')
    rt.torch.set_num_threads(1)
    return core, rt


def model_record(package, group, seed, hashes):
    index = read(verified(package, 'release_index.json', hashes))
    records = [m for m in index['models'] if m['group'] == group and m['model_seed'] == seed]
    require(len(records) == 1, f'Expected one released checkpoint: {group}/model{seed}')
    record = records[0]
    require(record['training_config'].get('objective') == 'modality_macro_ce', 'Expected modality-macro CE checkpoint')
    rel = f'weights/{group}/model{seed}'
    require(read(verified(package, rel + '/metadata.json', hashes)) == record, 'Index/metadata mismatch')
    path = verified(package, rel + '/model.pt', hashes)
    require(sha(path) == record['exported_weight_sha256'], 'Weight identity mismatch')
    return record, path


def normalize_saved_metrics(value):
    result = dict(value)
    if 'confusion_matrix' not in result and all(k in result for k in ('tn', 'fp', 'fn', 'tp')):
        result['confusion_matrix'] = {k: int(result[k]) for k in ('tn', 'fp', 'fn', 'tp')}
    return result


def load_model(core, rt, record, path, device):
    payload = rt.torch.load(path, map_location='cpu', weights_only=True)
    require(payload['model_config'] == record['model_config'], 'Model configuration mismatch')
    model = core.instantiate_model(rt, payload['model_config'], device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    return model


def input_frame(path):
    frame = pd.read_csv(path, dtype={'sample_id': str, 'sequence': str})
    require(len(frame) > 0 and 'sequence' in frame, 'Input must contain a nonempty sequence column')
    require(frame.sequence.notna().all() and frame.sequence.str.strip().ne('').all(), 'Empty SMILES')
    if 'sample_id' not in frame:
        frame.insert(0, 'sample_id', [f'row{i}' for i in range(len(frame))])
    require(frame.sample_id.notna().all() and frame.sample_id.is_unique, 'Missing or duplicate sample IDs')
    if 'label' in frame:
        require(frame.label.notna().all() and frame.label.isin([0, 1]).all(), 'Labels must be 0 or 1')
    return frame


def graphs(core, rt, frame):
    result = []
    for row in frame.itertuples():
        try:
            result.append(rt.Data(**core.graph_tensor_dict(row.sequence, int(getattr(row, 'label', 0)), rt.torch)))
        except Exception as exc:
            raise RuntimeError(f'Graph construction failed for {row.sample_id}: {exc}') from exc
    return result


def metrics(frame):
    from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef, balanced_accuracy_score
    y, p, pred = frame.label.to_numpy(), frame.probability_bbb_plus.to_numpy(), frame.prediction.to_numpy()
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probabilities')
    require(set(pred).issubset({0, 1}), 'Invalid predicted labels')
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {'n': len(frame), 'accuracy': float(np.mean(y == pred)),
            'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            'mcc': float(matthews_corrcoef(y, pred)), 'f1': float(f1_score(y, pred, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'sensitivity': tp / (tp + fn) if tp + fn else None,
            'specificity': tn / (tn + fp) if tn + fp else None,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}}


def compare(left, right, atol=1e-6, rtol=1e-5):
    require(len(left) == len(right), 'Prediction row-count mismatch')
    for col in ('sample_id', 'sequence', 'label'):
        require(col in left and col in right and left[col].tolist() == right[col].tolist(), f'Prediction order/identity mismatch: {col}')
    a, b = left.probability_bbb_plus.to_numpy(), right.probability_bbb_plus.to_numpy()
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite prediction')
    lm, rm = metrics(left), metrics(right)
    metric_ok = all(lm[k] == rm[k] if isinstance(lm[k], dict) or lm[k] is None or rm[k] is None
                    else abs(lm[k] - rm[k]) <= 1e-10 for k in lm)
    probability_ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    flips = int(np.count_nonzero(left.prediction.to_numpy() != right.prediction.to_numpy()))
    return {'pass': probability_ok and flips == 0 and metric_ok, 'probability_allclose': probability_ok,
            'max_abs_probability_difference': float(np.max(np.abs(a - b))),
            'mean_abs_probability_difference': float(np.mean(np.abs(a - b))),
            'class_disagreements': flips, 'metrics_match': metric_ok,
            'left_metrics': lm, 'right_metrics': rm, 'atol': atol, 'rtol': rtol}


def environment(rt):
    return {'python': platform.python_version(), 'executable': sys.executable,
            'cuda_build': rt.torch.version.cuda,
            'packages': {n: importlib.metadata.version(n) for n in ('torch', 'torch-geometric', 'rdkit', 'numpy', 'pandas', 'scikit-learn')}}
