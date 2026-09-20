import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from inference_common import checksums, verified, runtime, model_record, load_model, input_frame, graphs, metrics, compare, environment, read, write, sha, require, normalize_saved_metrics


CLUSTER = 'runs/peptide_transfer_cluster70_strict_split42'
GROUPS = {
    'macro_cluster70': (
        'runs/cross_modal_completion_cluster70_split42_v1/evaluation/modality_macro_ce/model{seed}/peptide/predictions.csv',
        'runs/cross_modal_completion_cluster70_split42_v1/evaluation/modality_macro_ce/model{seed}/peptide/metrics.json'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package-root', type=Path)
    p.add_argument('--group', choices=GROUPS, default='macro_cluster70')
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44,45,46])
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    require(args.batch_size > 0 and len(set(args.seeds)) == len(args.seeds), 'Invalid batch size or repeated seed')
    if args.package_root is None:
        candidates = list(Path.cwd().glob('releases/checkpoint_delivery_*/final/package'))
        require(len(candidates) == 1, 'Set --package-root to the directory containing release_index.json (multiple or no packages found)')
        args.package_root = candidates[0]
    package = args.package_root.resolve()
    hashes = checksums(package)
    manifest_path = verified(package, CLUSTER + '/fixed_split_manifest.csv', hashes)
    manifest = pd.read_csv(manifest_path)
    frame = manifest.loc[manifest.split.eq('test') & manifest.modality.eq('peptide')].reset_index(drop=True)
    require(len(frame) == 90 and frame.label.value_counts().to_dict() == {0:45, 1:45}, 'Expected fixed balanced 90-peptide test')
    records = []
    for seed in args.seeds:
        record, weight = model_record(package, args.group, seed, hashes)
        require(record['manifest_sha256'] == sha(manifest_path), 'Checkpoint/split mismatch')
        pred_rel, result_rel = [t.format(seed=seed) for t in GROUPS[args.group]]
        pred_path = verified(package, pred_rel, hashes)
        saved = input_frame(pred_path)
        result = read(verified(package, result_rel, hashes))
        h = result.get('checkpoint_sha256') or result.get('checkpoint', {}).get('sha256')
        require(h == record['original_checkpoint_sha256'], 'Saved predictions refer to another original weight')
        h = result.get('predictions_sha256') or result.get('predictions', {}).get('sha256')
        require(h == sha(pred_path), 'Saved prediction/result hash mismatch')
        for col in ('sample_id','sequence','label'):
            require(frame[col].tolist() == saved[col].tolist(), f'Saved prediction/frozen split mismatch: {col}')
        recorded_metrics = normalize_saved_metrics(result.get('test_metrics', result.get('metrics')))
        calculated = metrics(saved)
        for key, value in calculated.items():
            require(key in recorded_metrics and (value == recorded_metrics[key] if isinstance(value, dict) or value is None
                    else abs(value - recorded_metrics[key]) < 1e-7), f'Saved metric mismatch: {key}')
        records.append((seed, record, weight, saved))
    print(f'PREFLIGHT PASS: {len(records)} checkpoints; frozen 90 peptides; no training', flush=True)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / 'fixed_samples.csv', index=False)
    core, rt = runtime(package, hashes, local=False)
    device = core.resolve_device(rt.torch, args.device)
    write(out / 'protocol.json', {'package_root': str(package), 'group': args.group, 'seeds': args.seeds,
          'manifest_sha256': sha(manifest_path), 'device': str(device), 'batch_size': args.batch_size,
          'environment': environment(rt), 'package_environment': read(package / 'environment_verified.json') if (package / 'environment_verified.json').exists() else None,
          'scope': 'Inference parity for supplied checkpoint exports and their stored predictions; not a training reproducibility test'})
    results = []
    for seed, record, weight, saved in records:
        d = out / f'model{seed}'
        d.mkdir()
        try:
            core.set_deterministic_seeds(rt, seed, 'warn', device)
            model = load_model(core, rt, record, weight, device)
            loader = rt.DataLoader(graphs(core, rt, frame), batch_size=args.batch_size, shuffle=False, num_workers=0)
            old_metrics, arrays = core.evaluate_model(rt, model, loader, rt.torch.nn.CrossEntropyLoss(), device, return_predictions=True)
            require(np.array_equal(arrays['labels'], frame.label.to_numpy()), 'Reference evaluation changed row order')
            reference = frame.copy()
            reference['probability_bbb_plus'], reference['prediction'] = arrays['probabilities'], arrays['predictions']
            reference.to_csv(d / 'reference_predictions.csv', index=False)
            write(d / 'reference_metrics.json', old_metrics)
            del model, loader
            if device.type == 'cuda':
                rt.torch.cuda.empty_cache()
            command = [sys.executable, str(Path(__file__).with_name('predict.py')), '--package-root', str(package),
                       '--group', args.group, '--model-seed', str(seed), '--input', str(out / 'fixed_samples.csv'),
                       '--output-dir', str(d / 'new_entry'), '--device', str(device), '--batch-size', str(args.batch_size)]
            subprocess.run(command, check=True)
            current = input_frame(d / 'new_entry/predictions.csv')
            a, b, c = compare(reference, current), compare(saved, reference), compare(saved, current)
            diff = frame[['sample_id','sequence','label']].copy()
            diff['saved_score'], diff['reference_score'], diff['new_score'] = saved.probability_bbb_plus, reference.probability_bbb_plus, current.probability_bbb_plus
            diff['saved_prediction'], diff['reference_prediction'], diff['new_prediction'] = saved.prediction, reference.prediction, current.prediction
            diff['abs_reference_new'] = np.abs(diff.reference_score - diff.new_score)
            diff['abs_saved_new'] = np.abs(diff.saved_score - diff.new_score)
            diff.to_csv(d / 'per_sample_comparison.csv', index=False)
            report = {'seed': seed, 'weight_sha256': sha(weight), 'reference_vs_new': a,
                      'saved_vs_reference': b, 'saved_vs_new': c,
                      'status': 'PASS' if all(x['pass'] for x in (a,b,c)) else 'FAIL'}
            write(d / 'comparison.json', report)
            results.append(report)
            print(f'model{seed}: {report["status"]}; new/reference max score delta={a["max_abs_probability_difference"]:.3g}; saved/new={c["max_abs_probability_difference"]:.3g}', flush=True)
        except Exception as exc:
            report = {'seed': seed, 'status':'ERROR', 'error':f'{type(exc).__name__}: {exc}'}
            write(d / 'comparison.json', report)
            results.append(report)
            print(f'model{seed}: ERROR {exc}', flush=True)
    passed = len(results) == len(records) and all(r['status'] == 'PASS' for r in results)
    write(out / 'parity_summary.json', {'status':'PASS' if passed else 'FAIL', 'group':args.group, 'seeds':args.seeds,
          'scope':'Supplied checkpoint inference only; no training or checkpoint selection',
          'shared_components':'Verified original model and graph builder intentionally reused. Prediction entry and metric calculation are checked separately.',
          'results':results})
    print(f'OVERALL {"PASS" if passed else "FAIL"}: {out / "parity_summary.json"}', flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
