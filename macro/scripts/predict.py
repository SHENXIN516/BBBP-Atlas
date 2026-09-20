import argparse
from pathlib import Path
import numpy as np
from inference_common import checksums, runtime, model_record, load_model, input_frame, graphs, metrics, environment, write, sha, require


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package-root', type=Path, required=True)
    p.add_argument('--group', choices=['macro_cluster70'], default='macro_cluster70')
    p.add_argument('--model-seed', type=int, choices=range(42, 47), required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    require(args.batch_size > 0, 'Batch size must be positive')
    package = args.package_root.resolve()
    hashes = checksums(package)
    record, weight = model_record(package, args.group, args.model_seed, hashes)
    frame = input_frame(args.input)
    core, rt = runtime(package, hashes)
    device = core.resolve_device(rt.torch, args.device)
    core.set_deterministic_seeds(rt, args.model_seed, 'warn', device)
    model = load_model(core, rt, record, weight, device)
    loader = rt.DataLoader(graphs(core, rt, frame), batch_size=args.batch_size, shuffle=False, num_workers=0)
    probs, labels, logits_all = [], [], []
    with rt.torch.inference_mode():
        for batch in loader:
            logits = model(batch.to(device))
            require(logits.ndim == 2 and logits.shape[1] == 2 and bool(rt.torch.isfinite(logits).all()), 'Invalid logits')
            logits_all.append(logits.cpu().numpy())
            probs.append(rt.torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            labels.append(logits.argmax(dim=-1).cpu().numpy())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = frame.copy()
    result['probability_bbb_plus'] = np.concatenate(probs).astype(np.float64)
    result['prediction'] = np.concatenate(labels)
    logits = np.concatenate(logits_all)
    result['logit_bbb_minus'], result['logit_bbb_plus'] = logits[:, 0], logits[:, 1]
    result.to_csv(out / 'predictions.csv', index=False)
    write(out / 'inference.json', {'group': args.group, 'model_seed': args.model_seed,
          'model_config': record['model_config'], 'weight_sha256': sha(weight), 'input_sha256': sha(args.input),
          'predictions_sha256': sha(out / 'predictions.csv'), 'device': str(device), 'batch_size': args.batch_size,
          'strict_load': True, 'decision_rule': 'softmax(logits)[:,1]; argmax(logits); exact ties -> 0',
          'graph_policy': 'Original verified graph builder, original SMILES order; no graph cache or silent row exclusion',
          'environment': environment(rt), 'metrics': metrics(result) if 'label' in result else None})
    print(f'INFERENCE COMPLETE: {out}', flush=True)


if __name__ == '__main__':
    main()
