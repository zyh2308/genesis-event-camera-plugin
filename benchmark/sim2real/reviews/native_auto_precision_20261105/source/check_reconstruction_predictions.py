"""Frozen-checkpoint CPU recurrent sensitivity probe; no training or real data.

Consumes actual formal-loader H5 copies made by check_formal_loaders.py. It
does not adjust model output, quantize input timestamps, or accept tolerances.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

from check_formal_loaders import definitions, difference, sha, as_numpy


def model_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_count(value):
    if torch.is_tensor(value):
        return 1
    if isinstance(value, dict):
        return sum(tensor_count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_count(v) for v in value)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--trainer-source', type=Path, required=True)
    p.add_argument('--training-recipe', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-checkpoint-sha256', required=True)
    p.add_argument('--reference-h5', type=Path, required=True)
    p.add_argument('--candidate-h5', type=Path, required=True)
    p.add_argument('--reference-repeat-h5', type=Path, required=True)
    p.add_argument('--start-s', type=float, required=True)
    p.add_argument('--end-s', type=float, required=True)
    p.add_argument('--expected-windows', type=int, default=100)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    sys.path.insert(0, str(args.model_root))
    from data_loader.dataset import DynamicH5Dataset
    from utils.data import data_sources
    from model.model import E2VIDRecurrent
    ctx = definitions(args.trainer_source, ['Stage4E2VIDDataset'],
                      {'DynamicH5Dataset': DynamicH5Dataset, 'data_sources': data_sources})
    Dataset = ctx['Stage4E2VIDDataset']
    config = json.loads(args.training_recipe.read_text())['reconstruction']
    model = E2VIDRecurrent(dict(config['model_args'])).cpu()
    if sha(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError('Frozen checkpoint file SHA256 differs from registered checkpoint')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('schema') != 'e2vid_stage4_resume_v1':
        raise RuntimeError('Unexpected formal checkpoint schema')
    model.load_state_dict(checkpoint['model'], strict=True)
    expected_hash = checkpoint['model_sha256']
    if model_hash(model) != expected_hash:
        raise RuntimeError('Loaded frozen model differs from checkpoint hash')
    model.eval()
    source_paths = [args.trainer_source, Path(inspect.getfile(DynamicH5Dataset)),
                    Path(inspect.getfile(E2VIDRecurrent))]
    report = {'status': 'running', 'training_started': False, 'real_data_used': False,
              'device': 'cpu', 'threads': 2, 'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': sha(args.checkpoint),
              'model_state_sha256': expected_hash,
              'training_recipe_sha256': sha(args.training_recipe),
              'sources': {str(x): sha(x) for x in source_paths},
              'interval_s': [args.start_s, args.end_s],
              'tolerance_acceptance_used': False,
              'mse_definition': 'float64 mean squared difference of native float32 model image and formal-loader true-frame tensor; no range rescaling',
              'full_episode_from_zero_claimed': False,
              'note': 'Recurrent states reset once at shared segment start; then continue across every selected physical window. No histogram matching, rescaling, clipping, or optimizer step.',
              'streams': {}, 'pairs': {}}
    def save():
        (args.output_root/'reconstruction_prediction_probe.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    predictions, targets = {}, {}
    paths = {'reference': args.reference_h5, 'candidate': args.candidate_h5,
             'reference_repeat': args.reference_repeat_h5}
    fixed_indices = None
    for name, path in paths.items():
        dataset = Dataset(str(path), num_bins=5, combined_voxel_channels=True,
                          filter_hot_events=False,
                          transforms={'CenterCrop': {'size': [160, 160]}})
        try:
            indices = []
            for i in range(len(dataset)):
                right = float(dataset.frame_ts[i])
                left = float(dataset.frame_ts[i-1]) if i else 0.
                if left >= args.start_s-1e-12 and right <= args.end_s+1e-12 and right > left:
                    indices.append(i)
            if len(indices) != args.expected_windows:
                raise RuntimeError(f'{name}: selected {len(indices)} windows, expected {args.expected_windows}')
            if indices != list(range(indices[0],indices[-1]+1)):
                raise RuntimeError('Selected windows must be consecutive')
            if fixed_indices is None:
                fixed_indices = indices
            elif indices != fixed_indices:
                raise RuntimeError('Different physical indices across streams')
            model.reset_states()
            if tensor_count(model.states) != 0:
                raise RuntimeError('Recurrent reset did not clear state')
            stream = {'path': str(path), 'sha256': sha(path), 'indices': indices,
                      'state_reset_at_segment_start': True, 'rows': []}
            report['streams'][name] = stream
            outputs, gt = [], []
            with torch.inference_mode():
                for i in indices:
                    item = dataset.__getitem__(i, seed=0)
                    source = item['events'].unsqueeze(0).float().cpu()
                    image = model(source)['image']
                    target = item['frame'].unsqueeze(0).float().cpu()
                    if image.shape != target.shape:
                        raise RuntimeError(f'Native model/target shape mismatch {image.shape} != {target.shape}; no guessed crop/pad')
                    if not torch.isfinite(image).all() or not torch.isfinite(target).all():
                        raise RuntimeError('Non-finite frozen-model output or true frame')
                    state_tensors = tensor_count(model.states)
                    if state_tensors == 0:
                        raise RuntimeError('Expected nonempty recurrent state')
                    array = image.detach().cpu().numpy().copy()
                    truth = target.detach().cpu().numpy().copy()
                    outputs.append(array);gt.append(truth)
                    stream['rows'].append({'index': i, 'right_s': float(item['timestamp']),
                        'input_sha256': hashlib.sha256(np.ascontiguousarray(as_numpy(source)).tobytes()).hexdigest(),
                        'prediction_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                        'prediction_range': [float(array.min()), float(array.max())],
                        'target_range': [float(truth.min()), float(truth.max())],
                        'mse_to_true_frame': float(np.mean((array.astype(np.float64)-truth.astype(np.float64))**2)),
                        'recurrent_state_tensors': state_tensors})
            if model_hash(model) != expected_hash:
                raise RuntimeError('Frozen model parameters/buffers changed during eval')
            predictions[name] = np.stack(outputs)
            targets[name] = np.stack(gt)
            # Numeric predictions, not display imagery or compressed illustrations.
            np.save(args.output_root/(name+'_predictions.npy'), predictions[name])
            stream['mean_mse_to_true_frame'] = float(np.mean([r['mse_to_true_frame'] for r in stream['rows']]))
            stream['model_state_unchanged'] = True
            save()
        finally:
            dataset.h5_file.close()
    for name in ('candidate','reference_repeat'):
        if not np.array_equal(targets['reference'],targets[name]):
            raise RuntimeError('True target frames differ across supposedly shared physical streams')
        rows = []
        for index, a, b, ra, rb in zip(fixed_indices,predictions['reference'],predictions[name],
                                      report['streams']['reference']['rows'],report['streams'][name]['rows']):
            rows.append({'index': index, 'prediction_difference': difference(a,b),
                         'reference_mse': ra['mse_to_true_frame'],
                         'other_mse': rb['mse_to_true_frame'],
                         'mse_delta_other_minus_reference': rb['mse_to_true_frame']-ra['mse_to_true_frame']})
        deltas = np.array([r['mse_delta_other_minus_reference'] for r in rows])
        report['pairs']['reference_vs_'+name] = {
            'whole_sequence_prediction_difference': difference(predictions['reference'],predictions[name]),
            'mean_mse_delta': float(deltas.mean()),
            'max_abs_per_frame_mse_delta': float(np.abs(deltas).max(initial=0)),
            'rows': rows}
    report.update(status='complete', continuous_windows=args.expected_windows,
                  recurrent_states_reset_between_windows=False,
                  checkpoint_parameters_frozen=True,
                  quality_acceptance_pass_claimed=False,
                  general_quality_or_real_test_result_claimed=False)
    save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('streams','pairs')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
