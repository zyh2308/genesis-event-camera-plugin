"""One CPU zero-event recurrent stream; sensitivity control, not quality scoring."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

from check_formal_loaders import definitions, difference, sha
from check_reconstruction_predictions import model_hash, tensor_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--completed-probe-json', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--trainer-source', type=Path, required=True)
    p.add_argument('--training-recipe', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    metadata = json.loads(args.completed_probe_json.read_text())
    if metadata.get('status') != 'complete' or metadata.get('continuous_windows') != 100:
        raise RuntimeError('Expected completed registered 100-window probe metadata')
    args.output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    if sha(args.training_recipe) != metadata['training_recipe_sha256']:
        raise RuntimeError('Training recipe changed since completed probe')
    if sha(args.trainer_source) != metadata['sources'][str(args.trainer_source)]:
        raise RuntimeError('Trainer changed since completed probe')
    sys.path.insert(0, str(args.model_root))
    from data_loader.dataset import DynamicH5Dataset
    from utils.data import data_sources
    from model.model import E2VIDRecurrent
    for obj in (DynamicH5Dataset, E2VIDRecurrent):
        source = Path(inspect.getfile(obj))
        if sha(source) != metadata['sources'][str(source)]:
            raise RuntimeError('Installed model/loader source changed since completed probe')
    Dataset = definitions(args.trainer_source, ['Stage4E2VIDDataset'],
                          {'DynamicH5Dataset': DynamicH5Dataset, 'data_sources': data_sources})['Stage4E2VIDDataset']
    reference_meta = metadata['streams']['reference']
    template = Path(reference_meta['path'])
    if sha(template) != reference_meta['sha256']:
        raise RuntimeError('Actual reference input H5 changed')
    baseline_path = args.completed_probe_json.parent/'reference_predictions.npy'
    baseline = np.load(baseline_path, allow_pickle=False)
    if len(baseline) != 100:
        raise RuntimeError('Reference prediction count differs from completed probe')
    checkpoint_path = Path(metadata['checkpoint'])
    if sha(checkpoint_path) != metadata['checkpoint_sha256']:
        raise RuntimeError('Frozen checkpoint changed; expected hash read from completed metadata')
    config = json.loads(args.training_recipe.read_text())['reconstruction']
    model = E2VIDRecurrent(dict(config['model_args'])).cpu()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    expected_model_hash = metadata['model_state_sha256']
    if model_hash(model) != expected_model_hash:
        raise RuntimeError('Loaded checkpoint model hash mismatch')
    model.eval();model.reset_states()
    if tensor_count(model.states) != 0:
        raise RuntimeError('Segment-start reset did not clear states')
    dataset = Dataset(str(template), num_bins=5, combined_voxel_channels=True,
                      filter_hot_events=False, transforms={'CenterCrop': {'size': [160,160]}})
    outputs, rows = [], []
    try:
        with torch.inference_mode():
            for position, index in enumerate(reference_meta['indices']):
                item = dataset.__getitem__(index, seed=0)
                original = item['events'].unsqueeze(0).float().cpu()
                if hashlib.sha256(np.ascontiguousarray(original.numpy()).tobytes()).hexdigest() != reference_meta['rows'][position]['input_sha256']:
                    raise RuntimeError('Formal reference event tensor changed since completed probe')
                if hashlib.sha256(np.ascontiguousarray(baseline[position]).tobytes()).hexdigest() != reference_meta['rows'][position]['prediction_sha256']:
                    raise RuntimeError('Saved baseline prediction changed since completed probe')
                zero = torch.zeros_like(original)
                image = model(zero)['image']
                frame = item['frame'].unsqueeze(0).float().cpu()
                if image.shape != frame.shape or not torch.isfinite(image).all():
                    raise RuntimeError('Zero-input prediction shape/nonfinite failure')
                array = image.detach().cpu().numpy().copy()
                if array.shape != baseline[position].shape:
                    raise RuntimeError('Prediction shape differs from completed probe')
                if float(item['timestamp']) != reference_meta['rows'][position]['right_s']:
                    raise RuntimeError('Physical window differs from completed probe')
                nstate = tensor_count(model.states)
                if nstate == 0:
                    raise RuntimeError('Recurrent state unexpectedly empty')
                outputs.append(array)
                rows.append({'index': index, 'right_s': float(item['timestamp']),
                    'original_input_nonzero_elements': int(torch.count_nonzero(original)),
                    'zero_input_nonzero_elements': int(torch.count_nonzero(zero)),
                    'prediction_difference_from_original': difference(baseline[position],array),
                    'zero_input_mse_to_true_frame': float(np.mean((array.astype(np.float64)-frame.numpy().astype(np.float64))**2)),
                    'original_input_mse_to_true_frame': reference_meta['rows'][position]['mse_to_true_frame'],
                    'recurrent_state_tensors': nstate})
    finally:
        dataset.h5_file.close()
    if model_hash(model) != expected_model_hash:
        raise RuntimeError('Frozen model parameters/buffers changed during eval')
    zero_predictions = np.stack(outputs)
    comparison = difference(baseline,zero_predictions)
    # Positive control only: nonzero effect, no invented quality tolerance.
    sensitive = (comparison['changed_elements'] > 0 and comparison['max_abs'] > 0
                 and sum(r['original_input_nonzero_elements'] for r in rows) > 0)
    report = {'status': 'complete', 'training_started': False, 'real_data_used': False,
              'device': 'cpu', 'threads': 2, 'continuous_windows': len(rows),
              'completed_probe_metadata_sha256': sha(args.completed_probe_json),
              'checkpoint': str(checkpoint_path), 'checkpoint_sha256': metadata['checkpoint_sha256'],
              'model_state_sha256': expected_model_hash, 'model_state_unchanged': True,
              'baseline_predictions': str(baseline_path), 'baseline_predictions_sha256': sha(baseline_path),
              'control': 'zeros_like(actual_formal_loader_events)',
              'recurrent_reset': 'once_at_segment_start_only',
              'prediction_difference': comparison,
              'positive_control_detected_nonzero_event_input_response': sensitive,
              'positive_control_passed': sensitive,
              'quality_tolerance_used': False, 'quality_acceptance_claimed': False,
              'note': 'Detects that the frozen checkpoint responds to removing events. Does not establish universal sensitivity to every possible perturbation.',
              'rows': rows}
    np.save(args.output_root/'zero_event_predictions.npy',zero_predictions)
    (args.output_root/'reconstruction_positive_control.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    return 0 if sensitive else 2


if __name__ == '__main__':
    raise SystemExit(main())
