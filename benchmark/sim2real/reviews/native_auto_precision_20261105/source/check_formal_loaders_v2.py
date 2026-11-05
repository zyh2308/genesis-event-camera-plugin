"""CPU-only paired raw-event audit using actual, pinned production encoders.

No simulation, training, scoring, tolerance acceptance, or input modification.
Production AST definitions are compiled unchanged, avoiding trainer main().
H5 templates are copied to a new audit directory before event replacement.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import inspect
import json
import random
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def difference(a, b):
    a, b = as_numpy(a), as_numpy(b)
    result = {'shape_a': list(a.shape), 'shape_b': list(b.shape),
              'dtype_a': str(a.dtype), 'dtype_b': str(b.dtype),
              'sha_a': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
              'sha_b': hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()}
    exact = a.shape == b.shape and a.dtype == b.dtype and result['sha_a'] == result['sha_b']
    result['bitwise_equal'] = exact
    if a.shape == b.shape:
        delta = a.astype(np.float64) - b.astype(np.float64)
        result.update(changed_elements=int(np.count_nonzero(delta)),
                      max_abs=float(np.abs(delta).max(initial=0)),
                      rms=float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.,
                      finite_a=bool(np.isfinite(a).all()), finite_b=bool(np.isfinite(b).all()))
    return result


def definitions(path, names, namespace=None):
    """Extract the named production functions/classes verbatim, plus constants."""
    path = Path(path)
    tree = ast.parse(path.read_text())
    ctx = dict(np=np, cv2=cv2, torch=torch, F=F, h5py=h5py,
               Path=Path, random=random, __name__='formal_loader_audit')
    if namespace:
        ctx.update(namespace)
    # Import literal production constants, never substitute guessed values.
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                ctx[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    for name in names:
        nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        if len(nodes) != 1:
            raise RuntimeError(f'Expected one production definition {name} in {path}, got {len(nodes)}')
        module = ast.Module(body=[nodes[0]], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ctx)
    return ctx


def events(path):
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            result = np.asarray(value['events'])
        finally:
            value.close()
    else:
        result = np.asarray(value)
    if result.ndim != 2 or result.shape[1] != 4 or result.dtype != np.float64:
        raise ValueError('Raw events must be native float64 Nx4 [seconds,x,y,+/-1]; do not quantize')
    if not np.isfinite(result).all() or (np.diff(result[:, 0]) < 0).any():
        raise ValueError('Non-finite or unsorted raw events')
    if len(result) and (not np.isin(result[:, 3], [-1., 1.]).all()):
        raise ValueError('Raw polarity must be +/-1')
    return result


def window(raw, left, right):
    i, j = np.searchsorted(raw[:, 0], [left, right], side='right')
    return raw[i:j], [int(i), int(j)]


def clone_reconstruction(template, raw, dest):
    shutil.copyfile(template, dest)
    with h5py.File(dest, 'r+') as h:
        keys = {'xs': raw[:, 1], 'ys': raw[:, 2], 'ts': raw[:, 0],
                'ps': (raw[:, 3] > 0).astype(np.uint8)}
        for key, values in keys.items():
            node = h[f'events/{key}']
            dtype, attrs = node.dtype, dict(node.attrs)
            if key == 'ts' and dtype != raw[:, 0].dtype:
                raise RuntimeError('Formal H5 timestamp dtype differs; no implicit precision change allowed')
            del h[f'events/{key}']
            item = h['events'].create_dataset(key, data=values.astype(dtype))
            for name, value in attrs.items():
                item.attrs[name] = value
        h.attrs['num_events'] = len(raw)
        h.attrs['precision_audit_only'] = True
    # Only events and num_events change; true frames/GT remain copied unchanged.
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases-json', type=Path, required=True)
    parser.add_argument('--scripts-root', type=Path, required=True)
    parser.add_argument('--flow-trainer-source', type=Path, required=True)
    parser.add_argument('--e2vid-trainer-source', type=Path, required=True)
    parser.add_argument('--e2vid-model-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    scripts = args.scripts_root
    source_paths = [scripts / 'prepare_steflow_stage0.py', scripts / 'run_pose_stage2_pilot.py',
                    args.flow_trainer_source, args.e2vid_trainer_source]
    manifest = json.loads(args.cases_json.read_text())
    report = {'status': 'running', 'real_data_used': False, 'training_started': False,
              'all_exact_required': True, 'tolerance_acceptance_used': False,
              'cases_manifest_sha256': sha(args.cases_json),
              'sources': {str(p): sha(p) for p in source_paths}, 'cases': []}
    def save():
        (args.output_root / 'formal_loader_results.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    flow = definitions(source_paths[0], ['gaussian_count_representation'])
    pose = definitions(source_paths[1], ['encode_event_window'])
    flow_runner = definitions(args.flow_trainer_source,
                              ['transform_matrix_and_grid', 'apply_joint_transform', 'val_input'])
    # Import the actual installed pinned E2VID dataset, not the archived old mirror.
    sys.path.insert(0, str(args.e2vid_model_root))
    from data_loader.dataset import DynamicH5Dataset
    from utils.data import data_sources
    dataset_source = Path(inspect.getfile(DynamicH5Dataset)).resolve()
    report['sources'][str(dataset_source)] = sha(dataset_source)
    recon = definitions(args.e2vid_trainer_source, ['Stage4E2VIDDataset'],
                        {'DynamicH5Dataset': DynamicH5Dataset, 'data_sources': data_sources})
    Dataset = recon['Stage4E2VIDDataset']
    upstream = sys.modules[DynamicH5Dataset.__module__]
    voxel_fn = getattr(upstream, 'events_to_voxel_torch', None)
    if voxel_fn is not None:
        voxel_source = Path(inspect.getfile(voxel_fn)).resolve()
        report['sources'][str(voxel_source)] = sha(voxel_source)
    for spec in manifest['cases']:
        case = {'id': spec['id'], 'inputs': {}, 'windows_10ms': [], 'flow': [], 'pose': [], 'reconstruction': []}
        report['cases'].append(case)
        try:
            raw_a, raw_b = events(spec['reference']), events(spec['candidate'])
            for key in ('reference', 'candidate', 'physical_h5'):
                case['inputs'][key] = {'path': spec[key], 'sha256': sha(spec[key])}
            case['raw'] = difference(raw_a, raw_b)
            case['address_polarity_order'] = difference(raw_a[:, 1:], raw_b[:, 1:])
            with h5py.File(spec['physical_h5'], 'r') as h:
                g = h['episodes'][spec.get('episode_id', spec['id'])]
                ts = np.asarray(g['frame_timestamps_s'], dtype=np.float64)
                height, width = g['depth_m'].shape[1:]
            start, end = float(spec['start_s']), float(spec['end_s'])
            if not start < end:
                raise ValueError('Explicit shared physical start_s/end_s required')
            pairs = [(i, float(ts[i]), float(ts[i + 1])) for i in range(len(ts)-1)
                     if ts[i] >= start - 1e-12 and ts[i+1] <= end + 1e-12]
            if not pairs:
                raise ValueError('No complete physical 10ms windows covered')
            for i, left, right in pairs:
                wa, ia = window(raw_a, left, right)
                wb, ib = window(raw_b, left, right)
                boundary = {'index': i, 'left_s': left, 'right_s': right,
                            'reference_range': ia, 'candidate_range': ib,
                            'membership_index_equal': ia == ib,
                            'window_address_polarity': difference(wa[:, 1:], wb[:, 1:])}
                case['windows_10ms'].append(boundary)
                tensors = [pose['encode_event_window'](
                    r[:, 0], r[:, 1].astype(int), r[:, 2].astype(int),
                    (r[:, 3] > 0).astype(np.uint8), left, right, width, height, 8.)
                    for r in (raw_a, raw_b)]
                case['pose'].append({'index': i, 'input': difference(*tensors)})
            # Exact non-overlapping original 30ms physical grid (indices 0,3,...).
            flow_template = spec.get('flow_template_npz')
            if flow_template:
                case['inputs']['flow_template'] = {'path': flow_template, 'sha256': sha(flow_template)}
                with np.load(flow_template, allow_pickle=False) as h:
                    data = {k: np.asarray(h[k]) for k in h.files}
                for i in range(0, len(ts)-3, 3):
                    left, right = float(ts[i]), float(ts[i+3])
                    if left < start - 1e-12 or right > end + 1e-12:
                        continue
                    reps = [flow['gaussian_count_representation'](window(r, left, right)[0], height, width)
                            for r in (raw_a, raw_b)]
                    row = {'index': i//3, 'left_s': left, 'right_s': right,
                           'representation': difference(*reps), 'training': [], 'validation': None}
                    # Preserve actual common images and labels from the formal bank.
                    if (height, width) != (260, 346):
                        raise ValueError('Formal Stage9 flow expects native260x346; do not silently resize')
                    sample_ix = i // 3
                    for seed in manifest.get('augmentation_seeds', [0, 1, 2]):
                        transformed = []
                        for rep in reps:
                            item = {k: data[k][sample_ix:sample_ix+1] for k in ('gray_prev','gray_next','flow_gt','flow_valid')}
                            item['events'] = rep[None]
                            transformed.append(flow_runner['apply_joint_transform'](item, np.random.default_rng(seed), torch.device('cpu'))[0])
                        row['training'].append({'seed': seed, 'input': difference(*transformed)})
                    vals = [flow_runner['val_input']({'events': rep[None],
                              'gray_prev': data['gray_prev'][sample_ix:sample_ix+1],
                              'gray_next': data['gray_next'][sample_ix:sample_ix+1]}, 0, torch.device('cpu'))[0]
                            for rep in reps]
                    row['validation'] = difference(*vals)
                    case['flow'].append(row)
            else:
                case['flow_not_run'] = 'A real corresponding formal flow_template_npz is required; no dummy GT/frames generated'
            template = spec.get('reconstruction_template_h5')
            if template:
                case['inputs']['reconstruction_template'] = {'path': template, 'sha256': sha(template)}
                folder = args.output_root / spec['id']
                folder.mkdir(exist_ok=False)
                paths = [clone_reconstruction(template, r, folder/(name+'.h5'))
                         for name, r in zip(('reference','candidate'), (raw_a, raw_b))]
                transforms = [('native', {}) , ('validation', {'CenterCrop': {'size': [160, 160]}}),
                              ('training', {'RandomCrop': {'size': [112,112]}, 'RandomFlip': {'p_hflip': .5, 'p_vflip': .5}})]
                for label, settings in transforms:
                    sets = [Dataset(str(p), num_bins=5, combined_voxel_channels=True,
                                    filter_hot_events=False, transforms=settings.copy()) for p in paths]
                    try:
                        if sets[0].frame_ts != sets[1].frame_ts:
                            raise ValueError('Common frame timestamps changed')
                        for index in range(min(len(s) for s in sets)):
                            right = sets[0].frame_ts[index]
                            left = sets[0].frame_ts[index-1] if index else ts[0]
                            if left < start - 1e-12 or right > end + 1e-12 or not right > left:
                                continue
                            actual_ranges = [list(map(int, d.get_event_indices(index))) for d in sets]
                            expected_ranges = [list(map(int, np.searchsorted(r[:, 0], [left, right], side='right')))
                                               for r in (raw_a, raw_b)]
                            if actual_ranges != expected_ranges:
                                raise RuntimeError(f'Formal E2VID index/window mismatch at {index}: '
                                                   f'{actual_ranges} != {expected_ranges}')
                            seeds = manifest.get('augmentation_seeds', [0,1,2]) if label=='training' else [0]
                            for seed in seeds:
                                items = [d.__getitem__(index, seed=seed) for d in sets]
                                if any(float(item['timestamp']) != float(right) for item in items):
                                    raise RuntimeError('Formal E2VID returned timestamp differs from actual frame endpoint')
                                case['reconstruction'].append({'index': index, 'left_s': float(left),
                                    'right_s': float(right), 'mode': label, 'seed': seed,
                                    'actual_event_index_ranges': actual_ranges,
                                    'inputs': {key: difference(items[0][key],items[1][key])
                                               for key in ('events','frame','flow','flow_valid') if key in items[0]}})
                    finally:
                        for dataset in sets:
                            dataset.h5_file.close()
            else:
                case['reconstruction_not_run'] = 'A real corresponding reconstruction_template_h5 is required'
            case['status'] = 'checked'
        except Exception:
            case['status'] = 'error'
            case['error'] = traceback.format_exc()
        save()
    def all_equal(value):
        if isinstance(value, dict):
            return value.get('bitwise_equal',True) and value.get('finite_a',True) and value.get('finite_b',True) and all(all_equal(v) for v in value.values())
        if isinstance(value, list):
            return all(all_equal(v) for v in value)
        return True
    # Raw timestamps are reported independently; no silent raw tolerance acceptance.
    outputs_equal = all(all_equal(c['flow']) and all_equal(c['pose']) and all_equal(c['reconstruction'])
                        and all(w['membership_index_equal'] and w['window_address_polarity']['bitwise_equal']
                                for w in c['windows_10ms']) for c in report['cases'])
    complete = bool(report['cases']) and all(c['status']=='checked' for c in report['cases'])
    requested = manifest.get('required_tasks', ['flow','pose','reconstruction'])
    covered = {task: sum(len(c[task]) for c in report['cases']) for task in ('flow','pose','reconstruction')}
    complete = complete and all(covered[t] > 0 for t in requested)
    report.update(status='complete' if complete else 'incomplete', actual_windows_by_task=covered,
                  tested_model_inputs_bitwise_equal=bool(outputs_equal),
                  passed=bool(complete and outputs_equal),
                  raw_float64_bitwise_equal=all(c['raw']['bitwise_equal'] for c in report['cases'] if 'raw' in c),
                  general_quality_or_universal_equivalence_claimed=False)
    save()
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
