"""Fresh-process, deterministic conformance of a static optimized plugin.

This is a correctness harness, not a performance benchmark.  It does not
rewrite production source or replace module-global functions.  An instance-
local observer hashes the actual frames/times passed to the unchanged sensor
method.  Observing device data introduces synchronization and host transfers.
Run the frozen and candidate --core paths in separate processes, then compare
events.npy, final_sensor_state.npz, and the physical descriptors in trace.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

# Set before importing Torch or creating a CUDA context.  This applies only to
# this validation process; it does not change other running jobs.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

import h5py
import numpy as np
import torch


PHYSICAL_STATE_KEYS = (
    'base_log_frame', 'lp_log_frame', 'new_frame', 'log_new_frame', 'diff_frame',
    'photoreceptor_noise_arr', 'photoreceptor_noise_vrms', 'noise_rate_array',
    'timestamp_mem', 'pos_thres', 'neg_thres', 'pos_thres_nominal',
    'neg_thres_nominal', 't_previous', 't_start', 'num_events_total',
    'num_events_on', 'num_events_off', 'frame_counter', '_initialized',
    '_warned_depth_noise', 'transfer_input', 'transfer_log_response',
    'calibrated_noise_rate_map', 'calibrated_noise_on_probability',
    'calibrated_hot_pixel_mask', 'correlated_noise_state',
)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_descriptor(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    a = np.ascontiguousarray(np.asarray(value))
    if a.dtype.hasobject:
        raise TypeError('Object arrays cannot be a physical numeric snapshot')
    result = {'kind': 'array', 'shape': list(a.shape), 'dtype': a.dtype.str,
              'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
    if np.issubdtype(a.dtype, np.number):
        result['finite_count'] = int(np.isfinite(a).sum())
        result['nan_count'] = int(np.isnan(a).sum())
        result['positive_infinity_count'] = int(np.isposinf(a).sum())
        result['negative_infinity_count'] = int(np.isneginf(a).sum())
    return result


def descriptor(value):
    if torch.is_tensor(value) or isinstance(value, (np.ndarray, np.generic)):
        return array_descriptor(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return {'kind': 'python_float', 'hex': value.hex()}
    if dataclasses.is_dataclass(value):
        return descriptor(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): descriptor(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [descriptor(v) for v in value]
    raise TypeError(f'Unsupported observable type {type(value)!r}')


def sensor_snapshot(sensor):
    result = {key: descriptor(getattr(sensor, key)) if hasattr(sensor, key)
              else {'kind': 'absent'} for key in PHYSICAL_STATE_KEYS}
    result['rng_state'] = array_descriptor(sensor.rng.get_state())
    result['config'] = descriptor(sensor.cfg)
    return result


def diagnostic_snapshot(interpolator):
    result = {}
    visibility = getattr(interpolator, 'last_visibility', {})
    # Access the underlying mapping without invoking a lazy accessor that
    # replaces device values in the production object's diagnostic cache.
    if hasattr(visibility, 'data') and isinstance(visibility.data, dict):
        visibility = visibility.data
    if not isinstance(visibility, dict):
        raise TypeError('Unexpected diagnostic mapping; do not silently omit it')
    result['visibility_and_coverage'] = descriptor(visibility)
    depth = getattr(interpolator, 'last_intermediate_z', None)
    if hasattr(depth, 'items') and isinstance(depth.items, list):
        depth = depth.items
    result['intermediate_depth'] = descriptor(depth)
    result['uncertainty'] = descriptor(getattr(interpolator, 'last_uncertainty', None))
    result['invalid_fraction'] = descriptor(getattr(interpolator, 'last_invalid_fraction', None))
    result['invalid_mask'] = descriptor(getattr(interpolator, 'last_invalid_mask', None))
    result['endpoint_invalid_mask'] = descriptor(getattr(interpolator, '_endpoint_invalid_mask', None))
    result['sampling'] = descriptor(getattr(interpolator, 'last_sampling', None))
    requests = []
    for request in getattr(interpolator, 'last_render_requests', []):
        # DeviceRenderRequest.mask materializes and mutates _mask; inspect the
        # stored field directly instead.  No renderer is called by this audit.
        fields = vars(request)
        requests.append({
            't': descriptor(request.t), 'alpha': descriptor(request.alpha),
            'reason': request.reason,
            'available_at': descriptor(getattr(request, 'available_at', None)),
            'mask': descriptor(fields.get('_mask', fields.get('mask'))),
            'diagnostics': descriptor(request.diagnostics),
        })
    result['render_requests'] = requests
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--intervals', type=int, required=True)
    parser.add_argument('--campaign', type=Path,
                        default=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1'))
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--start-index', type=int)
    parser.add_argument('--expected-samples', type=int, default=377)
    args = parser.parse_args()
    if args.intervals < 1 or not args.case.replace('_', '').isalnum():
        raise ValueError('A positive interval count and one rollout ID are required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('This registered deterministic test requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
    core = args.core.resolve()
    if not (core / 'genesis_event_plugin/optimized.py').is_file():
        raise FileNotFoundError('Expected a complete static optimized release at --core')
    if any(k == 'genesis_event_plugin' or k.startswith('genesis_event_plugin.') for k in sys.modules):
        raise RuntimeError('Run each --core in a fresh process; package already imported')
    profile_path = (args.profile or args.campaign/'sources/stage9_sensor_profile.json').resolve()
    profile = json.loads(profile_path.read_text())
    for name, expected in profile['core_canonical_LF_sha256'].items():
        actual = hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Frozen reference source mismatch: {name}')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One explicitly assigned visible CUDA device is required')
    sys.path.insert(0, str(core))
    from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
    package = importlib.import_module('genesis_event_plugin')
    if not Path(package.__file__).resolve().is_relative_to(core):
        raise RuntimeError('Imported package is outside the requested static core')
    source_files = {}
    for path in core.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py', '.json', '.cu', '.cpp'}:
            source_files[str(path.relative_to(core))] = file_sha(path)
    args.out.mkdir(parents=True, exist_ok=False)

    kind = 'pose' if args.case.startswith('train') else 'rich'
    start = args.start_index if args.start_index is not None else (0 if kind == 'pose' else 100)
    physical = args.campaign/(kind+'_bank/physical')/(args.case+'.h5')
    with h5py.File(physical, 'r') as file:
        g = file['episodes'][args.case]
        original_ts = np.asarray(g['frame_timestamps_s'][start:start+args.intervals+1], dtype=np.float64)
        K = np.asarray(g['camera_intrinsics_K_px'])
        ids = np.asarray(g['object_segmentation_link_ids'])
        fields = {key: np.asarray(g[key][start:start+args.intervals+1]) for key in
                  ('radiance_linear_rgb', 'depth_m', 'segmentation_link_idx',
                   'object_to_world', 'camera_to_world_cv')}
    if len(original_ts) != args.intervals+1 or not np.all(np.diff(original_ts)>0):
        raise RuntimeError('Physical fixture does not contain requested consecutive endpoints')
    ts = original_ts-original_ts[0]
    height, width = fields['depth_m'].shape[1:]
    cursor = {'i': 0}
    scene = SimpleNamespace(cur_t=0.)
    camera = SimpleNamespace(res=(width, height), intrinsics=K)
    def frame_provider(*_):
        i = cursor['i']
        return {'radiance': fields['radiance_linear_rgb'][i],
                'depth': fields['depth_m'][i], 'seg': fields['segmentation_link_idx'][i]}
    def motion_provider(*_):
        i = cursor['i']
        return {'camera_to_world': fields['camera_to_world_cv'][i],
                'object_to_world': {int(s): fields['object_to_world'][i, k] for k,s in enumerate(ids)}}
    plugin = OptimizedGenesisPhysicsEventPlugin(
        noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),
        sampler_config=AdaptiveSamplingConfig(**profile['sampler_kwargs']),
        frame_provider=frame_provider, motion_state_provider=motion_provider,
        **profile['plugin_kwargs'])
    plugin.attach(scene, camera)
    if not plugin.optimization_active:
        raise RuntimeError(f'Actual optimized path inactive: {plugin.optimization_status}')
    original_generate = plugin._generate_device_sequence
    sensor_inputs = []
    def observe_sensor_input(frames, times):
        if len(frames) != len(times):
            raise RuntimeError('Sensor frames/times length mismatch')
        batch = torch.stack(frames).detach().cpu().numpy()
        if not np.isfinite(batch).all() or np.any(batch < 0):
            raise RuntimeError('Non-finite/negative actual sensor radiance')
        sensor_inputs.append({
            'frames_sha256': hashlib.sha256(np.ascontiguousarray(batch).tobytes()).hexdigest(),
            'frame_shape': list(batch.shape[1:]), 'frame_dtype': batch.dtype.str,
            'each_frame_sha256': [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in batch],
            'timestamps': descriptor(np.asarray(times, dtype=np.float64)),
            'timestamps_s': list(map(float, times)), 'frame_count': len(frames),
        })
        return original_generate(frames, times)
    # Test-only, instance-local, read-only observation.  No source rewrite and
    # no process-global module/function replacement.
    plugin._generate_device_sequence = observe_sensor_input
    plugin.start_episode()
    first = plugin.capture()
    if first.shape != (0,4) or sensor_inputs:
        raise RuntimeError('Unexpected initialization behavior')
    trace = {'schema': 'static_full_plugin_gate_v1', 'status': 'running',
             'core': str(core), 'core_source_sha256': source_files,
             'profile': str(profile_path), 'profile_sha256': file_sha(profile_path),
             'case': args.case, 'intervals': args.intervals, 'start_index': start,
             'physical_path': str(physical), 'physical_origin_s': float(original_ts[0]),
             'physical_inputs': {k: array_descriptor(v) for k,v in fields.items()},
             'K': array_descriptor(K), 'object_ids': array_descriptor(ids),
             'original_timestamps': array_descriptor(original_ts),
             'torch': torch.__version__, 'cuda': torch.version.cuda,
             'gpu_name': torch.cuda.get_device_name(),
             'gpu_uuid_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
             'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
             'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
             'sensor_initial': sensor_snapshot(plugin.dvs_emulator),
             'optimization_status': descriptor(plugin.optimization_status),
             'observed_sensor_method_source': str(Path(inspect.getsourcefile(original_generate)).resolve()),
             'observation_changes_timing': True, 'performance_claimed': False,
             'production_sources_modified': False, 'training_started': False,
             'real_data_used': False, 'rows': []}
    def save():
        tmp = args.out/'trace.json.tmp'
        tmp.write_text(json.dumps(trace, indent=2)+'\n')
        tmp.replace(args.out/'trace.json')
    save()
    chunks = []
    try:
        for i in range(1,len(ts)):
            cursor['i'] = i
            scene.cur_t = float(ts[i])
            before = len(sensor_inputs)
            begin = time.perf_counter()
            events = plugin.capture()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            if len(sensor_inputs) != before+1:
                raise RuntimeError('Each physical interval must invoke the actual optimized sensor exactly once')
            samples = int(plugin.interpolator.last_sampling.samples)
            if samples != args.expected_samples or sensor_inputs[-1]['frame_count'] != samples:
                raise RuntimeError(f'Unexpected actual sample grid {samples}')
            if not plugin.optimization_active or getattr(plugin,'fallbacks',0) or getattr(plugin.dvs_emulator,'_candidate_fallbacks',0):
                raise RuntimeError('Optimized path inactive or fallback occurred')
            if events.ndim != 2 or events.shape[1] != 4 or events.dtype != np.float64 or not np.isfinite(events).all():
                raise RuntimeError('Actual public event output contract failed')
            if len(events) and ((np.diff(events[:,0])<0).any() or events[:,0].min()<ts[i-1]-1e-12 or events[:,0].max()>ts[i]+1e-12):
                raise RuntimeError('Events unsorted or outside their physical interval')
            chunks.append(events)
            row = {'index': i, 'physical_left_s': float(ts[i-1]), 'physical_right_s': float(ts[i]),
                   'capture_instrumented_wall_s': elapsed, 'events': array_descriptor(events),
                   'event_count': len(events), 'sensor_inputs': sensor_inputs[-1],
                   'sensor_state': sensor_snapshot(plugin.dvs_emulator),
                   'geometry_diagnostics': diagnostic_snapshot(plugin.interpolator),
                   'plugin_total_steps': int(plugin.total_steps),
                   'plugin_total_events': int(plugin.total_events),
                   'fast_path_active': True, 'fallbacks': 0}
            trace['rows'].append(row)
            save()
            print('FULL_PLUGIN_INTERVAL', args.case, i, args.intervals, len(events), flush=True)
        events = np.concatenate(chunks, axis=0)
        if len(events) and (np.diff(events[:,0])<0).any():
            raise RuntimeError('Full actual public event stream is unsorted; no corrective resort applied')
        np.save(args.out/'events.npy', events, allow_pickle=False)
        state = {}
        for key in PHYSICAL_STATE_KEYS:
            value = getattr(plugin.dvs_emulator,key,None)
            if torch.is_tensor(value):
                state[key] = value.detach().cpu().numpy()
            elif isinstance(value,np.ndarray) or isinstance(value,(int,float,bool)):
                state[key] = np.asarray(value)
        state['rng_state'] = plugin.dvs_emulator.rng.get_state().cpu().numpy()
        np.savez(args.out/'final_sensor_state.npz', **state)
        for relative, expected in source_files.items():
            if file_sha(core/relative) != expected:
                raise RuntimeError('Static core source changed during the test')
        trace.update(status='complete', actual_public_events=array_descriptor(events),
                     actual_event_count=len(events), actual_sensor_invocations=len(sensor_inputs),
                     final_sensor_state=sensor_snapshot(plugin.dvs_emulator),
                     passed_local_contract=True,
                     pairwise_equivalence_not_decided_by_this_single_run=True)
        save()
        print('FULL_PLUGIN_GATE_COMPLETE', args.case, len(events), flush=True)
    except BaseException as error:
        trace.update(status='failed', passed_local_contract=False,
                     error_type=type(error).__name__, error=str(error))
        save()
        raise
    finally:
        plugin.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
