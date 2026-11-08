"""Optional detailed profiling around an unchanged official task worker.

Use in a separate profile/parity pass. Per-stage barriers and extra telemetry
perturb timing, so this wrapper is never the primary throughput measurement.
No source file, model parameter, input pixel or event row is changed.
"""
from __future__ import annotations
import argparse
import functools
import importlib
import json
import os
from pathlib import Path
import pickle
import runpy
import sys
import time
sys.dont_write_bytecode = True

from cost_ledger import WallLedger, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-script', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--method', choices=['ours', 'v2e'], required=True)
    p.add_argument('--sidecar', type=Path, required=True)
    p.add_argument('--synchronize-stages', action='store_true')
    args = p.parse_args()
    expected_name = 'event_worker.py' if args.method == 'ours' else 'v2e_event_worker.py'
    if args.worker_script.name != expected_name or args.worker_script.parent.resolve() != args.source_dir.resolve():
        p.error('Use the actual source-directory worker, not a copied algorithm implementation')
    if args.sidecar.exists():
        p.error('Detailed-profile sidecar already exists')
    sys.path.insert(0, str(args.source_dir.resolve()))
    import torch
    def sync():
        if args.synchronize_stages and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    current = [WallLedger(sync=sync if args.synchronize_stages else None)]
    records, hooks = [], []
    startup_ns = time.perf_counter_ns()

    def wrap(cls, name, stage, generator=False):
        if not hasattr(cls, name):
            hooks.append(dict(symbol=cls.__name__+'.'+name, installed=False))
            return
        original = getattr(cls, name)
        if generator:
            @functools.wraps(original)
            def measured(*a, **kw):
                iterator = iter(original(*a, **kw))
                while True:
                    try:
                        with current[0].span(stage):
                            item = next(iterator)
                    except StopIteration:
                        return
                    yield item
        else:
            @functools.wraps(original)
            def measured(*a, **kw):
                with current[0].span(stage):
                    return original(*a, **kw)
        setattr(cls, name, measured)
        module = sys.modules[cls.__module__]
        path = Path(module.__file__).resolve()
        hooks.append(dict(symbol=cls.__module__+'.'+cls.__name__+'.'+name, stage=stage,
                          installed=True, source_path=str(path), source_sha256=sha256_file(path)))

    if args.method == 'ours':
        physics = importlib.import_module('genesis_event_plugin.physics_plugin')
        interpolation = importlib.import_module('genesis_event_plugin.physics_interpolator')
        dvs = importlib.import_module('genesis_event_plugin.dvs_emulator')
        wrap(physics.GenesisPhysicsEventPlugin, 'capture', 'ours_capture_inclusive')
        wrap(interpolation.PhysicsInterpolator, 'interpolate', 'ours_sparse_reconstruction')
        wrap(dvs.DvsEmulator, 'generate_events', 'event_response')
        wrap(dvs.DvsEmulator, 'initialize', 'event_response_initialization')
        if hasattr(dvs, 'DvsBatchEmulator'):
            wrap(dvs.DvsBatchEmulator, 'generate_events_batch', 'event_response')
            wrap(dvs.DvsBatchEmulator, 'initialize_batch', 'event_response_initialization')
    else:
        v2e_source = Path(os.environ['CATCHIT_V2E_SOURCE'])
        sys.path.insert(0, str(v2e_source))
        streaming = importlib.import_module('streaming_v2e')
        emulator = importlib.import_module('v2ecore.emulator')
        wrap(streaming.StreamingV2E, '__init__', 'v2e_startup_inclusive')
        wrap(streaming.StreamingV2E, 'push', 'v2e_capture_inclusive')
        # Timing next(iterator) excludes the DVS work done by its caller between
        # successive yields. It includes native SSM, frame conversion and resize.
        wrap(streaming.StreamingV2E, 'pair_frames', 'v2e_ssm_and_frame_conversion', generator=True)
        wrap(emulator.EventEmulator, 'generate_events', 'event_response')

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    load_original, dump_original = pickle.load, pickle.dump
    packet = [None]
    sequence = [0]

    def measured_load(file, *a, **kw):
        begin = time.perf_counter_ns()
        value = load_original(file, *a, **kw)
        if file is stdin and isinstance(value, dict) and value.get('kind') in ('reset', 'frame'):
            # Input waiting/deserialization is kept separate from compute.
            sequence[0] += 1
            packet[0] = dict(sequence=sequence[0], kind=value['kind'],
                input_load_blocking_wall_s=(time.perf_counter_ns()-begin)/1e9,
                packet_end_s=float(value['packet']['time_s']))
            current[0] = WallLedger(sync=sync if args.synchronize_stages else None)
            current[0].start_hot()
        return value

    def measured_dump(value, file, *a, **kw):
        if file is stdout and isinstance(value, dict):
            if value.get('ready') is True:
                value = dict(value)
                value['_cost_profile_ready'] = dict(method=args.method,
                    startup_to_ready_wall_s=(time.perf_counter_ns()-startup_ns)/1e9,
                    instrumented=True, primary_throughput=False,
                    per_stage_cuda_barriers=args.synchronize_stages)
            elif 'events' in value and packet[0] is not None:
                current[0].stop_hot()
                profile = dict(packet[0], **current[0].summary())
                profile['original_worker_wall_s'] = value.get('wall_s')
                profile['gpu_cumulative_time_s'] = None
                if torch.cuda.is_initialized():
                    profile['torch_peak_allocated_bytes_lifetime'] = int(torch.cuda.max_memory_allocated())
                    profile['torch_peak_reserved_bytes_lifetime'] = int(torch.cuda.max_memory_reserved())
                # All telemetry consists of primitives, so the original recursive
                # wire decoder accepts this extra key. The events are untouched.
                value = dict(value)
                value['_cost_profile'] = profile
                records.append(profile)
                packet[0] = None
        return dump_original(value, file, *a, **kw)

    pickle.load, pickle.dump = measured_load, measured_dump
    status = 'ERROR'
    try:
        sys.argv = [str(args.worker_script)]
        runpy.run_path(str(args.worker_script), run_name='__main__')
        status = 'COMPLETE'
    finally:
        pickle.load, pickle.dump = load_original, dump_original
        args.sidecar.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(status=status, method=args.method, instrumented_profile_only=True,
            worker_path=str(args.worker_script.resolve()), worker_sha256=sha256_file(args.worker_script),
            wrapper_sha256=sha256_file(__file__), hooks=hooks, packets=records,
            event_mutation=False, gpu_cumulative_time_s=None,
            caveat='Per-stage inclusive time is not additive; use exclusive spans. Primary cost must use the unwrapped worker. GPU cumulative time is unknown.')
        args.sidecar.write_text(json.dumps(payload, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
