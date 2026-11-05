"""Geometry-floor pilot against withheld real1ms Stage9 renders, not event GT.

Run only on the root agent's leased GPU2. Does not call a sensor, train, contact
a server, supply real subframes to the candidate, or modify a frozen module.
The1ms reference cannot establish20us event accuracy or high-N convergence.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

EXPECTED_GEOMETRY = 'e74793c4805eabfa2a19d9db02b5d829c73a12a61d3fed6b4cbeb8e5995a060b'


def qualify_independent_new_clip(receipt, sha):
    """Qualify this new actual-rendered clip without passing historical replay.

    The historical receipt remains unchanged and false. Historical observations
    are read only here for scene/K/object identity; none enter the candidate.
    """
    import h5py
    import numpy as np
    pins = {
        'rich_physical_worker.py':'0eafc57043eb2a4b4e963fbb6ccf4e58d4286df12d3f1d20856637f839097d17',
        'stage9_replicad_scene_loader.py':'8aa173ff3bc0d91c7e432fb5edbd1414338f805f5d2f71865c6b94068adef5dd',
        'rich_bank_protocol.json':'e3fd835346e422deedeeb77d2d110fc27bf0d6e568708ec7c690f7473bbb13ff',
    }
    assert receipt['actual_genesis_render'] is True
    assert receipt['endpoint_interpolation_used_for_reference'] is False
    assert receipt['coarse_endpoint_replay_passed'] is False, 'Use the original qualified-replay branch if historical replay passed'
    assert receipt['repeated_actual_render_passed'] is True
    repeats = receipt['repeated_actual_render_checks']
    assert len(repeats) >= 3
    assert all(row['rng_unchanged'] is True and all(row['raw_bytes_equal'].values()) for row in repeats)
    assert receipt['specification']['source_sha256'] == pins
    old_path, new_path = Path(receipt['source_10ms_bank']), Path(receipt['output'])
    assert sha(old_path) == receipt['source_10ms_bank_sha256']
    assert sha(new_path) == receipt['output_sha256']
    coarse = receipt['coarse_endpoint_checks']
    assert coarse and all(row['fields']['object_to_world']['raw_bytes_equal'] is True for row in coarse)
    measurements = []
    def same(x,y):
        return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    with h5py.File(new_path,'r') as new, h5py.File(old_path,'r') as old:
        ng,og = new['episodes'][receipt['case']],old['episodes'][receipt['case']]
        assert str(ng.attrs['scene_id']) == str(og.attrs['scene_id'])
        assert same(np.asarray(ng['camera_intrinsics_K_px']),np.asarray(og['camera_intrinsics_K_px']))
        assert same(np.asarray(ng['object_segmentation_link_ids']),np.asarray(og['object_segmentation_link_ids']))
        assert json.loads(ng.attrs['motion_parameters']) == json.loads(og.attrs['motion_parameters'])
        for name,pin in [('source_worker_sha256',pins['rich_physical_worker.py']),
                         ('source_loader_sha256',pins['stage9_replicad_scene_loader.py']),
                         ('source_protocol_sha256',pins['rich_bank_protocol.json'])]:
            assert str(ng.attrs[name]) == pin
        for recorded in coarse:
            di,oi = recorded['dense_index'],recorded['original_index']
            fields = {}
            for name in ('radiance_linear_rgb','depth_m','segmentation_link_idx','camera_to_world_cv','object_to_world'):
                x,y = np.asarray(ng[name][di]),np.asarray(og[name][oi])
                fields[name] = dict(raw_bytes_equal=same(x,y),
                    max_abs=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))),
                    changed_values=int(np.count_nonzero(x != y)))
                assert fields[name] == recorded['fields'][name], 'Historical evidence no longer reproduces: '+name
            assert fields['object_to_world']['raw_bytes_equal'] is True
            measurements.append(dict(dense_index=di,original_index=oi,fields=fields,
                timestamp_delta_s=recorded['timestamp_delta_s']))
        scene_id = str(ng.attrs['scene_id'])
    return dict(qualification_mode='independent_new_dense_clip',
        historical_endpoint_bitwise_replay_passed=False,
        historical_replay_failed=True,
        original_receipt_false_preserved=True,
        historical_coarse_differences_reverified=measurements,
        actual_Genesis_middle_frames=True, repeated_actual_render_and_RNG_exact=True,
        same_Stage9_scene_id=scene_id, source_builder_and_trajectory_formula_pins_verified=True,
        intrinsics_raw_bytes_equal=True, object_identities_and_coarse_SE3_raw_bytes_equal=True,
        analytic_motion_parameters_equal=True,
        candidate_endpoints='Only current new dense clip indices0,10,20,...; historical images/depth/camera never supplied',
        reference_middle_frames='Current new dense clip actual1ms withheld renders',
        scope='A fresh Stage9-derived synthetic development clip, not a bitwise historical-bank replay',
        geometry_floor_only=True, reference_temporal_convergence_established=False,
        event_timestamp_20us_accuracy_established=False, sampling_recommendation_permitted=False)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--independent-new-clip', action='store_true', required=True)
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--reference-receipt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--Ns', default='63,126,189,377,754')
    p.add_argument('--interval-limit', type=int, default=10)
    p.add_argument('--expected-gpu', default='GPU-a02affe0-7598-579e-ca39-f551f35ad6e9')
    p.add_argument('--active-log-range', type=float, default=.02)
    p.add_argument('--no-query-array-export', action='store_true')
    a = p.parse_args()
    core = a.core.resolve(strict=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == a.expected_gpu, 'Require the root-leased physical GPU2 UUID'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert not any(k == 'genesis_event_plugin' or k.startswith('genesis_event_plugin.') for k in sys.modules)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(core))
    import cv2
    import h5py
    import numpy as np
    import torch
    from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
    from genesis_event_plugin.gpu_interpolator import ResidentPhysicsInterpolator
    from genesis_event_plugin.radiometry import prepare_linear_radiance

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    module = importlib.import_module('genesis_event_plugin.gpu_geometry')
    assert Path(module.__file__).resolve().is_relative_to(core)

    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024**2), b''):
                digest.update(block)
        return digest.hexdigest()

    inventory = {str(path.relative_to(core)):sha(path) for path in sorted(core.rglob('*.py'))
                 if '__pycache__' not in path.parts}
    assert inventory['genesis_event_plugin/gpu_geometry.py'] == EXPECTED_GEOMETRY
    profile_path = core/'stage9_sensor_profile.json'
    profile = json.loads(profile_path.read_text())
    receipt = json.loads(a.reference_receipt.read_text())
    assert receipt['status'] == 'complete' and receipt['actual_genesis_render'] is True
    assert receipt['endpoint_interpolation_used_for_reference'] is False
    assert a.independent_new_clip is True
    independent_qualification = qualify_independent_new_clip(receipt, sha)
    assert receipt['render_dt_s'] == .001 and receipt['physics_dt_s'] == .001
    reference_path = Path(receipt['output']).resolve(strict=True)
    assert sha(reference_path) == receipt['output_sha256']
    Ns = [int(x) for x in a.Ns.split(',')]
    assert len(Ns) == len(set(Ns)) and all(n >= 2 for n in Ns) and 377 in Ns
    assert 1 <= a.interval_limit <= 20 and a.active_log_range > 0
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='independent_actual_genesis_geometry_floor_pilot_v1', status='running',
        independent_new_clip=True, independent_clip_qualification=independent_qualification,
        historical_endpoint_bitwise_replay_passed=False, historical_replay_failed=True,
        original_reference_receipt_unmodified=True,
        base_scoring_script_sha256='d46add4eed499ffa632d6c9d7632b59c6cbdbd3e904ef0f8c4a7fe7a0bce29c3',
        reference=str(reference_path), reference_sha256=receipt['output_sha256'],
        reference_receipt=str(a.reference_receipt), reference_receipt_sha256=sha(a.reference_receipt),
        core=str(core), source_sha256=inventory, profile_sha256=sha(profile_path),
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], gpu_name=torch.cuda.get_device_name(),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        deterministic_algorithms=True, cublas_workspace_config=':4096:8',
        CPU_threads=2, TF32=False, sensor_called=False, sensor_noise_off_by_not_calling_sensor=True,
        production_sensor_profile_gate_exercised=False,
        candidate_uses_only_observed_10ms_endpoints=True,
        reference_frames_used_as_candidate_anchors=False,
        forced_Ns=Ns, rows=[], aggregates=[],
        metric_definition='log(max(linear_luminance,1e-6)) error normalized by nominal contrast threshold',
        active_mask_definition='reference10ms temporal log range >= registered active_log_range',
        active_log_range=a.active_log_range,
        query_reconstruction='piecewise linear radiance between actual candidate sample times; observed left endpoint prepended',
        query_support='conjunction of the two bracketing candidate support masks; exact-grid query uses that grid support',
        stratum_masks='reference only; all-finite target pixels, active, segmentation/depth boundary, temporal segmentation switches, thin medial bands, quiet background',
        reference_temporal_convergence_established=False,
        event_timestamp_20us_accuracy_established=False,
        sampling_recommendation_permitted=False,
        limitations=['Historical Stage9 endpoint replay failed and remains failed; this is an explicitly new independent synthetic clip.',
            'The Stage9 scene, trajectory formula and motion parameters are reused, but historical camera/image bytes are not claimed identical.',
            '1ms reference checks spatial geometry/occlusion error floor only.',
            'Does not prove microsecond event timestamps, stochastic-noise equivalence or a universally optimal N.',
            'Changing N cannot in general fix endpoint-only constant-twist/appearance/unknown-region error.',
            'Single timing observation per interval, deterministic diagnostic mode; not headline deployment efficiency.'],
        geometry_policy={k:profile['plugin_kwargs'].get(k) for k in ('warp_composite','disocclusion_policy','strict_fidelity','min_coverage_fraction')})

    def save():
        (out/'geometry_quality.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def summarize(values):
        x = np.asarray(values, dtype=np.float64)
        if x.size == 0:
            return dict(pixels=0, RMSE_over_C=None, p50_abs_over_C=None, p95_abs_over_C=None,
                        p99_abs_over_C=None, max_abs_over_C=None)
        z = np.abs(x)
        return dict(pixels=int(x.size), RMSE_over_C=float(np.sqrt(np.mean(x*x))),
            p50_abs_over_C=float(np.percentile(z,50)), p95_abs_over_C=float(np.percentile(z,95)),
            p99_abs_over_C=float(np.percentile(z,99)), max_abs_over_C=float(z.max()))

    def boundary(seg, depth):
        edge = np.zeros(seg.shape, dtype=np.uint8)
        for axis in (0,1):
            left, right = [slice(None),slice(None)], [slice(None),slice(None)]
            left[axis], right[axis] = slice(None,-1), slice(1,None)
            x,y = tuple(left),tuple(right)
            change = seg[x] != seg[y]
            good = (depth[x]>0)&(depth[y]>0)
            change |= good & (np.abs(depth[x]-depth[y])/np.maximum(np.minimum(depth[x],depth[y]),1e-6) > .05)
            edge[x] |= change
            edge[y] |= change
        return cv2.dilate(edge, np.ones((5,5), np.uint8)).astype(bool)

    def thin_medial_band(seg, depth):
        skeleton = np.zeros(seg.shape, dtype=np.uint8)
        for sid in np.unique(seg):
            mask = ((seg == sid)&(depth>1e-6)).astype(np.uint8)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            local_max = cv2.dilate(distance, np.ones((3,3),np.uint8))
            skeleton |= ((distance > 0)&(distance <= 2)&(distance >= local_max-1e-6)).astype(np.uint8)
        return cv2.dilate(skeleton, np.ones((3,3),np.uint8)).astype(bool)

    try:
        with h5py.File(reference_path, 'r') as h:
            g = h['episodes'][receipt['case']]
            times = np.asarray(g['frame_timestamps_s'], dtype=np.float64)
            assert np.allclose(np.diff(times),.001,atol=1e-9,rtol=0)
            assert (len(times)-1)%10 == 0
            K = np.asarray(g['camera_intrinsics_K_px'])
            ids = np.asarray(g['object_segmentation_link_ids'])
            interval_count = min((len(times)-1)//10, a.interval_limit)
            C = float(profile['noise_config_kwargs']['pos_thres'])
            assert C == profile['noise_config_kwargs']['neg_thres'] and C > 0
            tau = 1/(2*math.pi*profile['noise_config_kwargs']['cutoff_hz'])
            report.update(case=receipt['case'], intervals=interval_count, common_queries_per_interval=9,
                          radiometric_floor=1e-6, nominal_threshold_C=C, tau_s=tau)
            save()
            for N in Ns:
                pooled = {key:[] for key in ('all','active','boundary','occlusion','thin','background')}
                exported, exported_support, exported_times = [], [], []
                begin_N = time.perf_counter()
                for interval in range(interval_count):
                    l,r = interval*10,(interval+1)*10
                    t0,t1 = float(times[l]),float(times[r])
                    dt = t1-t0
                    endpoints = []
                    # Only these two observations and their poses enter the candidate.
                    for i in (l,r):
                        endpoints.append(dict(rad=np.asarray(g['radiance_linear_rgb'][i]),
                            depth=np.asarray(g['depth_m'][i]), seg=np.asarray(g['segmentation_link_idx'][i]),
                            cam=np.asarray(g['camera_to_world_cv'][i]),
                            objects={int(sid):value for sid,value in zip(ids,np.asarray(g['object_to_world'][i]))}))
                    left,right = endpoints
                    # This is an explicit sampler-control experiment, not a profile
                    # edit. Raise max_samples to N; never silently hit300/512 caps.
                    # The tiny factor avoids ceil(N+roundoff) at an integer boundary.
                    eps_override = (dt/tau)/N*(1+1e-10)
                    cfg = replace(AdaptiveSamplingConfig(**profile['sampler_kwargs']),
                                  max_samples=N, sensor_time_constant_s=tau,
                                  sensor_eps_target=eps_override, strict=False)
                    interp = ResidentPhysicsInterpolator(K, sampler_config=cfg,
                        disocclusion_policy='hold', composite='coverage_time_log_linear',
                        device='cuda:0', warp_backend='torch', strict_fidelity=False,
                        min_coverage_fraction=profile['plugin_kwargs']['min_coverage_fraction'])
                    assert interp.__class__.__module__ == 'genesis_event_plugin.gpu_interpolator'
                    free,total = torch.cuda.mem_get_info()
                    estimate = 320*N*left['depth'].size+512*1024**2
                    if estimate > .8*free:
                        raise RuntimeError(f'N={N} conservative full-stack memory estimate {estimate} exceeds80% of currently free {free}; no precision/shape/cap fallback permitted')
                    delta = {sid:right['objects'][sid]@np.linalg.inv(left['objects'][sid]) for sid in ids.tolist()}
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begun = time.perf_counter()
                    with torch.no_grad():
                        frames, grid_times = interp.interpolate_device(
                            left['rad'],left['depth'],left['seg'],right['rad'],right['depth'],right['seg'],
                            left['cam'],right['cam'],delta,t0,t1,input_space='linear',
                            object_world0=left['objects'],object_world1=right['objects'])
                    torch.cuda.synchronize()
                    wall = time.perf_counter()-begun
                    actual = asdict(interp.last_sampling)
                    assert len(frames) == len(grid_times) == actual['samples'] == N
                    assert not actual['capped'] and actual['uncapped_samples'] == N, 'A different physical bound or hidden cap dominated the requested N'
                    assert np.all(np.diff(np.r_[t0,grid_times]) > 0)
                    assert abs(grid_times[-1]-t1) < 1e-12
                    assert interp.last_uncertainty['extra_render_count'] == 0
                    assert interp.last_uncertainty['photometric_anchor_count'] == 2
                    resident = interp._resident_frames
                    assert resident.is_cuda and resident.dtype == torch.float32 and resident.shape[0] == N
                    visible = interp.last_visibility['intermediate_visibility']
                    # DeferredDiagnostics may expose this diagnostic as NumPy;
                    # copying its Boolean value back does not alter the generator.
                    visible = torch.as_tensor(visible, device='cuda', dtype=torch.bool)
                    initial = torch.as_tensor(prepare_linear_radiance(left['rad'], input_space='linear'),device='cuda',dtype=resident.dtype)
                    series = torch.cat((initial[None],resident),dim=0)
                    supports = torch.cat((torch.ones_like(visible[:1]),visible),dim=0)
                    qt = times[l+1:r]
                    grid = np.r_[t0,np.asarray(grid_times,dtype=np.float64)]
                    predicted, support_rows = [], []
                    for query in qt:
                        hi = int(np.searchsorted(grid,query,side='left'))
                        hi = min(max(1,hi),N)
                        lo = hi-1
                        weight = float((query-grid[lo])/(grid[hi]-grid[lo]))
                        assert -1e-10 <= weight <= 1+1e-10
                        weight = min(1.,max(0.,weight))
                        if abs(query-grid[hi]) < 1e-12:
                            value, supported = series[hi],supports[hi]
                        elif abs(query-grid[lo]) < 1e-12:
                            value, supported = series[lo],supports[lo]
                        else:
                            value = series[lo]*(1-weight)+series[hi]*weight
                            supported = supports[lo]&supports[hi]
                        predicted.append(value.detach().cpu().numpy())
                        support_rows.append(supported.detach().cpu().numpy())
                    predicted,supported = np.stack(predicted),np.stack(support_rows)
                    assert np.isfinite(predicted).all() and (predicted >= 0).all()
                    # Withheld observations are read only after candidate generation.
                    truth_rgb = np.asarray(g['radiance_linear_rgb'][l:r+1])
                    truth = np.stack([prepare_linear_radiance(x, input_space='linear') for x in truth_rgb])
                    logtruth = np.log(np.maximum(truth.astype(np.float64),1e-6))
                    difference = (np.log(np.maximum(predicted.astype(np.float64),1e-6))-logtruth[1:-1])/C
                    active = np.ptp(logtruth,axis=0) >= a.active_log_range
                    segs = np.asarray(g['segmentation_link_idx'][l:r+1])
                    depths = np.asarray(g['depth_m'][l:r+1])
                    strata = {key:[] for key in pooled}
                    for j in range(1,10):
                        edges = boundary(segs[j],depths[j])
                        masks = dict(all=np.ones(active.shape,dtype=bool),active=active,
                            boundary=edges,occlusion=(segs[j] != segs[j-1])|(segs[j] != segs[j+1]),
                            thin=thin_medial_band(segs[j],depths[j]), background=~active&~edges)
                        for name,mask in masks.items():
                            strata[name].append(difference[j-1][mask])
                    scored = {}
                    for name,values in strata.items():
                        values = np.concatenate(values)
                        scored[name] = summarize(values)
                        pooled[name].append(values)
                    row = dict(N=N,interval=interval,t0_s=t0,t1_s=t1,
                        actual_sampling=actual, sampler_config=asdict(cfg),
                        sampler_eps_override=eps_override, source_profile_eps_default_not_overwritten=True,
                        resident_geometry_active=True, inherited_numpy_endpoint_audit_retained=True,
                        sensor_fastpath_not_called=True, extra_render_count=0,
                        endpoint_anchors=2, hole_hold_fallback_pixels=int(interp._resident_last_hole_pixels),
                        common_query_unknown_fraction=float((~supported).mean()),
                        unsupported_query_error=summarize(difference[~supported]),
                        all_held_and_unknown_predictions_included_in_error=True,
                        geometry_wall_s=wall, peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                        peak_measurement_includes_common_query_reconstruction=True,
                        conservative_memory_precheck_bytes=estimate,
                        common_query_timestamps_s=qt.tolist(),strata=scored)
                    report['rows'].append(row)
                    if not a.no_query_array_export:
                        exported.append(predicted)
                        exported_support.append(supported)
                        exported_times.extend(qt.tolist())
                    save()
                    print('GEOMETRY_INTERVAL',json.dumps(dict(N=N,interval=interval,wall_s=wall,
                        active_RMSE_over_C=scored['active']['RMSE_over_C'],
                        boundary_RMSE_over_C=scored['boundary']['RMSE_over_C'],
                        unknown_fraction=row['common_query_unknown_fraction'])),flush=True)
                    del interp,frames,resident,visible,initial,series,supports,predicted,supported,endpoints
                    gc.collect()
                    torch.cuda.empty_cache()
                aggregate = dict(N=N,intervals=interval_count,
                    geometry_wall_s=sum(row['geometry_wall_s'] for row in report['rows'] if row['N'] == N),
                    diagnostic_wall_s=time.perf_counter()-begin_N,
                    strata={name:summarize(np.concatenate(values)) for name,values in pooled.items()})
                if exported:
                    folder = out/('N'+str(N))
                    folder.mkdir()
                    np.save(folder/'query_predictions.npy',np.concatenate(exported),allow_pickle=False)
                    np.save(folder/'query_support.npy',np.concatenate(exported_support),allow_pickle=False)
                    np.save(folder/'query_times.npy',np.asarray(exported_times,dtype=np.float64),allow_pickle=False)
                    aggregate['arrays'] = {path.name:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size) for path in folder.glob('*.npy')}
                report['aggregates'].append(aggregate)
                save()
        assert {str(path.relative_to(core)):sha(path) for path in sorted(core.rglob('*.py'))
                if '__pycache__' not in path.parts} == inventory
        assert sha(reference_path) == receipt['output_sha256']
        report.update(status='complete',core_unchanged=True,reference_unchanged=True)
        save()
        with (out/'geometry_quality.csv').open('w',newline='') as stream:
            columns = ['N','stratum','pixels','RMSE_over_C','p50_abs_over_C','p95_abs_over_C','p99_abs_over_C','max_abs_over_C','geometry_wall_s']
            writer = csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            for row in report['aggregates']:
                for name,metrics in row['strata'].items():
                    writer.writerow(dict(N=row['N'],stratum=name,geometry_wall_s=row['geometry_wall_s'],**metrics))
        print('GEOMETRY_QUALITY_COMPLETE',json.dumps(dict(case=report['case'],Ns=Ns,
            intervals=interval_count,reference_convergence_established=False,
            sampling_recommendation_permitted=False,output=str(out))),flush=True)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
