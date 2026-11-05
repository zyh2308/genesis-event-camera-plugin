"""Read-only CPU breakdown of geometry errors by true depth and support.

Consumes completed query arrays and actual rendered reference frames. It never
rerenders, imports Torch, calls a GPU, changes a simulator, or edits raw results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-receipt', '--reference', dest='reference_receipt', type=Path, required=True)
    p.add_argument('--geometry-results', type=Path, required=True,
                   help='Completed geometry output directory or its geometry_quality.json')
    p.add_argument('--out', type=Path, required=True, help='New auxiliary output directory')
    p.add_argument('--maps-N', type=int, help='Optionally save three raw query-map NPZs for this N')
    p.add_argument('--map-indices', help='Optional comma-separated query indices; default first/middle/last')
    a = p.parse_args()
    # Applied before NumPy/OpenCV imports; affects only this CPU analysis process.
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    sys.dont_write_bytecode = True
    import cv2
    import h5py
    import numpy as np
    cv2.setNumThreads(2)
    begun = time.perf_counter()
    result_path = a.geometry_results.resolve(strict=True)
    if result_path.is_dir():
        result_path = result_path/'geometry_quality.json'
    result_dir = result_path.parent
    receipt_path = a.reference_receipt.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    geometry = json.loads(result_path.read_text())
    assert receipt['status'] == geometry['status'] == 'complete'
    assert receipt['actual_genesis_render'] is True
    assert receipt['endpoint_interpolation_used_for_reference'] is False
    assert receipt['repeated_actual_render_passed'] is True
    assert geometry['case'] == receipt['case']
    assert geometry['reference_sha256'] == receipt['output_sha256']
    reference_path = Path(receipt['output']).resolve(strict=True)
    out = a.out.resolve()
    assert not out.exists(), 'Use a fresh auxiliary output directory'

    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024**2), b''):
                digest.update(chunk)
        return digest.hexdigest()

    inputs = {}
    def register(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = sha(path)
        assert expected is None or actual == expected, 'Input checksum differs: '+str(path)
        inputs[str(path)] = dict(sha256=actual, bytes=path.stat().st_size)
        return path

    register(receipt_path)
    register(result_path)
    register(reference_path,receipt['output_sha256'])
    # Load only the pinned pure-NumPy radiometry file, never the package initializer.
    core = Path(geometry['core']).resolve(strict=True)
    radiometry_path = register(core/'genesis_event_plugin/radiometry.py',
                               geometry['source_sha256']['genesis_event_plugin/radiometry.py'])
    module_spec = importlib.util.spec_from_file_location('geometry_aux_pinned_radiometry',radiometry_path)
    radiometry = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = radiometry
    module_spec.loader.exec_module(radiometry)
    assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules), 'CPU auxiliary must not import Torch'
    C = float(geometry['nominal_threshold_C'])
    floor = float(geometry['radiometric_floor'])
    active_threshold = float(geometry['active_log_range'])
    assert C > 0 and floor > 0 and active_threshold > 0
    intervals = int(geometry['intervals'])
    assert intervals > 0 and geometry['common_queries_per_interval'] == 9
    aggregate_by_N = {int(row['N']):row for row in geometry['aggregates']}
    Ns = list(geometry['forced_Ns'])
    assert set(Ns) == set(aggregate_by_N)
    if a.maps_N is not None:
        assert a.maps_N in Ns
    out.mkdir(parents=True)
    report = dict(schema='geometry_reference_depth_support_auxiliary_v1', status='running',
        case=geometry['case'], reference=str(reference_path), geometry_results=str(result_path),
        source_receipt=str(receipt_path), original_historical_replay_passed=receipt['coarse_endpoint_replay_passed'],
        independent_new_clip=geometry.get('independent_new_clip',False),
        historical_failed_receipt_preserved=True,
        CPU_only=True, Torch_or_GPU_called=False, CPU_threads=2,
        numpy_version=np.__version__, opencv_version=cv2.__version__,
        metric_definition='log(max(raw linear luminance,floor)) error / C; float64 metric arithmetic, native FP32 luminance',
        radiometry=dict(input_space='linear', HDR_clipped_to_one=False, per_frame_normalization=False,
                        radiometric_floor=floor, C=C, weights=radiometry.DEFAULT_LUMINANCE_WEIGHTS.tolist()),
        valid_depth_definition='isfinite(reference depth_m) AND reference depth_m > 0; exact zero excluded',
        support_definition='Stored actual common-query candidate support; no mask recomputation or threshold relaxation',
        active_definition='Reference11-frame temporal log-luminance range >= registered active_log_range',
        boundary_definition='Original pilot segmentation/depth discontinuity test and5x5 dilation, reference only',
        depth_invalid_is_sky_claimed=False,
        limitations=['Invalid depth is a background/unknown-depth proxy, not a semantic sky annotation.',
                     'Error on unsupported pixels is retained; masking does not improve or replace the original overall score.',
                     'This is a1ms geometry-floor analysis, not microsecond event accuracy or optimal-N qualification.'],
        per_N=[], maps=[], inputs=inputs)

    def save():
        (out/'geometry_support_auxiliary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

    def summarize(error,mask):
        selected = np.asarray(error[mask],dtype=np.float64)
        count = int(selected.size)
        if not count:
            return dict(pixels=0,RMSE_over_C=None,mean_signed_over_C=None,
                        p50_abs_over_C=None,p95_abs_over_C=None,p99_abs_over_C=None,
                        max_abs_over_C=None,sum_squared_error_over_C2=0.)
        assert np.isfinite(selected).all()
        magnitude = np.abs(selected)
        q50,q95,q99 = np.percentile(magnitude,[50,95,99])
        squares = float(np.sum(selected*selected,dtype=np.float64))
        return dict(pixels=count,RMSE_over_C=float(math.sqrt(squares/count)),
                    mean_signed_over_C=float(np.mean(selected)),
                    p50_abs_over_C=float(q50),p95_abs_over_C=float(q95),p99_abs_over_C=float(q99),
                    max_abs_over_C=float(magnitude.max()),sum_squared_error_over_C2=squares)

    def boundary(seg,depth):
        edge = np.zeros(seg.shape,dtype=np.uint8)
        for axis in (0,1):
            left,right = [slice(None),slice(None)],[slice(None),slice(None)]
            left[axis],right[axis] = slice(None,-1),slice(1,None)
            x,y = tuple(left),tuple(right)
            change = seg[x] != seg[y]
            good = (depth[x]>0)&(depth[y]>0)
            change |= good & (np.abs(depth[x]-depth[y])/np.maximum(np.minimum(depth[x],depth[y]),1e-6)>.05)
            edge[x] |= change
            edge[y] |= change
        return cv2.dilate(edge,np.ones((5,5),np.uint8)).astype(bool)

    def fraction(numerator,denominator):
        return float(numerator/denominator) if denominator else None

    try:
        with h5py.File(reference_path,'r') as h:
            g = h['episodes'][geometry['case']]
            assert len(g['frame_timestamps_s']) >= intervals*10+1
            all_times = np.asarray(g['frame_timestamps_s'][:intervals*10+1],dtype=np.float64)
            assert np.allclose(np.diff(all_times),.001,rtol=0,atol=1e-9)
            raw_rgb = np.asarray(g['radiance_linear_rgb'][:intervals*10+1])
            depth = np.asarray(g['depth_m'][:intervals*10+1])
            segmentation = np.asarray(g['segmentation_link_idx'][:intervals*10+1])
        assert raw_rgb.dtype == np.float32 and raw_rgb.shape[:3] == depth.shape == segmentation.shape
        assert raw_rgb.shape[-1] == 3
        assert np.isfinite(raw_rgb).all() and (raw_rgb>=0).all()
        luma = np.stack([radiometry.prepare_linear_radiance(x,input_space='linear') for x in raw_rgb])
        log_luma = np.log(np.maximum(luma.astype(np.float64),floor))
        query_indices = np.asarray([interval*10+j for interval in range(intervals) for j in range(1,10)],dtype=np.int64)
        qt = all_times[query_indices]
        truth = luma[query_indices]
        reference_log = log_luma[query_indices]
        query_depth = depth[query_indices]
        finite_radiance = np.isfinite(truth)&(truth>=0)
        valid_depth = np.isfinite(query_depth)&(query_depth>0)
        invalid_depth = ~valid_depth
        active = np.stack([np.ptp(log_luma[interval*10:(interval+1)*10+1],axis=0)>=active_threshold
                           for interval in range(intervals) for j in range(1,10)])
        boundaries = np.stack([boundary(segmentation[i],depth[i]) for i in query_indices])
        quiet_background = ~active&~boundaries
        assert finite_radiance.all()
        query_count = len(qt)
        pixel_count = int(finite_radiance.size)
        report['reference_counts'] = dict(queries=query_count,query_pixel_positions=pixel_count,
            valid_depth=int(valid_depth.sum()),invalid_depth=int(invalid_depth.sum()),
            zero_depth=int((query_depth==0).sum()),negative_depth=int((query_depth<0).sum()),
            nonfinite_depth=int((~np.isfinite(query_depth)).sum()),
            active=int(active.sum()),boundary=int(boundaries.sum()),quiet_background=int(quiet_background.sum()),
            invalid_depth_reference_luma_min=float(truth[invalid_depth].min()) if invalid_depth.any() else None,
            invalid_depth_reference_luma_max=float(truth[invalid_depth].max()) if invalid_depth.any() else None,
            invalid_depth_reference_luma_mean=float(truth[invalid_depth].mean()) if invalid_depth.any() else None)
        for N in Ns:
            data_folder = result_dir/('N'+str(N))
            paths = {name:register(data_folder/name,
                aggregate_by_N[N].get('arrays',{}).get(name,{}).get('sha256'))
                for name in ('query_predictions.npy','query_support.npy','query_times.npy')}
            predicted = np.load(paths['query_predictions.npy'],allow_pickle=False,mmap_mode='r')
            supported = np.load(paths['query_support.npy'],allow_pickle=False,mmap_mode='r')
            stored_times = np.load(paths['query_times.npy'],allow_pickle=False,mmap_mode='r')
            assert predicted.shape == supported.shape == truth.shape
            assert predicted.dtype == np.float32 and supported.dtype == np.bool_
            assert stored_times.dtype == np.float64 and np.array_equal(stored_times,qt)
            assert np.isfinite(predicted).all() and (predicted>=0).all()
            error = (np.log(np.maximum(np.asarray(predicted,dtype=np.float64),floor))-reference_log)/C
            unsupported = ~supported
            masks = dict(all=finite_radiance,valid_reference_depth=valid_depth,invalid_reference_depth=invalid_depth,
                supported=supported,unsupported=unsupported,
                valid_depth_supported=valid_depth&supported,valid_depth_unsupported=valid_depth&unsupported,
                invalid_depth_supported=invalid_depth&supported,invalid_depth_unsupported=invalid_depth&unsupported,
                active=active,active_supported=active&supported,active_unsupported=active&unsupported,
                active_valid_depth=active&valid_depth,
                active_valid_depth_supported=active&valid_depth&supported,
                active_valid_depth_unsupported=active&valid_depth&unsupported,
                boundary=boundaries,boundary_supported=boundaries&supported,boundary_unsupported=boundaries&unsupported,
                boundary_valid_depth_supported=boundaries&valid_depth&supported,
                quiet_background=quiet_background)
            metrics = {name:summarize(error,mask) for name,mask in masks.items()}
            # This is an independent recomputation check, not a quality tolerance.
            reproduced = {}
            for name,oldname in [('all','all'),('active','active'),('boundary','boundary'),('quiet_background','background')]:
                original = aggregate_by_N[N]['strata'][oldname]
                actual = metrics[name]
                assert actual['pixels'] == original['pixels']
                if actual['pixels']:
                    delta = abs(actual['RMSE_over_C']-original['RMSE_over_C'])
                    assert delta <= 1e-12, 'Auxiliary does not reproduce the original score: '+name
                    reproduced[name] = dict(original_RMSE_over_C=original['RMSE_over_C'],
                                             recomputed_RMSE_over_C=actual['RMSE_over_C'],absolute_delta=delta)
            unknown_count = int(unsupported.sum())
            invalid_unknown = int((invalid_depth&unsupported).sum())
            valid_unknown = int((valid_depth&unsupported).sum())
            total_sse = metrics['all']['sum_squared_error_over_C2']
            counts = dict(total=pixel_count,supported=int(supported.sum()),unsupported=unknown_count,
                unknown_fraction=fraction(unknown_count,pixel_count),
                unsupported_on_invalid_depth=invalid_unknown,unsupported_on_valid_depth=valid_unknown,
                fraction_unknown_on_invalid_depth=fraction(invalid_unknown,unknown_count),
                fraction_unknown_on_valid_depth=fraction(valid_unknown,unknown_count),
                fraction_valid_depth_unsupported=fraction(valid_unknown,int(valid_depth.sum())),
                fraction_invalid_depth_unsupported=fraction(invalid_unknown,int(invalid_depth.sum())))
            contributions = {name:fraction(value['sum_squared_error_over_C2'],total_sse)
                             for name,value in metrics.items()}
            # Equivalent physical luma summaries help distinguish missing depth
            # from black/sky without inventing a semantic sky label.
            unsupported_appearance = dict(reference_luma_mean=float(truth[unsupported].mean()) if unknown_count else None,
                prediction_luma_mean=float(np.asarray(predicted)[unsupported].mean()) if unknown_count else None,
                invalid_depth_supported_reference_luma_mean=float(truth[invalid_depth&supported].mean()) if (invalid_depth&supported).any() else None)
            item = dict(N=N,counts=counts,strata=metrics,error_energy_fraction=contributions,
                        original_metrics_reproduced=reproduced,unsupported_appearance=unsupported_appearance,
                        query_arrays_read_only=True)
            report['per_N'].append(item)
            if a.maps_N == N:
                chosen = [int(x) for x in a.map_indices.split(',')] if a.map_indices else [0,query_count//2,query_count-1]
                assert len(chosen) == len(set(chosen)) and all(0<=i<query_count for i in chosen)
                for qi in chosen:
                    map_path = out/('N'+str(N)+'_query'+str(qi).zfill(3)+'_maps.npz')
                    np.savez_compressed(map_path,N=np.asarray(N),query_index=np.asarray(qi),t_s=np.asarray(qt[qi]),
                        reference_luma=truth[qi],prediction_luma=np.asarray(predicted[qi]),
                        log_error_over_C=error[qi],reference_depth_m=query_depth[qi],
                        reference_segmentation=segmentation[query_indices[qi]],
                        supported=np.asarray(supported[qi]),valid_reference_depth=valid_depth[qi],
                        active=active[qi],boundary=boundaries[qi],quiet_background=quiet_background[qi])
                    report['maps'].append(dict(path=str(map_path),sha256=sha(map_path),query_index=qi,t_s=float(qt[qi]),N=N))
            save()
            print('GEOMETRY_SUPPORT_AUX',json.dumps(dict(N=N,counts=counts,
                supported_RMSE_over_C=metrics['supported']['RMSE_over_C'],
                active_supported_RMSE_over_C=metrics['active_supported']['RMSE_over_C'],
                valid_depth_supported_RMSE_over_C=metrics['valid_depth_supported']['RMSE_over_C'],
                active_valid_depth_supported_RMSE_over_C=metrics['active_valid_depth_supported']['RMSE_over_C'])),flush=True)
            del error,predicted,supported,unsupported,masks
        for path,record in inputs.items():
            assert sha(path) == record['sha256'], 'An input changed during auxiliary analysis: '+path
        report.update(status='complete',inputs_unchanged=True,wall_s=time.perf_counter()-begun)
        save()
        with (out/'geometry_support_strata.csv').open('w',newline='') as stream:
            columns = ['N','stratum','pixels','RMSE_over_C','mean_signed_over_C','p50_abs_over_C',
                       'p95_abs_over_C','p99_abs_over_C','max_abs_over_C','sum_squared_error_over_C2','error_energy_fraction']
            writer = csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            for item in report['per_N']:
                for name,value in item['strata'].items():
                    writer.writerow(dict(N=item['N'],stratum=name,**value,error_energy_fraction=item['error_energy_fraction'][name]))
        with (out/'geometry_support_counts.csv').open('w',newline='') as stream:
            columns = ['N']+list(report['per_N'][0]['counts'])
            writer = csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            for item in report['per_N']:
                writer.writerow(dict(N=item['N'],**item['counts']))
        print('GEOMETRY_SUPPORT_AUXILIARY_COMPLETE',json.dumps(dict(output=str(out),Ns=Ns,
              elapsed_s=report['wall_s'],inputs_unchanged=True)),flush=True)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
