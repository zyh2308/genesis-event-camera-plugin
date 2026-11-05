"""Two short actual-Resident controls; no training or frozen-source edits.

1. Same image/depth/seg/camera/object endpoints across a10ms interval.
2. Real left/right pair: compare the observed t1 output with the true right image.
Only run under the root agent's fresh GPU2 lease. No numerical pass tolerance is
introduced; raw differences and support are reported without relabeling them.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core',type=Path,required=True)
    p.add_argument('--reference-receipt','--reference',dest='reference_receipt',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--N',type=int,default=63)
    p.add_argument('--interval-index',type=int,default=0)
    p.add_argument('--expected-gpu',default='GPU-a02affe0-7598-579e-ca39-f551f35ad6e9')
    a = p.parse_args()
    assert a.N>=2 and a.interval_index>=0
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.expected_gpu
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]='2'
    sys.dont_write_bytecode=True
    assert not any(k=='genesis_event_plugin' or k.startswith('genesis_event_plugin.') for k in sys.modules)
    core=a.core.resolve(strict=True)
    sys.path.insert(0,str(core))
    import h5py
    import numpy as np
    import torch
    from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
    from genesis_event_plugin.gpu_interpolator import ResidentPhysicsInterpolator
    from genesis_event_plugin.radiometry import prepare_linear_radiance
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()

    def sha(path):
        digest=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):
                digest.update(block)
        return digest.hexdigest()

    source={str(path.relative_to(core)):sha(path) for path in sorted(core.rglob('*.py')) if '__pycache__' not in path.parts}
    assert source['genesis_event_plugin/gpu_geometry.py']=='e74793c4805eabfa2a19d9db02b5d829c73a12a61d3fed6b4cbeb8e5995a060b'
    profile=json.loads((core/'stage9_sensor_profile.json').read_text())
    receipt=json.loads(a.reference_receipt.read_text())
    assert receipt['status']=='complete' and receipt['actual_genesis_render'] is True
    assert receipt['repeated_actual_render_passed'] is True
    reference=Path(receipt['output']).resolve(strict=True)
    assert sha(reference)==receipt['output_sha256']
    l,r=a.interval_index*10,(a.interval_index+1)*10
    with h5py.File(reference,'r') as h:
        g=h['episodes'][receipt['case']]
        assert r<len(g['frame_timestamps_s'])
        K=np.asarray(g['camera_intrinsics_K_px'])
        ids=np.asarray(g['object_segmentation_link_ids'])
        times=np.asarray(g['frame_timestamps_s'][[l,r]],dtype=np.float64)
        endpoints=[]
        for index in (l,r):
            endpoints.append(dict(rad=np.asarray(g['radiance_linear_rgb'][index]),
                depth=np.asarray(g['depth_m'][index]),seg=np.asarray(g['segmentation_link_idx'][index]),
                cam=np.asarray(g['camera_to_world_cv'][index]),
                objects={int(sid):value for sid,value in zip(ids,np.asarray(g['object_to_world'][index]))}))
    assert all(x['rad'].dtype==np.float32 and x['rad'].shape==(260,346,3) for x in endpoints)
    assert all(x['depth'].shape==x['seg'].shape==(260,346) for x in endpoints)
    C=float(profile['noise_config_kwargs']['pos_thres'])
    tau=1/(2*math.pi*profile['noise_config_kwargs']['cutoff_hz'])
    dt=float(times[1]-times[0])
    epsilon=(dt/tau)/a.N*(1+1e-10)
    cfg=replace(AdaptiveSamplingConfig(**profile['sampler_kwargs']),max_samples=a.N,
                sensor_time_constant_s=tau,sensor_eps_target=epsilon,strict=False)
    out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='resident_geometry_identity_control_v1',status='running',
        core=str(core),source_sha256=source,reference=str(reference),reference_sha256=receipt['output_sha256'],
        reference_receipt=str(a.reference_receipt),historical_replay_passed=receipt['coarse_endpoint_replay_passed'],
        case=receipt['case'],interval_index=a.interval_index,N=a.N,times_s=times.tolist(),
        sampler_config=asdict(cfg),sampler_eps_override=epsilon,
        gpu_uuid=a.expected_gpu,CPU_threads=2,deterministic_algorithms=True,TF32=False,
        sensor_called=False,training_started=False,core_modified=False,
        numerical_tolerance_acceptance_used=False,tests=[],
        limitations=['Identity verifies self-reprojection and radiometry, not independent renderer intrinsics/depth calibration.',
                     'Real t1 is an observed endpoint override; success alone does not validate middle-frame reconstruction.'])

    def save():
        (out/'geometry_identity_control.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

    def metrics(pred,target,mask=None):
        x=np.asarray(pred,dtype=np.float64)
        y=np.broadcast_to(np.asarray(target,dtype=np.float64),x.shape)
        selected=np.ones(x.shape,dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
        assert selected.shape==x.shape
        count=int(selected.sum())
        if not count:
            return dict(pixels=0,max_abs_luma=None,RMSE_luma=None,RMSE_log_over_C=None,p99_abs_log_over_C=None)
        delta=(x-y)[selected]
        log_delta=((np.log(np.maximum(x,1e-6))-np.log(np.maximum(y,1e-6)))/C)[selected]
        return dict(pixels=count,max_abs_luma=float(np.abs(delta).max()),RMSE_luma=float(np.sqrt(np.mean(delta*delta))),
                    RMSE_log_over_C=float(np.sqrt(np.mean(log_delta*log_delta))),
                    p99_abs_log_over_C=float(np.percentile(np.abs(log_delta),99)))

    try:
        for name,left,right in [('same_endpoint_identity',endpoints[0],endpoints[0]),
                                 ('real_pair_observed_t1',endpoints[0],endpoints[1])]:
            interp=ResidentPhysicsInterpolator(K,sampler_config=cfg,disocclusion_policy='hold',
                composite='coverage_time_log_linear',device='cuda:0',warp_backend='torch',strict_fidelity=False,
                min_coverage_fraction=profile['plugin_kwargs']['min_coverage_fraction'])
            delta={sid:right['objects'][sid]@np.linalg.inv(left['objects'][sid]) for sid in ids.tolist()}
            torch.cuda.synchronize()
            began=time.perf_counter()
            with torch.no_grad():
                frames,grid_times=interp.interpolate_device(left['rad'],left['depth'],left['seg'],
                    right['rad'],right['depth'],right['seg'],left['cam'],right['cam'],delta,
                    float(times[0]),float(times[1]),input_space='linear',
                    object_world0=left['objects'],object_world1=right['objects'])
            torch.cuda.synchronize()
            wall=time.perf_counter()-began
            sampling=asdict(interp.last_sampling)
            assert len(frames)==len(grid_times)==sampling['samples']==a.N
            assert not sampling['capped'] and sampling['uncapped_samples']==a.N
            assert abs(grid_times[-1]-times[1])<1e-12
            assert interp.last_uncertainty['extra_render_count']==0
            resident=interp._resident_frames
            assert resident.is_cuda and resident.dtype==torch.float32 and resident.shape==(a.N,260,346)
            output=resident.detach().cpu().numpy()
            supported=np.asarray(interp.last_visibility['intermediate_visibility'],dtype=bool)
            assert supported.shape==output.shape and np.isfinite(output).all()
            target_left=prepare_linear_radiance(left['rad'],input_space='linear')
            target_right=prepare_linear_radiance(right['rad'],input_space='linear')
            assert np.array_equal(target_left,interp._current_I0) and np.array_equal(target_right,interp._current_I1)
            record=dict(name=name,geometry_wall_s=wall,actual_sampling=sampling,
                resident_geometry_active=True,shape=list(output.shape),dtype=str(output.dtype),
                luma_range=[float(output.min()),float(output.max())],
                unknown_fraction=float((~supported).mean()),
                final_endpoint=metrics(output[-1],target_right),
                final_endpoint_supported=metrics(output[-1],target_right,supported[-1]),
                final_endpoint_unknown_fraction=float((~supported[-1]).mean()),
                final_endpoint_raw_bytes_equal=output[-1].tobytes()==target_right.tobytes(),
                hole_hold_fallback_pixels=int(interp._resident_last_hole_pixels),
                actual_left_luma_matches_production_current_I0=True,
                actual_right_luma_matches_production_current_I1=True)
            if name=='same_endpoint_identity':
                record.update(all_samples=metrics(output,target_left),
                    supported_samples=metrics(output,target_left,supported),
                    unsupported_samples=metrics(output,target_left,~supported),
                    all_samples_raw_bytes_equal=all(frame.tobytes()==target_left.tobytes() for frame in output),
                    max_identity_object_delta_deviation=float(max(np.max(np.abs(value-np.eye(4))) for value in delta.values())))
            report['tests'].append(record)
            save()
            print('GEOMETRY_IDENTITY_CONTROL',json.dumps(record),flush=True)
            np.save(out/(name+'_last_prediction.npy'),output[-1],allow_pickle=False)
            np.save(out/(name+'_target.npy'),target_right,allow_pickle=False)
            del interp,frames,resident,output,supported
            gc.collect()
            torch.cuda.empty_cache()
        assert {str(path.relative_to(core)):sha(path) for path in sorted(core.rglob('*.py')) if '__pycache__' not in path.parts}==source
        assert sha(reference)==receipt['output_sha256']
        report.update(status='complete',core_unchanged=True,reference_unchanged=True)
        save()
        print('GEOMETRY_IDENTITY_CONTROL_COMPLETE',str(out),flush=True)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        save()
        raise


if __name__=='__main__':
    main()
