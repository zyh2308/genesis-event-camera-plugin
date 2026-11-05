"""Paired display-domain geometry diagnostic using unchanged V2E SuperSloMo.

Only the coordinator launches this under the physical GPU2 lease.  No sensor,
training, real dataset, package installation or frozen-source edit is performed.
Native midpoint images retain their native times, including auto per-interval N.
"""
from __future__ import annotations
import argparse
import atexit
import contextlib
import csv
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import traceback

EXPECTED_SLOMO = '78be368ed9e6e4c9fba054c5446d0402d7a44b75bba6fc8791f44ec96e4fbcf2d'
V2E_REVISION = 'cac638444557a224ba6ee5c29260916f152c00bd'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def dump(p,value):
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v2e-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--core',type=Path,required=True)
    p.add_argument('--reference-receipt',type=Path,required=True)
    p.add_argument('--ours-results',type=Path,required=True,help='Prior geometry_independent_v2 directory')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fixed-N',default='63,126,377')
    p.add_argument('--interval-limit',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--expected-gpu',default='GPU-a02affe0-7598-579e-ca39-f551f35ad6e9')
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.expected_gpu
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert a.batch_size==1, 'This pilot registers one interval per inference batch; do not silently change auto max-flow aggregation'
    assert 2<=a.interval_limit<=10
    fixed=[int(x) for x in a.fixed_N.split(',') if x]
    assert fixed and len(fixed)==len(set(fixed)) and all(2<=x<=754 for x in fixed)
    sys.dont_write_bytecode=True
    import numpy as np
    import h5py
    import cv2
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    logging.basicConfig(level=logging.WARNING)
    vroot=a.v2e_root.resolve(strict=True);core=a.core.resolve(strict=True)
    assert sha(vroot/'v2ecore/slomo.py')==EXPECTED_SLOMO
    assert not any(n=='v2ecore' or n.startswith('v2ecore.') for n in sys.modules)
    sys.path[:0]=[str(vroot),str(core)]
    from v2ecore.slomo import SuperSloMo
    from genesis_event_plugin.radiometry import prepare_linear_radiance
    radiometry=importlib.import_module('genesis_event_plugin.radiometry')
    assert Path(radiometry.__file__).resolve().is_relative_to(core)
    ours_dir=a.ours_results.resolve(strict=True)
    original=json.loads((ours_dir/'geometry_quality.json').read_text())
    assert original['status']=='complete' and original['core_unchanged']
    assert sha(Path(radiometry.__file__))==original['source_sha256']['genesis_event_plugin/radiometry.py']
    receipt=json.loads(a.reference_receipt.read_text())
    assert receipt['status']=='complete' and receipt['actual_genesis_render']
    assert receipt['repeated_actual_render_passed']
    assert not receipt['endpoint_interpolation_used_for_reference']
    ref=Path(receipt['output']).resolve(strict=True)
    assert sha(ref)==receipt['output_sha256']==original['reference_sha256']
    assert original['reference_temporal_convergence_established'] is False
    vfiles={str(x.relative_to(vroot)):sha(x) for x in sorted(vroot.rglob('*.py'))}
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='v2e_paired_display_geometry_pilot_v1',status='running',
        source_script_sha256=sha(__file__),v2e_revision=V2E_REVISION,
        source_root=str(vroot),v2e_python_sha256=vfiles,checkpoint=str(a.checkpoint),
        checkpoint_sha256=sha(a.checkpoint),core=str(core),radiometry_sha256=sha(radiometry.__file__),
        reference=str(ref),reference_sha256=sha(ref),reference_receipt=str(a.reference_receipt),
        reference_receipt_sha256=sha(a.reference_receipt),historical_endpoint_replay_passed=receipt['coarse_endpoint_replay_passed'],
        independent_new_development_clip=True,real_data_used=False,sensor_called=False,noise_called=False,
        training_started=False,production_sources_modified=False,physical_GPU_UUID=a.expected_gpu,
        gpu_name=torch.cuda.get_device_name(),torch=torch.__version__,CPU_threads=2,
        deterministic_algorithms=True,TF32=False,batch_size=1,rows=[],
        fixed_Ns=fixed,native_auto_minimum_N=2,
        display_mapping='T(y)=round255(sRGB(y/(1+y))); y=max(raw FP32 linear luminance,0). Same scalar function for references, anchors and Ours predictions.',
        display_mapping_exposure=1.,display_range=[0,255],framewise_normalization=False,
        active_mask='true raw linear luminance10ms log range>=0.02',
        boundary_mask='segmentation discontinuity or relative depth discontinuity>5%, dilated5x5',
        query_reconstruction='V2E native midpoint display predictions plus original observed endpoints; piecewise linear DISPLAY intensity at common absolute1ms queries. Ours: previous raw-luminance common-query predictions, then same pointwise T.',
        native_midpoint_timestamps_relabelled=False,observed_endpoints_supplied_to_both=True,
        reference_subframes_supplied_to_candidate=False,reference_temporal_convergence_established=False,
        limitations=['Display-domain geometry only; not raw-HDR photoreceptor fidelity or an event-sensor comparison.',
            'One fresh Stage9-derived0.1s clip and one timing trial; not a global quality or throughput ranking.',
            '1ms true renders have not passed finer-render temporal convergence; no20us event claim.',
            'Ours uses physical geometry/depth/poses; V2E receives its native observed display frames.',
            'Ours geometry timings are imported from previous run, not a new paired performance trial.'])
    def save():dump(out/'v2e_geometry_quality.json',report)
    save()
    try:
        with h5py.File(ref,'r') as h:
            g=h['episodes'][receipt['case']]
            all_ts=np.asarray(g['frame_timestamps_s'],dtype=np.float64)
            assert np.allclose(np.diff(all_ts),.001,rtol=0,atol=1e-9)
            n=a.interval_limit;stop=n*10+1
            ts=all_ts[:stop];rgb=np.asarray(g['radiance_linear_rgb'][:stop])
            raw=np.stack([prepare_linear_radiance(x,input_space='linear') for x in rgb])
            seg=np.asarray(g['segmentation_link_idx'][:stop]);dep=np.asarray(g['depth_m'][:stop])
        assert raw.dtype==np.float32 and raw.shape[1:]==(260,346) and np.isfinite(raw).all()
        def mapping(x):
            x=np.asarray(x,dtype=np.float32)
            assert np.isfinite(x).all()
            y=np.maximum(np.float32(0),x);tone=y/(np.float32(1)+y)
            srgb=np.where(tone<=.0031308,12.92*tone,1.055*np.power(tone,1./2.4)-.055)
            return np.clip(np.rint(255.*srgb),0,255).astype(np.uint8)
        display=mapping(raw)
        # Record, rather than conceal, scalar-luma rounding relative to the
        # original RGB expression used to create Stage9 display files.
        rr=np.maximum(0.,.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2])
        registered_display=mapping(rr)
        diff=display.astype(np.int16)-registered_display.astype(np.int16)
        report['scalar_vs_registered_RGB_display_mapping']=dict(changed_values=int(np.count_nonzero(diff)),
            total_values=int(diff.size),max_abs_DN=int(np.max(np.abs(diff))),
            why='Production luma operator and explicit RGB multiplication/addition can have last-bit FP32 differences; all compared paths here use the same scalar mapping.')
        anchors=display[::10];anchor_ts=ts[::10];H,W=raw.shape[1:]
        inputs=out/'inputs';inputs.mkdir()
        image_lookup={}
        for i,img in enumerate(anchors):
            path=inputs/(str(i).zfill(6)+'.npy');np.save(path,img,allow_pickle=False)
            image_lookup[str(path)]=img
        report.update(case=receipt['case'],intervals=n,observed_endpoints=n+1,
                      common_queries=9*n,shape_hw=[H,W],simulated_s=float(ts[-1]-ts[0]),
                      native_endpoint_display_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in anchors])
        save()
        slomo=SuperSloMo(str(a.checkpoint),auto_upsample=False,upsampling_factor=2,
                        batch_size=1,video_path=None,vid_orig=None,vid_slomo=None,preview=False)
        @contextlib.contextmanager
        def memory_images(cache):
            osave=Image.Image.save;oload=np.load;oall=slomo._SuperSloMo__all_images
            def image_save(image,path,*args,**kwargs):cache[int(Path(path).stem)]=np.array(image,copy=True)
            def array_load(path,*args,**kwargs):
                value=image_lookup.get(str(path))
                return value.copy() if value is not None else oload(path,*args,**kwargs)
            Image.Image.save=image_save;np.load=array_load
            slomo._SuperSloMo__all_images=lambda folder:[str(Path(folder)/(str(i)+'.png')) for i in sorted(cache)]
            try:yield
            finally:Image.Image.save=osave;np.load=oload;slomo._SuperSloMo__all_images=oall
        def native_call(memory=True,folder=None):
            cache={};folder=folder or out/'virtual_png'
            torch.cuda.synchronize();begun=time.perf_counter()
            with memory_images(cache) if memory else contextlib.nullcontext():
                relative,average=slomo.interpolate(str(inputs),str(folder),(W,H))
            torch.cuda.synchronize();wall=time.perf_counter()-begun
            if memory:images=np.stack([cache[k] for k in sorted(cache)])
            else:images=np.stack([cv2.imread(str(folder/(str(i)+'.png')),cv2.IMREAD_GRAYSCALE) for i in range(len(relative))])
            relative=np.asarray(relative,dtype=np.float64)
            assert images.dtype==np.uint8 and images.shape[1:]==(H,W) and len(images)==len(relative)
            # Returned relative coordinates are left-aligned bookkeeping, but
            # the actual native image is generated at (j+0.5)/N.  Auto N can
            # differ per interval, so the fixed-N helper cannot be used.
            counts=[];absolute=[];positions=[]
            for i in range(n):
                idx=np.flatnonzero((relative>=i-1e-10)&(relative<i+1-1e-10))
                N=len(idx);assert N>=2
                assert np.allclose(relative[idx],i+np.arange(N)/N,rtol=0,atol=1e-10)
                counts.append(N);positions.append(idx)
                absolute.extend((anchor_ts[i]+(np.arange(N)+.5)/N*(anchor_ts[i+1]-anchor_ts[i])).tolist())
            absolute=np.asarray(absolute,dtype=np.float64)
            assert np.all(np.diff(absolute)>0) and sum(counts)==len(images)
            helper_diff=None
            if not slomo.auto_upsample:
                helper=slomo.get_interpolated_timestamps(anchor_ts)
                helper_diff=float(np.max(np.abs(helper-absolute)));assert helper_diff<1e-12
                assert counts==[slomo.upsampling_factor]*n
            return images,absolute,counts,positions,wall,float(average),helper_diff
        warm=native_call()
        report['warmup_including_lazy_checkpoint_load_s']=warm[4]
        disk=out/'adapter_disk_png';disk.mkdir()
        official=native_call(memory=False,folder=disk)
        adapter=dict(memory_vs_disk_predictions_byte_equal=np.array_equal(warm[0],official[0]),
            midpoint_times_equal=np.array_equal(warm[1],official[1]),frames=len(warm[0]),fixed_N=2,
            input_normalization_and_PIL_resize_unchanged=True,
            native_source_sha256=EXPECTED_SLOMO,external_previous_adapter_equivalence_not_assumed=True)
        dump(out/'adapter_disk_equivalence.json',adapter)
        assert adapter['memory_vs_disk_predictions_byte_equal'] and adapter['midpoint_times_equal']
        del warm,official
        def boundary(s,d):
            mask=np.zeros(s.shape,dtype=np.uint8)
            for axis in (0,1):
                left=[slice(None),slice(None)];right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
                x,y=tuple(left),tuple(right);good=(d[x]>0)&(d[y]>0)
                change=(s[x]!=s[y])|(good&(np.abs(d[x]-d[y])/np.maximum(np.minimum(d[x],d[y]),1e-6)>.05))
                mask[x]|=change;mask[y]|=change
            return cv2.dilate(mask,np.ones((5,5),np.uint8)).astype(bool)
        qtimes=np.concatenate([ts[i*10+1:i*10+10] for i in range(n)])
        truth=np.concatenate([display[i*10+1:i*10+10] for i in range(n)]).astype(np.float64)/255.
        masks={key:[] for key in ['all','active','boundary','background']}
        for i in range(n):
            active=np.ptp(np.log(np.maximum(raw[i*10:i*10+11].astype(np.float64),1e-6)),axis=0)>=.02
            for j in range(1,10):
                edges=boundary(seg[i*10+j],dep[i*10+j])
                masks['all'].append(np.ones((H,W),dtype=bool));masks['active'].append(active)
                masks['boundary'].append(edges);masks['background'].append(~active&~edges)
        masks={k:np.stack(v) for k,v in masks.items()}
        def score(pred):
            assert pred.shape==truth.shape and np.isfinite(pred).all()
            delta=pred-truth;result={}
            for key,mask in masks.items():
                values=delta[mask]
                if len(values):
                    mse=float(np.mean(values*values));rmse=math.sqrt(mse)
                    result[key]=dict(pixels=len(values),MSE=mse,RMSE=rmse,PSNR_dB=-10*math.log10(mse) if mse else None,
                        perfect_match=not bool(mse),p99_abs=float(np.percentile(np.abs(values),99)))
                else:result[key]=dict(pixels=0,MSE=None,RMSE=None,PSNR_dB=None,perfect_match=None,p99_abs=None)
            return result
        # A simple shared-endpoint control gives the interpolation task scale.
        baseline=np.concatenate([np.stack([(1-j/10)*anchors[i].astype(np.float64)/255.+j/10*anchors[i+1].astype(np.float64)/255. for j in range(1,10)]) for i in range(n)])
        report['rows'].append(dict(method='display_endpoint_linear',mode='no_model_control',N=None,strata=score(baseline)))
        for N in fixed:
            ag=next((x for x in original['aggregates'] if x['N']==N),None)
            if ag is None:
                report.setdefault('ours_unavailable_Ns',[]).append(N);continue
            arrays=ag['arrays'];predpath=Path(arrays['query_predictions.npy']['path'])
            timepath=Path(arrays['query_times.npy']['path'])
            assert sha(predpath)==arrays['query_predictions.npy']['sha256']
            assert sha(timepath)==arrays['query_times.npy']['sha256']
            previous=np.load(predpath,mmap_mode='r')[:n*9];previous_times=np.load(timepath)[:n*9]
            assert np.array_equal(previous_times,qtimes)
            pred=mapping(previous).astype(np.float64)/255.
            row=dict(method='EventGenesis',mode='prior_raw_prediction_same_display_mapping',N=N,
                geometry_wall_s=sum(x['geometry_wall_s'] for x in original['rows'] if x['N']==N and x['interval']<n),
                timing_imported_from_previous_run=True,strata=score(pred))
            report['rows'].append(row);save()
        modes=[('auto_min2',True,2)]+[(f'fixed{N}',False,N) for N in fixed]
        for label,auto,N in modes:
            slomo.auto_upsample=auto;slomo.upsampling_factor=N
            images,times,counts,positions,wall,avg,helperdiff=native_call()
            prediction=[]
            for i,idx in enumerate(positions):
                kt=np.r_[anchor_ts[i],times[idx],anchor_ts[i+1]]
                stack=np.concatenate((anchors[i][None],images[idx],anchors[i+1][None])).astype(np.float64)/255.
                for query in ts[i*10+1:i*10+10]:
                    hi=int(np.searchsorted(kt,query,side='left'));hi=min(max(1,hi),len(kt)-1);lo=hi-1
                    weight=float((query-kt[lo])/(kt[hi]-kt[lo]));assert -1e-10<=weight<=1+1e-10
                    prediction.append((1-weight)*stack[lo]+weight*stack[hi])
            prediction=np.stack(prediction)
            folder=out/label;folder.mkdir()
            np.save(folder/'query_display_predictions.npy',prediction.astype(np.float32),allow_pickle=False)
            np.save(folder/'query_times.npy',qtimes,allow_pickle=False)
            np.save(folder/'native_prediction_times.npy',times,allow_pickle=False)
            row=dict(method='V2E_SuperSloMo',mode=label,N=N if not auto else None,
                actual_N_by_interval=counts,actual_frame_count=len(images),native_average_N=avg,
                max_native_step_s=max((anchor_ts[i+1]-anchor_ts[i])/count for i,count in enumerate(counts)),
                geometry_wall_s=wall,checkpoint_load_excluded=True,disk_PNG_output_excluded=True,
                sensor_excluded=True,fixed_helper_timestamp_max_abs_difference_s=helperdiff,
                native_midpoint_timestamps_unchanged=True,strata=score(prediction),
                prediction_arrays={x.name:dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in folder.glob('*.npy')})
            report['rows'].append(row);save()
            print('V2E_GEOMETRY',json.dumps(dict(mode=label,actual_Ns=counts,wall_s=wall,all=row['strata']['all'],active=row['strata']['active'])),flush=True)
            del images,prediction
        slomo.cleanup();atexit.unregister(slomo.cleanup)
        assert sha(ref)==report['reference_sha256']
        assert {str(x.relative_to(vroot)):sha(x) for x in sorted(vroot.rglob('*.py'))}==vfiles
        assert sha(radiometry.__file__)==report['radiometry_sha256']
        report.update(status='complete',v2e_source_unchanged=True,reference_unchanged=True)
        save()
        with (out/'v2e_geometry_quality.csv').open('w',newline='') as f:
            keys=['method','mode','N','stratum','pixels','MSE','RMSE','PSNR_dB','p99_abs','geometry_wall_s']
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            for row in report['rows']:
                for key,val in row['strata'].items():
                    writer.writerow(dict(method=row['method'],mode=row['mode'],N=row['N'],stratum=key,
                        geometry_wall_s=row.get('geometry_wall_s'),**{k:val[k] for k in ['pixels','MSE','RMSE','PSNR_dB','p99_abs']}))
        dump(out/'artifact_index.json',[dict(path=str(x.relative_to(out)),bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(out.rglob('*')) if x.is_file()])
        print('V2E_GEOMETRY_COMPLETE',str(out),flush=True)
    except Exception as exc:
        report.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        save();raise

if __name__=='__main__':main()
