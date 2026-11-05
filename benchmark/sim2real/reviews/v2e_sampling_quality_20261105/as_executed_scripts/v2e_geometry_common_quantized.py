"""Read-only CPU re-score of the native-output display-domain pilot.

Every candidate is finally clipped and rounded into the same uint8 DN domain.
Original results/arrays survive unchanged. V2E's saved float32 query arrays are
used; their rounding relative to the original float64 interpolation is declared.
This does not make native raw/display temporal interpolation commute, and is not
a universal geometric-algorithm or event-fidelity ranking.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import cv2
import h5py
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def dump(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def mapping(x):
    x=np.asarray(x,dtype=np.float32)
    assert np.isfinite(x).all()
    y=np.maximum(np.float32(0),x);tone=y/(np.float32(1)+y)
    srgb=np.where(tone<=.0031308,12.92*tone,1.055*np.power(tone,1./2.4)-.055)
    return np.clip(np.rint(255.*srgb),0,255).astype(np.uint8)


def boundary(s,d):
    mask=np.zeros(s.shape,dtype=np.uint8)
    for axis in (0,1):
        left=[slice(None),slice(None)];right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
        x,y=tuple(left),tuple(right);good=(d[x]>0)&(d[y]>0)
        changed=(s[x]!=s[y])|(good&(np.abs(d[x]-d[y])/np.maximum(np.minimum(d[x],d[y]),1e-6)>.05))
        mask[x]|=changed;mask[y]|=changed
    return cv2.dilate(mask,np.ones((5,5),np.uint8)).astype(bool)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-receipt',type=Path,required=True)
    p.add_argument('--geometry-results',type=Path,required=True,help='V2E v1 JSON or its containing directory')
    p.add_argument('--ours-results',type=Path,required=True,help='Original Ours geometry JSON or containing directory')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result_path=a.geometry_results/'v2e_geometry_quality.json' if a.geometry_results.is_dir() else a.geometry_results
    ours_path=a.ours_results/'geometry_quality.json' if a.ours_results.is_dir() else a.ours_results
    v=json.loads(result_path.read_text());o=json.loads(ours_path.read_text());r=json.loads(a.reference_receipt.read_text())
    assert v['status']==o['status']=='complete' and v['sensor_called'] is False
    assert r['actual_genesis_render'] and r['repeated_actual_render_passed']
    ref=Path(r['output']);assert sha(ref)==r['output_sha256']==v['reference_sha256']==o['reference_sha256']
    a.out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='common_final_uint8_display_geometry_rescore_v1',status='running',
        source_script_sha256=sha(__file__),CPU_only=True,GPU_or_Torch_called=False,
        original_results_modified=False,original_V2E_results=str(result_path),ours_results=str(ours_path),
        input_sha256={str(x):sha(x) for x in [result_path,ours_path,a.reference_receipt,ref]},rows=[],
        final_query_quantization='uint8 round-to-nearest-even(np.clip(prediction_normalized,0,1)*255), all methods',
        reference_domain='same fixed linear-luma -> Reinhard -> sRGB -> uint8 DN as V2E v1',
        native_V2E_forward_changed=False,V2E_saved_query_dtype='float32; original v1 metric was computed before this save from float64 queries',
        raw_vs_display_temporal_interpolation_equal_claimed=False,
        limitations=['Native-output display-domain pilot on one0.1s clip; not universal geometry, sensor or realism ranking.',
            'Final common-query quantization is symmetric; native temporal interpolation domains still differ.',
            'Saved V2E float32 query rounding can resolve exact half-DN ties differently from unavailable original float64 queries; near-tie counts are reported.',
            'No denser real-render convergence was established and no event timing is scored.'])
    save=lambda:dump(a.out/'common_quantized_geometry.json',report)
    save();begun=time.perf_counter()
    try:
        n=int(v['intervals']);assert 2<=n<=10
        with h5py.File(ref,'r') as h:
            g=h['episodes'][r['case']];stop=n*10+1
            times=np.asarray(g['frame_timestamps_s'][:stop],dtype=np.float64)
            rgb=np.asarray(g['radiance_linear_rgb'][:stop],dtype=np.float32)
            seg=np.asarray(g['segmentation_link_idx'][:stop]);dep=np.asarray(g['depth_m'][:stop])
        assert np.allclose(np.diff(times),.001,rtol=0,atol=1e-9)
        weights=np.asarray([.2126,.7152,.0722],dtype=np.float32)
        raw=np.stack([np.tensordot(x,weights,axes=([-1],[0])).astype(np.float32,copy=False) for x in rgb])
        display=mapping(raw);anchors=display[::10]
        assert [hashlib.sha256(x.tobytes()).hexdigest() for x in anchors]==v['native_endpoint_display_sha256']
        truth=np.concatenate([display[i*10+1:i*10+10] for i in range(n)])
        qt=np.concatenate([times[i*10+1:i*10+10] for i in range(n)])
        masks={key:[] for key in ['all','active','boundary','background']}
        H,W=raw.shape[1:]
        for i in range(n):
            active=np.ptp(np.log(np.maximum(raw[i*10:i*10+11].astype(np.float64),1e-6)),axis=0)>=.02
            for j in range(1,10):
                edges=boundary(seg[i*10+j],dep[i*10+j])
                masks['all'].append(np.ones((H,W),dtype=bool));masks['active'].append(active)
                masks['boundary'].append(edges);masks['background'].append(~active&~edges)
        masks={k:np.stack(x) for k,x in masks.items()}
        report.update(case=r['case'],intervals=n,common_queries=len(qt),shape_hw=[H,W],
            masks_count={k:int(x.sum()) for k,x in masks.items()},reference_anchors_match_V2E_v1_bytewise=True)
        def score(pred):
            pred=np.asarray(pred,dtype=np.float64);assert pred.shape==truth.shape and np.isfinite(pred).all()
            scaled=np.clip(pred,0,1)*255
            quant=np.rint(scaled).astype(np.uint8)
            delta=(quant.astype(np.float64)-truth.astype(np.float64))/255
            # A small, conservative flag for saved-float32 half-DN sensitivity;
            # do not perturb/tie-correct the actual stored prediction values.
            near_half=np.abs(scaled-(np.floor(scaled)+.5))<=4*np.finfo(np.float32).eps*255
            summary={}
            for key,mask in masks.items():
                z=delta[mask];mse=float(np.mean(z*z)) if z.size else None
                summary[key]=dict(pixels=int(z.size),MSE=mse,RMSE=math.sqrt(mse) if mse is not None else None,
                    PSNR_dB=-10*math.log10(mse) if mse else None,perfect_match=(mse==0) if mse is not None else None,
                    p99_abs=float(np.percentile(np.abs(z),99)) if z.size else None,
                    saved_float32_near_half_DN_count=int(near_half[mask].sum()))
            return summary,dict(clipped_query_values=int(np.count_nonzero((pred<0)|(pred>1))),
                near_half_DN_count=int(near_half.sum()),near_half_detection_tolerance_DN=float(4*np.finfo(np.float32).eps*255))
        def append(method,mode,N,pred,extra=None):
            strata,numeric=score(pred)
            report['rows'].append(dict(method=method,mode=mode,N=N,strata=strata,numerical_query_metadata=numeric,**(extra or {})));save()
        baseline=np.concatenate([np.stack([(1-j/10)*anchors[i].astype(np.float64)/255+j/10*anchors[i+1].astype(np.float64)/255 for j in range(1,10)]) for i in range(n)])
        append('display_endpoint_linear','common_final_uint8',None,baseline)
        for old in v['rows']:
            if old['method']=='EventGenesis':
                N=old['N'];ag=next(x for x in o['aggregates'] if x['N']==N);arrays=ag['arrays']
                pp=Path(arrays['query_predictions.npy']['path']);tp=Path(arrays['query_times.npy']['path'])
                assert sha(pp)==arrays['query_predictions.npy']['sha256'] and sha(tp)==arrays['query_times.npy']['sha256']
                report['input_sha256'].update({str(pp):sha(pp),str(tp):sha(tp)})
                assert np.array_equal(np.load(tp)[:n*9],qt)
                previous=np.load(pp,mmap_mode='r')[:n*9]
                append('EventGenesis','prior_raw_query_then_T_common_final_uint8',N,mapping(previous).astype(np.float64)/255,
                       dict(timing_imported_from_previous_run=True,geometry_wall_s=old.get('geometry_wall_s')))
            elif old['method']=='V2E_SuperSloMo':
                arrays=old['prediction_arrays'];pp=Path(arrays['query_display_predictions.npy']['path']);tp=Path(arrays['query_times.npy']['path'])
                assert sha(pp)==arrays[pp.name]['sha256'] and sha(tp)==arrays[tp.name]['sha256']
                report['input_sha256'].update({str(pp):sha(pp),str(tp):sha(tp)})
                assert np.array_equal(np.load(tp),qt)
                pred=np.load(pp,mmap_mode='r');assert pred.dtype==np.float32
                append('V2E_SuperSloMo',old['mode']+'_common_final_uint8',old['N'],pred,
                    dict(original_query_prediction_dtype='float32',original_float64_query_interpolation_not_available=True,
                         actual_N_by_interval=old['actual_N_by_interval'],geometry_wall_s=old.get('geometry_wall_s')))
        assert all(sha(path)==expected for path,expected in report['input_sha256'].items())
        report.update(status='complete',inputs_unchanged=True,wall_s=time.perf_counter()-begun);save()
        with (a.out/'common_quantized_geometry.csv').open('w',newline='') as f:
            fields=['method','mode','N','stratum','pixels','MSE','RMSE','PSNR_dB','p99_abs','saved_float32_near_half_DN_count']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in report['rows']:
                for key,value in row['strata'].items():writer.writerow(dict(method=row['method'],mode=row['mode'],N=row['N'],stratum=key,**{k:value[k] for k in fields[4:]}))
        print('COMMON_QUANTIZED_GEOMETRY_COMPLETE',len(report['rows']),flush=True)
    except Exception as exc:
        report.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));save();raise


if __name__=='__main__':main()
