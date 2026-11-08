"""Synthetic-only full15-checkpoint STE correlation-forward equivalence gate.

No real events, frames, GT or scores are read. A parent must lease GPU5 and
enforce a ten-minute hard timeout. This worker never grants scoring permission.
--cpu-self-test tests the extracted original functions and fixed fixtures only.
"""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F

UUID='GPU-06ee62ba-c171-40d4-b4ae-ea63391dc885'
METHODS=('eventgenesis','v2e','vid2e','film_esim','rife_esim')
SEEDS=(0,1,2)
CASE_IDS=('sparse_gaussian_counts','structured_gaussian_counts')
THRESHOLDS=dict(max_abs_px=1e-3,rms_px=1e-4)
BASE_SEED=20261108
NATIVE_HW=(260,346)
TARGET=352

def require(test,message):
    if not test:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def checked(path,expected):
    require(path.is_file() and sha(path)==expected,'Pinned file changed: '+str(path))
    return dict(path=str(path.resolve()),sha256=expected,size_bytes=path.stat().st_size)

def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def write_new(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def extract(path,names):
    text=path.read_text();tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    require({n.name for n in nodes}==set(names),'Expected source functions absent')
    scope=dict(torch=torch,F=F,np=np)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope,{n.name:hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest() for n in nodes}

def load_module(path,name):
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def model_tree(root,bundle):
    records=[r for r in json.loads(bundle.read_text())['repositories'] if r['repo']=='STE-FlowNet']
    require(len(records)==1 and records[0]['commit']=='eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6','Model source commit differs')
    record=records[0];rows=[]
    for item in record['files']:
        relative=Path(item['path']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe model source path')
        checked(root/relative,item['sha256']);require((root/relative).stat().st_size==item['size_bytes'],'Model source size differs')
        rows.append([item['path'],item['size_bytes'],item['sha256']])
    tree=hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    require(len(rows)==record['included_file_count']==14 and sum(r[1] for r in rows)==record['included_size_bytes'] and tree==record['included_tree_sha256'],'Model source tree differs')
    return dict(repository='STE-FlowNet',commit=record['commit'],verification='manifest_file_sha256_and_tree_sha256',
                files_verified=14,bytes_verified=sum(r[1] for r in rows),tree_sha256=tree)

def fixtures(pad):
    # Prespecified nonnegative ON/OFF count-like volumes, not signed voxels.
    rng=np.random.default_rng(BASE_SEED);h,w=NATIVE_HW;bins=np.arange(5,dtype=np.float64)[None,None,None,:]
    support=rng.random((2,h,w,1))<.025
    amplitudes=(1+rng.poisson(3,size=(2,h,w,1))).astype(np.float64)
    centres=rng.uniform(0,4,size=(2,h,w,1))
    sparse=(support*amplitudes*np.exp(-.5*((bins-centres)/.7)**2)).astype(np.float32)
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    phase=rng.uniform(0,2*np.pi,size=2)
    structured=[]
    for c in range(2):
        amp=1.+2.*np.sin(xx/17.+phase[c])**2+1.5*np.cos(yy/23.+phase[c])**2
        amp+=2.*((xx>90+30*c)&(xx<230)&(yy>65)&(yy<190))
        centre=(.5+3.*(xx/(w-1))+.3*np.sin(yy/29.+c))[None,:,:,None]
        structured.append((amp[None,:,:,None]*np.exp(-.5*((bins-centre)/.9)**2))[0])
    structured=np.stack(structured).astype(np.float32)
    result=[]
    for name,native in zip(CASE_IDS,(sparse,structured)):
        require(native.shape==(2,h,w,5) and native.dtype==np.float32 and np.isfinite(native).all() and native.min()>=0,'Invalid synthetic count fixture')
        padded,offset=pad(native,TARGET)
        require(padded.shape==(2,TARGET,TARGET,5) and offset==(46,3),'Padding changed')
        require(np.array_equal(padded[:,46:306,3:349,:],native),'Pad changed sensor values')
        result.append(dict(case_id=name,native=native,padded=padded,offset=offset,
                           metadata=dict(case_id=name,base_seed=BASE_SEED,native_shape=list(native.shape),
                           input_shape=[1,2,TARGET,TARGET,5],native_sha256=array_sha(native),input_sha256=array_sha(padded),
                           native_min=float(native.min()),native_max=float(native.max()),nonzero_fraction=float(np.count_nonzero(native)/native.size))))
    return result

def cpu_oracle(reference,adapter):
    # An independent scalar float64 oracle checks displacement direction/borders.
    rng=np.random.default_rng(BASE_SEED);a=rng.standard_normal((1,3,6,7));b=rng.standard_normal(a.shape)
    expected=np.zeros((1,9,9,6,7),np.float64)
    for dy in range(9):
        for dx in range(9):
            for y in range(6):
                for x in range(7):
                    sy,sx=y+dy-4,x+dx-4
                    if 0<=sy<6 and 0<=sx<7:
                        expected[0,dy,dx,y,x]=sum(a[0,c,y,x]*b[0,c,sy,sx] for c in range(3))
    with torch.inference_mode():
        ref=reference(torch.from_numpy(a),torch.from_numpy(b)).numpy()
        ad=adapter(torch.from_numpy(a),torch.from_numpy(b)).numpy()
    errors=dict(reference_max_abs=float(np.max(np.abs(ref-expected))),adapter_max_abs=float(np.max(np.abs(ad-expected))),
                reference_adapter_max_abs=float(np.max(np.abs(ref-ad))))
    passed=all(math.isfinite(v) and v<=1e-12 for v in errors.values())
    return dict(status='passed' if passed else 'failed',passed=passed,dtype='float64',shape=list(a.shape),
                base_seed=BASE_SEED,max_abs_threshold=1e-12,**errors)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-self-test',action='store_true')
    for name in ('reference','adapter'):
        p.add_argument('--'+name+'-source',type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--compat-source',type=Path);p.add_argument('--compat-sha256')
    p.add_argument('--manifest',type=Path);p.add_argument('--manifest-sha256')
    p.add_argument('--model-root',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();require(not args.out.exists(),'Refusing to overwrite result')
    require(args.out.parent.is_dir(),'Parent output directory must already exist')
    started=time.monotonic();deadline=started+600
    result=dict(schema='mvsec_synthetic_forward_equivalence_v1',status='failed',thresholds=THRESHOLDS,
                fixture_seed=BASE_SEED,flow_unit='px_per_interval',dtype='float32',
                comparison_domain='all native260x346 pixels and both flow components, no GT/event mask; padded diagnostics separate',
                real_data_read=False,real_scores_computed=False,training_started=False,GPU_used=False,
                scoring_authorization_created=False,checkpoints={},sources={})
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        result['sources']['reference']=checked(args.reference_source,args.reference_sha256)
        result['sources']['adapter']=checked(args.adapter_source,args.adapter_sha256)
        ref_scope,ref_hashes=extract(args.reference_source,['_MemoryEfficientSpatialCorrelation','pure_torch_spatial_correlation'])
        ad_scope,ad_hashes=extract(args.adapter_source,['_torch_corr_extension_compat','center_pad_event_volume','crop_flow_to_sensor','strict_model_output_to_hw'])
        reference=ref_scope['pure_torch_spatial_correlation'];adapter_corr=ad_scope['_torch_corr_extension_compat']
        result['sources']['extracted_reference_function_sha256']=ref_hashes
        result['sources']['extracted_adapter_function_sha256']=ad_hashes
        result['fp64_oracle']=cpu_oracle(reference,adapter_corr)
        require(result['fp64_oracle']['passed'],'FP64 scalar mathematical oracle failed')
        cases=fixtures(ad_scope['center_pad_event_volume']);result['fixtures']=[c['metadata'] for c in cases]
        if args.cpu_self_test:
            result.update(status='passed',cpu_self_test_only=True,elapsed_seconds=time.monotonic()-started)
            write_new(args.out,result);print('CPU_SELF_TEST_PASSED',flush=True);return
        require(all(x is not None for x in [args.compat_source,args.compat_sha256,args.manifest,args.manifest_sha256,args.model_root]),'GPU probe requires pinned manifest, compatibility and model root')
        result['sources']['compat']=checked(args.compat_source,args.compat_sha256)
        result['manifest']=checked(args.manifest,args.manifest_sha256)
        result['manifest_sha256']=args.manifest_sha256
        result['sources']['probe_script']=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).resolve()))
        manifest=json.loads(args.manifest.read_text())
        require(manifest['schema']=='stage9_mvsec_matrix_manifest_v2' and manifest['complete_checkpoint_matrix'] and not manifest['missing_training_runs'],'Full15 completed checkpoint matrix required')
        keys={f'{m}/seed{s}' for m in METHODS for s in SEEDS}
        require(set(manifest['checkpoint_binding'])==keys,'Matrix must contain exactly15 weights')
        require(os.environ.get('CUDA_VISIBLE_DEVICES')==UUID and torch.cuda.is_available() and torch.cuda.device_count()==1,'Parent must lease GPU5 UUID with exactly one visible device')
        torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        result.update(GPU_used=True,gpu_uuid=UUID,gpu_name=torch.cuda.get_device_name(0),tf32=False,eval_mode=True,inference_mode=True)
        adapter=load_module(args.adapter_source,'synthetic_equivalence_pinned_adapter')
        compat=load_module(args.compat_source,'synthetic_equivalence_pinned_compat')
        result['model_source']=model_tree(args.model_root,adapter.SOURCE_BUNDLE_MANIFEST)
        for seed in SEEDS:
            for method in METHODS:
                require(time.monotonic()<deadline,'Ten-minute worker budget exhausted')
                key=f'{method}/seed{seed}';binding=manifest['checkpoint_binding'][key]
                require(binding['method']==method and binding['seed']==seed and binding['completed_updates']==8910,'Checkpoint identity/budget mismatch')
                cp=Path(binding['selected_checkpoint']);checked(cp,binding['selected_checkpoint_sha256'])
                require(cp.name=='synthetic_best.pt' and cp.parent==Path(binding['run']) and cp.stat().st_size==binding['selected_checkpoint_size_bytes'],'Only exact registered synthetic_best accepted')
                model,checkpoint,commit,verification=adapter.load_pinned_model(args.model_root,cp,1,torch.device('cuda:0'))
                require(commit=='eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6','Model pin differs')
                for name,value in [('generator',method),('seed',seed),('config_sha256',manifest['recipe_sha256']),('data_manifest_sha256',binding['data_manifest_sha256']),('initial_state_sha256',binding['initial_state_sha256'])]:
                    require(checkpoint.get(name)==value,'Checkpoint metadata differs: '+name)
                compatibility=compat.patch_ste(importlib.import_module('models'),importlib.import_module('multiscaleloss'),args.model_root)
                corr=importlib.import_module('models.corr');model.eval();rows=[]
                for case in cases:
                    require(time.monotonic()<deadline,'Ten-minute worker budget exhausted')
                    tensor=torch.from_numpy(case['padded'][None]).to(device='cuda:0',dtype=torch.float32)
                    predictions=[];padded_predictions=[]
                    for function in (reference,adapter._torch_corr_extension_compat):
                        corr.spatial_correlation_sample=function
                        with torch.inference_mode():output=model(tensor,TARGET,use_dt4=False,flow_seq=False)
                        torch.cuda.synchronize()
                        padded=adapter.strict_model_output_to_hw(output).copy()
                        prediction=adapter.crop_flow_to_sensor(padded,NATIVE_HW,case['offset']).copy()
                        require(padded.shape==(2,TARGET,TARGET) and prediction.shape==(2,*NATIVE_HW),'Final predictor shape differs')
                        padded_predictions.append(padded);predictions.append(prediction)
                        del output
                    difference=predictions[0].astype(np.float64)-predictions[1].astype(np.float64)
                    padded_difference=padded_predictions[0].astype(np.float64)-padded_predictions[1].astype(np.float64)
                    maximum=float(np.max(np.abs(difference)));rms=float(np.sqrt(np.mean(difference**2)))
                    passed=math.isfinite(maximum) and math.isfinite(rms) and maximum<=THRESHOLDS['max_abs_px'] and rms<=THRESHOLDS['rms_px']
                    rows.append(dict(case_id=case['case_id'],input_sha256=case['metadata']['input_sha256'],max_abs_px=maximum,rms_px=rms,passed=passed,
                                     padded_max_abs_px=float(np.max(np.abs(padded_difference))),padded_rms_px=float(np.sqrt(np.mean(padded_difference**2))),
                                     reference_prediction_sha256=array_sha(predictions[0]),adapter_prediction_sha256=array_sha(predictions[1])))
                    del tensor
                result['checkpoints'][key]=dict(checkpoint_sha256=binding['selected_checkpoint_sha256'],cases=rows,passed=all(r['passed'] for r in rows),
                                               model_commit=commit,model_source_verification=verification,training_compatibility=compatibility)
                partial=args.out.with_name(args.out.stem+'.partial.json');temporary=partial.with_suffix('.tmp')
                temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(partial)
                print('SYNTHETIC_EQUIVALENCE',key,result['checkpoints'][key]['passed'],[(r['max_abs_px'],r['rms_px']) for r in rows],flush=True)
                del model,checkpoint;gc.collect();torch.cuda.empty_cache()
        require(len(result['checkpoints'])==15,'Not all15 checkpoints tested')
        result['status']='passed' if all(c['passed'] for c in result['checkpoints'].values()) else 'failed'
        result['elapsed_seconds']=time.monotonic()-started
        write_new(args.out,result)
        if result['status']!='passed':raise SystemExit(1)
    except BaseException:
        if not args.out.exists():
            result.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
            write_new(args.out,result)
        raise

if __name__=='__main__':main()
