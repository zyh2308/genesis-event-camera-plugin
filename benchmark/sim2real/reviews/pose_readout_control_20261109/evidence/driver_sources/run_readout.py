"""C1 RAW native-global versus fixed predicted-peak soft-focus; zero updates."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
import pose_adapter as A
sys.dont_write_bytecode=True
REG_SHA='cb0a02ee6f8a9f8e089d81ecc534a712f7a8ef5e7d4e924841a0fb8ef24dccca'
CONTRACT_SHA='d2368c9f246a61efc4382238741f1e7a62cf891aa8d6d653136832ed64ea3b62'
DIAGNOSTIC_SHA='9e604e49ae861ab3831d641cd8c6d57d506017b060ebd30ee7a0b23f80e8771b'
EXPORT_SHA='fea5095c363b2d4a02602c4fe2dec6144058564be877c55caf0ef1dc24631093'
MATH_MANIFEST='b46848fa5acf2d7eb21c0a53694e1b4f412057bd1bc30d00b51fc512f66c7055'
MATH_SHA='8432c5b67deada0697acb64caa1481d6a1462d005aa73e3e8ff81a4a986c9265'
SHARED_SHA='bccc8e057eec4b506029eeb14c9e4d6c90ddef2d2240b2b6e4da9367b26308c7'

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def record(p):
    p=Path(p);return dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def default(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False,default=default)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def checked_bundle(p,digest,records):
    need(sha(p/'source_manifest.json')==digest,'Externally pinned source manifest differs');records.append(record(p/'source_manifest.json'))
    for name,h in read(p/'source_manifest.json').items():
        need(Path(name).name==name and sha(p/name)==h,'Frozen bundle source differs: '+name);records.append(record(p/name))
def references(E,pds,records):
    cp=pds.parent/'export/completion.json';need(sha(cp)==EXPORT_SHA,'Original export receipt changed');c=read(cp)
    need(c['status']=='complete' and c['optimizer_updates']==0 and c['real_data_used'] is False and c['contract_sha256']==CONTRACT_SHA,'Original export scope differs');records.append(record(cp))
    byname={Path(r['path']).name:r for r in c['artifacts']};refs={}
    for name in ('predictions.jsonl','native_scores.json'):
        r=byname[name];need(sha(r['path'])==r['sha256'],'Original reference artifact changed');records.append(r)
    with Path(byname['predictions.jsonl']['path']).open() as f:
        for line in f:
            r=json.loads(line)
            if r['arm']=='C1':refs[(r['method'],r['seed'],r['condition'],r['sample_id'])]=r
    scores={(r['method'],r['seed'],r['condition']):r['score'] for r in read(byname['native_scores.json']['path']) if r['arm']=='C1'}
    need(len(refs)==1152 and len(scores)==18,'Complete C1 matched64 reference matrix required');return refs,scores
def baseline_admission(path,digest,source_manifest):
    need(path is not None and sha(path)==digest,'Actual SHA-bound baseline required');v=read(path)
    need(v['schema']=='pose_readout_control_driver_v1' and v['status']=='complete' and v['mode']=='baseline' and v['optimizer_updates']==0 and v['real_data_used'] is False,'Baseline incomplete')
    need(v['registration_sha256']==REG_SHA and v['contract_sha256']==CONTRACT_SHA and v['source_manifest_sha256']==source_manifest and v['math_manifest_sha256']==MATH_MANIFEST,'Baseline source/registration mismatch')
    need(v['paired_cases']==384 and v['score_rows']==12 and v['all_reference_checks_passed'] and v['all_models_unchanged'],'Baseline inventory/parity/state failed')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('baseline','full'))
    for n in ('contract','diagnostic-sources','math-sources','registration','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--baseline-receipt',type=Path);p.add_argument('--baseline-sha256');a=p.parse_args()
    need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'Fresh absolute output required');a.out.mkdir();start=time.monotonic()
    try:
        need(sha(a.registration)==REG_SHA and sha(a.contract)==CONTRACT_SHA,'Registration/base contract changed');reg=read(a.registration);c=read(a.contract)
        need(reg['schema']=='pose_predicted_peak_readout_control_v1' and reg['legacy_arm']=='physical_supervised' and reg['input']=='RAW' and reg['sigma_normalized']==1/16 and reg['optimizer_updates']==0 and reg['real_data_used'] is False,'Registered scientific scope changed')
        torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        sources=[];own=Path(__file__).parent;checked_bundle(own,a.source_manifest_sha256,sources);checked_bundle(a.math_sources,MATH_MANIFEST,sources);checked_bundle(a.diagnostic_sources,DIAGNOSTIC_SHA,sources)
        need(sha(a.math_sources/'softfocus_probability.py')==MATH_SHA and sha(own/'native_pose_shared.py')==SHARED_SHA,'Registered math/native score source changed')
        binding=os.environ.get('CUDA_VISIBLE_DEVICES');M=load('readout_registered_math',a.math_sources/'softfocus_probability.py');E=load('readout_frozen_export',a.diagnostic_sources/'export_pose_factors.py')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')==binding,'Dependency import changed assigned GPU visibility');ctx=E.load_context(c,check_gpu=True)
        need(sha(ctx.shared.__file__)==SHARED_SHA,'Actual native pose metric implementation changed')
        E.remember(ctx.source_records,a.registration,REG_SHA);E.remember(ctx.source_records,a.contract,CONTRACT_SHA)
        refs,score_refs=references(E,a.diagnostic_sources,sources)
        if a.mode=='full':baseline_admission(a.baseline_receipt,a.baseline_sha256,a.source_manifest_sha256);sources.append(record(a.baseline_receipt))
        scores=[];cases=[];models=[];parity=[];maps=[];weights=[];tree=cKDTree(ctx.vertices);keypoints=None
        for method in reg['methods']:
            rows,_=E.aligned_rows(ctx,c,method,with_gains=False);views={k:ctx.shared.tensors(v,ctx.device) for k,v in rows.items()}
            expected={f'{rid}/{k:03d}' for rid in reg['rids'] for k in reg['k']}
            need(set(rows)==set(reg['conditions']) and all({r['id'] for r in v}==expected for v in rows.values()),'Fixed64 physical inventory changed')
            for values in rows.values():
                for row in values:
                    if keypoints is None:keypoints=row['keys'].copy()
                    need(np.array_equal(row['keys'],keypoints),'Keypoint CAD changes within fixed object cohort')
            for condition,values in rows.items():
                bank=ctx.D/'pose_bank' if condition=='normal' else Path(c['radiance_validation'])/condition/'pose_bank'
                E.remember(ctx.source_records,bank/'events'/method/'manifest.json')
                for rid in reg['rids']:
                    for q in (bank/'physical'/(rid+'.h5'),bank/'events'/method/(rid+'.npy')):E.remember(ctx.source_records,q)
            for seed in ([0] if a.mode=='baseline' else reg['seeds']):
                depth,_,_=E.load_branch(ctx,method,seed,'physical_supervised','depth');center,_,_=E.load_branch(ctx,method,seed,'physical_supervised','center')
                parent=next(r for r in ctx.completion['results'] if (r['method'],r['seed'])==(method,seed));rp=ctx.A/f'jobs/117_pose_rotation_paired_20261106/results/{method}_seed{seed}_rotation.pt';E.remember(ctx.source_records,rp,parent['parents']['rotation_sha256'])
                head=ctx.rotation.RotationHead().to(ctx.device).eval();head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
                for v in (depth,center,head):v.requires_grad_(False)
                states=[E.state_sha(v.state_dict()) for v in (depth,center,head)]
                for condition,value in rows.items():
                    need(time.monotonic()-start<1200,'Readout diagnostic runtime exceeded');images=views[condition];K=[r['K'] for r in value]
                    readout=A.center_readouts(center,images,M);d=E.predict_raw(depth,images,ctx.scale,K);gt=ctx.shared.labels(ctx.scale,value,ctx.vertices,ctx.device).cpu().numpy()
                    for i,row in enumerate(value):
                        ref=refs[(method,seed,condition,row['id'])];np.testing.assert_array_equal(row['uv'][0],np.asarray(ref['uv_gt'],np.float32));checks={}
                        for name,current,original in [('center_uv',readout['control_uv'][i],ref['pred_uv']),('depth_uv',d['uv'][i],ref['depth_model_uv']),('log_size',d['log_size'][i],ref['log_size']),('log_shape',d['log_shape'][i],ref['log_shape']),('z',d['z'][i],ref['z_m']),('size',d['size'][i],ref['size'])]:
                            error=float(np.max(np.abs(np.asarray(current)-original)));need(np.isfinite(error) and error<=2e-6,'Original native reference parity failed: '+name);checks[name]=error
                        parity.append(dict(method=method,seed=seed,condition=condition,sample_id=row['id'],max_abs=checks))
                    pair=A.pose_pair(ctx,head,images,value,readout['control_uv'],readout['candidate_uv'],d['z'],d['size'],gt[:,2],tree,score_refs[(method,seed,condition)])
                    need(np.max(np.abs(pair['control']['R']-np.stack([refs[(method,seed,condition,row['id'])]['R_pred'] for row in value])))<=2e-6,'Control rotation differs from frozen reference')
                    for arm in ('control','candidate'):scores.append(dict(method=method,seed=seed,condition=condition,arm=arm,readout='native_global' if arm=='control' else 'predicted_peak_softfocus',score=pair[arm]['score'],windows=64,window_ids=[r['id'] for r in value]))
                    for i,row in enumerate(value):
                        diag=readout['diagnostics'][i];r0=pair['control']['R'][i];r1=pair['candidate']['R'][i]
                        finite=lambda u,R:bool(np.isfinite([*u,*R.ravel(),d['z'][i],d['size'][i]]).all())
                        cases.append(dict(map_index=len(cases),method=method,seed=seed,condition=condition,rid=row['rid'],sample_id=row['id'],control_uv=readout['control_uv'][i],candidate_uv=readout['candidate_uv'][i],gt_uv=row['uv'][0],z_m=d['z'][i],size=d['size'][i],R_control=r0,R_candidate=r1,
                            finite_prediction_control=finite(readout['control_uv'][i],r0),finite_prediction_candidate=finite(readout['candidate_uv'][i],r1),peak_uv=diag['peak']['uv'],original_fp64_uv=diag['original_fp64_uv'],softfocus_numerical=diag['numerical'],K=row['K'],T_gt=row['T'],anchor=row['keys'][0]))
                    maps.extend(readout['attention']);weights.extend(readout['weights'])
                need(all(E.state_sha(v.state_dict())==old for v,old in zip((depth,center,head),states)),'Frozen model/rotation state changed');models.append(dict(method=method,seed=seed,center_state=states[1],depth_state=states[0],rotation_state=states[2],unchanged=True))
                del depth,center,head;torch.cuda.empty_cache()
        n=384 if a.mode=='baseline' else 1152;need(len(cases)==len(parity)==n and len(scores)==n//32 and all(r['finite_prediction_control'] and r['finite_prediction_candidate'] for r in cases),'Incomplete/nonfinite paired inventory')
        sources+=ctx.source_records;need(all(sha(r['path'])==r['sha256'] for r in sources),'Frozen source/input/weight changed')
        write(a.out/'scores.json',scores);write(a.out/'native_parity.json',dict(passed=True,checks=parity))
        with (a.out/'paired_cases.jsonl').open('x') as f:
            for r in cases:f.write(json.dumps(r,allow_nan=False,default=default)+'\n')
        np.savez_compressed(a.out/'readout_maps.npz',original_attention=np.stack(maps),focused_weights=np.stack(weights));write(a.out/'readout_map_index.json',[{k:r[k] for k in ('map_index','method','seed','condition','rid','sample_id')} for r in cases]);np.save(a.out/'vertices.npy',ctx.vertices);np.save(a.out/'keypoints.npy',keypoints)
        write(a.out/'completion.json',dict(schema='pose_readout_control_driver_v1',status='complete',mode=a.mode,optimizer_updates=0,real_data_used=False,GT_forward_inputs=False,paired_cases=n,score_rows=len(scores),models=models,
            registration_sha256=REG_SHA,contract_sha256=CONTRACT_SHA,source_manifest_sha256=a.source_manifest_sha256,math_manifest_sha256=MATH_MANIFEST,all_reference_checks_passed=True,all_models_unchanged=True,
            formal30_status='not_retested_not_passed_by_this_probe',formal8_status='not_retested_not_passed_by_this_probe',automatic_real_scoring=False,scope='4 fixed synthetic rollouts64 windows; C1RAW only; normal is matched64 not clean384',
            source_records=sources,artifacts=[record(q) for q in a.out.iterdir() if q.is_file()],elapsed_seconds=time.monotonic()-start))
    except BaseException as error:
        write(a.out/'failure.json',dict(status='failed',optimizer_updates=0,real_data_used=False,error=repr(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
