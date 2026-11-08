"""Frozen synthetic-only pose-factor forward export; zero optimizer updates."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.dont_write_bytecode=True
CORE='62da4f35ce3c5c96f04268bd1687a27657af9238'
GPU4='GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3'
NATIVE_SHA='85951e6d43e59140b81d7303370a6b252fec902dc065ff17bb30faf8e4fc6186'
PRIOR_MANIFEST='e237e55a02a359d224d9be3ba33c404100fd7a98509f2994e715c1d7c16e4502'
ARM_MAPPING={'clean_control':'C0','physical_supervised':'C1','paired_consistency':'slow'}
GAINS=(.5,1.,2.)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def state_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False,default=json_default)


def write_line(stream,value):
    stream.write(json.dumps(value,allow_nan=False,default=json_default)+'\n')


def validate_contract(c):
    assert c['schema']=='pose_factor_conflict_diagnostic_v1' and c['frozen_core']==CORE
    assert c['gpu_uuid']==GPU4 and c['timeout_seconds']==1200
    assert c['methods']==['eventgenesis','v2e'] and c['seeds']==[0,1,2] and c['arm_mapping']==ARM_MAPPING
    assert c['native_source_sha256']==NATIVE_SHA and c['prior_source_manifest_sha256']==PRIOR_MANIFEST
    for key in ('campaign','prior_source_dir','slow_study','radiance_train','radiance_validation','data_audit_path'):
        assert Path(c[key]).is_absolute()
    for key in ('slow_source_manifest_sha256','slow_receipt_sha256','slow_completion_sha256','data_audit_sha256'):
        assert len(c[key])==64 and all(ch in '0123456789abcdef' for ch in c[key])
    assert len(c['source_dependencies'])==6
    assert all(Path(p).is_absolute() and len(h)==64 for p,h in c['source_dependencies'].items())


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def remember(records,path,expected=None):
    p=Path(path);digest=sha(p)
    if expected is not None:assert digest==expected,str(p)
    record=dict(path=str(p),size_bytes=p.stat().st_size,sha256=digest)
    prior=[r for r in records if r['path']==str(p)]
    if prior:assert prior[0]==record
    else:records.append(record)
    return record


def check_gpu(c):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (GPU4,'4')
    actual=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).strip()
    assert actual==c['gpu_uuid']==GPU4 and torch.cuda.device_count()==1
    active=subprocess.check_output(['nvidia-smi','-i','4','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    pids={int(x.strip()) for x in active.splitlines() if x.strip().isdigit()}
    assert pids<={335186,os.getpid()},'Only the assigned queue parent and this child may use GPU4'


def load_context(c,check_gpu=True):
    validate_contract(c)
    if check_gpu:globals()['check_gpu'](c)
    records=[];R=Path(c['campaign']);A=R/'recovery_20261105_mainline';study=Path(c['slow_study'])
    recipe=R/'sources/training_recipe.json';remember(records,recipe)
    assert json.loads(recipe.read_text())['frozen_simulator_core']==CORE
    prior=Path(c['prior_source_dir']);mp=prior/'source_manifest.json';remember(records,mp,c['prior_source_manifest_sha256'])
    for name,digest in json.loads(mp.read_text()).items():
        assert Path(name).name==name;remember(records,prior/name,digest)
    native_path=prior/'pose_two_branch_coverage.py';remember(records,native_path,c['native_source_sha256'])
    sm=study.parent/'sources/source_manifest.json';remember(records,sm,c['slow_source_manifest_sha256'])
    for name,digest in json.loads(sm.read_text()).items():
        assert Path(name).name==name;remember(records,sm.parent/name,digest)
    receipt_path=study/'slow_features_execution_receipt.json';completion_path=study/'completion.json'
    remember(records,receipt_path,c['slow_receipt_sha256']);remember(records,completion_path,c['slow_completion_sha256'])
    receipt=json.loads(receipt_path.read_text());completion=json.loads(completion_path.read_text())
    assert receipt['status']=='complete_synthetic_only' and receipt['phase']=='study' and receipt['new_optimizer_updates']==3456
    assert not receipt['real_scores_computed'] and receipt['prior_controls_unchanged'] and receipt['all_controls_metrics_verified_against_original']
    assert receipt['source_manifest_sha256']==c['slow_source_manifest_sha256'] and len(receipt['records'])==36
    assert completion['status']=='complete' and not completion['real_data_used'] and len(completion['results'])==6
    assert len(completion['gate']['checks'])==30 and sum(bool(r['passed']) for r in completion['gate']['checks'])==18
    assert completion['original_confirmation']['passed'] and len(completion['original_confirmation']['comparisons'])==8
    assert receipt['thirty_gate']==completion['gate'] and receipt['original_confirmation']==completion['original_confirmation']
    expected={(m,s,a,b) for m in c['methods'] for s in c['seeds'] for a in ARM_MAPPING for b in ('depth','center')}
    assert {(r['method'],r['seed'],r['arm'],r['branch']) for r in receipt['records']}==expected
    for path,digest in c['source_dependencies'].items():remember(records,path,digest)
    audit_path=Path(c['data_audit_path']);remember(records,audit_path,c['data_audit_sha256'])
    audit=json.loads(audit_path.read_text());assert audit['status']=='passed' and audit['all_train_selected']
    assert all(audit['representation'][m]['normal_npz_vs_fresh_base_max_abs']==0 for m in c['methods'])
    assert not audit['real_data_used']
    sys.path.insert(0,str(prior));native=load_module('factor_bound_native',native_path);shared=native.shared
    scale=shared.load('factor_scale',A/'pose_scale_pilot_20261106.py')
    rotation=shared.load('factor_rotation',A/'pose_rotation_pilot_20261106.py')
    pilot=shared.load('factor_pilot',A/'pose_depth_pilot_20261106.py')
    fresh=shared.load('factor_fresh',A/'pose_fresh_confirm_20261106_v3.py')
    encoder_path=R/'sources/benchmark/sim2real/scripts/run_pose_stage2_pilot.py'
    tree=ast.parse(encoder_path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='encode_event_window')
    network_size=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NETWORK_SIZE' for t in n.targets))
    scope=dict(np=np,cv2=cv2,NETWORK_SIZE=network_size)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(encoder_path),'exec'),scope)
    assets=json.loads((A/'pose_prepared_full_v5/protocol.json').read_text())['assets']
    remember(records,assets['vertices']['path'],assets['vertices']['sha256'])
    vertices=np.load(assets['vertices']['path'],allow_pickle=False)
    assert vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all()
    return SimpleNamespace(R=R,A=A,D=A/'jobs/119_pose_fresh_confirm_paired_20261106/results',study=study,
        receipt=receipt,completion=completion,native=native,shared=shared,scale=scale,rotation=rotation,pilot=pilot,fresh=fresh,
        encode=scope['encode_event_window'],network_size=network_size,vertices=vertices,diameter=assets['diameter_m'],
        threshold=.1*assets['diameter_m'],device=torch.device('cuda' if check_gpu else 'cpu'),source_records=records)


def load_branch(ctx,method,seed,legacy_arm,branch):
    assert method in ('eventgenesis','v2e') and seed in (0,1,2) and legacy_arm in ARM_MAPPING and branch in ('depth','center')
    record=next(r for r in ctx.receipt['records'] if (r['method'],r['seed'],r['arm'],r['branch'])==(method,seed,legacy_arm,branch))
    path=ctx.study/(f'{method}_seed{seed}_{legacy_arm}'+('_localizer' if branch=='center' else '')+'.pt')
    remember(ctx.source_records,path)
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state_sha(state)==record['final_state_sha256']
    model=ctx.scale.CenterScale().to(ctx.device);model.load_state_dict(state,strict=True);model.eval()
    return model,record,path


def raw_count_gain_images(encode,network_size,t,x,y,p,start,end,width,height,count_scale):
    """Diagnostic count amplitudes before log/clip/resize, never inverse-resized counts."""
    base=encode(t,x,y,p,start,end,width,height,count_scale)
    selected=(t>start)&(t<=end);xs,ys,ps=x[selected],y[selected],p[selected]
    assert np.isin(ps,(0,1)).all()
    on=np.zeros((height,width),np.float32);off=np.zeros_like(on);total=np.zeros_like(on)
    np.add.at(on,(ys[ps==1],xs[ps==1]),1.);np.add.at(off,(ys[ps==0],xs[ps==0]),1.)
    np.add.at(total,(ys,xs),1.)
    scale=np.float32(np.log1p(count_scale));result={}
    for gain in GAINS:
        channels=[np.minimum(np.log1p(v*np.float32(gain))/scale,1.) for v in (on,off,total)]
        frame=np.ascontiguousarray(np.stack([cv2.resize(v,(network_size,network_size),interpolation=cv2.INTER_LINEAR) for v in channels]),dtype=np.float32)
        if gain==1:assert np.array_equal(frame,base),'Native encoder parity required before any gain diagnosis';frame=base
        assert np.array_equal(frame>0,base>0),'Gain must preserve spatial support'
        result[gain]=frame
    return base,result


def aligned_rows(ctx,c,method,with_gains=True):
    banks=dict(normal=ctx.D/'pose_bank',half=Path(c['radiance_validation'])/'half/pose_bank',eighth=Path(c['radiance_validation'])/'eighth/pose_bank')
    ids=sorted(p.stem for p in (banks['half']/'physical').glob('*.h5'));assert len(ids)==4 and all(s.startswith('validation_') for s in ids)
    rows={};gains={}
    for condition,bank in banks.items():
        files=[bank/'physical'/(rid+'.h5') for rid in ids];assert all(p.is_file() for p in files)
        cache=[]
        def callback(*args):
            base,current=raw_count_gain_images(ctx.encode,ctx.network_size,*args);cache.append(current);return base
        rows[condition]=ctx.fresh.adapter(bank,files,method,callback if with_gains else ctx.encode)
        assert len(rows[condition])==64 and len({r['rid'] for r in rows[condition]})==4
        if with_gains:
            assert len(cache)==len(rows[condition]);gains[condition]={gain:[r[gain] for r in cache] for gain in GAINS}
    identity=ctx.native.identity(rows['normal'])
    assert all(ctx.native.identity(value)==identity for value in rows.values())
    return rows,gains


@torch.inference_mode()
def predict_raw(model,images,scale,intrinsics):
    values={k:[] for k in ('uv','log_size','log_shape','z','size','local','global','entropy','normalized_entropy','peak_probability','argmax_uv','local3_uv')}
    for i in range(0,len(images),32):
        block=images[i:i+32];f=model.features(block);flat=f.flatten(2)
        logits=model.center(f).flatten(1);attention=logits.softmax(1);uv=attention@model.grid
        local=(flat*attention[:,None]).sum(2);context=flat.mean(2)
        log_size,log_shape=(model.size(torch.cat((local,context),1))-2.3).unbind(1)
        ordinary=model(block)
        for ours,native in zip((logits,uv,log_size,log_shape),ordinary):
            torch.testing.assert_close(ours,native,rtol=0,atol=0)
        log_f=torch.tensor([scale.focal(K) for K in intrinsics[i:i+32]],device=images.device)
        z=(log_shape+log_f-log_size).exp();size=log_size.exp()
        entropy=-(attention*attention.clamp_min(1e-30).log()).sum(1)
        maximum_indices=attention.argmax(1);maximum=model.grid[maximum_indices]
        width=f.shape[-1];locations=torch.arange(len(model.grid),device=images.device)
        in_local3=((locations%width-maximum_indices[:,None]%width).abs()<=1)&((locations//width-maximum_indices[:,None]//width).abs()<=1)
        local3=logits.masked_fill(~in_local3,-torch.inf).softmax(1)@model.grid
        for key,value in [('uv',uv),('log_size',log_size),('log_shape',log_shape),('z',z),('size',size),
                          ('local',local),('global',context),('entropy',entropy),('normalized_entropy',entropy/math.log(len(model.grid))),
                          ('peak_probability',attention.max(1).values),('argmax_uv',maximum),('local3_uv',local3)]:
            assert torch.isfinite(value).all();values[key].append(value.detach().cpu())
    return {key:torch.cat(value).numpy() for key,value in values.items()}


@torch.inference_mode()
def swapped_global(depth_model,base,donor,scale,intrinsics,device):
    out=dict(log_size=[],log_shape=[],z=[])
    for i in range(0,len(intrinsics),32):
        local=torch.as_tensor(base['local'][i:i+32],device=device);context=torch.as_tensor(donor['global'][i:i+32],device=device)
        log_size,log_shape=(depth_model.size(torch.cat((local,context),1))-2.3).unbind(1)
        log_f=torch.tensor([scale.focal(K) for K in intrinsics[i:i+32]],device=device)
        for key,value in [('log_size',log_size),('log_shape',log_shape),('z',(log_shape+log_f-log_size).exp())]:out[key].append(value.cpu().numpy())
    return {key:np.concatenate(value) for key,value in out.items()}


def compare_summary(current,reference):
    assert current.keys()==reference.keys()
    for key in current:
        assert np.isfinite(current[key]) and np.isclose(current[key],reference[key],rtol=1e-6,atol=1e-8),(key,current[key],reference[key])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert args.out.is_absolute() and args.out.parent.is_dir() and not args.out.exists() and not args.out.is_symlink()
    c=json.loads(args.contract.read_text());validate_contract(c);start=time.monotonic()
    def bound():
        if time.monotonic()-start>c['timeout_seconds']:raise TimeoutError('Bounded zero-update diagnostic exceeded1200s')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ctx=load_context(c);remember(ctx.source_records,args.contract)
    own=Path(__file__).resolve().parent;mp=own/'source_manifest.json'
    if mp.exists():
        for name,digest in json.loads(mp.read_text()).items():
            assert Path(name).name==name;remember(ctx.source_records,own/name,digest)
        remember(ctx.source_records,mp)
    else:remember(ctx.source_records,Path(__file__))
    args.out.mkdir(exist_ok=False);assetdir=args.out/'assets';assetdir.mkdir()
    vp=assetdir/'vertices.npy';np.save(vp,ctx.vertices)
    prediction_path=args.out/'predictions.jsonl';gain_path=args.out/'count_gain.jsonl';context_path=args.out/'context_swap.jsonl'
    native_scores=[];parity=[];counters=dict(predictions=0,count_gain=0,context_swap=0);keypoints=None
    tree=cKDTree(ctx.vertices)
    with prediction_path.open('x') as predstream,gain_path.open('x') as gainstream,context_path.open('x') as contextstream:
        for method in c['methods']:
            bound();rows,gain_images=aligned_rows(ctx,c,method)
            views={condition:ctx.shared.tensors(value,ctx.device) for condition,value in rows.items()}
            gain_views={condition:{gain:ctx.shared.tensors([dict(image=frame) for frame in images],ctx.device) for gain,images in table.items()} for condition,table in gain_images.items()}
            fullfiles=sorted((ctx.D/'pose_bank/physical').glob('*.h5'));assert len(fullfiles)==96
            full=ctx.fresh.adapter(ctx.D/'pose_bank',fullfiles,method,ctx.encode);grouped={}
            for row in full:grouped.setdefault(row['rid'],[]).append(row)
            clean=[g[i] for g in grouped.values() for i in np.linspace(0,len(g)-1,4,dtype=int)]
            assert len(grouped)==96 and len(clean)==384;clean_tensor=ctx.shared.tensors(clean,ctx.device)
            for condition,value in rows.items():
                for row in value:
                    if keypoints is None:keypoints=row['keys'].copy();np.save(assetdir/'keypoints.npy',keypoints)
                    assert np.array_equal(row['keys'],keypoints)
            gtlabels={condition:ctx.shared.labels(ctx.scale,value,ctx.vertices,ctx.device).cpu().numpy() for condition,value in rows.items()}
            for seed in c['seeds']:
                run=next(r for r in ctx.completion['results'] if (r['method'],r['seed'])==(method,seed))
                rp=ctx.A/f'jobs/117_pose_rotation_paired_20261106/results/{method}_seed{seed}_rotation.pt'
                remember(ctx.source_records,rp,run['parents']['rotation_sha256'])
                rotation_head=ctx.rotation.RotationHead().to(ctx.device).eval()
                rotation_head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
                rotation_state=state_sha(rotation_head.state_dict())
                for legacy,canonical in ARM_MAPPING.items():
                    bound();depth,dr,_=load_branch(ctx,method,seed,legacy,'depth');center,cr,_=load_branch(ctx,method,seed,legacy,'center')
                    depth_before=state_sha(depth.state_dict());center_before=state_sha(center.state_dict())
                    output={};centers={}
                    for condition,value in rows.items():
                        intrinsics=[r['K'] for r in value]
                        output[condition]=predict_raw(depth,views[condition],ctx.scale,intrinsics)
                        centers[condition]=predict_raw(center,views[condition],ctx.scale,intrinsics)
                        native_uv,native_z,native_size=ctx.scale.estimate(depth,views[condition],intrinsics)
                        np.testing.assert_array_equal(output[condition]['uv'],native_uv);np.testing.assert_array_equal(output[condition]['z'],native_z);np.testing.assert_array_equal(output[condition]['size'],native_size)
                        uv=centers[condition]['uv'];z=output[condition]['z'];size=output[condition]['size']
                        rotations=ctx.shared.frozen_rotation(ctx.rotation,rotation_head,views[condition],uv,size,intrinsics)
                        score=ctx.shared.score(ctx.rotation,value,uv,z,size,gtlabels[condition][:,2],rotations,ctx.vertices,tree,ctx.threshold)
                        native_scores.append(dict(method=method,seed=seed,arm=canonical,condition=condition,score=score))
                        if condition!='normal':
                            compare_summary(score['summary'],run['arms'][legacy][condition]['summary'])
                            parity.append(dict(method=method,seed=seed,arm=canonical,condition=condition,scope='native full4-relit-rollout64windows',passed=True))
                        gt_uv=np.stack([r['uv'][0] for r in value]);gt_size=np.exp(gtlabels[condition][:,2])
                        crop_oracles={
                            'R_crop_gt_center':ctx.shared.frozen_rotation(ctx.rotation,rotation_head,views[condition],gt_uv,size,intrinsics),
                            'R_crop_gt_size':ctx.shared.frozen_rotation(ctx.rotation,rotation_head,views[condition],uv,gt_size,intrinsics),
                            'R_crop_gt_center_and_size':ctx.shared.frozen_rotation(ctx.rotation,rotation_head,views[condition],gt_uv,gt_size,intrinsics)}
                        for i,row in enumerate(value):
                            focal=ctx.scale.focal(row['K']);gt_z=float((row['T'][:3,:3]@row['keys'][0]+row['T'][:3,3])[2])
                            factor=dict(method=method,seed=seed,arm=canonical,legacy_arm=legacy,condition=condition,rid=row['rid'],sample_id=row['id'],
                                K=row['K'],T_gt=row['T'],anchor=row['keys'][0],uv_gt=row['uv'][0],pred_uv=uv[i],z_m=z[i],size=size[i],R_pred=rotations[i],
                                log_size=output[condition]['log_size'][i],log_shape=output[condition]['log_shape'][i],focal_log=focal,
                                gt_z_m=gt_z,gt_log_size=gtlabels[condition][i,2],gt_log_shape=np.log(gt_z)-focal+gtlabels[condition][i,2],
                                depth_model_uv=output[condition]['uv'][i],center_attention_entropy=centers[condition]['entropy'][i],
                                center_attention_normalized_entropy=centers[condition]['normalized_entropy'][i],center_peak_probability=centers[condition]['peak_probability'][i],
                                depth_attention_entropy=output[condition]['entropy'][i],center_argmax_uv=centers[condition]['argmax_uv'][i],
                                alt_uv_peak=centers[condition]['argmax_uv'][i],alt_uv_local3=centers[condition]['local3_uv'][i],
                                diagnostic_only_fields=['gt_*','R_crop_gt_*'],crop_oracles_diagnostic_only=True)
                            factor.update({name:matrices[i] for name,matrices in crop_oracles.items()});write_line(predstream,factor);counters['predictions']+=1
                        for gain in GAINS:
                            gdepth=predict_raw(depth,gain_views[condition][gain],ctx.scale,intrinsics)
                            gcenter=predict_raw(center,gain_views[condition][gain],ctx.scale,intrinsics)
                            if gain==1:
                                for key in ('uv','z','size','log_size','log_shape'):np.testing.assert_array_equal(gdepth[key],output[condition][key])
                                np.testing.assert_array_equal(gcenter['uv'],uv)
                            for i,row in enumerate(value):
                                write_line(gainstream,dict(method=method,seed=seed,arm=canonical,condition=condition,rid=row['rid'],sample_id=row['id'],gain=gain,
                                    pred_uv=gcenter['uv'][i],depth_model_uv=gdepth['uv'][i],log_size=gdepth['log_size'][i],log_shape=gdepth['log_shape'][i],z_m=gdepth['z'][i],
                                    delta_log_size=gdepth['log_size'][i]-output[condition]['log_size'][i],delta_log_shape=gdepth['log_shape'][i]-output[condition]['log_shape'][i],
                                    delta_z_m=gdepth['z'][i]-z[i],diagnostic_only=True,physical_event_resimulation=False,spatial_support_preserved=True));counters['count_gain']+=1
                    for target in rows:
                        intrinsics=[r['K'] for r in rows[target]]
                        for donor in rows:
                            swap=swapped_global(depth,output[target],output[donor],ctx.scale,intrinsics,ctx.device)
                            if donor==target:
                                for key in ('log_size','log_shape','z'):np.testing.assert_array_equal(swap[key],output[target][key])
                            for i,row in enumerate(rows[target]):
                                write_line(contextstream,dict(method=method,seed=seed,arm=canonical,target_condition=target,global_donor_condition=donor,
                                    rid=row['rid'],sample_id=row['id'],log_size=swap['log_size'][i],log_shape=swap['log_shape'][i],z_m=swap['z'][i],
                                    delta_log_size=swap['log_size'][i]-output[target]['log_size'][i],delta_log_shape=swap['log_shape'][i]-output[target]['log_shape'][i],
                                    delta_z_m=swap['z'][i]-output[target]['z'][i],pred_uv=centers[target]['uv'][i],depth_model_uv=output[target]['uv'][i],
                                    center_change=0.,local_features_unchanged=True,GT_input_used=False,diagnostic_only=True));counters['context_swap']+=1
                    intrinsics=[r['K'] for r in clean];_,clean_z,clean_size=ctx.scale.estimate(depth,clean_tensor,intrinsics)
                    clean_uv=ctx.native.estimate_centers(center,clean_tensor)
                    clean_R=ctx.shared.frozen_rotation(ctx.rotation,rotation_head,clean_tensor,clean_uv,clean_size,intrinsics)
                    clean_labels=ctx.shared.labels(ctx.scale,clean,ctx.vertices,ctx.device).cpu().numpy()
                    clean_score=ctx.shared.score(ctx.rotation,clean,clean_uv,clean_z,clean_size,clean_labels[:,2],clean_R,ctx.vertices,tree,ctx.threshold)
                    compare_summary(clean_score['summary'],run['arms'][legacy]['clean']['summary'])
                    parity.append(dict(method=method,seed=seed,arm=canonical,condition='clean384',scope='native96-rollout4windows each,not matched-normal64',passed=True,score=clean_score))
                    assert state_sha(depth.state_dict())==depth_before==dr['final_state_sha256']
                    assert state_sha(center.state_dict())==center_before==cr['final_state_sha256']
                    assert state_sha(rotation_head.state_dict())==rotation_state
                    del depth,center;bound()
                    print('FACTOR_CELL',method,seed,canonical,'matched64/parity/gain/context complete',flush=True)
                del rotation_head
            del views,gain_views,clean_tensor
    assert counters==dict(predictions=3456,count_gain=10368,context_swap=10368) and len(native_scores)==54 and len(parity)==54
    write_json(args.out/'native_scores.json',native_scores);write_json(args.out/'native_parity.json',parity)
    assert all(sha(r['path'])==r['sha256'] for r in ctx.source_records);bound()
    assets=dict(vertices=dict(path=str(vp),sha256=sha(vp),size_bytes=vp.stat().st_size),
                keypoints=dict(path=str(assetdir/'keypoints.npy'),sha256=sha(assetdir/'keypoints.npy'),size_bytes=(assetdir/'keypoints.npy').stat().st_size),diameter_m=ctx.diameter)
    artifacts=[dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size) for p in (prediction_path,gain_path,context_path,args.out/'native_scores.json',args.out/'native_parity.json')]
    write_json(args.out/'completion.json',dict(schema='pose_factor_forward_export_v1',status='complete',optimizer_updates=0,
        real_data_used=False,GT_used_only_in_scores_and_explicit_oracle_diagnostics=True,models_and_sources_unchanged=True,
        contract_sha256=sha(args.contract),source_script_sha256=sha(Path(__file__)),source_records=ctx.source_records,
        assets=assets,artifacts=artifacts,counters=counters,native_parity_checks=len(parity),native_parity_passed=True,
        matched_normal_scope='4 aligned independent relit validation trajectories,64windows; distinct from nativeclean384',
        gains=list(GAINS),gain_stage='raw ON/OFF/TOTAL counts before original log/clip/native resize; no physical event regeneration',
        context_scope='local feature fixed,only same-geometry other-brightness global pooled feature substituted; not deployable',
        elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
