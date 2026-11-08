"""Fixed BAF2ms synthetic-only CPU center diagnostic; zero optimizer updates."""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import numpy as np
import torch
from baf import EpisodeEncoder

DIAGNOSTIC_MANIFEST='9e604e49ae861ab3831d641cd8c6d57d506017b060ebd30ee7a0b23f80e8771b'
EXPORT_SHA='fdd5d96dfd6191fb73923cfa06699d5995ab0750a8e1579223ef8a1b3b5c61cd'
FORWARD_COMPLETION_SHA='fea5095c363b2d4a02602c4fe2dec6144058564be877c55caf0ef1dc24631093'
RAW_PARITY_ATOL=2e-6
CONDITIONS=('normal','half','eighth')

def need(value,message):
    if not value: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2,allow_nan=False)

def record(path):
    p=Path(path);return dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p))

def load_export(directory,contract_path):
    manifest=directory/'source_manifest.json'
    need(sha(manifest)==DIAGNOSTIC_MANIFEST,'Unregistered diagnostic source manifest')
    files=read(manifest)
    for name,digest in files.items():
        need(Path(name).name==name and sha(directory/name)==digest,'Frozen diagnostic source changed: '+name)
    need(files['export_pose_factors.py']==EXPORT_SHA and sha(contract_path)==files['contract.json'],'Frozen contract/source mismatch')
    spec=importlib.util.spec_from_file_location('baf_bound_export',directory/'export_pose_factors.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,[record(manifest)]+[record(directory/name) for name in files]

def load_reference(directory,contract_sha):
    completion_path=directory.parent/'export/completion.json'
    need(sha(completion_path)==FORWARD_COMPLETION_SHA,'Forward reference receipt changed')
    completion=read(completion_path)
    need(completion['schema']=='pose_factor_forward_export_v1' and completion['status']=='complete','Incomplete reference export')
    need(completion['optimizer_updates']==0 and completion['real_data_used'] is False and completion['native_parity_passed'] is True,'Reference scope invalid')
    need(completion['contract_sha256']==contract_sha and completion['source_script_sha256']==EXPORT_SHA,'Reference binding invalid')
    item=next(v for v in completion['artifacts'] if Path(v['path']).name=='predictions.jsonl')
    need(sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['size_bytes'],'Predictions SHA/size mismatch')
    rows={}
    with Path(item['path']).open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['arm']=='C1':
                key=(r['method'],r['seed'],r['condition'],r['sample_id'])
                need(key not in rows,'Duplicate reference window');rows[key]=r
    need(len(rows)==1152,'Complete C1 reference inventory required')
    return rows,[record(completion_path),item]

def select_indices(rows,rids):
    selected=[]
    for rid in rids:
        indices=sorted((i for i,r in enumerate(rows) if r['rid']==rid),key=lambda i:rows[i]['id'])
        need(len(indices)>=4,'Fewer than4 TRAIN windows')
        selected.extend(indices[i] for i in np.linspace(0,len(indices)-1,4,dtype=int))
    need(len(selected)==32 and len(set(selected))==32,'Exactly32 TRAIN windows required')
    return selected

def prepare(ctx,c,method,split):
    if split=='train':
        manifest=read(ctx.R/'pose_bank/bank_manifest.json')
        all_rids=sorted(r.get('rollout_id',r.get('rollout')) for r in manifest['rollouts'] if r['split']=='train')
        completion=read(Path(c['radiance_train'])/'completion.json')
        need(len(all_rids)==96 and set(all_rids)==set(completion['selected']) and completion['status']=='complete','All96 original TRAIN trajectories required')
        rids=all_rids[:8];normal=ctx.R/'pose_bank';relit=Path(c['radiance_train'])
    else:
        normal=ctx.D/'pose_bank';relit=Path(c['radiance_validation'])
        rids=sorted(p.stem for p in (relit/'half/pose_bank/physical').glob('*.h5'))
        need(len(rids)==4 and all(r.startswith('validation_') for r in rids),'Exactly4 independent VAL trajectories required')
    result={};identity=None
    for condition,bank in dict(normal=normal,**{s:relit/s/'pose_bank' for s in ('half','eighth')}).items():
        files=[bank/'physical'/(rid+'.h5') for rid in rids]
        callback=EpisodeEncoder(ctx.encode)
        raw=ctx.fresh.adapter(bank,files,method,callback)
        need(len(raw)==len(callback.windows),'Adapter skipped/added encoded windows')
        indices=select_indices(raw,rids) if split=='train' else list(range(len(raw)))
        need(len(indices)==(32 if split=='train' else 64),'Wrong cohort size')
        rows=[raw[i] for i in indices];windows=[callback.windows[i] for i in indices]
        # Independent ordinary adapter call verifies unchanged RAW pathway exactly.
        ordinary=ctx.fresh.adapter(bank,files,method,ctx.encode)
        need(ctx.native.identity(raw)==ctx.native.identity(ordinary),'Raw adapter identity changed')
        need(all(np.array_equal(a['image'],b['image']) for a,b in zip(raw,ordinary)),'Raw image encoding changed')
        current=ctx.native.identity(rows)
        if identity is None: identity=current
        else: need(current==identity,'Relighting changed physical windows/GT')
        filtered=[dict(row,image=win['filtered_image']) for row,win in zip(rows,windows)]
        raw_tensor=ctx.shared.tensors(rows,ctx.device);filtered_tensor=ctx.shared.tensors(filtered,ctx.device)
        need(raw_tensor.shape==filtered_tensor.shape==(len(rows),3,128,128),'Native area128 input required')
        need(torch.isfinite(raw_tensor).all() and torch.isfinite(filtered_tensor).all(),'Nonfinite input')
        result[condition]=(rows,windows,raw_tensor,filtered_tensor)
        print('PREPARED',method,split,condition,len(rows),'episodes',len(callback.episodes),flush=True)
        del callback,ordinary,raw,filtered;gc.collect()
    return result

def check_reference(raw_uv,rows,reference,method,seed,condition):
    maximum=0.
    for uv,row in zip(raw_uv,rows):
        r=reference[(method,seed,condition,row['id'])]
        np.testing.assert_array_equal(np.asarray(r['uv_gt'],np.float32),row['uv'][0])
        need(r['rid']==row['rid'],'Reference rollout differs')
        error=float(np.max(np.abs(uv-np.asarray(r['pred_uv']))));maximum=max(maximum,error)
        need(np.isfinite(error) and error<=RAW_PARITY_ATOL,'CPU raw C1 UV differs from frozen GPU reference: '+str(error))
    return maximum

def predict_three(estimate,model,raw,filtered):
    original_raw=raw.clone();original_filtered=filtered.clone();zero=torch.zeros_like(raw)
    with torch.inference_mode(): values=tuple(estimate(model,x) for x in (raw,filtered,zero))
    need(torch.equal(raw,original_raw) and torch.equal(filtered,original_filtered),'Forward modified input tensor')
    need(torch.count_nonzero(zero)==0,'Zero-input negative control changed')
    return values

def summarize(rows):
    groups={}
    for r in rows: groups.setdefault((r['method'],r['seed'],r['split'],r['condition']),[]).append(r)
    output=[]
    for key,items in sorted(groups.items()):
        rollouts={rid:[v for v in items if v['rid']==rid] for rid in sorted({v['rid'] for v in items})}
        macro={name:float(np.mean([np.mean([v[name] for v in values]) for values in rollouts.values()]))
               for name in ('raw_center_error_px','filtered_center_error_px','zero_center_error_px','center_shift_px')}
        raw=sum(v['raw_events'] for v in items);retained=sum(v['retained_events'] for v in items)
        output.append(dict(zip(('method','seed','split','condition'),key),windows=len(items),rollouts=len(rollouts),
                           **macro,raw_events=raw,retained_events=retained,event_retention_ratio=retained/raw if raw else None))
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--diagnostic-sources',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        E,source_records=load_export(args.diagnostic_sources,args.contract);c=read(args.contract)
        ctx=E.load_context(c,check_gpu=False);need(ctx.device.type=='cpu' and not torch.cuda.is_initialized(),'CPU-only diagnostic required')
        reference,reference_records=load_reference(args.diagnostic_sources,sha(args.contract));source_records+=reference_records
        source_records.extend(record(Path(__file__).parent/n) for n in ('baf.py','diagnose_centers.py','PROTOCOL.md'))
        rows_out=[];parity=[];models=[];identities={}
        for method in c['methods']:
            data={split:prepare(ctx,c,method,split) for split in ('train','validation')}
            for split in data:
                identity=ctx.native.identity(data[split]['normal'][0])
                if split in identities: need(identity==identities[split],'Cross-generator window/GT differs')
                identities[split]=identity
            for seed in c['seeds']:
                model,training_record,path=E.load_branch(ctx,method,seed,'physical_supervised','center')
                before=E.state_sha(model.state_dict());need(before==training_record['final_state_sha256'],'C1 model binding differs')
                for split,conditions in data.items():
                    for condition,(rows,windows,raw,filtered) in conditions.items():
                        need(time.monotonic()-start<1200,'CPU diagnostic exceeded1200s bound')
                        raw_uv,filtered_uv,zero_uv=predict_three(ctx.native.estimate_centers,model,raw,filtered)
                        need(raw_uv.shape==filtered_uv.shape==zero_uv.shape==(len(rows),2),'Wrong prediction shape')
                        need(all(np.isfinite(v).all() for v in (raw_uv,filtered_uv,zero_uv)),'Nonfinite prediction')
                        if split=='validation':
                            maximum=check_reference(raw_uv,rows,reference,method,seed,condition)
                            parity.append(dict(method=method,seed=seed,condition=condition,max_abs_normalized_uv=maximum,passed=True))
                        for i,(row,win) in enumerate(zip(rows,windows)):
                            gt=np.asarray(row['uv'][0]);pixels=np.array([346.,260.])
                            value=dict(method=method,seed=seed,arm='C1',split=split,condition=condition,rid=row['rid'],sample_id=row['id'],
                                raw_pred_uv=raw_uv[i].tolist(),filtered_pred_uv=filtered_uv[i].tolist(),zero_pred_uv=zero_uv[i].tolist(),uv_gt=gt.tolist(),
                                raw_center_error_px=float(np.linalg.norm((raw_uv[i]-gt)*pixels)),
                                filtered_center_error_px=float(np.linalg.norm((filtered_uv[i]-gt)*pixels)),
                                zero_center_error_px=float(np.linalg.norm((zero_uv[i]-gt)*pixels)),
                                raw_input_nonzero_values=int(torch.count_nonzero(raw[i])),
                                filtered_input_nonzero_values=int(torch.count_nonzero(filtered[i])),
                                center_shift_px=float(np.linalg.norm((filtered_uv[i]-raw_uv[i])*pixels)),
                                **{k:win[k] for k in ('raw_events','retained_events','start_s','end_s')})
                            value['event_retention_ratio']=value['retained_events']/value['raw_events'] if value['raw_events'] else None
                            rows_out.append(value)
                need(E.state_sha(model.state_dict())==before,'C1 model state/buffers changed')
                models.append(dict(method=method,seed=seed,path=str(path),state_sha256=before,unchanged=True))
                del model;gc.collect();print('FINISHED_C1',method,seed,flush=True)
            del data;gc.collect()
        need(len(rows_out)==1728 and len(parity)==18 and len(models)==6,'Incomplete fixed inventory')
        all_sources=source_records+ctx.source_records
        need(all(sha(r['path'])==r['sha256'] for r in all_sources),'Source/model bytes changed')
        need(not torch.cuda.is_initialized() and time.monotonic()-start<1200,'CPU/runtime boundary violated')
        prediction_path=args.out/'center_diagnostic.jsonl'
        with prediction_path.open('x',encoding='utf-8') as stream:
            for row in rows_out: stream.write(json.dumps(row,allow_nan=False)+'\n')
        write(args.out/'summary.json',summarize(rows_out))
        write(args.out/'completion.json',dict(schema='pose_baf_center_diagnostic_v2',status='complete',optimizer_updates=0,
            real_data_used=False,GPU_used=False,filter_threshold_us=2000,initial_time_surface=0,polarity_used=False,
            causal_whole_episode=True,GT_filter_input=False,models_unchanged=True,raw_uv_parity_atol=RAW_PARITY_ATOL,
            zero_input_control=True,zero_control_uses_GT=False,zero_control_scope='same frozen C1 center model and exact batch shape; no event information',
            rows=1728,parity=parity,models=models,source_records=all_sources,contract_sha256=sha(args.contract),
            artifacts=[record(prediction_path),record(args.out/'summary.json')],elapsed_seconds=time.monotonic()-start))
    except BaseException as error:
        write(args.out/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),optimizer_updates=0,elapsed_seconds=time.monotonic()-start))
        raise

if __name__=='__main__':main()
