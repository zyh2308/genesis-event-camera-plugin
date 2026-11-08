"""Frozen SOURCE-INIT teacher targets, RAW versus BAF; CPU, zero updates."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,statistics,sys,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
BAF_MANIFEST='1383a156058347ad673ded37760dd5529a019c287f06a8afb4441e5fb7853adf'
DIAGNOSTIC_MANIFEST='9e604e49ae861ab3831d641cd8c6d57d506017b060ebd30ee7a0b23f80e8771b'

def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def record(path):
    p=Path(path);return dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size)
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def checked_sources(directory,digest):
    mp=directory/'source_manifest.json';need(sha(mp)==digest,'Source manifest differs');records=[record(mp)]
    for name,expected in read(mp).items():
        need(Path(name).name==name and sha(directory/name)==expected,'Frozen source differs: '+name);records.append(record(directory/name))
    return records
def metrics(raw,filtered,gt):
    need(all(len(v)==2 and all(math.isfinite(float(x)) for x in v) for v in (raw,filtered,gt)),'Finite normalized UV pairs required')
    distance=lambda a,b:math.hypot((float(a[0])-float(b[0]))*346.,(float(a[1])-float(b[1]))*260.)
    return dict(raw_teacher_center_error_px=distance(raw,gt),baf_teacher_center_error_px=distance(filtered,gt),teacher_target_shift_px=distance(raw,filtered))
def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['method'],r['seed'],r['split'],r['condition']),[]).append(r)
    out=[]
    for key,items in sorted(groups.items()):
        rids=sorted({r['rid'] for r in items});macro={k:statistics.mean(statistics.mean(r[k] for r in items if r['rid']==rid) for rid in rids)
            for k in ('raw_teacher_center_error_px','baf_teacher_center_error_px','teacher_target_shift_px')}
        out.append(dict(zip(('method','seed','split','condition'),key),**macro,windows=len(items),rollouts=len(rids)))
    return out
def load_teacher(ctx,E,method,seed):
    import torch
    run=next(r for r in ctx.completion['results'] if (r['method'],r['seed'])==(method,seed))
    path=Path(run['localization_path']);E.remember(ctx.source_records,path,run['localization_sha256'])
    state=torch.load(path,map_location='cpu',weights_only=True)
    source=next(r for r in ctx.receipt['records'] if (r['method'],r['seed'],r['arm'],r['branch'])==(method,seed,'physical_supervised','center'))
    digest=E.state_sha(state);need(digest==source['source_state_sha256'],'Teacher must equal original C1 SOURCE initialization, not C1 endpoint')
    model=ctx.scale.CenterScale().to(ctx.device).eval();model.load_state_dict(state,strict=True);model.requires_grad_(False)
    need(E.state_sha(model.state_dict())==digest,'Teacher load changed state');return model,digest,path
def remember_inputs(ctx,E,c,method,data):
    E.remember(ctx.source_records,Path(c['radiance_train'])/'completion.json')
    for split,conditions in data.items():
        for condition,(rows,_,_,_) in conditions.items():
            bank=(ctx.R if split=='train' else ctx.D)/'pose_bank' if condition=='normal' else Path(c['radiance_train' if split=='train' else 'radiance_validation'])/condition/'pose_bank'
            E.remember(ctx.source_records,bank/'events'/method/'manifest.json')
            if (bank/'bank_manifest.json').is_file():E.remember(ctx.source_records,bank/'bank_manifest.json')
            for rid in sorted({r['rid'] for r in rows}):
                for path in (bank/'physical'/(rid+'.h5'),bank/'events'/method/(rid+'.npy')):E.remember(ctx.source_records,path)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('contract','diagnostic-sources','baf-sources','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'Unique absolute output required');a.out.mkdir();start=time.monotonic()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        sources=checked_sources(a.baf_sources,BAF_MANIFEST)+checked_sources(a.diagnostic_sources,DIAGNOSTIC_MANIFEST)
        sys.path.insert(0,str(a.baf_sources));spec=importlib.util.spec_from_file_location('teacher_target_bound_baf',a.baf_sources/'diagnose_centers.py')
        D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
        need(Path(sys.modules['baf'].__file__).resolve()==(a.baf_sources/'baf.py').resolve(),'BAF module shadowed')
        E,extra=D.load_export(a.diagnostic_sources,a.contract);sources+=extra;c=read(a.contract);ctx=E.load_context(c,check_gpu=False)
        need(ctx.device.type=='cpu' and not torch.cuda.is_initialized(),'CPU-only required');sources.extend(record(Path(__file__).parent/n) for n in ('probe_teacher_targets.py','PROTOCOL.md'))
        rows_out=[];models=[]
        for method in c['methods']:
            data={split:D.prepare(ctx,c,method,split) for split in ('train','validation')}
            remember_inputs(ctx,E,c,method,data)
            for seed in c['seeds']:
                model,before,path=load_teacher(ctx,E,method,seed)
                for split,conditions in data.items():
                    for condition,(rows,windows,raw,filtered) in conditions.items():
                        need(time.monotonic()-start<1200,'Fixed1200s runtime bound exceeded')
                        raw_before=raw.clone();filtered_before=filtered.clone()
                        with torch.inference_mode():u=ctx.native.estimate_centers(model,raw);v=ctx.native.estimate_centers(model,filtered)
                        need(u.shape==v.shape==(len(rows),2) and np.isfinite(u).all() and np.isfinite(v).all(),'Invalid teacher output')
                        need(torch.equal(raw,raw_before) and torch.equal(filtered,filtered_before),'Teacher forward changed input')
                        for i,(row,window) in enumerate(zip(rows,windows)):
                            gt=row['uv'][0].tolist();raw_uv=u[i].tolist();baf_uv=v[i].tolist()
                            rows_out.append(dict(method=method,seed=seed,split=split,condition=condition,rid=row['rid'],sample_id=row['id'],
                                model_role='frozen_source_init_teacher_NOT_C1_endpoint',raw_teacher_uv=raw_uv,baf_teacher_uv=baf_uv,uv_gt=gt,
                                **metrics(raw_uv,baf_uv,gt),**{k:window[k] for k in ('start_s','end_s','raw_events','retained_events')}))
                need(E.state_sha(model.state_dict())==before,'Teacher state/buffers changed')
                models.append(dict(method=method,seed=seed,role='source_init_teacher',path=str(path),state_sha256=before,unchanged=True))
                del model;print('TEACHER_DONE',method,seed,flush=True)
            del data
        need(len(rows_out)==1728 and len(models)==6,'Incomplete fixed matrix')
        sources+=ctx.source_records;need(all(sha(r['path'])==r['sha256'] for r in sources),'Frozen source/model bytes changed')
        need(not torch.cuda.is_initialized() and time.monotonic()-start<1200,'CPU/runtime boundary violated')
        pred=a.out/'teacher_targets.jsonl'
        with pred.open('x') as f:
            for row in rows_out:f.write(json.dumps(row,allow_nan=False)+'\n')
        write(a.out/'summary.json',summarize(rows_out))
        write(a.out/'completion.json',dict(schema='pose_source_teacher_target_diagnostic_v1',status='complete',optimizer_updates=0,
            model_role='original_source_init_teacher_NOT_C1_NOT_BAF_candidate',rows=1728,models=models,source_records=sources,
            real_data_used=False,GPU_used=False,teacher_modified=False,GT_used_only_in_diagnostic_errors=True,contract_sha256=sha(a.contract),
            filter_threshold_us=2000,causal_whole_episode=True,raw_encode_exact_parity=True,
            C1_endpoint_UV_parity='not_applicable_different_weights',artifacts=[record(pred),record(a.out/'summary.json')],elapsed_seconds=time.monotonic()-start))
    except BaseException as error:
        write(a.out/'failure.json',dict(status='failed',optimizer_updates=0,error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start));raise

if __name__=='__main__':main()
