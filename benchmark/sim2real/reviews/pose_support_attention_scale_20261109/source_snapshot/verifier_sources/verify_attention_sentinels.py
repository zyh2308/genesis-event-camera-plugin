"""Independent fixed k=12 attention-map audit. Worker only; never trains."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,sys,time,traceback
from pathlib import Path

DIAGNOSTIC_SHA='9e604e49ae861ab3831d641cd8c6d57d506017b060ebd30ee7a0b23f80e8771b'
BAF_SHA='1383a156058347ad673ded37760dd5529a019c287f06a8afb4441e5fb7853adf'
SUPPORT_SHA='203525b580a0d520e6f15d7ea2e9acdac029d3c78e11d83c8d514840441e592f'
ATTENTION_SHA='24121b23d94a6ce6a5112cff796deced796b86be6c5b83f17c15884bd31e052f'
GPU4='GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3'
K_SENTINEL=12
UV_ATOL=2e-6
MASS_ATOL=2e-6

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def record(p):
    p=Path(p);return dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size)
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def verify_bundle(directory,expected):
    need(expected is not None and sha(directory/'source_manifest.json')==expected,'Frozen source manifest mismatch')
    records=[record(directory/'source_manifest.json')]
    for name,digest in read(directory/'source_manifest.json').items():
        need(Path(name).name==name and sha(directory/name)==digest,'Pinned source changed: '+name);records.append(record(directory/name))
    return records
def safe_cpu_helper_import(torch,name,path,device):
    """Restore only the queue-assigned environment in this worker process."""
    assigned=os.environ.get('CUDA_VISIBLE_DEVICES')
    need(not torch.cuda.is_initialized(),'CUDA must remain uninitialized before CPU-helper import')
    if device=='cuda':need(assigned in (GPU4,'4'),'Only original GPU4 assignment allowed')
    else:need(assigned=='','CPU worker must explicitly hide CUDA')
    try:
        module=load(name,path)
        need(not torch.cuda.is_initialized(),'CPU helper unexpectedly initialized CUDA')
    finally:
        if not torch.cuda.is_initialized():
            if assigned is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=assigned
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==assigned,'Assigned device changed during helper import')
    return module

def direct_regions(path,rid,row):
    """No support-module numeric helpers. Dynamic target slot and native masks."""
    import h5py,numpy as np
    with h5py.File(path,'r') as f:
        g=f['episodes'][rid];target=g['target_pose'][:];objects=g['object_to_world'][:];ids=g['object_segmentation_link_ids'][:]
        need(target.shape==(21,4,4) and objects.shape==(21,len(ids),4,4),'Physical pose inventory changed')
        slots=[i for i in range(len(ids)) if np.array_equal(target,objects[:,i])]
        need(len(slots)==1 and len(np.unique(ids))==len(ids),'Target object identity ambiguous')
        slot=slots[0];target_id=int(ids[slot]);seg=g['segmentation_link_idx'][:];depth=g['depth_m'][:];counts=g['target_visible_pixels'][:];times=g['frame_timestamps_s'][:]
        need(seg.shape==depth.shape==(21,260,346) and times.shape==(21,) and np.allclose(np.diff(times),.01,atol=1e-7,rtol=0),'Physical image/cadence changed')
        masks=seg==target_id
        need(np.array_equal(masks.sum((1,2)),counts),'All21 visible target counts differ')
        need(np.isfinite(depth[masks]).all() and (depth[masks]>0).all(),'Invalid target-visible depth')
        need(np.array_equal(row['K'],g['camera_intrinsics_K_px'][:]) and np.array_equal(row['keys'],g['cad_keypoints'][:]),'Sentinel camera/CAD changed')
        end=K_SENTINEL+1;T=np.linalg.inv(g['camera_to_world_cv'][end])@target[end]
        need(np.array_equal(T,row['T']),'Sentinel physical pose endpoint changed')
        return dict(startpoint=masks[K_SENTINEL].copy(),endpoint=masks[end].copy(),endpoint_union_approx=(masks[K_SENTINEL]|masks[end]).copy()),dict(target_slot=slot,target_id=target_id,start_s=float(times[K_SENTINEL]),end_s=float(times[end]),physical_source=record(path),visibility='exact at frame endpoints; union approximation, not per-event visibility')

def independent_fraction(mask):
    import cv2,numpy as np,torch
    encoded=cv2.resize(mask.astype(np.float32),(416,416),interpolation=cv2.INTER_LINEAR)
    x=torch.from_numpy(encoded)[None,None]
    x=torch.nn.functional.interpolate(x,(128,128),mode='area')
    a=torch.nn.functional.interpolate(x,(16,16),mode='area')[0,0].numpy()
    return a,float(a.mean()),float(mask.astype(np.float32).mean())

def member(mask,uv):
    import numpy as np
    if not mask.any():return None
    x,y=np.asarray(uv)*[346.,260.]
    return bool(mask[int(math.floor(y)),int(math.floor(x))]) if 0<=x<346 and 0<=y<260 else False

def independent_metrics(mask,fraction,attention,uv,peak_uv):
    import numpy as np
    frac,area,native_area=fraction
    available=bool(mask.any() and area>0)
    mass=float(np.sum(attention.astype(np.float64)*frac)) if available else None
    return dict(region_attention_mass=mass,region_fraction_native=native_area,region_grid_area_fraction=area,
        region_mass_enrichment=mass/area if available else None,prediction_in_region=member(mask,uv),peak_in_region=member(mask,peak_uv),available=available)

def compare_region(actual,reference):
    errors={};passed=True
    for key in ('available','prediction_in_region','peak_in_region'):
        same=actual[key]==reference[key];passed &= same;errors[key]=dict(actual=actual[key],reference=reference[key],passed=same)
    for key in ('region_fraction_native','region_grid_area_fraction'):
        delta=abs(actual[key]-reference[key]);same=delta<=1e-12;passed &= same;errors[key]=dict(max_abs=delta,atol=1e-12,passed=same)
    for key in ('region_attention_mass','region_mass_enrichment'):
        x,y=actual[key],reference[key]
        if x is None or y is None:
            same=x is None and y is None;errors[key]=dict(actual=x,reference=y,passed=same)
        else:
            atol=MASS_ATOL if key=='region_attention_mass' else MASS_ATOL/actual['region_grid_area_fraction']
            delta=abs(x-y);same=delta<=atol;errors[key]=dict(max_abs=delta,atol=atol,passed=same)
        passed &= same
    return bool(passed),errors

def original_chunks(model,images,torch):
    """Keep producer64 ordering/batch32; callers verify only fixed sentinels."""
    need(images.device==next(model.parameters()).device,'Original native input/model device mismatch')
    for offset in range(0,len(images),32):
        block=images[offset:offset+32];before=block.clone();captured={}
        def feature_hook(_m,_i,out):captured['features']=out.detach().clone()
        def logit_hook(_m,_i,out):captured['logits']=out.detach().clone().flatten(1)
        hooks=[model.features.register_forward_hook(feature_hook),model.center.register_forward_hook(logit_hook)]
        try:native=model(block)
        finally:
            for hook in hooks:hook.remove()
        need(torch.equal(block,before),'Original forward changed input')
        need(captured['features'].shape==(len(block),64,16,16) and captured['logits'].shape==(len(block),256) and torch.equal(native[0],captured['logits']),'Unmodified original native forward/hook shapes differ')
        probs=torch.softmax(captured['logits'],dim=1);uv=probs@model.grid;peak=model.grid[probs.argmax(1)]
        need(torch.equal(uv,native[1]),'Independent native softmax/grid UV differs within same forward')
        need(all(torch.isfinite(x).all() for x in (probs,uv,peak)),'Nonfinite native output')
        yield offset,len(block),probs,uv,peak

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('full-completion','attention-sources','diagnostic-sources','baf-sources','support-sources','contract','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--full-completion-sha256',required=True);p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=p.parse_args()
    need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'Unique absolute verifier output required');a.out.mkdir();start=time.monotonic()
    def bound():need(time.monotonic()-start<1200,'Fixed1200s independent verifier bound exceeded')
    try:
        if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
        import cv2,numpy as np,torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        sources=verify_bundle(Path(__file__).resolve().parent,a.source_manifest_sha256)
        for directory,digest in [(a.attention_sources,ATTENTION_SHA),(a.diagnostic_sources,DIAGNOSTIC_SHA),(a.baf_sources,BAF_SHA),(a.support_sources,SUPPORT_SHA)]:sources+=verify_bundle(directory,digest)
        need(sha(a.full_completion)==a.full_completion_sha256,'Actual full completion byte binding differs');full=read(a.full_completion)
        need(full['status']=='complete' and full['mode']=='full' and full['source_manifest_sha256']==ATTENTION_SHA and full['support_source_manifest_sha256']==SUPPORT_SHA and full['contract_sha256']==sha(a.contract),'Actual complete fixed full diagnostic required')
        need(full['diagnostic_rows']==9216 and full['raw_hybrid_rows']==6144 and full['optimizer_updates']==0 and full['real_data_used'] is False and full['GT_modified_forward'] is False,'Full matrix/isolation differs')
        item=next(r for r in full['artifacts'] if Path(r['path']).name=='per_window.jsonl');file=Path(item['path'])
        need(file.resolve()==(a.full_completion.parent/'per_window.jsonl').resolve() and sha(file)==item['sha256'] and file.stat().st_size==item['size_bytes'],'Full per-window artifact is pinned')
        reference={}
        for line in file.open():
            r=json.loads(line)
            if int(r['sample_id'].rsplit('/',1)[1])==K_SENTINEL:
                key=tuple(r[k] for k in ('method','seed','arm','branch','input_view','condition','rid','sample_id'))
                need(key not in reference,'Duplicate fixed sentinel row');reference[key]=r
        need(len(reference)==576,'All576 prespecified k12 reference rows required');sources +=[record(a.full_completion),record(file),record(a.contract)]
        sys.path.insert(0,str(a.baf_sources));D=safe_cpu_helper_import(torch,'sentinel_frozen_baf_prepare',a.baf_sources/'diagnose_centers.py',a.device)
        need(Path(sys.modules['baf'].__file__).resolve()==(a.baf_sources/'baf.py').resolve(),'BAF helper shadowed')
        E,extra=D.load_export(a.diagnostic_sources,a.contract);sources+=extra;c=read(a.contract);ctx=E.load_context(c,check_gpu=a.device=='cuda')
        need(ctx.network_size==416,'Native416 encoder required');maps=[];map_index=[];checks=[];models=[];mask_bank={};mask_index=[];total_forward_images=0;total_forward_calls=0
        with torch.inference_mode():
            for method in c['methods']:
                bound();data=D.prepare(ctx,c,method,'validation');need(set(data)=={'normal','half','eighth'},'Three fixed lights required');selected={};regions={};region_meta={}
                for condition,(rows,_windows,raw,baf) in data.items():
                    need(len(rows)==64 and len({r['rid'] for r in rows})==4,'Complete fixed original64 window data required')
                    indices=[i for i,r in enumerate(rows) if int(r['id'].rsplit('/',1)[1])==K_SENTINEL]
                    need(len(indices)==4 and len({rows[i]['rid'] for i in indices})==4,'Exactly k12 at each original4 trajectories')
                    selected[condition]=[(i,rows[i]) for i in indices]
                    bank=ctx.D/'pose_bank' if condition=='normal' else Path(c['radiance_validation'])/condition/'pose_bank'
                    for rid in sorted({rows[i]['rid'] for i in indices}):
                        for path in (bank/'physical'/(rid+'.h5'),bank/'events'/method/(rid+'.npy'),bank/'events'/method/'manifest.json'):E.remember(ctx.source_records,path)
                    for i,row in selected[condition]:
                        masks,meta=direct_regions(bank/'physical'/(row['rid']+'.h5'),row['rid'],row)
                        regions[condition,row['id']]={name:(mask,independent_fraction(mask)) for name,mask in masks.items()}
                        region_meta[condition,row['id']]=meta
                        for name,mask in masks.items():
                            digest=hashlib.sha256(mask.tobytes()).hexdigest();mask_bank.setdefault(digest,mask)
                            mask_index.append(dict(method=method,condition=condition,rid=row['rid'],sample_id=row['id'],kind=name,sha256=digest,**meta))
                for seed in c['seeds']:
                    for arm,legacy in [('C0','clean_control'),('C1','physical_supervised')]:
                        for branch in ('center','depth'):
                            bound();model,_,path=E.load_branch(ctx,method,seed,legacy,branch);model.requires_grad_(False);before=E.state_sha(model.state_dict())
                            original_model=next(r for r in full['model_records'] if (r['method'],r['seed'],r['arm'],r['branch'])==(method,seed,arm,branch))
                            need(original_model['state_sha256']==before and original_model['path']==str(path) and original_model['unchanged'],'Sentinel must use exactly the full-run frozen model')
                            for condition,(rows,_windows,raw,baf) in data.items():
                                for view,images in [('raw',raw),('baf',baf)]:
                                    for offset,count,probs,uv,peak in original_chunks(model,images,torch):
                                        bound();total_forward_calls+=1;total_forward_images+=count
                                        for index,row in selected[condition]:
                                            if not offset<=index<offset+count:continue
                                            n=index-offset
                                            key=(method,seed,arm,branch,view,condition,row['rid'],row['id']);old=reference[key];attention=probs[n].cpu().numpy().reshape(16,16);puv=uv[n].cpu().numpy();pk=peak[n].cpu().numpy()
                                            meta=region_meta[condition,row['id']]
                                            need(old['target_identity']['object_slot']==meta['target_slot'] and old['target_identity']['segmentation_id']==meta['target_id'] and old['support']['start_s']==meta['start_s'] and old['support']['end_s']==meta['end_s'],'Independent target identity and native time window must match full record')
                                            duv=float(np.max(np.abs(puv-np.asarray(old['pred_uv']))));dpeak=float(np.max(np.abs(pk-np.asarray(old['peak_uv']))));passed=duv<=UV_ATOL and dpeak==0;rr={}
                                            for name,(mask,fraction) in regions[condition,row['id']].items():
                                                actual=independent_metrics(mask,fraction,attention,puv,pk);ok,errors=compare_region(actual,old['attention'][name]);passed &= ok;rr[name]=errors
                                            map_index.append(dict(index=len(maps),method=method,seed=seed,arm=arm,branch=branch,input_view=view,condition=condition,rid=row['rid'],sample_id=row['id']))
                                            maps.append(attention.astype(np.float32,copy=True));checks.append(dict(**map_index[-1],passed=bool(passed),max_abs_uv=duv,UV_atol=UV_ATOL,max_abs_peak_uv=dpeak,regions=rr))
                            need(E.state_sha(model.state_dict())==before,'Frozen model/buffers changed');models.append(dict(method=method,seed=seed,arm=arm,branch=branch,path=str(path),state_sha256=before,unchanged=True));del model
                del data
        need(len(maps)==len(checks)==576 and len(models)==24 and total_forward_images==9216 and total_forward_calls==288,'Incomplete fixed producer-forward/sentinel inventory');sources+=ctx.source_records;need(all(sha(r['path'])==r['sha256'] for r in sources),'Original source/data/model bytes changed');bound()
        np.savez_compressed(a.out/'attention_maps_k12.npz',attention=np.stack(maps));write(a.out/'attention_map_index.json',map_index)
        np.savez_compressed(a.out/'native_region_masks_deduplicated.npz',**{'m_'+d:m for d,m in mask_bank.items()});write(a.out/'native_region_mask_index.json',mask_index)
        write(a.out/'sentinel_comparisons.json',dict(checks=checks))
        passed=all(r['passed'] for r in checks)
        write(a.out/'completion.json',dict(schema='independent_attention_k12_verifier_v1',status='passed' if passed else 'failed',sentinel_k=12,attention_maps=576,original_full_rows=9216,total_forward_images=total_forward_images,total_forward_calls=total_forward_calls,verified_attention_maps=576,producer_batch_size=32,scope='original64/batch32 forward; only fixed one window per trajectory verified/saved; not all9216 attention maps audited',
            methods=c['methods'],seeds=c['seeds'],models=models,passed_cases=sum(r['passed'] for r in checks),failed_cases=sum(not r['passed'] for r in checks),
            thresholds=dict(uv_max_abs=UV_ATOL,mass_abs=MASS_ATOL,enrichment_abs='mass_abs / region_area',region_fraction_abs=1e-12,peak_membership='exact discrete match'),
            full_completion_sha256=sha(a.full_completion),source_records=sources,source_script_sha256=sha(__file__),source_manifest_sha256=a.source_manifest_sha256,device=a.device,optimizer_updates=0,real_data_used=False,GT_inputs_to_model=False,
            numeric_oracle_uses_support_summarize_attention=False,all_model_and_source_unchanged=True,elapsed_seconds=time.monotonic()-start,artifacts=[record(p) for p in a.out.iterdir() if p.is_file()]))
        if not passed:raise SystemExit(1)
    except BaseException:
        if not (a.out/'failure.json').exists():write(a.out/'failure.json',dict(status='failed',error=traceback.format_exc(),optimizer_updates=0,real_data_used=False))
        raise
if __name__=='__main__':main()
