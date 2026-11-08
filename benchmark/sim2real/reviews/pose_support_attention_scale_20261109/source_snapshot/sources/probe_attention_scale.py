"""Frozen C0/C1 RAW/BAF attention, physical support and scale diagnostic.

No updates, new decoder, GT-guided filtering/crop, or real-data evaluation.
RAW feature/attention cross-light hybrids are mechanism probes, not deployable
predictions: the second illumination is unavailable at ordinary test time.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,itertools,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np

DIAGNOSTIC_MANIFEST='9e604e49ae861ab3831d641cd8c6d57d506017b060ebd30ee7a0b23f80e8771b'
BAF_MANIFEST='1383a156058347ad673ded37760dd5529a019c287f06a8afb4441e5fb7853adf'
EXPORT_COMPLETION='fea5095c363b2d4a02602c4fe2dec6144058564be877c55caf0ef1dc24631093'
ARMS={'C0':'clean_control','C1':'physical_supervised'}
LIGHTS=('normal','half','eighth')


def need(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):
    p=Path(p);return dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size)
def write(p,x):
    with Path(p).open('x') as h:json.dump(x,h,indent=2,allow_nan=False)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def checked_bundle(directory,expected):
    mp=directory/'source_manifest.json';need(sha(mp)==expected,'Source manifest differs: '+str(directory))
    result=[record(mp)]
    for name,digest in read(mp).items():
        need(Path(name).name==name and sha(directory/name)==digest,'Source bytes differ: '+name);result.append(record(directory/name))
    return result


def load_reference(directory,contract_sha):
    cp=directory.parent/'export/completion.json';need(sha(cp)==EXPORT_COMPLETION,'Frozen original export receipt differs')
    c=read(cp);need(c['status']=='complete' and c['contract_sha256']==contract_sha and c['optimizer_updates']==0 and not c['real_data_used'],'Reference scope differs')
    raw=next(x for x in c['artifacts'] if Path(x['path']).name=='predictions.jsonl')
    path=Path(raw['path']);need(sha(path)==raw['sha256'] and path.stat().st_size==raw['size_bytes'],'Original predictions changed')
    ref={}
    for line in path.open():
        r=json.loads(line)
        if r['arm'] in ARMS:
            key=(r['method'],r['seed'],r['arm'],r['condition'],r['sample_id'])
            need(key not in ref,'Duplicate reference row');ref[key]=r
    need(len(ref)==2304,'Complete C0/C1 original reference required')
    return ref,[record(cp),raw]


def check_raw(observed,reference,branch):
    checks=[]
    names=[('uv','pred_uv' if branch=='center' else 'depth_model_uv')]
    if branch=='depth':names += [('log_size','log_size'),('log_shape','log_shape'),('z','z_m')]
    for actual,old in names:
        a=np.asarray(observed[actual],float);b=np.asarray(reference[old],float)
        delta=float(np.max(np.abs(a-b)))
        need(np.isfinite(delta) and delta<=2e-6,'RAW native reference parity failed '+actual+': '+str(delta))
        checks.append(dict(field=actual,reference_field=old,max_abs=delta,tolerance=2e-6))
    return checks


def observe_all(model,images,scale,intrinsics,observer,bound):
    import torch
    chunks=[]
    for offset in range(0,len(images),32):
        bound();chunks.append(observer.observe_native(model,images[offset:offset+32],scale,intrinsics[offset:offset+32]))
    return {name:torch.cat([r[name] for r in chunks]) for name in chunks[0]}


def cpu_array(t):return t.detach().cpu().numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('baseline','full'))
    for key in ('contract','diagnostic-sources','baf-sources','support-sources','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--support-manifest-sha256',required=True)
    p.add_argument('--source-manifest-sha256',required=True)
    p.add_argument('--baseline-receipt',type=Path)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    a=p.parse_args();need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'Fresh absolute output required')
    a.out.mkdir();start=time.monotonic()
    def bound():need(time.monotonic()-start<3600,'Fixed3600s bound exceeded')
    try:
        if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
        import torch,cv2
        torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        import observe_native as O
        import attention_scale_metrics as M
        own=Path(__file__).parent;own_manifest=sha(own/'source_manifest.json')
        need(own_manifest==a.source_manifest_sha256,'Own externally locked source manifest differs')
        sources=checked_bundle(own,own_manifest)+checked_bundle(a.baf_sources,BAF_MANIFEST)+checked_bundle(a.diagnostic_sources,DIAGNOSTIC_MANIFEST)
        sources+=checked_bundle(a.support_sources,a.support_manifest_sha256)
        S=load('bound_event_region_support',a.support_sources/'event_support.py')
        sys.path.insert(0,str(a.baf_sources));D=load('attention_bound_baf',a.baf_sources/'diagnose_centers.py')
        need(Path(sys.modules['baf'].__file__).resolve()==(a.baf_sources/'baf.py').resolve(),'BAF module shadowed')
        E,extra=D.load_export(a.diagnostic_sources,a.contract);sources+=extra;c=read(a.contract)
        ctx=E.load_context(c,check_gpu=a.device=='cuda');need(ctx.network_size==416,'Original encoder416 required')
        ref,extra=load_reference(a.diagnostic_sources,sha(a.contract));sources+=extra
        if a.mode=='full':
            need(a.baseline_receipt is not None,'Actual RAW baseline admission required')
            baseline=read(a.baseline_receipt)
            need(baseline['status']=='complete' and baseline['mode']=='baseline' and baseline['source_manifest_sha256']==own_manifest and
                 baseline['support_source_manifest_sha256']==a.support_manifest_sha256 and baseline['contract_sha256']==sha(a.contract) and
                 baseline['device']==a.device and baseline['raw_parity_passed'] is True and baseline['optimizer_updates']==0,'Baseline admission mismatch')
            sources.append(record(a.baseline_receipt))
        results=[];hybrids=[];models=[];parity=[];support_rows=[];diagparity=[];geometry_parity=[]
        for method in c['methods']:
            bound();data=D.prepare(ctx,c,method,'validation')
            need(set(data)==set(LIGHTS),'All three illumination conditions required')
            region_data={};support={};pyramids={};labels={}
            for condition,(rows,windows,raw,filtered) in data.items():
                need(len(rows)==64 and len({r['rid'] for r in rows})==4,'Fixed four-rollout64 cohort changed')
                bank=ctx.D/'pose_bank' if condition=='normal' else Path(c['radiance_validation'])/condition/'pose_bank'
                E.remember(ctx.source_records,bank/'events'/method/'manifest.json')
                for rid in sorted({r['rid'] for r in rows}):
                    for path in (bank/'physical'/(rid+'.h5'),bank/'events'/method/(rid+'.npy')):E.remember(ctx.source_records,path)
                if a.mode=='baseline':continue
                bound();region_data[condition]=S.regions_for_rows(bank,rows,strict=True)
                support[condition],extra=S.window_support_for_rows(bank,rows,method,sys.modules['baf'],region_data[condition]);sources+=extra
                labels[condition]=cpu_array(ctx.shared.labels(ctx.scale,rows,ctx.vertices,ctx.device))
                pyramids[condition]={}
                for row in rows:
                    rr=region_data[condition][row['id']]
                    pyramids[condition][row['id']]={name:S.region_pyramid(rr[name]) for name in ('startpoint','endpoint','endpoint_union_approx')}
                    support_rows.append(dict(method=method,condition=condition,**support[condition][row['id']]))
            if a.mode=='full':
                for condition in ('half','eighth'):
                    geometry_parity.append(dict(method=method,condition=condition,**S.verify_matched_regions(region_data['normal'],region_data[condition])))
            for seed in ([0] if a.mode=='baseline' else c['seeds']):
                for arm,legacy in ARMS.items():
                    for branch in ('center','depth'):
                        bound();model,model_record,path=E.load_branch(ctx,method,seed,legacy,branch)
                        model.requires_grad_(False);before=E.state_sha(model.state_dict());raw_saved={}
                        for condition,(rows,windows,raw,filtered) in data.items():
                            count=8 if a.mode=='baseline' else 64
                            current=rows[:count];intrinsics=[r['K'] for r in current]
                            for view,images in ([('raw',raw[:count])] if a.mode=='baseline' else [('raw',raw),('baf',filtered)]):
                                observed=observe_all(model,images,ctx.scale,intrinsics,O,bound)
                                arrays={name:cpu_array(value) for name,value in observed.items() if name!='features_map'}
                                if view=='raw':
                                    raw_saved[condition]=observed
                                    for i,row in enumerate(current):
                                        reference=ref[(method,seed,arm,condition,row['id'])]
                                        need(np.array_equal(reference['uv_gt'],row['uv'][0]),'Original label differs')
                                        need(np.array_equal(reference['K'],row['K']) and np.array_equal(reference['T_gt'],row['T']) and
                                             np.array_equal(reference['anchor'],row['keys'][0]),'Original camera/pose/anchor labels differ')
                                        gt_depth=float((row['T'][:3,:3]@row['keys'][0]+row['T'][:3,3])[2])
                                        need(reference['gt_z_m']==gt_depth,'Original GT anchor depth differs')
                                        checks=check_raw({k:v[i] for k,v in arrays.items()},reference,branch)
                                        parity.append(dict(method=method,seed=seed,arm=arm,branch=branch,condition=condition,sample_id=row['id'],checks=checks))
                                if a.mode=='baseline':continue
                                nominal=O.nominal_center_target(model.grid,np.stack([r['uv'][0] for r in current]))
                                nominal={name:cpu_array(value) for name,value in nominal.items()}
                                for i,row in enumerate(current):
                                    rr=region_data[condition][row['id']];uv=arrays['uv'][i];peak=arrays['peak_uv'][i]
                                    attn={}
                                    for name in ('startpoint','endpoint','endpoint_union_approx'):
                                        attn[name]=S.summarize_attention(rr[name],arrays['attention'][i].reshape(16,16),uv,peak,pyramid=pyramids[condition][row['id']][name])
                                        attn[name]['center_error_px']=float(np.linalg.norm((uv-row['uv'][0])*[346,260]))
                                    gt_heat_reference={name:S.summarize_attention(rr[name],nominal['attention'][i].reshape(16,16),nominal['uv'][i],nominal['peak_uv'][i],pyramid=pyramids[condition][row['id']][name])
                                                       for name in ('startpoint','endpoint','endpoint_union_approx')}
                                    gt_heat_reference.update(target_decoded_uv=nominal['uv'][i].tolist(),
                                        target_decoded_center_bias_px=float(np.linalg.norm((nominal['uv'][i]-row['uv'][0])*[346,260])),
                                        GT_anchor_outside_grid_range=bool(nominal['outside_grid_range'][i]),
                                        use='posthoc_original_soft_supervision_reference_never_forward_input')
                                    gt_z=float((row['T'][:3,:3]@row['keys'][0]+row['T'][:3,3])[2])
                                    gt=dict(gt_log_size=float(labels[condition][i,2]),gt_z_m=gt_z,focal_log=ctx.scale.focal(row['K']))
                                    scale=(M.scale_statistics(float(arrays['log_size'][i]),float(arrays['log_shape'][i]),gt['focal_log'],gt['gt_log_size'],gt_z,float(arrays['z'][i])) if branch=='depth' else None)
                                    results.append(dict(method=method,seed=seed,arm=arm,branch=branch,input_view=view,condition=condition,rid=row['rid'],sample_id=row['id'],
                                        pred_uv=uv.tolist(),peak_uv=peak.tolist(),attention=attn,labels=gt,scale=scale,
                                        features=M.feature_statistics(arrays['local'][i],arrays['context'][i]),motion=rr['motion'],
                                        target_identity=rr['target_identity'],geometric_sanity=rr['geometric_sanity'],
                                        nominal_GT_heat_reference=gt_heat_reference,
                                        visible_geometry={name:S.region_geometry(rr[name]) for name in ('startpoint','endpoint','endpoint_union_approx')},
                                        support=support[condition][row['id']],GT_region_usage='posthoc_only_no_forward_input',
                                        scale_used_in_pose=branch=='depth'))
                                bound()
                        if a.mode=='full' and branch=='depth':
                            for condition in ('half','eighth'):
                                for feature_light,attention_light in itertools.product(('normal',condition),repeat=2):
                                    bound();fs=raw_saved[feature_light];ats=raw_saved[attention_light]
                                    hybrid=O.hybrid_scale(model,fs['features_map'],ats['attention'],fs['focal'])
                                    values={k:cpu_array(v) for k,v in hybrid.items()}
                                    if feature_light==attention_light:
                                        errors={k:float(np.max(np.abs(values[k]-cpu_array(fs[k])))) for k in values}
                                        need(all(v<=2e-6 for v in errors.values()),'Hybrid diagonal native parity failed')
                                        diagparity.append(dict(method=method,seed=seed,arm=arm,paired_light=condition,diagonal=feature_light,max_abs=errors,tolerance=2e-6))
                                    rows=data['normal'][0]
                                    for i,row in enumerate(rows):
                                        gt_z=float((row['T'][:3,:3]@row['keys'][0]+row['T'][:3,3])[2])
                                        hybrids.append(dict(method=method,seed=seed,arm=arm,paired_light=condition,rid=row['rid'],sample_id=row['id'],
                                            feature_source=feature_light,attention_source=attention_light,input_view='raw',
                                            log_size=float(values['log_size'][i]),log_shape=float(values['log_shape'][i]),z_m=float(values['z'][i]),gt_z_m=gt_z,
                                            gt_log_size=float(labels['normal'][i,2]),
                                            gt_log_shape=math.log(gt_z)-ctx.scale.focal(row['K'])+float(labels['normal'][i,2]),
                                            signed_depth_error_m=float(values['z'][i]-gt_z),diagnostic_only=True,GT_used_only_for_error=True,
                                            deployable=False,context_source='same_as_features'))
                        need(E.state_sha(model.state_dict())==before,'Frozen model or buffer changed')
                        models.append(dict(method=method,seed=seed,arm=arm,branch=branch,path=str(path),state_sha256=before,unchanged=True))
                        del model,raw_saved
                        if a.device=='cuda':torch.cuda.empty_cache()
            del data,region_data,support,pyramids,labels
        expected=192 if a.mode=='baseline' else 4608
        need(len(parity)==expected,'Incomplete RAW parity cohort')
        if a.mode=='full':need(len(results)==9216 and len(hybrids)==6144 and len(models)==24 and len(support_rows)==384,'Incomplete fixed diagnostic matrix')
        else:need(len(models)==8,'Incomplete baseline models')
        sources+=ctx.source_records
        need(all(sha(r['path'])==r['sha256'] for r in sources),'Source/weights/data changed during diagnostic');bound()
        write(a.out/'raw_native_parity.json',dict(status='passed',checks=parity))
        if a.mode=='full':
            for name,rows in (('per_window',results),('raw_feature_attention_hybrids',hybrids),('event_support',support_rows),('cross_light_deltas',M.paired_light_deltas(results))):
                with (a.out/(name+'.jsonl')).open('x') as handle:
                    for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
            write(a.out/'macro_summaries.json',M.scalar_rollout_summaries(results))
            write(a.out/'cross_light_geometry_parity.json',dict(status='passed',checks=geometry_parity))
            hybrid_summary=M.hybrid_summaries(hybrids)
            need(len(hybrid_summary['per_seed'])==96 and len(hybrid_summary['three_seed_means'])==32,'Incomplete hybrid summary matrix')
            need(all(r['seeds']==[0,1,2] for r in hybrid_summary['three_seed_means']),'Three paired hybrid seeds required')
            write(a.out/'raw_hybrid_summaries.json',hybrid_summary)
            with (a.out/'raw_hybrid_three_seed_table.csv').open('x',newline='') as handle:
                fields=['method','arm','paired_light','feature_source','attention_source','depth_mae_cm','depth_bias_cm','log_size_mae','log_size_bias','log_shape_mae','log_shape_bias']
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                for row in hybrid_summary['three_seed_means']:
                    m=row['metrics'];writer.writerow({k:row[k] for k in fields[:5]}|dict(depth_mae_cm=100*m['depth_mae_m'],depth_bias_cm=100*m['depth_bias_m'],**{k:m[k] for k in fields[7:]}))
            write(a.out/'hybrid_diagonal_parity.json',dict(status='passed',checks=diagparity))
        completion=dict(schema='pose_support_attention_scale_frozen_v1',status='complete',mode=a.mode,device=a.device,
            source_manifest_sha256=own_manifest,support_source_manifest_sha256=a.support_manifest_sha256,contract_sha256=sha(a.contract),
            raw_parity_passed=True,raw_parity_records=len(parity),model_records=models,source_records=sources,
            diagnostic_rows=len(results),raw_hybrid_rows=len(hybrids),support_physical_method_light_rows=len(support_rows),
            optimizer_updates=0,real_data_used=False,GT_used_only_posthoc=True,GT_modified_forward=False,filters_adopted=False,
            attention_RF_interpretation=False,hybrids_deployable=False,clean384_mixed=False,
            elapsed_seconds=time.monotonic()-start,artifacts=[record(path) for path in a.out.iterdir() if path.is_file()])
        write(a.out/'completion.json',completion);print('ATTENTION_SCALE_COMPLETE',a.mode,len(results),len(hybrids),flush=True)
    except BaseException:
        write(a.out/'failure.json',dict(status='failed',optimizer_updates=0,real_data_used=False,error=traceback.format_exc()));raise


if __name__=='__main__':main()
