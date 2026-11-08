"""Standalone, equal-budget synthetic BAF center adaptation; frozen C1 depth/R."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from pose_priority_policy import BoundedFitter,draw_auxiliary,state_sha
sys.dont_write_bytecode=True
POLICY_SHA='e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c'
DIAGNOSTIC_SHA='9e604e49ae861ab3831d641cd8c6d57d506017b060ebd30ee7a0b23f80e8771b'
BAF_SHA='1383a156058347ad673ded37760dd5529a019c287f06a8afb4441e5fb7853adf'
CONTRACT_SHA='60db65c84cba5c6045542eca90640017f4e863e5e8ad9cab774dae778deee762'

def need(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def record(p):
    p=Path(p);return dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size)
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def verify_sources(path,digest,records):
    p=Path(path);need(sha(p/'source_manifest.json')==digest,'Source manifest mismatch')
    for name,expected in read(p/'source_manifest.json').items():
        need(Path(name).name==name and sha(p/name)==expected,'Source changed: '+name);records.append(record(p/name))
    records.append(record(p/'source_manifest.json'))
def verify_input_lineage(records):
    need(len(records)>0,'Input lineage missing')
    for r in records:need(Path(r['path']).stat().st_size==r['size_bytes'] and sha(r['path'])==r['sha256'],'Input bytes changed: '+r['path'])
def trajectory(rows):return [{k:r[k] for k in ('update','normal_indices','auxiliary_indices','half_mask')} for r in rows]
def read_log(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    need([r['update'] for r in rows]==list(range(1,len(rows)+1)),'Noncontiguous original training log');return rows
def validate_contract(c):
    need(c['schema']=='pose_baf_center_adaptation_v1' and c['candidate_name']=='fixed_baf_center','Wrong experiment')
    need(c['methods']==['eventgenesis','v2e'] and c['seeds']==[0,1,2],'Complete paired matrix required')
    need(c['new_center_optimizer_updates']==1728 and c['updates_per_center_branch']==288,'Budget changed')
    need(c['epochs']==6 and c['batch_size']==32 and c['half_probability']==.75,'Sampling changed')
    need(c['feature_learning_rate']==c['head_learning_rate']==.00015 and c['normal_teacher_weight']==2 and c['auxiliary_teacher_weight']==c['pair_weight']==0,'Objective changed')
    need(c['diagnostic_source_manifest_sha256']==DIAGNOSTIC_SHA and c['baf_source_manifest_sha256']==BAF_SHA,'Unregistered dependencies')
    need(c['candidate_initialization']=='original_C1_source_initialization_NOT_C1_finetune' and c['candidate_depth']=='reuse_C1_RAW_input_unchanged','Changed training/control roles')
    need(c['candidate_rotation']=='reuse_frozen_rotation_RAW_input_predicted_crop' and c['real_data_scoring_allowed'] is False,'Inference/isolation changed')
    need(c['filter_threshold_us']==2000 and c['filter_initial_time_surface']==0 and c['filter_polarity_used'] is False,'Filter changed')
    need(c['phase_runtime_bounds_seconds']=={'smoke':3600,'study':3600},'Runtime bound changed')

def verify_smoke(s,manifest_sha):
    need(s['schema']=='pose_baf_center_adaptation_v1' and s['status']=='complete_synthetic_only' and s['phase']=='smoke','Actual complete smoke required')
    need(s['new_optimizer_updates']==2 and s['contract_sha256']==CONTRACT_SHA and s['source_manifest_sha256']==manifest_sha,'Smoke binding/budget mismatch')
    need(s['real_scores_computed'] is False and s['canonical_candidate']=='fixed_baf_center','Smoke isolation changed')
    records=s['records'];need(len(records)==2 and {(r['method'],r['seed'],r['branch']) for r in records}=={(m,0,'center') for m in ('eventgenesis','v2e')},'Incomplete paired smoke')
    need(all(r['updates']==r['new_optimizer_updates']==1 and r['teacher_unchanged'] and r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal'] for r in records),'Actual optimizer recovery missing')
    need(len(s['full288_admissions'])==6 and all(r['full288_replay_exact'] for r in s['full288_admissions']),'Full-six preflight absent')

def replay(fitter,normal,y,aux,ay,clean,seed,smoke=False):
    rng=np.random.default_rng(71070+seed);h=hashlib.sha256();draws=[]
    for _ in range(1 if smoke else 6):
        order=rng.permutation(len(normal));order=order[:32] if smoke else order
        for off in range(0,len(order),32):
            idx=order[off:off+32];ai,half=draw_auxiliary(rng,len(aux)//2,len(idx))
            _,yy=fitter.shared.affine(normal[idx],y[idx],rng)
            _,_,bb=fitter.native.matched_affine(aux[ai],clean[ai],ay[ai],rng)
            h.update(yy.detach().cpu().numpy().tobytes());h.update(bb.detach().cpu().numpy().tobytes())
            draws.append(dict(update=len(draws)+1,normal_indices=idx.tolist(),auxiliary_indices=ai.tolist(),half_mask=half.tolist()))
    return h.hexdigest(),draws

def episode_inputs(ctx,B,bank,rids,method,lineage=None):
    """At most one episode's 416 images live; CUDA native area128, then CPU."""
    rows_out=[];raw=[];filtered=[];targets=[]
    if lineage is not None:
        if (bank/'bank_manifest.json').is_file():lineage.append(record(bank/'bank_manifest.json'))
        lineage.append(record(bank/'events'/method/'manifest.json'))
    for rid in rids:
        if lineage is not None:lineage.extend(record(p) for p in (bank/'physical'/(rid+'.h5'),bank/'events'/method/(rid+'.npy')))
        callback=B.EpisodeEncoder(ctx.encode);rows=ctx.fresh.adapter(bank,[bank/'physical'/(rid+'.h5')],method,callback)
        need(len(rows)==len(callback.windows),'Adapter callback inventory changed')
        frows=[dict(r,image=w['filtered_image']) for r,w in zip(rows,callback.windows)]
        raw.append(ctx.shared.tensors(rows,ctx.device).cpu());filtered.append(ctx.shared.tensors(frows,ctx.device).cpu())
        targets.append(ctx.shared.labels(ctx.scale,rows,ctx.vertices,ctx.device).cpu())
        rows_out.extend({k:v for k,v in r.items() if k!='image'} for r in rows)
        del rows,frows,callback;gc.collect()
    return dict(rows=rows_out,raw=torch.cat(raw),filtered=torch.cat(filtered),y=torch.cat(targets))

def select_view(view,indices):
    return dict(rows=[view['rows'][i] for i in indices],**{k:view[k][indices] for k in ('raw','filtered','y')})

def prepare_inputs(ctx,E,B,c,method):
    prep=read(Path(c['radiance_train'])/'completion.json');selected=prep['selected']
    need(prep['status']=='complete' and not prep['real_data_used'] and len(selected)==96,'Original96 TRAIN required')
    npzs=sorted((ctx.R/'adapters'/method/'pose'/method/'train').glob('*.npz'));need(len(npzs)==96,'Original96 NPZ input files required')
    lineage=[record(Path(c['radiance_train'])/'completion.json')]+[record(p) for p in npzs]
    loader=ctx.pilot.helpers(Path(c['causal_helpers_path']),ctx.R)['load'];rows=loader(method,'train');need(len(rows)==1529,'Original1529 NPZ order required')
    normal_y=ctx.shared.labels(ctx.scale,rows,ctx.vertices,ctx.device).cpu();ids=[r['id'] for r in rows]
    need(len(set(ids))==1529,'Duplicate NPZ IDs')
    del rows;gc.collect()
    base=episode_inputs(ctx,B,ctx.R/'pose_bank',selected,method,lineage)
    lookup={r['id']:i for i,r in enumerate(base['rows'])};need(set(lookup)==set(ids),'Raw/fresh TRAIN IDs differ')
    normal=base['filtered'][[lookup[s] for s in ids]]
    levels=[]
    for level in ('half','eighth'):
        current=episode_inputs(ctx,B,Path(c['radiance_train'])/level/'pose_bank',selected,method,lineage)
        need(ctx.native.identity(current['rows'])==ctx.native.identity(base['rows']),'Auxiliary identity/order changed');levels.append(current)
    aux=torch.cat([v['filtered'] for v in levels]);ay=torch.cat([v['y'] for v in levels]);clean=torch.cat([base['filtered'],base['filtered']])
    need(torch.equal(ay[:len(ay)//2],ay[len(ay)//2:]),'Auxiliary original labels differ')
    valbank=ctx.D/'pose_bank';allids=sorted(p.stem for p in (valbank/'physical').glob('*.h5'));need(len(allids)==96,'Original96 independent VAL trajectories required')
    allval=episode_inputs(ctx,B,valbank,allids,method,lineage);groups={rid:[] for rid in allids}
    for i,row in enumerate(allval['rows']):groups[row['rid']].append(i)
    clean_indices=[group[i] for group in groups.values() for i in np.linspace(0,len(group)-1,4,dtype=int)]
    views={'clean':select_view(allval,clean_indices)}
    for level in ('half','eighth'):
        bank=Path(c['radiance_validation'])/level/'pose_bank';rids=sorted(p.stem for p in (bank/'physical').glob('*.h5'));need(len(rids)==4,'Four relit VAL required')
        need(not(set(rids)&set(selected)),'TRAIN/VAL rollout overlap')
        views[level]=episode_inputs(ctx,B,bank,rids,method,lineage)
    matched_ids=[r['id'] for r in views['half']['rows']];by_id={r['id']:i for i,r in enumerate(allval['rows'])}
    views['matched_normal']=select_view(allval,[by_id[s] for s in matched_ids])
    need(all(ctx.native.identity(views[s]['rows'])==ctx.native.identity(views['matched_normal']['rows']) for s in ('half','eighth')),'Matched VAL GT differs')
    need(len(views['clean']['rows'])==384 and all(len(views[s]['rows'])==64 for s in ('matched_normal','half','eighth')),'VAL scope changed')
    lineage=list({r['path']:r for r in lineage}.values());verify_input_lineage(lineage)
    return dict(normal=normal,y=normal_y,aux=aux,ay=ay,clean=clean,normal_ids=ids,aux_ids=[r['id'] for v in levels for r in v['rows']],views=views,input_lineage=lineage)

def center_epe(rows,uv):
    values=np.linalg.norm((uv-np.stack([r['uv'][0] for r in rows]))*[346.,260.],axis=1)
    return float(np.mean([values[[i for i,r in enumerate(rows) if r['rid']==rid]].mean() for rid in sorted({r['rid'] for r in rows})]))

def admit_control(ctx,E,c,prior,fitter,method,seed,tensors):
    normal,y,aux,ay,clean=tensors
    run=next(r for r in ctx.completion['results'] if (r['method'],r['seed'])==(method,seed));sp=Path(run['localization_path'])
    E.remember(ctx.source_records,sp,run['localization_sha256']);state=torch.load(sp,map_location='cpu',weights_only=True)
    expected=next(r for r in prior['records'] if (r['method'],r['seed'],r['arm'],r['branch'])==(method,seed,'physical_supervised','center'))
    need(state_sha(state)==expected['source_state_sha256'] and expected['updates']==288,'Unequal source initialization/budget')
    need(expected['teacher_unchanged'] and expected['independent_model_optimizer_rebuild'] and expected['optimizer_state_equal'],'Original C1 recovery missing')
    old_log=Path(c['prior_study'])/'training_evidence'/f'{method}_seed{seed}_physical_supervised_center.jsonl'
    reused=next(r for r in ctx.receipt['records'] if (r['method'],r['seed'],r['arm'],r['branch'])==(method,seed,'physical_supervised','center'))
    need(all(reused[k]==expected[k] for k in ('source_state_sha256','final_state_sha256','labels_sha256','updates')),'Reused C1 control differs from original')
    need(reused['original_log']['path']==str(old_log),'Original control log mapping changed');E.remember(ctx.source_records,old_log,reused['original_log']['sha256'])
    labels,draws=replay(fitter,normal,y,aux,ay,clean,seed)
    need(len(draws)==288 and labels==expected['labels_sha256'] and draws==trajectory(read_log(old_log)),'Full288 original C1 labels/augmentation replay failed')
    return dict(method=method,seed=seed,full288_replay_exact=True,labels_sha256=labels,original_log=record(old_log),
                trajectory_sha256=hashlib.sha256(json.dumps(draws).encode()).hexdigest(),source_state_sha256=state_sha(state),source_model=record(sp))

def stage_gate(results):
    checks=[]
    for run in results:
        before=run['arms']['raw_C1'];after=run['arms']['baf_center']
        for condition in ('clean','half','eighth'):
            a,b=before[condition],after[condition]
            checks.append(dict(method=run['method'],seed=run['seed'],condition=condition,metric='center_epe_px',
                               passed=b['center_epe_px']<a['center_epe_px'] if condition=='eighth' else b['center_epe_px']<=a['center_epe_px']))
            checks.append(dict(method=run['method'],seed=run['seed'],condition=condition,metric='success',
                               passed=b['pose']['summary']['success']-a['pose']['summary']['success']>=(-.02 if condition=='clean' else 0.)))
    return dict(scope='center_stage_only_36_checks_not_global30',checks=checks,passed=all(r['passed'] for r in checks))

def evaluate(ctx,E,method,seed,model,views,reference):
    depth,_,dp=E.load_branch(ctx,method,seed,'physical_supervised','depth');raw_center,_,cp=E.load_branch(ctx,method,seed,'physical_supervised','center')
    run=next(r for r in ctx.completion['results'] if (r['method'],r['seed'])==(method,seed))
    rp=ctx.A/f'jobs/117_pose_rotation_paired_20261106/results/{method}_seed{seed}_rotation.pt';E.remember(ctx.source_records,rp,run['parents']['rotation_sha256'])
    head=ctx.rotation.RotationHead().to(ctx.device).eval();head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
    states=[(v,state_sha(v.state_dict())) for v in (depth,raw_center,head)];tree=cKDTree(ctx.vertices);output={arm:{} for arm in ('raw_C1','frozen_C1_BAF','baf_center')}
    for condition,view in views.items():
        rows=view['rows'];raw=view['raw'].to(ctx.device);filtered=view['filtered'].to(ctx.device);K=[r['K'] for r in rows]
        with torch.inference_mode():
            _,z,size=ctx.scale.estimate(depth,raw,K)
            uv_sets=dict(raw_C1=ctx.native.estimate_centers(raw_center,raw),frozen_C1_BAF=ctx.native.estimate_centers(raw_center,filtered),baf_center=ctx.native.estimate_centers(model,filtered))
        if condition!='clean':
            label='normal' if condition=='matched_normal' else condition
            for i,row in enumerate(rows):
                r=reference[(method,seed,label,row['id'])];np.testing.assert_array_equal(np.asarray(r['uv_gt'],np.float32),row['uv'][0])
                need(np.max(np.abs(uv_sets['raw_C1'][i]-r['pred_uv']))<=2e-6,'Frozen raw C1 UV parity failed')
        for arm,uv in uv_sets.items():
            R=ctx.shared.frozen_rotation(ctx.rotation,head,raw,uv,size,K)
            score=ctx.shared.score(ctx.rotation,rows,uv,z,size,view['y'][:,2].numpy(),R,ctx.vertices,tree,ctx.threshold)
            output[arm][condition]=dict(center_epe_px=center_epe(rows,uv),pose=score,windows=len(rows),rollouts=len({r['rid'] for r in rows}))
        if condition!='matched_normal':E.compare_summary(output['raw_C1'][condition]['pose']['summary'],run['arms']['physical_supervised'][condition]['summary'])
        del raw,filtered
    need(all(state_sha(v.state_dict())==s for v,s in states),'Frozen depth/center/rotation changed')
    return dict(method=method,seed=seed,arms=output)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--phase',choices=('smoke','study'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke-receipt',type=Path);p.add_argument('--smoke-receipt-sha256');a=p.parse_args()
    need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'Unique existing-parent output required');a.out.mkdir();start=time.monotonic()
    try:
        need(CONTRACT_SHA is not None and sha(a.contract)==CONTRACT_SHA,'Root-registered contract SHA required');c=read(a.contract);validate_contract(c)
        torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        own=Path(__file__).resolve().parent;sources=[];verify_sources(own,sha(own/'source_manifest.json'),sources)
        need(sha(own/'pose_priority_policy.py')==POLICY_SHA,'Native BoundedFitter source changed')
        D=Path(c['diagnostic_source_dir']);verify_sources(D,DIAGNOSTIC_SHA,sources);E=load('baf_center_export',D/'export_pose_factors.py')
        need(sha(c['base_contract_path'])==c['base_contract_sha256'],'Base contract changed');base=read(c['base_contract_path'])
        need(all(c[k]==v for k,v in base.items() if k!='schema'),'Base science fields differ');ctx=E.load_context(base,check_gpu=True)
        Bdir=Path(c['baf_source_dir']);verify_sources(Bdir,BAF_SHA,sources);B=load('baf_center_fixed_rule',Bdir/'baf.py')
        need(sha(own/'baf.py')==sha(Bdir/'baf.py'),'Tested BAF copy differs from deployed frozen rule')
        for path,digest in [(a.contract,CONTRACT_SHA),(c['base_contract_path'],c['base_contract_sha256']),(c['causal_helpers_path'],c['causal_helpers_sha256']),
                            (c['center_policy_path'],POLICY_SHA),(c['forward_completion'],c['forward_completion_sha256']),
                            (c['baf_diagnostic_completion'],c['baf_diagnostic_completion_sha256']),
                            (Path(c['prior_study'])/'completion.json',c['prior_study_completion_sha256']),
                            (Path(c['prior_study'])/'priority_execution_receipt.json',c['prior_execution_receipt_sha256'])]:E.remember(ctx.source_records,path,digest)
        baf_receipt=read(c['baf_diagnostic_completion']);need(baf_receipt['schema']=='pose_baf_center_diagnostic_v2' and baf_receipt['status']=='complete','Actual BAF diagnostic required')
        need(baf_receipt['optimizer_updates']==0 and baf_receipt['real_data_used'] is False and baf_receipt['models_unchanged'] and baf_receipt['zero_input_control'],'BAF diagnostic scope differs')
        need(len(baf_receipt['parity'])==18 and all(r['passed'] and r['max_abs_normalized_uv']<=2e-6 for r in baf_receipt['parity']),'Actual raw C1 BAF diagnostic parity required')
        prior=read(Path(c['prior_study'])/'priority_execution_receipt.json');need(prior['status']=='complete_synthetic_only' and prior['real_scores_computed'] is False,'Reliable C1 controls required')
        reference={};forward=read(c['forward_completion'])
        need(forward['schema']=='pose_factor_forward_export_v1' and forward['status']=='complete' and forward['optimizer_updates']==0,'Complete zero-update reference required')
        need(forward['real_data_used'] is False and forward['models_and_sources_unchanged'] and forward['native_parity_passed'],'Reference isolation/parity failed')
        need(forward['contract_sha256']==c['base_contract_sha256'],'Reference contract differs')
        for artifact in forward['artifacts']:
            if Path(artifact['path']).name=='predictions.jsonl':
                E.remember(ctx.source_records,artifact['path'],artifact['sha256'])
                with Path(artifact['path']).open() as f:
                    for line in f:
                        r=json.loads(line)
                        if r['arm']=='C1':
                            key=(r['method'],r['seed'],r['condition'],r['sample_id']);need(key not in reference,'Duplicate C1 reference window');reference[key]=r
        E.remember(ctx.source_records,c['forward_completion'],c['forward_completion_sha256']);need(len(reference)==1152,'Full matched C1 reference required')
        smoke=a.phase=='smoke';seeds=[0] if smoke else [0,1,2];smoke_receipt=None
        if not smoke:
            need(a.smoke_receipt is not None and sha(a.smoke_receipt)==a.smoke_receipt_sha256,'SHA-bound smoke required');smoke_receipt=read(a.smoke_receipt)
            verify_smoke(smoke_receipt,sha(own/'source_manifest.json'))
        fitter=BoundedFitter(ctx.native,a.out,smoke);fitter.order=[(m,s,'physical_supervised','center') for m in c['methods'] for s in seeds]
        caches=[];admissions=[];results=[]
        # All SIX complete288-step admissions precede ANY candidate update,
        # including the two-update smoke. Never train one method while another
        # method's labels or augmentation replay remain unverified.
        for method in c['methods']:
            if smoke:
                data=prepare_inputs(ctx,E,B,c,method);cache_path=a.out/f'{method}_inputs128.pt';torch.save(dict(contract_sha256=CONTRACT_SHA,source_manifest_sha256=sha(own/'source_manifest.json'),data=data),cache_path);caches.append(record(cache_path))
            else:
                item=next(r for r in smoke_receipt['input_caches'] if Path(r['path']).name==f'{method}_inputs128.pt');E.remember(ctx.source_records,item['path'],item['sha256'])
                cached=torch.load(item['path'],map_location='cpu',weights_only=False);need(cached['contract_sha256']==CONTRACT_SHA and cached['source_manifest_sha256']==sha(own/'source_manifest.json'),'Cache provenance mismatch');data=cached['data'];caches.append(item)
                verify_input_lineage(data['input_lineage'])
            ctx.source_records.extend(r for r in data['input_lineage'] if not any(s['path']==r['path'] for s in ctx.source_records))
            normal,y,aux,ay,clean=(data[k].to(ctx.device) for k in ('normal','y','aux','ay','clean'))
            need(len(normal)==1529 and len(aux)==3058 and len(clean)==3058,'Original complete TRAIN dimensions required')
            for seed in c['seeds']:admissions.append(admit_control(ctx,E,c,prior,fitter,method,seed,(normal,y,aux,ay,clean)))
            del data,normal,y,aux,ay,clean;gc.collect();torch.cuda.empty_cache()
        need(len(admissions)==6 and not fitter.records,'All-six pre-update admission required')
        for seed in c['seeds']:need(len({r['labels_sha256'] for r in admissions if r['seed']==seed})==1,'Cross-generator labels differ')
        write(a.out/'pre_update_admission.json',dict(status='passed',optimizer_updates=0,full288_admissions=admissions,input_caches=caches))
        for method in c['methods']:
            item=next(r for r in caches if Path(r['path']).name==f'{method}_inputs128.pt');need(sha(item['path'])==item['sha256'],'Cache changed before training')
            data=torch.load(item['path'],map_location='cpu',weights_only=False)['data']
            normal,y,aux,ay,clean=(data[k].to(ctx.device) for k in ('normal','y','aux','ay','clean'))
            for seed in seeds:
                need(time.monotonic()-start<c['phase_runtime_bounds_seconds'][a.phase],'Fixed phase runtime bound exceeded')
                admission=next(r for r in admissions if (r['method'],r['seed'])==(method,seed));state=torch.load(admission['source_model']['path'],map_location='cpu',weights_only=True)
                need(state_sha(state)==admission['source_state_sha256'],'Source init changed after admission');old=read_log(admission['original_log']['path'])
                model,history=fitter.fit_center(ctx.scale,state,normal,y,aux,ay,seed,smoke,clean,0)
                actual=fitter.records[-1];actual['canonical_arm']=c['candidate_name'];actual['new_optimizer_updates']=actual['updates'];actual['input_policy']='same_fixed_BAF2ms_on_normal_aux_and_teacher_normal'
                log=a.out/'training_evidence'/f'{method}_seed{seed}_physical_supervised_center.jsonl';need(trajectory(read_log(log))==trajectory(old)[:actual['updates']],'Actual candidate draw sequence differs')
                expected_labels=replay(fitter,normal,y,aux,ay,clean,seed,smoke=smoke)[0];need(actual['labels_sha256']==expected_labels,'Actual candidate labels differ')
                model_path=a.out/f'{method}_seed{seed}_baf_center.pt';torch.save(model.state_dict(),model_path)
                actual['final_model']=record(model_path);actual['training_log']=record(log);actual['native_bounded_fitter_receipt']=record(log.with_name(log.stem+'_receipt.json'))
                if not smoke:results.append(evaluate(ctx,E,method,seed,model,data['views'],reference))
                del model,state;gc.collect()
            del data,normal,y,aux,ay,clean;gc.collect();torch.cuda.empty_cache()
        need(len(fitter.records)==(2 if smoke else 6),'Incomplete branch inventory')
        all_sources=sources+ctx.source_records;need(all(sha(r['path'])==r['sha256'] for r in all_sources),'Frozen dependencies changed')
        need(time.monotonic()-start<c['phase_runtime_bounds_seconds'][a.phase],'Phase runtime exceeded')
        gate=None if smoke else stage_gate(results);need(smoke or len(gate['checks'])==36,'Wrong center-stage gate count')
        write(a.out/'completion.json',dict(schema='pose_baf_center_adaptation_v1',status='complete_synthetic_only',phase=a.phase,
            canonical_candidate=c['candidate_name'],result_arm_mapping=dict(baf_center=c['candidate_name'],physical_supervised='legacy implementation slot only'),new_optimizer_updates=sum(r['updates'] for r in fitter.records),records=fitter.records,
            contract_sha256=CONTRACT_SHA,source_manifest_sha256=sha(own/'source_manifest.json'),source_records=all_sources,input_caches=caches,
            full288_admissions=admissions,results=results,center_stage_gate=gate,real_scores_computed=False,
            original30_gate=dict(status='not_evaluated',thresholds_unchanged=True,reason='center-only study; original depth remains unchanged'),
            original8_confirmation=dict(status='not_evaluated',thresholds_unchanged=True,reason='full1536 inference not repeated'),
            checkpoint_verification_scope='model_optimizer_readback_only_no_resume_claim',resume_supported=False,
            frozen_depth_input='raw',frozen_rotation_input='raw with predicted center crop/ray frame',elapsed_seconds=time.monotonic()-start))
    except BaseException as error:
        write(a.out/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start));raise

if __name__=='__main__':main()
