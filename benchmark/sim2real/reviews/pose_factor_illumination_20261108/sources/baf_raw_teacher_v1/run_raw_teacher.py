"""Equal-budget BAF student / RAW-normal teacher; synthetic center branch only."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from raw_teacher_policy import BoundedFitter,state_sha,draw_auxiliary
sys.dont_write_bytecode=True
CONTRACT_SHA='d682813400a1709c90b47b4dcc73a83c8f7ea3fe830e9d837316555dea11ca41'
ORIGINAL_POLICY_SHA='e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c'
POLICY_SHA='9e713cc63aefe10c53b27505101f1a4271483a20a1657b5c64ca2de838d9aa04'
PREVIOUS_SOURCE_SHA='db07f866cc0ce6535a2f1917ea324b954bf5b3bc3d96966e9553bc56be4b8037'
PREVIOUS_CONTRACT_SHA='60db65c84cba5c6045542eca90640017f4e863e5e8ad9cab774dae778deee762'

def need(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def validate_contract(c):
    need(c['schema']=='pose_baf_raw_teacher_adaptation_v1' and c['candidate_name']=='raw_teacher_baf_center','Wrong candidate')
    need(c['candidate_teacher_input']=='RAW_normal_same_augmentation_as_BAF_student','Wrong teacher route')
    need(c['previous_baf_source_manifest_sha256']==PREVIOUS_SOURCE_SHA and c['previous_baf_contract_sha256']==PREVIOUS_CONTRACT_SHA,'Unregistered BAF controls')
    need(c['methods']==['eventgenesis','v2e'] and c['seeds']==[0,1,2],'Incomplete pairing')
    need(c['new_center_optimizer_updates']==1728 and c['updates_per_center_branch']==288 and c['epochs']==6 and c['batch_size']==32,'Budget changed')
    need(c['feature_learning_rate']==c['head_learning_rate']==.00015 and c['normal_teacher_weight']==2 and c['auxiliary_teacher_weight']==c['pair_weight']==0 and c['half_probability']==.75,'Objective changed')
    need(c['candidate_initialization']=='original_C1_source_initialization_NOT_C1_finetune' and c['candidate_depth']=='reuse_C1_RAW_input_unchanged','Initialization/depth changed')
    need(c['candidate_rotation']=='reuse_frozen_rotation_RAW_input_predicted_crop' and c['real_data_scoring_allowed'] is False,'Inference/isolation changed')
    need(c['filter_threshold_us']==2000 and c['filter_initial_time_surface']==0 and c['filter_polarity_used'] is False,'BAF changed')
    need(c['phase_runtime_bounds_seconds']=={'smoke':3600,'study':3600},'Runtime bound changed')
def replay(fitter,normal,y,aux,ay,clean,seed,smoke=False):
    rng=np.random.default_rng(71070+seed);h=hashlib.sha256();draws=[]
    for _ in range(1 if smoke else 6):
        order=rng.permutation(len(normal));order=order[:32] if smoke else order
        for off in range(0,len(order),32):
            idx=order[off:off+32];ai,half=draw_auxiliary(rng,len(aux)//2,len(idx))
            _,_,yy=fitter.native.matched_affine(normal[idx],fitter.raw_normal[idx],y[idx],rng)
            _,_,bb=fitter.native.matched_affine(aux[ai],clean[ai],ay[ai],rng)
            h.update(yy.detach().cpu().numpy().tobytes());h.update(bb.detach().cpu().numpy().tobytes())
            draws.append(dict(update=len(draws)+1,normal_indices=idx.tolist(),auxiliary_indices=ai.tolist(),half_mask=half.tolist()))
    return h.hexdigest(),draws
def verify_smoke(s,manifest_sha):
    need(s['schema']=='pose_baf_raw_teacher_adaptation_v1' and s['status']=='complete_synthetic_only' and s['phase']=='smoke','Actual new smoke required')
    need(s['contract_sha256']==CONTRACT_SHA and s['source_manifest_sha256']==manifest_sha and s['new_optimizer_updates']==2,'Smoke binding/budget mismatch')
    need(s['real_scores_computed'] is False and s['canonical_candidate']=='raw_teacher_baf_center','Smoke isolation changed')
    rows=s['records'];need(len(rows)==2 and {(r['method'],r['seed'],r['branch']) for r in rows}=={(m,0,'center') for m in ('eventgenesis','v2e')},'Smoke matrix changed')
    need(all(r['updates']==1 and r['teacher_unchanged'] and r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal'] for r in rows),'Smoke recovery incomplete')
    need(len(s['full288_admissions'])==6 and all(r['raw_teacher_paired_replay_exact'] for r in s['full288_admissions']) and len(s['raw_teacher_caches'])==2,'Admission/cache missing')
def previous_data(P,E,ctx,c,previous_smoke,method):
    item=next(r for r in previous_smoke['input_caches'] if Path(r['path']).name==f'{method}_inputs128.pt')
    E.remember(ctx.source_records,item['path'],item['sha256']);value=torch.load(item['path'],map_location='cpu',weights_only=False)
    need(value['contract_sha256']==PREVIOUS_CONTRACT_SHA and value['source_manifest_sha256']==PREVIOUS_SOURCE_SHA,'BAF cache provenance changed')
    data=value['data'];P.verify_input_lineage(data['input_lineage']);ctx.source_records.extend(r for r in data['input_lineage'] if not any(s['path']==r['path'] for s in ctx.source_records))
    return data,item
def prepare_raw(ctx,P,c,method,data):
    rows=ctx.pilot.helpers(Path(c['causal_helpers_path']),ctx.R)['load'](method,'train')
    need(len(rows)==1529 and [r['id'] for r in rows]==data['normal_ids'],'RAW normal order changed')
    y=ctx.shared.labels(ctx.scale,rows,ctx.vertices,ctx.device).cpu();need(torch.equal(y,data['y']),'RAW labels not exact original NPZ labels')
    raw=ctx.shared.tensors(rows,ctx.device).cpu();need(raw.shape==data['normal'].shape==(1529,3,128,128) and torch.isfinite(raw).all(),'RAW representation invalid')
    lineage=[r for r in data['input_lineage'] if Path(r['path']).suffix=='.npz'];need(len(lineage)==96,'Original96 NPZ source pins missing');P.verify_input_lineage(lineage)
    return dict(raw_normal=raw,labels=y,normal_ids=data['normal_ids'],source_records=lineage,role='RAW_teacher_only_BAF_student_unchanged')
def check_raw(P,value,data,manifest_sha):
    need(value['contract_sha256']==CONTRACT_SHA and value['source_manifest_sha256']==manifest_sha,'RAW teacher cache binding changed')
    v=value['data'];need(v['role']=='RAW_teacher_only_BAF_student_unchanged' and v['normal_ids']==data['normal_ids'] and torch.equal(v['labels'],data['y']),'RAW teacher cache role/labels changed')
    need(v['raw_normal'].shape==data['normal'].shape and v['raw_normal'].dtype==data['normal'].dtype and torch.isfinite(v['raw_normal']).all(),'RAW teacher cache invalid');P.verify_input_lineage(v['source_records']);return v['raw_normal']
def evaluate(P,ctx,E,method,seed,model,views,reference,old_runs):
    result=P.evaluate(ctx,E,method,seed,model,views,reference);old=old_runs[(method,seed)]
    for arm in ('raw_C1','frozen_C1_BAF'):
        for condition,current in result['arms'][arm].items():
            before=old['arms'][arm][condition];need(current['windows']==before['windows'] and current['rollouts']==before['rollouts'],'Prior evaluation scope differs')
            need(np.isclose(current['center_epe_px'],before['center_epe_px'],rtol=1e-6,atol=1e-8),'Fixed baseline center changed')
            E.compare_summary(current['pose']['summary'],before['pose']['summary'])
    result['arms']['raw_teacher_baf_center']=result['arms'].pop('baf_center');result['arms']['filtered_teacher_BAF']=old['arms']['baf_center']
    result['teacher_route_effect']={condition:dict(center_epe_delta_px=v['center_epe_px']-old['arms']['baf_center'][condition]['center_epe_px'],
        pose_success_delta=v['pose']['summary']['success']-old['arms']['baf_center'][condition]['pose']['summary']['success']) for condition,v in result['arms']['raw_teacher_baf_center'].items()}
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--phase',choices=('smoke','study'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke-receipt',type=Path);p.add_argument('--smoke-receipt-sha256');a=p.parse_args()
    need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'Unique output required');a.out.mkdir();start=time.monotonic()
    try:
        need(sha(a.contract)==CONTRACT_SHA,'Registered contract SHA required');c=read(a.contract);validate_contract(c)
        torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        previous=Path(c['previous_baf_sources']);need(sha(previous/'source_manifest.json')==PREVIOUS_SOURCE_SHA,'Previous worker unregistered')
        need(sha(previous/'run_baf_center.py')==read(previous/'source_manifest.json')['run_baf_center.py'],'Previous helper source changed before import')
        P=load('raw_teacher_fixed_previous_worker',previous/'run_baf_center.py');sources=[];P.verify_sources(previous,PREVIOUS_SOURCE_SHA,sources)
        own=Path(__file__).resolve().parent;manifest_sha=sha(own/'source_manifest.json');P.verify_sources(own,manifest_sha,sources)
        need(sha(own/'pose_priority_policy.py')==ORIGINAL_POLICY_SHA and sha(own/'raw_teacher_policy.py')==POLICY_SHA,'Policy derivation changed')
        D=Path(c['diagnostic_source_dir']);P.verify_sources(D,c['diagnostic_source_manifest_sha256'],sources);E=load('raw_teacher_export',D/'export_pose_factors.py')
        need(sha(c['base_contract_path'])==c['base_contract_sha256'],'Base contract changed');base=read(c['base_contract_path']);need(all(c[k]==v for k,v in base.items() if k!='schema'),'Base fields changed');ctx=E.load_context(base,check_gpu=True)
        for path,digest in [(a.contract,CONTRACT_SHA),(c['base_contract_path'],c['base_contract_sha256']),(c['causal_helpers_path'],c['causal_helpers_sha256']),
            (c['center_policy_path'],ORIGINAL_POLICY_SHA),(c['forward_completion'],c['forward_completion_sha256']),(c['baf_diagnostic_completion'],c['baf_diagnostic_completion_sha256']),
            (Path(c['prior_study'])/'priority_execution_receipt.json',c['prior_execution_receipt_sha256']),(c['previous_baf_smoke_completion'],c['previous_baf_smoke_completion_sha256']),
            (Path(c['previous_baf_study'])/'completion.json',c['previous_baf_study_completion_sha256']),(c['teacher_target_diagnostic'],c['teacher_target_diagnostic_sha256'])]:E.remember(ctx.source_records,path,digest)
        prior=read(Path(c['prior_study'])/'priority_execution_receipt.json');previous_smoke=read(c['previous_baf_smoke_completion']);old=read(Path(c['previous_baf_study'])/'completion.json');teacher_diag=read(c['teacher_target_diagnostic'])
        need(prior['status']=='complete_synthetic_only' and prior['real_scores_computed'] is False,'Original C1 isolation changed')
        for value in (previous_smoke,old):need(value['status']=='complete_synthetic_only' and value['contract_sha256']==PREVIOUS_CONTRACT_SHA and value['source_manifest_sha256']==PREVIOUS_SOURCE_SHA and value['real_scores_computed'] is False,'Previous experiment binding/isolation changed')
        need(previous_smoke['phase']=='smoke' and previous_smoke['new_optimizer_updates']==2 and old['phase']=='study' and old['new_optimizer_updates']==1728,'Previous budgets incomplete')
        need(teacher_diag['status']=='complete' and teacher_diag['schema']=='pose_source_teacher_target_diagnostic_v1' and teacher_diag['optimizer_updates']==0 and teacher_diag['real_data_used'] is False and teacher_diag['contract_sha256']==c['base_contract_sha256'],'Teacher diagnostic incomplete')
        old_runs={(r['method'],r['seed']):r for r in old['results']};need(len(old_runs)==6,'Full previous matrix required');reference={}
        for item in read(c['forward_completion'])['artifacts']:
            if Path(item['path']).name=='predictions.jsonl':
                E.remember(ctx.source_records,item['path'],item['sha256'])
                with Path(item['path']).open() as f:
                    for line in f:
                        r=json.loads(line)
                        if r['arm']=='C1':reference[(r['method'],r['seed'],r['condition'],r['sample_id'])]=r
        need(len(reference)==1152,'Full C1 reference required');smoke=a.phase=='smoke';seeds=[0] if smoke else c['seeds'];s=None
        if not smoke:
            need(a.smoke_receipt is not None and sha(a.smoke_receipt)==a.smoke_receipt_sha256,'Actual SHA-bound new smoke required');s=read(a.smoke_receipt);verify_smoke(s,manifest_sha)
        fitter=BoundedFitter(ctx.native,a.out,smoke);fitter.order=[(m,seed,'physical_supervised','center') for m in c['methods'] for seed in seeds]
        admissions=[];caches=[];raw_caches=[];results=[]
        for method in c['methods']:
            data,cache=previous_data(P,E,ctx,c,previous_smoke,method);caches.append(cache)
            if smoke:
                raw_value=dict(contract_sha256=CONTRACT_SHA,source_manifest_sha256=manifest_sha,data=prepare_raw(ctx,P,c,method,data));rp=a.out/f'{method}_RAW_normal128.pt';torch.save(raw_value,rp);raw_caches.append(P.record(rp))
            else:
                item=next(r for r in s['raw_teacher_caches'] if Path(r['path']).name==f'{method}_RAW_normal128.pt');E.remember(ctx.source_records,item['path'],item['sha256']);raw_value=torch.load(item['path'],map_location='cpu',weights_only=False);raw_caches.append(item)
            raw_normal=check_raw(P,raw_value,data,manifest_sha).to(ctx.device);fitter.raw_normal=raw_normal
            normal,y,aux,ay,clean=(data[k].to(ctx.device) for k in ('normal','y','aux','ay','clean'))
            for seed in c['seeds']:
                admission=P.admit_control(ctx,E,c,prior,fitter,method,seed,(normal,y,aux,ay,clean));h,draws=replay(fitter,normal,y,aux,ay,clean,seed)
                need(h==admission['labels_sha256'] and draws==P.trajectory(P.read_log(admission['original_log']['path'])),'RAW teacher paired replay changed labels/RNG')
                admissions.append(dict(admission,raw_teacher_paired_replay_exact=True))
            del data,raw_value,raw_normal,normal,y,aux,ay,clean;fitter.raw_normal=None;gc.collect();torch.cuda.empty_cache()
        need(len(admissions)==6 and not fitter.records,'All-six admission must precede updates')
        for seed in c['seeds']:need(len({r['labels_sha256'] for r in admissions if r['seed']==seed})==1,'Cross-generator labels differ')
        P.write(a.out/'pre_update_admission.json',dict(status='passed',optimizer_updates=0,admissions=admissions,BAF_input_caches=caches,RAW_teacher_caches=raw_caches))
        for method in c['methods']:
            data,_=previous_data(P,E,ctx,c,previous_smoke,method);item=next(r for r in raw_caches if Path(r['path']).name==f'{method}_RAW_normal128.pt');need(sha(item['path'])==item['sha256'],'RAW cache changed')
            raw_value=torch.load(item['path'],map_location='cpu',weights_only=False);fitter.raw_normal=check_raw(P,raw_value,data,manifest_sha).to(ctx.device)
            normal,y,aux,ay,clean=(data[k].to(ctx.device) for k in ('normal','y','aux','ay','clean'))
            for seed in seeds:
                need(time.monotonic()-start<c['phase_runtime_bounds_seconds'][a.phase],'Runtime exceeded');ad=next(r for r in admissions if (r['method'],r['seed'])==(method,seed))
                state=torch.load(ad['source_model']['path'],map_location='cpu',weights_only=True);need(state_sha(state)==ad['source_state_sha256'],'Source init changed')
                model,_=fitter.fit_center(ctx.scale,state,normal,y,aux,ay,seed,smoke,clean,0);r=fitter.records[-1]
                log=a.out/'training_evidence'/f'{method}_seed{seed}_physical_supervised_center.jsonl';h,draws=replay(fitter,normal,y,aux,ay,clean,seed,smoke)
                need(r['labels_sha256']==h and P.trajectory(P.read_log(log))==draws==P.trajectory(P.read_log(ad['original_log']['path']))[:r['updates']],'Actual paired trajectory differs')
                mp=a.out/f'{method}_seed{seed}_raw_teacher_baf_center.pt';torch.save(model.state_dict(),mp)
                r.update(canonical_arm=c['candidate_name'],new_optimizer_updates=r['updates'],teacher_input='RAW_normal_paired_affine',student_input='unchanged_fixed_BAF',final_model=P.record(mp),training_log=P.record(log))
                if not smoke:results.append(evaluate(P,ctx,E,method,seed,model,data['views'],reference,old_runs))
                del model,state;gc.collect()
            del data,raw_value,normal,y,aux,ay,clean;fitter.raw_normal=None;gc.collect();torch.cuda.empty_cache()
        gate=None if smoke else P.stage_gate([dict(r,arms=dict(r['arms'],baf_center=r['arms']['raw_teacher_baf_center'])) for r in results])
        need(len(fitter.records)==(2 if smoke else 6) and (smoke or len(gate['checks'])==36),'Incomplete matrix')
        all_sources=sources+ctx.source_records;need(all(sha(r['path'])==r['sha256'] for r in all_sources),'Source/input bytes changed');need(time.monotonic()-start<c['phase_runtime_bounds_seconds'][a.phase],'Phase bound exceeded')
        P.write(a.out/'completion.json',dict(schema='pose_baf_raw_teacher_adaptation_v1',status='complete_synthetic_only',phase=a.phase,canonical_candidate=c['candidate_name'],
            new_optimizer_updates=sum(r['updates'] for r in fitter.records),records=fitter.records,results=results,center_stage_gate=gate,full288_admissions=admissions,input_caches=caches,raw_teacher_caches=raw_caches,
            contract_sha256=CONTRACT_SHA,source_manifest_sha256=manifest_sha,source_records=all_sources,policy_sha256=POLICY_SHA,real_scores_computed=False,
            filtered_teacher_control=dict(path=str(Path(c['previous_baf_study'])/'completion.json'),sha256=c['previous_baf_study_completion_sha256'],immutable_reuse=True,new_optimizer_updates=0),
            student_input='unchanged_previous_fixed_BAF128_cache',normal_teacher_input='RAW_normal_paired_same_augmentation',
            checkpoint_verification_scope='model_optimizer_readback_only_no_resume_claim',resume_supported=False,
            original30_gate=dict(status='not_evaluated',thresholds_unchanged=True),original8_confirmation=dict(status='not_evaluated',thresholds_unchanged=True),elapsed_seconds=time.monotonic()-start))
    except BaseException as error:
        with (a.out/'failure.json').open('x') as f:json.dump(dict(status='failed',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start),f,indent=2)
        raise

if __name__=='__main__':main()
