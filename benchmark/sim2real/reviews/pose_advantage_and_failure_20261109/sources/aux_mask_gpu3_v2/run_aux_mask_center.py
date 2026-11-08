"""Registered RAW C1 + training-only visible-mask auxiliary center short trial."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree
import event_support as S
import legacy_center_bridge as P
import gpu3_resource_guard as G
from aux_center_policy import fit_center,balanced_mask_bce,state_sha
sys.dont_write_bytecode=True
POLICY_SHA='e1883b68715ecf067a9f4dccbd7a61af5f5dc26b557e061859f4066e7de8d38c'
REGION_SHA='6102a77a44c6604cc5349b9e58d2787fb8080c779561b7dc0b2bc1a143c0f91b'
DIAGNOSTIC_SHA='9e604e49ae861ab3831d641cd8c6d57d506017b060ebd30ee7a0b23f80e8771b'
PEAK_SHA='fff7c115df32a409ff242486dff654541ff8c67c130de40781ec5f9f11fd7bc8'
need=P.need;read=P.read;sha=P.sha;record=P.record;write=P.write


def validate_contract(c):
    need(c['schema']=='pose_aux_visible_mask_center_v1' and c['candidate_name']=='aux_visible_mask_center','Wrong candidate contract')
    need(c['methods']==['eventgenesis','v2e'] and c['seeds']==[0,1,2],'Complete paired matrix required')
    need(c['candidate_input']=='RAW' and c['teacher_input']=='same_RAW_normal_as_original_C1','RAW student/teacher required')
    need(c['auxiliary_loss']=='per_sample_class_balanced_fractional_BCE' and c['auxiliary_loss_coefficient']==1.0,'Auxiliary objective changed')
    need(c['auxiliary_head']=='independent_Conv2d_64_1_kernel1_training_only' and c['auxiliary_initialization_seed_base']==892100,'Auxiliary head/init changed')
    need(c['auxiliary_empty_class_policy']=='zero_contribution_no_class_renormalization_keep_window','Empty-class rule changed')
    need(c['updates_per_center_branch']==288 and c['new_center_optimizer_updates']==1728 and c['smoke_center_optimizer_updates']==2,'Update budget changed')
    need(c['epochs']==6 and c['batch_size']==32 and c['half_probability']==.75 and c['normal_teacher_weight']==2,'Original C1 sampling/objective changed')
    need(c['feature_learning_rate']==c['head_learning_rate']==.00015 and c['auxiliary_teacher_weight']==c['pair_weight']==0,'Original C1 optimizer/teacher changed')
    need(c['source_policy_sha256']==c['center_policy_sha256']==POLICY_SHA and c['region_source_sha256']==REGION_SHA and c['diagnostic_source_manifest_sha256']==DIAGNOSTIC_SHA,'Unregistered source dependency')
    need(c['wrong_peak_diagnostic_completion_sha256']==PEAK_SHA,'Wrong evidence trigger')
    need(c['candidate_initialization']=='original_C1_source_initialization_NOT_C1_finetune' and c['candidate_depth']=='reuse_C1_RAW_input_unchanged','Unequal source init/depth role')
    need(c['candidate_rotation']=='reuse_frozen_rotation_RAW_input_predicted_crop' and c['real_data_scoring_allowed'] is False and c['GT_used_only_for_training_loss_and_posthoc_scores'] is True,'Inference/isolation contract changed')
    need(c['phase_runtime_bounds_seconds']=={'smoke':3600,'study':3600} and c['resume_supported'] is False,'Runtime/resume contract changed')


def mask128(rows,bank):
    rows=[dict(r,rid=r.get('rid',r['id'].rsplit('/',1)[0])) for r in rows]
    regions=S.regions_for_rows(bank,rows,strict=True)
    masks=torch.from_numpy(np.stack([S.region_pyramid(regions[r['id']]['endpoint'])['input_fraction'] for r in rows]))[:,None]
    need(masks.shape==(len(rows),1,128,128) and torch.isfinite(masks).all() and ((masks>=0)&(masks<=1)).all(),'Physical mask128 invalid')
    return masks,regions


def raw_episode_inputs(ctx,bank,rids,method,lineage):
    """One original416 episode at a time; immutable RAW inputs and loss labels."""
    if (bank/'bank_manifest.json').is_file():lineage.append(record(bank/'bank_manifest.json'))
    lineage.append(record(bank/'events'/method/'manifest.json'));out=[];images=[];targets=[];masks=[];region_records={}
    for rid in rids:
        lineage.extend(record(p) for p in (bank/'physical'/(rid+'.h5'),bank/'events'/method/(rid+'.npy')))
        rows=ctx.fresh.adapter(bank,[bank/'physical'/(rid+'.h5')],method,ctx.encode)
        x=ctx.shared.tensors(rows,ctx.device).cpu();y=ctx.shared.labels(ctx.scale,rows,ctx.vertices,ctx.device).cpu();mask,rr=mask128(rows,bank)
        images.append(x);targets.append(y);masks.append(mask);region_records.update(rr)
        out.extend({k:v for k,v in r.items() if k!='image'} for r in rows);del rows;gc.collect()
    return dict(rows=out,raw=torch.cat(images),y=torch.cat(targets),mask=torch.cat(masks),regions=region_records)


def select(view,indices):
    return dict(rows=[view['rows'][i] for i in indices],**{k:view[k][indices] for k in ('raw','y','mask')})


def tensor_sha(value):return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def prepare_inputs(ctx,c,method):
    prep=read(Path(c['radiance_train'])/'completion.json');selected=prep['selected']
    need(prep['status']=='complete' and prep['real_data_used'] is False and len(selected)==96 and all(r.startswith('train_') for r in selected),'Original96 TRAIN required')
    npzs=sorted((ctx.R/'adapters'/method/'pose'/method/'train').glob('*.npz'));need(len(npzs)==96,'Original96 NPZ files required')
    lineage=[record(Path(c['radiance_train'])/'completion.json')]+[record(p) for p in npzs]
    loader=ctx.pilot.helpers(Path(c['causal_helpers_path']),ctx.R)['load'];rows=loader(method,'train');need(len(rows)==1529,'Original1529 normal NPZ windows required')
    ids=[r['id'] for r in rows];need(len(set(ids))==len(ids),'Duplicate NPZ windows')
    normal=ctx.shared.tensors(rows,ctx.device).cpu();y=ctx.shared.labels(ctx.scale,rows,ctx.vertices,ctx.device).cpu();nm,nr=mask128(rows,ctx.R/'pose_bank');del rows;gc.collect()
    base=raw_episode_inputs(ctx,ctx.R/'pose_bank',selected,method,lineage);lookup={r['id']:i for i,r in enumerate(base['rows'])}
    need(set(lookup)==set(ids),'Fresh/NPZ TRAIN inventory differs');indices=[lookup[s] for s in ids]
    need(torch.equal(normal,base['raw'][indices]) and torch.equal(nm,base['mask'][indices]),'NPZ/fresh RAW input or physical mask parity failed')
    levels=[]
    for level in ('half','eighth'):
        v=raw_episode_inputs(ctx,Path(c['radiance_train'])/level/'pose_bank',selected,method,lineage)
        need(ctx.native.identity(v['rows'])==ctx.native.identity(base['rows']),'Auxiliary original geometry/order differs');S.verify_matched_regions(base['regions'],v['regions']);levels.append(v)
    aux=torch.cat([v['raw'] for v in levels]);ay=torch.cat([v['y'] for v in levels]);am=torch.cat([v['mask'] for v in levels]);clean=torch.cat([base['raw'],base['raw']])
    need(torch.equal(ay[:1529],ay[1529:]),'Half/eighth original labels differ')
    valbank=ctx.D/'pose_bank';rids=sorted(p.stem for p in (valbank/'physical').glob('*.h5'));need(len(rids)==96,'Original96 independent VAL trajectories required')
    allval=raw_episode_inputs(ctx,valbank,rids,method,lineage);groups={rid:[] for rid in rids}
    for i,row in enumerate(allval['rows']):groups[row['rid']].append(i)
    views={'clean':select(allval,[g[i] for g in groups.values() for i in np.linspace(0,len(g)-1,4,dtype=int)])}
    for level in ('half','eighth'):
        bank=Path(c['radiance_validation'])/level/'pose_bank';vr=sorted(p.stem for p in (bank/'physical').glob('*.h5'));need(len(vr)==4 and not(set(vr)&set(selected)),'Independent four relit VAL required')
        v=raw_episode_inputs(ctx,bank,vr,method,lineage);S.verify_matched_regions({r['id']:allval['regions'][r['id']] for r in v['rows']},v['regions']);views[level]={k:v[k] for k in ('rows','raw','y','mask')}
    byid={r['id']:i for i,r in enumerate(allval['rows'])};views['matched_normal']=select(allval,[byid[r['id']] for r in views['half']['rows']])
    need(len(views['clean']['rows'])==384 and all(len(views[s]['rows'])==64 for s in ('matched_normal','half','eighth')),'Validation inventory changed')
    need(all(ctx.native.identity(views[s]['rows'])==ctx.native.identity(views['matched_normal']['rows']) for s in ('half','eighth')),'Matched validation labels differ')
    lineage=list({r['path']:r for r in lineage}.values());P.verify_input_lineage(lineage)
    masks=dict(normal_sha256=tensor_sha(nm),auxiliary_sha256=tensor_sha(am),normal_positive_mass=float(nm.sum()),auxiliary_positive_mass=float(am.sum()),normal_empty_masks=int((nm.sum((1,2,3))==0).sum()),auxiliary_empty_masks=int((am.sum((1,2,3))==0).sum()))
    return dict(normal=normal,y=y,aux=aux,ay=ay,clean=clean,normal_mask=nm,aux_mask=am,normal_ids=ids,aux_ids=[r['id'] for v in levels for r in v['rows']],views=views,input_lineage=lineage,mask_labels=masks)


def center_with_aux(model,head,images,mask):
    uv=[];losses=[];positive=[];negative=[];intersection=union=0.;buffer=[]
    hook=model.features.register_forward_hook(lambda module,args,output:buffer.append(output))
    try:
        with torch.inference_mode():
            for i in range(0,len(images),32):
                buffer.clear();p=model(images[i:i+32]);logits=head(buffer[0]);target=F.interpolate(mask[i:i+32],(16,16),mode='area');loss,stats=balanced_mask_bce(logits,target)
                uv.extend(p[1].cpu().numpy());losses.append((float(loss),len(p[1])));positive+=stats['foreground_mass'];negative+=stats['background_mass']
                # Fixed threshold solely reports coarse-label occupancy quality.
                pred=logits.sigmoid()>=.5;truth=target>=.5;intersection+=float((pred&truth).sum());union+=float((pred|truth).sum())
    finally:hook.remove()
    return np.asarray(uv),dict(balanced_bce=sum(v*n for v,n in losses)/len(images),foreground_mass=sum(positive),background_mass=sum(negative),empty_foreground_samples=sum(v==0 for v in positive),occupancy_iou_threshold05=intersection/union if union else None)


def evaluate(ctx,E,method,seed,model,auxhead,views,reference):
    depth,_,_=E.load_branch(ctx,method,seed,'physical_supervised','depth');control,_,_=E.load_branch(ctx,method,seed,'physical_supervised','center')
    parent=next(r for r in ctx.completion['results'] if (r['method'],r['seed'])==(method,seed));rp=ctx.A/f'jobs/117_pose_rotation_paired_20261106/results/{method}_seed{seed}_rotation.pt';E.remember(ctx.source_records,rp,parent['parents']['rotation_sha256'])
    head=ctx.rotation.RotationHead().to(ctx.device).eval();head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
    states=[(v,state_sha(v.state_dict())) for v in (depth,control,head,model,auxhead)];tree=cKDTree(ctx.vertices);output={arm:{} for arm in ('raw_C1','aux_visible_mask_center')}
    for condition,v in views.items():
        rows=v['rows'];raw=v['raw'].to(ctx.device);K=[r['K'] for r in rows]
        with torch.inference_mode():
            _,z,size=ctx.scale.estimate(depth,raw,K);old=ctx.native.estimate_centers(control,raw);new,maskscore=center_with_aux(model,auxhead,raw,v['mask'].to(ctx.device))
        if condition!='clean':
            label='normal' if condition=='matched_normal' else condition
            for i,row in enumerate(rows):
                r=reference[(method,seed,label,row['id'])];np.testing.assert_array_equal(np.asarray(r['uv_gt'],np.float32),row['uv'][0]);need(np.max(abs(old[i]-r['pred_uv']))<=2e-6,'Frozen RAW C1 center parity failed')
        for arm,uv in [('raw_C1',old),('aux_visible_mask_center',new)]:
            R=ctx.shared.frozen_rotation(ctx.rotation,head,raw,uv,size,K);score=ctx.shared.score(ctx.rotation,rows,uv,z,size,v['y'][:,2].numpy(),R,ctx.vertices,tree,ctx.threshold)
            output[arm][condition]=dict(center_epe_px=P.center_epe(rows,uv),pose=score,windows=len(rows),rollouts=len({r['rid'] for r in rows}),window_ids=[r['id'] for r in rows],auxiliary_mask_diagnostic=maskscore if arm!='raw_C1' else None)
        if condition!='matched_normal':E.compare_summary(output['raw_C1'][condition]['pose']['summary'],parent['arms']['physical_supervised'][condition]['summary'])
        del raw
    need(all(state_sha(v.state_dict())==old for v,old in states),'Frozen inference state changed')
    return dict(method=method,seed=seed,arms=output)


def stage_gates(results):
    checks=[];matched=[]
    for run in results:
        for condition in ('clean','half','eighth','matched_normal'):
            before=run['arms']['raw_C1'][condition];after=run['arms']['aux_visible_mask_center'][condition];rows=matched if condition=='matched_normal' else checks
            rows.append(dict(method=run['method'],seed=run['seed'],condition=condition,metric='center_epe_px',passed=after['center_epe_px']<before['center_epe_px'] if condition=='eighth' else after['center_epe_px']<=before['center_epe_px']))
            rows.append(dict(method=run['method'],seed=run['seed'],condition=condition,metric='success',passed=after['pose']['summary']['success']-before['pose']['summary']['success']>=(-.02 if condition=='clean' else 0.)))
    return dict(center_stage36=dict(checks=checks,passed=all(r['passed'] for r in checks)),matchednormal12=dict(checks=matched,passed=all(r['passed'] for r in matched)),scope='synthetic_center_stage_only_not_formal30_or8')


def verify_smoke(s,contract_sha,source_sha):
    need(s['schema']=='pose_aux_visible_mask_center_v1' and s['status']=='complete_synthetic_only' and s['phase']=='smoke','Actual complete smoke required')
    need(s['contract_sha256']==contract_sha and s['source_manifest_sha256']==source_sha and s['real_scores_computed'] is False and s['new_optimizer_updates']==2,'Smoke binding/isolation/budget differs')
    need(len(s['records'])==2 and {(r['method'],r['seed']) for r in s['records']}=={('eventgenesis',0),('v2e',0)},'Incomplete paired smoke')
    need(all(r['updates']==1 and r['first_native_loss_oracle']['passed'] and r['teacher_unchanged'] and r['size_unchanged'] and r['independent_model_optimizer_rebuild'] and r['optimizer_state_equal'] for r in s['records']),'Smoke objective/readback failed')
    need(len(s['full288_admissions'])==6 and all(r['full288_replay_exact'] for r in s['full288_admissions']),'Six full tapes not admitted')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--contract-sha256',required=True);p.add_argument('--source-manifest-sha256',required=True)
    p.add_argument('--resource-contract',type=Path,required=True);p.add_argument('--resource-contract-sha256',required=True)
    p.add_argument('--phase',choices=('smoke','study'),required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke-receipt',type=Path);p.add_argument('--smoke-receipt-sha256');a=p.parse_args()
    need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'New absolute output required');a.out.mkdir();start=time.monotonic()
    try:
        need(sha(a.contract)==a.contract_sha256,'Root-registered contract differs');c=read(a.contract);validate_contract(c)
        torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        own=Path(__file__).parent;sources=[];P.verify_sources(own,a.source_manifest_sha256,sources);need(sha(own/'pose_priority_policy.py')==POLICY_SHA and sha(own/'event_support.py')==REGION_SHA,'Frozen helper copy differs')
        D=Path(c['diagnostic_source_dir']);P.verify_sources(D,DIAGNOSTIC_SHA,sources);E=P.load('aux_mask_original_export',D/'export_pose_factors.py')
        need(sha(c['base_contract_path'])==c['base_contract_sha256'],'Base contract differs');base=read(c['base_contract_path']);need(all(c[k]==v for k,v in base.items() if k!='schema'),'Inherited physical/source fields differ')
        resource_start=G.check_resource(a.resource_contract,a.resource_contract_sha256,a.contract_sha256)
        ctx=E.load_context(base,check_gpu=False);ctx.device=torch.device('cuda:0')
        G.check_resource(a.resource_contract,a.resource_contract_sha256,a.contract_sha256)
        E.remember(ctx.source_records,a.resource_contract,a.resource_contract_sha256);E.remember(ctx.source_records,resource_start['parent_script']['path'],resource_start['parent_script']['sha256'])
        for path,digest in [(a.contract,a.contract_sha256),(c['base_contract_path'],c['base_contract_sha256']),(c['causal_helpers_path'],c['causal_helpers_sha256']),(c['center_policy_path'],POLICY_SHA),
            (c['forward_completion'],c['forward_completion_sha256']),(Path(c['prior_study'])/'completion.json',c['prior_study_completion_sha256']),(Path(c['prior_study'])/'priority_execution_receipt.json',c['prior_execution_receipt_sha256']),
            (c['wrong_peak_diagnostic_completion'],PEAK_SHA)]:E.remember(ctx.source_records,path,digest)
        peak=read(c['wrong_peak_diagnostic_completion']);need(peak['schema']=='pose_wrong_peak_full_cohort_cpu_v1' and peak['status']=='complete' and peak['prediction_cases']==1152 and peak['optimizer_updates']==0 and peak['real_data_used'] is False,'Actual frozen wrong-peak evidence missing')
        prior=read(Path(c['prior_study'])/'priority_execution_receipt.json');need(prior['status']=='complete_synthetic_only' and prior['real_scores_computed'] is False,'Reliable immutable C1 required')
        forward=read(c['forward_completion']);need(forward['status']=='complete' and forward['optimizer_updates']==0 and forward['real_data_used'] is False and forward['models_and_sources_unchanged'] and forward['native_parity_passed'] and forward['contract_sha256']==c['base_contract_sha256'],'Original C1 reference parity/isolation missing')
        ref={}
        for r in forward['artifacts']:
            if Path(r['path']).name=='predictions.jsonl':
                E.remember(ctx.source_records,r['path'],r['sha256'])
                with Path(r['path']).open() as f:
                    for line in f:
                        row=json.loads(line)
                        if row['arm']=='C1':
                            k=(row['method'],row['seed'],row['condition'],row['sample_id']);need(k not in ref,'Duplicate reference');ref[k]=row
        need(len(ref)==1152,'Complete C1 fixed64 references required');smoke=a.phase=='smoke';seeds=[0] if smoke else c['seeds'];smoke_receipt=None
        if not smoke:
            need(a.smoke_receipt is not None and sha(a.smoke_receipt)==a.smoke_receipt_sha256,'SHA-bound smoke required');smoke_receipt=read(a.smoke_receipt);verify_smoke(smoke_receipt,a.contract_sha256,a.source_manifest_sha256);E.remember(ctx.source_records,a.smoke_receipt,a.smoke_receipt_sha256)
            need(smoke_receipt['runtime_resource_contract_sha256']==a.resource_contract_sha256 and smoke_receipt['runtime_reference_baseline']['status']=='passed','Smoke runtime binding/numerical baseline differs')
        tape=SimpleNamespace(shared=ctx.shared,native=ctx.native);caches=[];admissions=[];masks=[];results=[];records=[]
        # Every method/seed's complete288 original replay precedes any update.
        for method in c['methods']:
            if smoke:
                data=prepare_inputs(ctx,c,method);cache=a.out/f'{method}_RAW_inputs_masks128.pt';torch.save(dict(contract_sha256=a.contract_sha256,source_manifest_sha256=a.source_manifest_sha256,data=data),cache);item=record(cache)
            else:
                item=next(r for r in smoke_receipt['input_caches'] if Path(r['path']).name==f'{method}_RAW_inputs_masks128.pt');E.remember(ctx.source_records,item['path'],item['sha256']);cached=torch.load(item['path'],map_location='cpu',weights_only=False)
                need(cached['contract_sha256']==a.contract_sha256 and cached['source_manifest_sha256']==a.source_manifest_sha256,'Cache source/contract differs');data=cached['data'];P.verify_input_lineage(data['input_lineage'])
            caches.append(item);ctx.source_records.extend(r for r in data['input_lineage'] if not any(s['path']==r['path'] for s in ctx.source_records));masks.append(dict(method=method,**data['mask_labels']))
            tensors=tuple(data[k].to(ctx.device) for k in ('normal','y','aux','ay','clean'));need(tuple(len(x) for x in tensors)==(1529,1529,3058,3058,3058),'Original complete RAW TRAIN dimensions differ')
            for seed in c['seeds']:admissions.append(P.admit_control(ctx,E,c,prior,tape,method,seed,tensors))
            del data,tensors;gc.collect();torch.cuda.empty_cache()
        need(len(admissions)==6 and all(len({r['labels_sha256'] for r in admissions if r['seed']==s})==1 for s in c['seeds']),'Cross-generator full288 labels differ')
        need(masks[0]['normal_sha256']==masks[1]['normal_sha256'] and masks[0]['auxiliary_sha256']==masks[1]['auxiliary_sha256'],'Shared physical mask labels differ across methods')
        write(a.out/'pre_update_admission.json',dict(status='passed',optimizer_updates=0,full288_admissions=admissions,mask_labels=masks,input_caches=caches))
        runtime_baseline=G.verify_reference_baseline(ctx,E,caches,ref,forward) if smoke else smoke_receipt['runtime_reference_baseline']
        write(a.out/'runtime_gpu3_reference_parity.json',runtime_baseline)
        for method in c['methods']:
            item=next(r for r in caches if Path(r['path']).name==f'{method}_RAW_inputs_masks128.pt');need(sha(item['path'])==item['sha256'],'Cache changed before training');data=torch.load(item['path'],map_location='cpu',weights_only=False)['data'];P.verify_input_lineage(data['input_lineage'])
            normal,y,aux,ay,clean,nm,am=(data[k].to(ctx.device) for k in ('normal','y','aux','ay','clean','normal_mask','aux_mask'))
            for seed in seeds:
                need(time.monotonic()-start<c['phase_runtime_bounds_seconds'][a.phase],'Phase runtime bound exceeded');admission=next(r for r in admissions if (r['method'],r['seed'])==(method,seed));state=torch.load(admission['source_model']['path'],map_location='cpu',weights_only=True);old=P.read_log(admission['original_log']['path'])
                need(state_sha(state)==admission['source_state_sha256'],'Original source init changed');model,head,history,r=fit_center(ctx.native,ctx.scale,state,normal,y,aux,ay,clean,nm,am,seed,smoke,a.out,method,old)
                tag=f'{method}_seed{seed}_aux_visible_mask_center';log=a.out/'training_evidence'/(tag+'.jsonl');need(P.trajectory(P.read_log(log))==P.trajectory(old)[:r['updates']],'Actual candidate tape differs from original C1')
                need(r['labels_sha256']==P.replay(tape,normal,y,aux,ay,clean,seed,smoke=smoke)[0],'Actual candidate labels differ')
                mp=a.out/(tag+'.pt');hp=a.out/(tag+'_auxiliary.pt');torch.save(model.state_dict(),mp);torch.save(head.state_dict(),hp);r.update(final_model=record(mp),final_auxiliary_head=record(hp),training_log=record(log));records.append(r)
                if not smoke:results.append(evaluate(ctx,E,method,seed,model,head,data['views'],ref))
                del model,head,state;gc.collect()
            del data,normal,y,aux,ay,clean,nm,am;gc.collect();torch.cuda.empty_cache()
        need(len(records)==(2 if smoke else 6) and sum(r['updates'] for r in records)==(2 if smoke else 1728),'Incomplete fixed update budget')
        sources+=ctx.source_records;need(all(sha(r['path'])==r['sha256'] for r in sources),'Frozen source/input/control changed');gates=None if smoke else stage_gates(results)
        need(smoke or (len(gates['center_stage36']['checks'])==36 and len(gates['matchednormal12']['checks'])==12),'Stage-gate inventory differs')
        need(time.monotonic()-start<c['phase_runtime_bounds_seconds'][a.phase],'Phase runtime exceeded')
        resource_end=G.check_resource(a.resource_contract,a.resource_contract_sha256,a.contract_sha256)
        write(a.out/'completion.json',dict(schema='pose_aux_visible_mask_center_v1',status='complete_synthetic_only',phase=a.phase,canonical_candidate='aux_visible_mask_center',new_optimizer_updates=sum(r['updates'] for r in records),records=records,
            contract_sha256=a.contract_sha256,source_manifest_sha256=a.source_manifest_sha256,source_records=sources,input_caches=caches,mask_labels=masks,full288_admissions=admissions,results=results,stage_gates=gates,real_scores_computed=False,
            runtime_resource_contract_sha256=a.resource_contract_sha256,resource_start=resource_start,resource_end=resource_end,runtime_reference_baseline=runtime_baseline,execution_gpu_uuid=G.GPU_UUID,original_frozen_source_gpu_uuid=base['gpu_uuid'],
            GT_forward_inputs=False,inference_auxiliary_head_used=False,pose_prediction_uses_auxiliary_head=False,auxiliary_head_evaluated_only_for_posthoc_mask_diagnostics=True,original30_gate=dict(status='not_retested_not_passed_by_this_probe',thresholds_unchanged=True),original8_confirmation=dict(status='not_retested_not_passed_by_this_probe',thresholds_unchanged=True),
            checkpoint_verification_scope='model_auxiliary_optimizer_readback_only_no_resume_claim',resume_supported=False,elapsed_seconds=time.monotonic()-start))
    except BaseException as error:
        write(a.out/'failure.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
