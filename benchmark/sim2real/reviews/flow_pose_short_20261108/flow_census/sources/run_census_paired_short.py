#!/usr/bin/env python3
"""Isolated canonical Census/native-control study; preparation does not execute it."""
import argparse,copy,gc,hashlib,json,math,os,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import flow_synthetic_diagnostic as D
import flow_sampling_short as B
import soft_census_v2 as C
import census_contract as K

HELPER_SHA='5340990638eecf0865d2a322e45cb903f990a9d4077cf047f7cb40b5de3f4319'
MODEL_TREE='99f719b925e92d71fd10d97c29ae011a16440df56923f80c54d9b5dd128af80a'
LOSS_SHA='ff401317e49ce37ae0f8e9bfd6b82f133549b4a0271be35095f74bdc50a439c0'

def write(path,value):D.write(path,value)
def hash_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def sources(root):
    manifest=K.read(root/'sources/source_manifest.json')
    for name,digest in manifest.items():
        K.need(Path(name).name==name and D.sha(root/'sources'/name)==digest,'Changed new-study source: '+name)
    K.need(D.sha(Path(B.__file__))==HELPER_SHA,'Historical helper bytes changed')
    return D.sha(root/'sources/source_manifest.json')
def input_sources(c):
    f=c['frozen_inputs'];R=D.validate_contract(f)
    K.need(D.sha(f['training_source'])==f['training_source_sha256'] and D.sha(f['compat_source'])==f['compat_sha256'],'Original trainer/compat changed')
    bundle=R/'sources/downstream_source_bundle.json'
    K.need(D.sha(bundle)==c['downstream_source_bundle_sha256'],'Pinned source bundle changed')
    scripts=Path(f['training_source']).parent;sys.path.insert(0,str(scripts))
    K.need(D.sha(scripts/'run_steflow_stage2_pilot.py')==c['original_pilot_sha256'],'Original objective/sampler helper changed')
    from verify_pinned_source import verify_pinned_source
    pin=next(r for r in K.read(bundle)['repositories'] if r['repo']=='STE-FlowNet')
    K.need(pin['commit']=='eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6' and pin['included_file_count']==14 and pin['included_tree_sha256']==MODEL_TREE,'Different14-file model source tree')
    result=verify_pinned_source(Path(f['model_root']),'STE-FlowNet',pin['commit'],bundle)
    K.need(D.sha(Path(f['model_root'])/'multiscaleloss.py')==LOSS_SHA,'Native loss source changed')
    return R,result
def checkpoint(c,method):
    f=c['frozen_inputs'];b=f['methods_binding'][method];path=Path(b['last_checkpoint'])
    K.need(D.sha(path)==b['last_checkpoint_sha256'],'Original complete lastCP changed')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    K.need(cp['generator']==method and cp['seed']==0 and cp['optimizer_update']==8910 and cp['resume_phase']=='complete' and cp['resume_allowed'] is False,'Not original completed8910 diagnostic source')
    K.need(cp['config_sha256']==f['recipe_sha256'] and cp['data_manifest_sha256']==b['data_manifest_sha256'] and cp['real_data_used'] is False,'Original CP lineage differs')
    K.need(cp['initial_state_sha256']==b['initial_state_sha256'],'Original initial-state identity differs')
    K.need(len(cp['optimizer']['state'])==70,'Need all70 original Adam state entries')
    return cp,dict(path=str(path),sha256=b['last_checkpoint_sha256'],model_state_sha256=cp['current_state_sha256'])
def load_cache(c,splits):
    f=c['frozen_inputs'];look={s:D.entries(f,s) for s in splits};cache={m:{} for m in K.METHODS};domains=[]
    for split in splits:
        for m in K.METHODS:cache[m][split]={}
        for rid in sorted(look[split]['eventgenesis']):
            a,b=[D.load_episode(look[split][m][rid]) for m in K.METHODS];D.pair_equal(a,b)
            for method,data in zip(K.METHODS,(a,b)):
                domains.append(dict(method=method,split=split,rollout=rid,raw_endpoints=K.raw_endpoint_domain(data)))
                cache[method][split][rid]=data
    return cache,look,domains
def batch(trainer,cache,replay,u):
    events=[];grays=[];labels=[];ids=[];rngs=[]
    for micro,ix in enumerate(replay['arms']['native_control'][u*8:u*8+8]):
        rid,i=replay['samples'][ix];data=cache[rid]
        sample={k:data[k][i:i+1] for k in ('events','gray_prev','gray_next','flow_gt','flow_valid')}
        seed=replay['augmentation_seed_sequences'][u*8+micro]
        event,gray,flow,valid=trainer.apply_joint_transform(sample,np.random.default_rng(np.random.SeedSequence(seed)),torch.device('cuda:0'))
        # Raw NPZ was strictly checked first; this tensor has known native bilinear provenance.
        C._image(gray[:,0:1],interpolated=True);C._image(gray[:,1:2],interpolated=True)
        events.append(event);grays.append(gray);labels.append(B.digest_tensors([gray,flow,valid]));ids.append(str(data['sample_ids'][i]));rngs.append(seed)
    return torch.cat(events),torch.cat(grays),dict(sample_ids=ids,label_augmentation_sha256=labels,augmentation_seed_sequences=rngs)
def parameter_signature(model):
    rows=[dict(name=n,shape=list(p.shape),requires_grad=p.requires_grad,dtype=str(p.dtype)) for n,p in model.named_parameters()]
    selected=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    return rows,hash_json(rows),selected
def gradient_norm(grads,parameters,expected_non_none=70):
    K.need(len(grads)==len(parameters),'Gradient list is not the full requires_grad set')
    connected=[g for g in grads if g is not None]
    K.need(len(connected)==expected_non_none and all(torch.isfinite(g).all() for g in connected),'Unexpected count/nonfinite connected gradients')
    # Unused branches contribute explicit zeros to the complete trainable vector.
    value=float(sum(p.new_zeros((),dtype=torch.float64) if g is None else g.detach().double().square().sum()
                    for p,g in zip(parameters,grads)).sqrt())
    K.need(math.isfinite(value) and value>0,'Zero/nonfinite full-model gradient');return value
def rng_fingerprint():
    return dict(python=hash_json(random.getstate()),numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
                torch=B.digest_tensors([torch.get_rng_state()]),cuda=[B.digest_tensors([v]) for v in torch.cuda.get_rng_state_all()])
def setup(c):
    f=c['frozen_inputs'];R,verified=input_sources(c)
    K.need(os.environ.get('CUDA_VISIBLE_DEVICES')==f['GPU_UUID'] and torch.cuda.device_count()==1,'Only root-leased registered GPU is permitted')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    trainer=D.load_module(f['training_source'],'census_fixed_original_trainer');sys.path.insert(0,f['model_root'])
    import models,models.corr,multiscaleloss
    models.corr.spatial_correlation_sample=trainer.pure_torch_spatial_correlation
    trainer.patch_ste(models,multiscaleloss,Path(f['model_root']))
    return trainer,models,multiscaleloss,verified
def cpu_admission(c,root,out):
    R,verified=input_sources(c);rows=[];initial=[]
    K.need(not torch.cuda.is_initialized(),'CPU admission must not create or inherit a CUDA context')
    sys.path.insert(0,c['frozen_inputs']['model_root']);import models
    inventory_model=models.steflow()
    inventory,inventory_sha,trainable=parameter_signature(inventory_model)
    K.need(len(inventory)==82 and len(trainable)==82,'Pinned actual inventory differs from verified82 named/trainable parameters')
    del inventory_model;gc.collect()
    for method in K.METHODS:
        cp,source=checkpoint(c,method);initial.append(cp['initial_state_sha256']);del cp;gc.collect()
        for split in ('train','validation'):
            for rid,row in D.entries(c['frozen_inputs'],split)[method].items():
                K.need(D.sha(row['path'])==row['sha256'],'Bank hash changed')
                with np.load(row['path'],allow_pickle=False) as f:
                    domain=K.raw_endpoint_domain({k:np.asarray(f[k]) for k in ('gray_prev','gray_next')})
                rows.append(dict(method=method,split=split,rollout=rid,file=dict(path=row['path'],sha256=row['sha256']),raw_endpoint_domain=domain))
    K.need(len(set(initial))==1,'Different original generator initializations')
    K.need(not torch.cuda.is_initialized(),'CPU admission unexpectedly initialized CUDA')
    write(out/'completion.json',dict(schema='flow_census_cpu_admission_v1',status='passed',source_manifest_sha256=sources(root),
        model_source_verification=verified,bank_files=rows,checkpoint_inputs_unchanged=True,original_initial_state_sha256=initial[0],
        parameter_inventory=inventory,parameter_signature_sha256=inventory_sha,total_named_parameters=len(inventory),total_requires_grad_parameters=len(trainable),
        raw_endpoint_domain='strict finite[0,1]; no encoding tolerance',internal_interpolation_domain='8dtypeeps numerical check only; no clamp/tensor changes',
        CUDA_initialized=False,optimizer_updates=0,real_data_used=False))
def phase0(c,root,out):
    trainer,models,loss,verified=setup(c);cache,look,domains=load_cache(c,('train',))
    replay=K.paired_replay({rid:r['stratum'] for rid,r in look['train']['eventgenesis'].items()});write(out/'control_replay.json',replay)
    records=[];deadline=time.monotonic()+c['phase0_deadline_seconds']
    for method in K.METHODS:
        cp,source=checkpoint(c,method);model,opt,scheduler,lr=B.restore(trainer,models,cp)
        signature,signature_sha,named=parameter_signature(model);parameters=tuple(p for _,p in named)
        K.need(signature_sha==K.read(root/'cpu/completion.json')['parameter_signature_sha256'],'GPU parameter inventory differs from pinned CPU inventory')
        write(out/(method+'_parameter_signature.json'),dict(parameters=signature,sha256=signature_sha,
            total_named_parameters=len(signature),total_requires_grad_parameters=len(parameters)))
        for i in range(8):
            K.need(time.monotonic()<deadline,'Phase0 deadline exceeded')
            event,gray,meta=batch(trainer,cache[method]['train'],replay,i);output=model(event,256,use_dt4=False,flow_seq=False)
            _,photo,_=trainer.objective_dt1(output,gray[:,0:1],gray[:,1:2],event,loss)
            census,details=C.multiscale_census_sum(output,gray[:,0:1],gray[:,1:2],patch_size=7,inputs_interpolated=True)
            gn=torch.autograd.grad(photo,parameters,retain_graph=True,allow_unused=True)
            gcensus=torch.autograd.grad(census,parameters,allow_unused=True)
            a,b=gradient_norm(gn,parameters),gradient_norm(gcensus,parameters)
            records.append(dict(method=method,batch=i,split='train',optimizer_updates=0,GT_in_lambda=False,
                replay_sha256=replay['replay_sha256'],gradient_space='all_requires_grad_parameters_zero_unused',model_state_sha256=source['model_state_sha256'],
                total_requires_grad_parameters=len(parameters),parameter_signature_sha256=signature_sha,
                native_unused_parameter_names=[n for (n,_),g in zip(named,gn) if g is None],
                census_unused_parameter_names=[n for (n,_),g in zip(named,gcensus) if g is None],
                labels_and_augmentation_sha256=hash_json(meta),native_photo_gradient_l2=a,census_photo_gradient_l2=b,census_to_native_ratio=b/a,
                native_finite_gradient_tensors=70,census_finite_gradient_tensors=70,support_fractions=[float(d['loss_support'].mean()) for d in details],
                source_checkpoint=source,actual_learning_rates=lr))
            K.need(B.counts(opt)==[8910] and trainer.state_sha256(model)==source['model_state_sha256'],'Phase0 changed weights/Adam')
            del output,photo,census,gn,gcensus,event,gray,details;gc.collect();torch.cuda.empty_cache()
        K.need(D.sha(source['path'])==source['sha256'],'OriginalCP bytes changed')
        del model,opt,scheduler,cp;gc.collect();torch.cuda.empty_cache()
    result=K.shared_lambda(records,replay['replay_sha256'])
    result.update(schema='flow_census_full_parameter_median_calibration_v1',status='calibrated_train_only',source_manifest_sha256=sources(root),
        model_source_verification=verified,raw_endpoint_domains=domains,old_checkpoints_unchanged=True,optimizer_updates=0,
        real_data_used=False,real_scores_used=False,GT_flow_optimizer_supervision=False,GT_direction_oracle=None,
        GT_note='GT only used in paired label hashes. No augmented-GT direction oracle; native bilinear-GT/NN-valid boundaries are not a training bug claim.')
    _,end_verified=input_sources(c);K.need(end_verified==verified,'Original14-file model source changed during calibration')
    write(out/'completion.json',result)
    write(out/'calibration_lock.json',dict(schema='flow_census_calibration_lock_v1',status='frozen_before_optimizer',
        completion=K.record(out/'completion.json'),shared_census_weight=result['shared_census_weight'],optimizer_updates=0,
        source_manifest_sha256=sources(root)))
def load_calibration(root,replay_sha):
    path=root/'phase0/completion.json';p=K.read(path)
    lock=K.read(root/'phase0/calibration_lock.json')
    K.need(lock['schema']=='flow_census_calibration_lock_v1' and lock['status']=='frozen_before_optimizer' and
        K.check(lock['completion'])==path and lock['shared_census_weight']==p['shared_census_weight'] and lock['source_manifest_sha256']==sources(root),'Calibration was not frozen before optimizer')
    K.need(p['schema']=='flow_census_full_parameter_median_calibration_v1' and p['status']=='calibrated_train_only' and
        p['source_manifest_sha256']==sources(root) and p['optimizer_updates']==0 and p['real_data_used'] is False and p['GT_in_lambda'] is False,'Invalid phase0 receipt')
    checked=K.shared_lambda(p['records'],replay_sha)
    for k,v in checked.items():K.need(p[k]==v,'Frozen calibration value differs: '+k)
    return p,K.record(path)
def one_update(trainer,model,opt,cache,replay,arm,u,weight,loss):
    event,gray,meta=batch(trainer,cache,replay,u);opt.zero_grad(set_to_none=True)
    output=model(event,256,use_dt4=False,flow_seq=False)
    native_total,native_photo,smooth=trainer.objective_dt1(output,gray[:,0:1],gray[:,1:2],event,loss)
    support=None;raw_census=None
    if arm=='native_control':objective=native_total
    else:
        census,details=C.multiscale_census_sum(output,gray[:,0:1],gray[:,1:2],patch_size=7,inputs_interpolated=True)
        objective=weight*census+10*smooth;raw_census=float(census.detach());support=[float(d['loss_support'].mean()) for d in details]
    K.need(all(torch.isfinite(v) for v in (objective,native_photo,smooth)),'Nonfinite study loss')
    objective.backward();signature,signature_sha,named=parameter_signature(model);parameters=tuple(p for _,p in named)
    norm=gradient_norm([p.grad for p in parameters],parameters)
    opt.step();K.need(B.counts(opt)==[8911+u] and len(opt.state)==70,'Adam did not advance all70 states exactly once')
    return dict(additional_update=u+1,optimizer_update=8911+u,**meta,loss=float(objective.detach()),native_photo=float(native_photo.detach()),
        census_raw=raw_census,shared_census_weight=weight,smooth=float(smooth.detach()),support_fractions=support,
        gradient_l2=norm,finite_gradient_tensors=70,learning_rates=[g['lr'] for g in opt.param_groups],
        total_requires_grad_parameters=len(parameters),parameter_signature_sha256=signature_sha,
        rng_after_update=rng_fingerprint(),GT_flow_optimizer_supervision=False)
def before_validation(c,trainer,model,cache,look,method,source,root,out):
    reuse=c['warm_before_reuse'].get(method)
    if reuse:
        receipt=K.read(K.check(reuse['receipt']));before=K.read(K.check(reuse['before']));parity=K.read(K.check(reuse['parity']))
        K.need(receipt['passed'] is True and receipt['source_checkpoint']==source and
            receipt['data_manifest_sha256']==c['frozen_inputs']['methods_binding'][method]['data_manifest_sha256'] and
            receipt['training_source_sha256']==c['frozen_inputs']['training_source_sha256'] and receipt['before']==reuse['before'] and receipt['parity']==reuse['parity'],'Warm reuse model/source binding insufficient')
        K.need(before['window_count']==528 and len(before['rollouts'])==8 and set(before['strata'])==set(K.STRATA) and parity['passed'] is True,'Warm reuse is not full528')
        K.need(parity['current']==before['regions']['all_physical_valid']['epe_px'],'Warm reuse EPE/parity differs')
        write(out/(method+'_warm_reuse.json'),dict(used=True,receipt=reuse,source_checkpoint=source));return before
    saved=B.rng_state();before=B.evaluate(trainer,model,cache,look['validation'][method])
    K.need(rng_fingerprint()==dict(python=hash_json(saved['python']),numpy=hashlib.sha256(repr(saved['numpy']).encode()).hexdigest(),
        torch=B.digest_tensors([saved['torch']]),cuda=[B.digest_tensors([v]) for v in saved['cuda']]),'Validation altered restored RNG')
    history=Path(c['frozen_inputs']['methods_binding'][method]['run'])/'training_metrics.jsonl'
    matches=[json.loads(line) for line in history.open() if '"synthetic_validation"' in line]
    old=next(r for r in reversed(matches) if r['optimizer_updates']==8910)
    actual=before['regions']['all_physical_valid']['epe_px'];expected=old['model_mean_epe_px_per_30ms']
    K.need(abs(actual-expected)<=1e-5,'Warm full528 parity against formal8910 failed')
    write(out/(method+'_before.json'),before)
    write(out/(method+'_warm_validation_parity.json'),dict(passed=True,current=actual,original=expected,absolute_tolerance=1e-5,
        history=K.record(history),source_checkpoint=source,model_state_sha256=trainer.state_sha256(model),reused=False))
    return before
def study(c,root,out,smoke):
    trainer,models,loss,verified=setup(c);cache,look,domains=load_cache(c,('train',) if smoke else ('train','validation'))
    replay=K.paired_replay({rid:r['stratum'] for rid,r in look['train']['eventgenesis'].items()});write(out/'control_replay.json',replay)
    calibration,calibration_record=load_calibration(root,replay['replay_sha256']);weight=calibration['shared_census_weight']
    results=[];all_lr=[];deadline=time.monotonic()+c['study_deadline_seconds'];initial=[]
    for method in K.METHODS:
        cp,source=checkpoint(c,method);initial.append(cp['initial_state_sha256']);before=None
        for arm in K.ARMS:
            K.need(time.monotonic()<deadline,'Bounded study deadline exceeded')
            run=out/(method+'_'+arm);run.mkdir(exist_ok=False)
            model,opt,scheduler,lr=B.restore(trainer,models,cp);model._diagnostic_method=method;all_lr.append(lr)
            start_rng=rng_fingerprint()
            if not smoke and before is None:before=before_validation(c,trainer,model,cache,look,method,source,root,out)
            rows=[];n=1 if smoke else 200
            with (run/'training_metrics.jsonl').open('x') as log:
                for u in range(n):
                    K.need(time.monotonic()<deadline,'Bounded study deadline exceeded')
                    row=one_update(trainer,model,opt,cache[method]['train'],replay,arm,u,weight,loss)
                    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                    if u+1==198:scheduler.step();K.need([g['lr'] for g in opt.param_groups]==lr,'Original control scheduler boundary changed')
                    if (u+1)%20==0:print('FLOW_CENSUS_UPDATE',method,arm,u+1,flush=True)
            saved=B.save_checkpoint(run/'last.pt',trainer,models,model,opt,scheduler,source,n)
            loaded=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            K.need(loaded['scheduler']==scheduler.state_dict(),'Saved scheduler differs')
            K.need(set(loaded['rng_state'])=={'python','numpy','torch','cuda'},'Incomplete saved RNG');del loaded
            validation=None if smoke else B.evaluate(trainer,model,cache,look['validation'][method])
            row=dict(method=method,seed=0,arm=arm,additional_updates=n,optimizer_update=8910+n,source_checkpoint=source,
                initial_learning_rates=lr,initial_rng_fingerprint=start_rng,shared_calibration=calibration_record,
                shared_census_weight=weight,source_manifest_sha256=sources(root),
                label_sequence_sha256=hash_json([r['label_augmentation_sha256'] for r in rows]),
                sample_sequence_sha256=hash_json([r['sample_ids'] for r in rows]),
                augmentation_sequence_sha256=hash_json([r['augmentation_seed_sequences'] for r in rows]),
                finite_gradient_tensors=70,checkpoint=saved,checkpoint_format_note='Unchanged historical save_checkpoint storage helper; this receipt carries canonical Census study identity',
                independent_model_and_Adam_restore_passed=True,scheduler_and_RNG_saved=True,validation=validation,real_data_used=False)
            write(run/'completion.json',row);results.append(row)
            del model,opt,scheduler;gc.collect();torch.cuda.empty_cache()
        K.need(D.sha(source['path'])==source['sha256'],'Formal lastCP bytes were modified');del cp;gc.collect()
    K.need(len(set(initial))==1 and all(lr==all_lr[0] for lr in all_lr),'Paired original initialization/LR differ')
    for name in ('label_sequence_sha256','sample_sequence_sha256','augmentation_sequence_sha256'):
        K.need(len({r[name] for r in results})==1,'Every arm/method must use identical control replay: '+name)
    for method in K.METHODS:
        pair=[r for r in results if r['method']==method]
        K.need(pair[0]['initial_rng_fingerprint']==pair[1]['initial_rng_fingerprint'],'Same-method arm RNG differs')
    result=dict(schema='flow_census_paired_short_completion_v1',status='complete',mode='smoke' if smoke else 'study',results=results,
        pairing_passed=True,source_manifest_sha256=sources(root),model_source_verification=verified,raw_endpoint_domains=domains,
        shared_calibration=calibration_record,shared_census_weight=weight,formal_weights_modified=False,GT_flow_optimizer_supervision=False,
        real_data_used=False,real_scores_used=False,automatic_long_training_or_real_scoring=False)
    _,end_verified=input_sources(c);K.need(end_verified==verified,'Original14-file model source changed during study')
    if not smoke:result['development_quality_gate']=K.quality(results)
    write(out/'completion.json',result)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['cpu','phase0','smoke','study'])
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    c=K.validate(K.read(a.contract));root=Path(c['root']).resolve(strict=True);digest=sources(root)
    K.need(a.contract.resolve()==root/'sources/contract.json' and a.out.resolve()==root/a.mode and not a.out.exists(),'Use frozen contract and fresh canonical output')
    if a.mode!='cpu':
        admission=K.read(root/'cpu/completion.json');K.need(admission['status']=='passed' and admission['source_manifest_sha256']==digest,'CPU admission required')
    if a.mode=='study':
        smoke=K.read(root/'smoke/completion.json');K.need(smoke['status']=='complete' and smoke['mode']=='smoke' and smoke['pairing_passed'] is True and smoke['source_manifest_sha256']==digest and
            len(smoke['results'])==4 and all(r['additional_updates']==1 and r['finite_gradient_tensors']==70 and r['independent_model_and_Adam_restore_passed'] is True for r in smoke['results']),'All four actual smoke updates required before study')
        K.need(K.check(smoke['shared_calibration'])==root/'phase0/completion.json','Study calibration differs from actual four-arm smoke')
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        if a.mode=='cpu':cpu_admission(c,root,a.out)
        elif a.mode=='phase0':phase0(c,root,a.out)
        else:study(c,root,a.out,a.mode=='smoke')
    except BaseException:
        write(a.out/'failure.json',dict(status='failed',mode=a.mode,error=traceback.format_exc(),real_data_used=False));raise


if __name__=='__main__':main()
