"""Registered sampler-only, synthetic-only diagnostic fork of complete flow CPs."""
import argparse,copy,gc,hashlib,json,math,os,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import flow_synthetic_diagnostic as D
from sampling import schedules,STRATA

def need(x,m):
    if not x:raise ValueError(m)
def write(p,v):D.write(p,v)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def counts(opt):return sorted(set(int(v['step'].item()) for v in opt.state.values() if 'step' in v))
def digest_tensors(values):
    h=hashlib.sha256()
    for value in values:h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def restore(trainer,models,cp):
    model=models.steflow().cuda();model.load_state_dict(cp['state_dict'],strict=True)
    need(not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()),'Unexpected BN state in frozen model')
    need(trainer.state_sha256(model)==cp['current_state_sha256'],'Warm model SHA differs')
    opt=torch.optim.Adam([{'params':model.bias_parameters(),'weight_decay':0.},{'params':model.weight_parameters(),'weight_decay':.0004}],lr=.0004,betas=(.9,.999))
    opt.load_state_dict(copy.deepcopy(cp['optimizer']))
    scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[5,10,20],gamma=.7);scheduler.load_state_dict(copy.deepcopy(cp['scheduler']))
    lr=[float(g['lr']) for g in opt.param_groups]
    need(all(math.isfinite(x) and x>0 for x in lr) and counts(opt)==[8910],'Warm LR/Adam counters invalid')
    need(scheduler.last_epoch==45 and cp['next_epoch']==45 and cp['next_position_samples']==0,'Original complete45-epoch boundary differs')
    restore_rng(dict(python=cp['python_rng'],numpy=cp['numpy_rng'],torch=cp['torch_rng'],cuda=cp['cuda_rng']))
    model.train();return model,opt,scheduler,lr

def aggregate(rows):
    result={}
    for name in rows[0]:
        if name=='invalid_geometry_or_visibility':continue
        e=[r[name]['epe_px'] for r in rows if r[name].get('epe_px') is not None]
        z=[r[name]['zero_epe_px'] for r in rows if r[name].get('zero_epe_px') is not None]
        result[name]=dict(epe_px=float(np.mean(e)) if e else None,zero_epe_px=float(np.mean(z)) if z else None,supported_members=len(e))
    return result

def evaluate(trainer,model,cache,val_entries):
    saved=rng_state();model.eval();rollouts=[]
    try:
        for rid in sorted(cache['eventgenesis']['validation']):
            a=cache['eventgenesis']['validation'][rid];b=cache['v2e']['validation'][rid]
            data=cache[model._diagnostic_method]['validation'][rid];rows=[]
            for i in range(66):
                with torch.inference_mode():
                    event,_,_=trainer.val_input(data,i,torch.device('cuda:0'))
                    pred=trainer.model_output_to_flow_hw(model(event,352,use_dt4=False,flow_seq=False))[:,46:306,3:349]
                gt=data['flow_gt'][i];valid=data['flow_valid'][i];active=D.active(a['events'][i])&D.active(b['events'][i])
                metrics=D.region_summary(pred,gt,valid,active);zero=D.region_summary(np.zeros_like(pred),gt,valid,active)
                for name,value in metrics.items():
                    if name!='invalid_geometry_or_visibility':value['zero_epe_px']=zero[name]['epe_px']
                official,_=trainer.compute_epe(pred,gt,valid,prediction_unit='px_per_interval',gt_unit='px_per_interval')
                need(abs(official-metrics['all_physical_valid']['epe_px'])<=1e-6,'Native primary metric differs from original')
                metrics['all_physical_valid']['epe_px']=official
                rows.append(metrics)
            rollouts.append(dict(rollout=rid,stratum=val_entries[rid]['stratum'],regions=aggregate(rows),windows=rows))
            print('SYNTHETIC_FULL_VALIDATION',model._diagnostic_method,rid,flush=True)
        result=dict(rollouts=rollouts,regions=aggregate([r['regions'] for r in rollouts]),strata={},window_count=528,real_data_used=False)
        for st in STRATA:
            rr=[r for r in rollouts if r['stratum']==st];need(len(rr)==2,'Each validation stratum must contain2 rollouts')
            result['strata'][st]=aggregate([r['regions'] for r in rr])
        return result
    finally:model.train();restore_rng(saved)

def one_update(trainer,model,opt,cache,replay,arm,u):
    events=[];grays=[];labels=[];ids=[]
    for micro,ix in enumerate(replay['arms'][arm][8*u:8*u+8]):
        rid,i=replay['samples'][ix];data=cache[rid]
        sample={k:data[k][i:i+1] for k in ('events','gray_prev','gray_next','flow_gt','flow_valid')}
        seed=replay['augmentation_seed_sequences'][8*u+micro]
        event,gray,flow,valid=trainer.apply_joint_transform(sample,np.random.default_rng(np.random.SeedSequence(seed)),torch.device('cuda:0'))
        events.append(event);grays.append(gray);labels.append(digest_tensors([gray,flow,valid]));ids.append(str(data['sample_ids'][i]))
    opt.zero_grad(set_to_none=True);event=torch.cat(events);gray=torch.cat(grays)
    output=model(event,256,use_dt4=False,flow_seq=False)
    # Exactly the frozen original objective: no GT flow/valid argument.
    objective,photo,smooth=trainer.objective_dt1(output,gray[:,0:1],gray[:,1:2],event,sys.modules['multiscaleloss'])
    need(all(torch.isfinite(x) for x in (objective,photo,smooth)),'Nonfinite original losses')
    objective.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
    need(len(grads)==70 and all(torch.isfinite(g).all() for g in grads),'Expected70 finite parameter gradients')
    norm=float(torch.stack([g.detach().square().sum() for g in grads]).sum().sqrt());need(math.isfinite(norm) and norm>0,'Nonfinite/zero gradient norm')
    opt.step();need(counts(opt)==[8911+u],'Adam counter did not advance exactly once')
    return dict(additional_update=u+1,optimizer_update=8911+u,sample_ids=ids,label_augmentation_sha256=labels,
                augmentation_seed_sequences=replay['augmentation_seed_sequences'][8*u:8*u+8],loss=float(objective.detach()),photo=float(photo.detach()),
                smooth=float(smooth.detach()),gradient_l2=norm,finite_gradient_tensors=70,learning_rates=[g['lr'] for g in opt.param_groups],GT_flow_optimizer_supervision=False)

def save_checkpoint(path,trainer,models,model,opt,scheduler,source,additional):
    saved=rng_state();state=dict(schema='flow_motion_sampler_short_checkpoint_v1',state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        current_state_sha256=trainer.state_sha256(model),rng_state=saved,source_checkpoint=source,additional_updates=additional,
        optimizer_update=8910+additional,resume_allowed=False,resume_phase='diagnostic_only',real_data_used=False)
    torch.save(state,path);loaded=torch.load(path,map_location='cpu',weights_only=False)
    rebuilt=models.steflow().cuda();rebuilt.load_state_dict(loaded['state_dict'],strict=True)
    need(trainer.state_sha256(rebuilt)==state['current_state_sha256'],'Saved model failed independent reconstruction')
    opt2=torch.optim.Adam([{'params':rebuilt.bias_parameters(),'weight_decay':0.},{'params':rebuilt.weight_parameters(),'weight_decay':.0004}],lr=.0004,betas=(.9,.999))
    opt2.load_state_dict(loaded['optimizer']);need(counts(opt2)==[8910+additional],'Saved Adam counter differs')
    for a,b in zip(opt.state.values(),opt2.state.values()):
        need(a.keys()==b.keys(),'Saved Adam state keys differ')
        for k in a:need(torch.equal(a[k].cpu(),b[k].cpu()),'Saved Adam tensor differs')
    del rebuilt,opt2,loaded;restore_rng(saved)
    return dict(path=str(path),sha256=D.sha(path),model_rebuild_passed=True,optimizer_state_equal=True,additional_updates=additional)

def quality(results):
    checks=[]
    for method in ('eventgenesis','v2e'):
        c=next(r['validation'] for r in results if r['method']==method and r['arm']=='control')
        m=next(r['validation'] for r in results if r['method']==method and r['arm']=='motion')
        for name in ('all_physical_valid','common_event_active_valid'):
            x,y=c['regions'][name]['epe_px'],m['regions'][name]['epe_px'];checks.append(dict(method=method,region=name,control=x,candidate=y,passed=y<x))
        for st in STRATA:
            x,y=c['strata'][st]['all_physical_valid']['epe_px'],m['strata'][st]['all_physical_valid']['epe_px']
            checks.append(dict(method=method,stratum=st,control=x,candidate=y,threshold=1.05*x,passed=y<=1.05*x))
    return dict(passed=all(r['passed'] for r in checks),checks=checks,automatic_long_training_or_real_scoring=False,single_seed_development=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','study']);p.add_argument('--contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.contract.read_text());f=c['frozen_inputs'];R=D.validate_contract(f);root=Path(c['root'])
    need(c['schema']=='flow_motion_sampler_short_v1' and c['updates_per_arm']==200 and c['real_data_used'] is False and c['real_scores_used'] is False,'Wrong scope/budget')
    need(a.out.resolve()==root/a.mode and not a.out.exists(),'Independent output directory required')
    manifest=json.loads((root/'sources/source_manifest.json').read_text())
    for name,h in manifest.items():need(Path(name).name==name and D.sha(root/'sources'/name)==h,'Changed trial source: '+name)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==f['GPU_UUID'] and torch.cuda.device_count()==1,'Only leasedGPU3 may run this trial')
    need(D.sha(Path(f['training_source']))==f['training_source_sha256'] and D.sha(Path(f['compat_source']))==f['compat_sha256'],'Original training/warp changed')
    if a.mode=='study':need(json.loads((root/'smoke/completion.json').read_text())['status']=='complete','Smoke must pass before study')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    trainer=D.load_module(f['training_source'],'paired_short_original_trainer');sys.path.insert(0,f['model_root'])
    from verify_pinned_source import verify_pinned_source
    bundle=R/'sources/downstream_source_bundle.json';pin=next(r for r in json.loads(bundle.read_text())['repositories'] if r['repo']=='STE-FlowNet')
    need(pin['commit']=='eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6','Unexpected downstream model commit')
    source_verification=verify_pinned_source(Path(f['model_root']),'STE-FlowNet',pin['commit'],bundle)
    import models,models.corr,multiscaleloss
    models.corr.spatial_correlation_sample=trainer.pure_torch_spatial_correlation;trainer.patch_ste(models,multiscaleloss,Path(f['model_root']))
    cache={m:{} for m in f['methods']};lookups={s:D.entries(f,s) for s in ('train','validation')}
    for split in ('train','validation'):
        for m in f['methods']:cache[m][split]={}
        for rid in sorted(lookups[split]['eventgenesis']):
            aa,bb=[D.load_episode(lookups[split][m][rid]) for m in f['methods']];D.pair_equal(aa,bb)
            cache['eventgenesis'][split][rid]=aa;cache['v2e'][split][rid]=bb
    replay=schedules({rid:r['stratum'] for rid,r in lookups['train']['eventgenesis'].items()});write(a.out/'sampler_replay.json',replay)
    results=[];all_lr=[];deadline=time.monotonic()+c['deadline_seconds']
    try:
        for method in f['methods']:
            binding=f['methods_binding'][method];source=dict(path=binding['last_checkpoint'],sha256=binding['last_checkpoint_sha256'])
            need(D.sha(Path(source['path']))==source['sha256'],'Fixed complete last checkpoint changed')
            cp=torch.load(source['path'],map_location='cpu',weights_only=False)
            need(cp['generator']==method and cp['seed']==0 and cp['optimizer_update']==8910 and cp['resume_phase']=='complete' and cp['resume_allowed'] is False,'Only complete8910 diagnostic fork accepted')
            need(cp['config_sha256']==f['recipe_sha256'] and cp['data_manifest_sha256']==binding['data_manifest_sha256'] and cp['real_data_used'] is False,'Source CP lineage differs')
            before=None
            for arm in ('control','motion'):
                need(time.monotonic()<deadline,'Bounded trial deadline exceeded')
                run=a.out/(method+'_'+arm);run.mkdir(exist_ok=False);model,opt,scheduler,lr=restore(trainer,models,cp);model._diagnostic_method=method;all_lr.append(lr)
                if a.mode=='study' and before is None:
                    before=evaluate(trainer,model,cache,lookups['validation'][method]);write(a.out/(method+'_before.json'),before)
                    history=Path(binding['run'])/'training_metrics.jsonl'
                    matches=[json.loads(line) for line in history.open() if '"synthetic_validation"' in line]
                    last=next(r for r in reversed(matches) if r['optimizer_updates']==8910)
                    actual=before['regions']['all_physical_valid']['epe_px'];expected=last['model_mean_epe_px_per_30ms']
                    need(abs(actual-expected)<=1e-5,'Warm full-validation EPE disagrees with original8910 checkpoint')
                    write(a.out/(method+'_warm_validation_parity.json'),dict(passed=True,current=actual,original=expected,absolute_tolerance=1e-5,history_sha256=D.sha(history)))
                rows=[];n=1 if a.mode=='smoke' else 200
                with (run/'training_metrics.jsonl').open('x') as log:
                    for u in range(n):
                        need(time.monotonic()<deadline,'Bounded trial deadline exceeded')
                        row=one_update(trainer,model,opt,cache[method]['train'],replay,arm,u);rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                        if (u+1)%20==0:print('FLOW_PAIRED_UPDATE',method,arm,u+1,flush=True)
                        if u+1==198:scheduler.step();need([g['lr'] for g in opt.param_groups]==lr,'Unexpected post45 LR change')
                checkpoint=save_checkpoint(run/'last.pt',trainer,models,model,opt,scheduler,source,n)
                validation=evaluate(trainer,model,cache,lookups['validation'][method]) if a.mode=='study' else None
                r=dict(method=method,arm=arm,additional_updates=n,source_checkpoint=source,initial_learning_rates=lr,
                       label_sequence_sha256=hashlib.sha256(json.dumps([r['label_augmentation_sha256'] for r in rows]).encode()).hexdigest(),
                       sample_sequence_sha256=hashlib.sha256(json.dumps([r['sample_ids'] for r in rows]).encode()).hexdigest(),validation=validation,checkpoint=checkpoint)
                write(run/'completion.json',r);results.append(r);del model,opt,scheduler;gc.collect();torch.cuda.empty_cache()
            need(D.sha(Path(source['path']))==source['sha256'],'Formal last checkpoint was modified')
            del cp
        need(all(x==all_lr[0] for x in all_lr),'Actual warm learning rates differ')
        for arm in ('control','motion'):
            rows=[r for r in results if r['arm']==arm]
            need(len({r['label_sequence_sha256'] for r in rows})==1 and len({r['sample_sequence_sha256'] for r in rows})==1,'Cross-generator replay/physical label pairing failed')
        out=dict(status='complete',results=results,pairing_passed=True,real_data_used=False,real_scores_used=False,
                 GT_flow_optimizer_supervision=False,formal_weights_modified=False,source_manifest_sha256=D.sha(root/'sources/source_manifest.json'),model_source_verification=source_verification)
        if a.mode=='study':out['development_quality_gate']=quality(results)
        write(a.out/'completion.json',out)
    except BaseException:
        write(a.out/'failure.json',dict(status='failed',error=traceback.format_exc(),real_data_used=False));raise

if __name__=='__main__':main()
