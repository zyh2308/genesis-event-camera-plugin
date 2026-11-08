"""Registered synthetic-only 4800->5100 paired loss continuation; no real loader."""
import argparse,copy,gc,hashlib,inspect,json,random,textwrap,time,types
from pathlib import Path
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from recon_short_diagnostic import setup,sha,save,state_sha
from activation_budget_diagnostic import (rng_state,restore_rng,rng_digest,digest_sequence,counts,metric_setup,evaluate)
from auxiliary_losses import PhotometricL1,LaplacianEdgeLoss

ARMS=('control','photometric','detail')

def setup_trial(campaign):
    context=setup(campaign)
    from trainer.trainer import Trainer
    original=Trainer.forward_sequence
    source=textwrap.dedent(inspect.getsource(original))
    needle="if loss_name == 'l2_loss':"
    assert source.count(needle)==1
    context.native_forward=original
    context.forward_source_sha256=hashlib.sha256(source.encode()).hexdigest()
    original_tc=context.temporal_class.__call__
    def count_calls(self,*a,**k):
        value=original_tc(self,*a,**k)
        if value is not None and not k.get('output_images',False):self.calls=getattr(self,'calls',0)+1
        return value
    context.temporal_class.__call__=count_calls
    return context

def shim_for(context,model,arm,original=False):
    shim=context.shim_for(model,'sigmoid');shim.loss_ftns[1].calls=0
    shim.forward_sequence=types.MethodType(context.native_forward,shim)
    if arm=='control':
        assert shim.forward_sequence.__func__ is context.native_forward
        return shim
    auxiliary=[('photometric_l1',PhotometricL1(weight=1.))]
    if arm=='detail':auxiliary.append(('laplacian_edge',LaplacianEdgeLoss(weight=.05)))
    def forward_with_aux(self,seq,all_losses=False):
        assert not all_losses
        images=[]
        hook=model.register_forward_hook(lambda module,inputs,output:images.append(output['image']))
        try:values=context.native_forward(self,seq)
        finally:hook.remove()
        assert len(images)==len(seq)==40
        source=next(k.split('/',1)[1] for k in values if k.startswith('perceptual_loss/'))
        additions=[]
        for name,fn in auxiliary:
            per_frame=[fn(pred,item['frame'].float().to(pred.device)) for pred,item in zip(images,seq)]
            value=sum(per_frame)/len(per_frame);values[name+'/'+source]=value;additions.append(value)
        values['loss']=values['loss']+sum(additions);values['loss/'+source]=values['loss']
        return values
    shim.forward_sequence=types.MethodType(forward_with_aux,shim)
    return shim

def load_start(context,campaign,method):
    root=campaign.parent/'20261107-recon-sigmoid-formal-v2/training/seed0'/method
    binding=json.loads((root/'synthetic_best.json').read_text());path=Path(binding['checkpoint'])
    assert path==root/'synthetic_best.pt' and binding['optimizer_update']==4800
    assert path.stat().st_size==binding['size_bytes'] and sha(path)==binding['sha256']
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['generator']==method and cp['seed']==0 and cp['optimizer_update']==4800
    assert cp['next_epoch']==80 and cp['next_batch_in_epoch']==0
    assert cp['model_args']['final_activation']=='sigmoid'
    assert cp['data_manifest_sha256']==sha(campaign/'adapters'/method/'reconstruction/adapter_manifest.json')
    assert cp['rng_state']['torch_cpu'].device.type=='cpu'
    return cp,binding

def restore_start(context,cp):
    model=context.model_for(0,'sigmoid');model.load_state_dict(cp['model'],strict=True)
    assert state_sha(model)==cp['model_sha256']
    c=context.cfg['optimizer']
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],betas=tuple(c['betas']),weight_decay=c['weight_decay'],amsgrad=c['amsgrad'])
    optimizer.load_state_dict(copy.deepcopy(cp['optimizer']))
    assert counts(optimizer)==[4800] and all(g['lr']==1e-4 for g in optimizer.param_groups)
    r=cp['rng_state'];restore_rng(dict(python=r['python'],numpy=r['numpy'],cpu=r['torch_cpu'],cuda=r['torch_cuda']))
    return model,optimizer

def verify_native(context,cp,sequence):
    records=[]
    # CUDA grid_sample backward uses atomic accumulation. Compare native/native
    # repeat variation as well as dispatch, never require bit-identical CUDA grads.
    for native in (True,True,True):
        model,opt=restore_start(context,cp);shim=shim_for(context,model,'control',original=native)
        loss=shim.forward_sequence(sequence);loss['loss'].backward()
        records.append(({k:float(v.detach()) for k,v in loss.items()},[p.grad.detach().cpu().clone() for p in model.parameters() if p.grad is not None]))
        del model,opt,shim,loss;gc.collect();torch.cuda.empty_cache()
    assert records[0][0]==records[1][0]==records[2][0]
    comparisons=[]
    for name,index in [('native_repeat_1',1),('native_repeat_2',2)]:
        errors=[float((a-b).abs().max()) for a,b in zip(records[0][1],records[index][1])]
        relative=float(torch.stack([(a-b).square().sum() for a,b in zip(records[0][1],records[index][1])]).sum().sqrt()/torch.stack([a.square().sum() for a in records[0][1]]).sum().sqrt())
        good=all(torch.allclose(a,b,rtol=1e-5,atol=1e-7) for a,b in zip(records[0][1],records[index][1]))
        aa=torch.cat([a.flatten().double() for a in records[0][1]]);bb=torch.cat([b.flatten().double() for b in records[index][1]])
        cosine=float(torch.nn.functional.cosine_similarity(aa,bb,dim=0))
        comparisons.append(dict(comparison=name,max_absolute_gradient_error=max(errors),relative_gradient_l2=relative,cosine=cosine,allclose=good))
    # Native CUDA self-repeat is diagnostic, not an assertion that atomic
    # gradient accumulation became deterministic. C0 uses this exact callable.
    result=dict(loss_exact=True,gradient_tensors=30,rtol=1e-5,atol=1e-7,comparisons=comparisons,losses=records[0][0],
        control_uses_native_function_identity=True,strict_cuda_gradient_reproducibility_claimed=False)
    return result

def auxiliary_cuda_oracle():
    from test_auxiliary_losses import official_rgb_edge_loss
    saved=rng_state();rows=[]
    try:
        for dtype in (torch.float32,torch.float64):
            torch.manual_seed(17);x=torch.rand(1,1,11,13,device='cuda',dtype=dtype,requires_grad=True);y=torch.rand_like(x)
            ours=LaplacianEdgeLoss(weight=1.)(x,y);oracle=official_rgb_edge_loss(x.repeat(1,3,1,1),y.repeat(1,3,1,1))
            ga=torch.autograd.grad(ours,x,retain_graph=True)[0];gb=torch.autograd.grad(oracle,x)[0]
            assert torch.allclose(ours,oracle,rtol=1e-6,atol=1e-8) and torch.allclose(ga,gb,rtol=1e-5,atol=1e-8)
            rows.append(dict(dtype=str(dtype),loss_error=float((ours-oracle).abs()),gradient_max_error=float((ga-gb).abs().max())))
        return rows
    finally:restore_rng(saved)

def detailed_evaluate(*args,**kwargs):
    # Preserve the verified full-rollout evaluator; add pointwise photometric
    # and gradient residuals instead of treating stronger edges as better edges.
    source=textwrap.dedent(inspect.getsource(evaluate))
    needle='frames.append(dict(index=i,'
    assert source.count(needle)==1
    extra="frames.append(dict(intensity_mae=float((pred-target).abs().mean()), intensity_bias=float((pred-target).mean()), edge_mae=float(.5*(((pred[...,1:,:]-pred[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])).abs().mean()+((pred[...,:,1:]-pred[...,:,:-1])-(target[...,:,1:]-target[...,:,:-1])).abs().mean())), index=i,"
    scope=dict(evaluate.__globals__);exec(compile(source.replace(needle,extra),'<registered_pointwise_metrics>','exec'),scope)
    return scope['evaluate'](*args,**kwargs)

def update(context,model,optimizer,shim,loader,absolute_update,probe=False):
    epoch,batch=divmod(absolute_update,60)
    order=np.random.default_rng(epoch).permutation(120).tolist();indices=order[batch*2:batch*2+2]
    optimizer.zero_grad(set_to_none=True);labels=[];events=[];losses=[];before=shim.loss_ftns[1].calls;tc_grad=[]
    for index in indices:
        seq=default_collate([loader.dataset[index]])
        labels.append(digest_sequence(seq,['frame','flow','flow_valid']));events.append(digest_sequence(seq,['events']))
        values=shim.forward_sequence(seq);assert all(torch.isfinite(x) for x in values.values())
        components=[v for k,v in values.items() if k!='loss' and not k.startswith('loss/')]
        assert torch.allclose(sum(components),values['loss'],rtol=1e-6,atol=1e-8), 'Loss components do not sum to total'
        if probe:
            temporal=next(v for k,v in values.items() if k.startswith('temporal_consistency_loss/'))
            g=torch.autograd.grad(temporal,tuple(model.parameters()),retain_graph=True,allow_unused=True)
            norm=float(torch.stack([x.square().sum() for x in g if x is not None]).sum().sqrt());assert norm>0;tc_grad.append(norm)
        (values['loss']/2).backward();losses.append({k:float(v.detach()) for k,v in values.items()})
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads)==30 and all(torch.isfinite(g).all() for g in grads)
    norm=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt());assert norm>0
    assert shim.loss_ftns[1].calls-before==76
    optimizer.step();assert counts(optimizer)==[absolute_update+1]
    return dict(optimizer_update=absolute_update+1,indices=indices,labels=labels,events=events,
        losses={k:float(np.mean([x[k] for x in losses])) for k in losses[0]},finite_gradient_tensors=30,gradient_l2=norm,tc_calls=76,tc_gradient_probes=tc_grad)

def checkpoint(context,path,model,optimizer,method,arm,binding,update_count):
    payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng_state=rng_state(),model_sha256=state_sha(model),
        optimizer_update=update_count,next_epoch=update_count//60,next_batch=update_count%60,method=method,arm=arm,seed=0,
        parent_checkpoint=binding,model_args=dict(final_activation='sigmoid'),real_data_used=False,trial_only=True)
    torch.save(payload,path);restored=torch.load(path,map_location='cpu',weights_only=False)
    h=hashlib.sha256()
    for k,v in sorted(restored['model'].items()):h.update(k.encode());h.update(v.numpy().tobytes())
    assert h.hexdigest()==payload['model_sha256'] and rng_digest(restored['rng_state'])==rng_digest(rng_state())
    steps=sorted(set(int(x['step'].item()) for x in restored['optimizer']['state'].values()));assert steps==[update_count]
    saved=rng_state()
    try:
        rebuilt=context.model_for(0,'sigmoid');rebuilt.load_state_dict(restored['model'],strict=True)
        assert state_sha(rebuilt)==payload['model_sha256']
        c=context.cfg['optimizer'];rebuilt_opt=torch.optim.Adam(rebuilt.parameters(),lr=c['lr'],betas=tuple(c['betas']),weight_decay=c['weight_decay'],amsgrad=c['amsgrad'])
        rebuilt_opt.load_state_dict(restored['optimizer']);assert counts(rebuilt_opt)==[update_count]
        for old,new in zip(optimizer.state.values(),rebuilt_opt.state.values()):
            assert old.keys()==new.keys()
            for key in old:
                assert torch.equal(old[key].detach().cpu(),new[key].detach().cpu())
        del rebuilt,rebuilt_opt
    finally:restore_rng(saved)
    save(path.with_suffix('.json'),dict(path=str(path),sha256=sha(path),size_bytes=path.stat().st_size,
        model_hash_match=True,rng_roundtrip_match=True,optimizer_counters=steps,independent_model_optimizer_rebuild=True,optimizer_state_equal=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=['eventgenesis','v2e'],required=True);p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    assert torch.cuda.device_count()==1;torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.out.mkdir(exist_ok=False);context=setup_trial(a.campaign);metrics=metric_setup(a.campaign)
    cp,binding=load_start(context,a.campaign,a.method);train=context.loader_for(a.method,True);val=context.loader_for(a.method)
    # Construct a real augmented sequence after restoring exactly the registered RNG.
    temp,opt=restore_start(context,cp);sequence=default_collate([train.dataset[np.random.default_rng(80).permutation(120)[0]]]);del temp,opt
    if a.smoke_only:
        parity=verify_native(context,cp,sequence);cuda_oracle=auxiliary_cuda_oracle();results=[]
        for arm in ARMS:
            model,optimizer=restore_start(context,cp);shim=shim_for(context,model,arm)
            row=update(context,model,optimizer,shim,train,4800,True)
            fixture=detailed_evaluate(context,metrics,model,val,'sigmoid',rollout_count=1,sequence_length=6)
            checkpoint(context,a.out/(arm+'.pt'),model,optimizer,a.method,arm,binding,4801)
            results.append(dict(arm=arm,update=row,fixture=fixture['all200']))
            del model,optimizer,shim;gc.collect();torch.cuda.empty_cache()
        assert len({json.dumps(r['update']['labels']) for r in results})==1
        assert len({json.dumps(r['update']['events']) for r in results})==1
        return save(a.out/'completion.json',dict(status='complete',results=results,native_parity=parity,auxiliary_cuda_oracle=cuda_oracle,real_data_used=False))
    protocol=dict(method=a.method,seed=0,arms=list(ARMS),start_update=4800,additional_updates_per_arm=300,
        parent=binding,recipe=context.cfg,dispatch_source_sha256=context.forward_source_sha256,
        script_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).parent/'PROTOCOL.md'),real_data_used=False,real_scoring_enabled=False)
    save(a.out/'protocol.json',protocol);results=[]
    # All branches start from exactly this model, optimizer and RNG, not each other's output.
    baseline_model,baseline_opt=restore_start(context,cp)
    baseline=detailed_evaluate(context,metrics,baseline_model,val,'sigmoid');save(a.out/'before_004800.json',baseline)
    del baseline_model,baseline_opt;gc.collect();torch.cuda.empty_cache()
    for arm in ARMS:
        run=a.out/arm;run.mkdir(exist_ok=False);model,optimizer=restore_start(context,cp);shim=shim_for(context,model,arm)
        start=time.monotonic();labels=[];events=[]
        with (run/'training_metrics.jsonl').open('x') as stream:
            for u in range(4800,5100):
                row=update(context,model,optimizer,shim,train,u,u==4800);row['elapsed_s']=time.monotonic()-start
                labels.append(row['labels']);events.append(row['events']);stream.write(json.dumps(row)+'\n');stream.flush()
                if (u+1)%20==0:print('UPDATE',a.method,arm,u+1,flush=True)
            checkpoint(context,run/'update_005100.pt',model,optimizer,a.method,arm,binding,5100)
            after=detailed_evaluate(context,metrics,model,val,'sigmoid');save(run/'validation_005100.json',after)
        result=dict(method=a.method,arm=arm,seed=0,additional_updates=300,absolute_update=5100,parent=binding,validation=after,
            label_sequence_sha256=hashlib.sha256(json.dumps(labels).encode()).hexdigest(),
            event_sequence_sha256=hashlib.sha256(json.dumps(events).encode()).hexdigest(),elapsed_s=time.monotonic()-start,real_data_used=False)
        save(run/'completion.json',result);results.append(result);save(a.out/'partial_results.json',results)
        del model,optimizer,shim;gc.collect();torch.cuda.empty_cache()
    assert len({r['label_sequence_sha256'] for r in results})==1 and len({r['event_sequence_sha256'] for r in results})==1
    save(a.out/'completion.json',dict(status='complete',method=a.method,arms=results,before=baseline,pairing_passed=True,real_data_used=False))

if __name__=='__main__':main()
