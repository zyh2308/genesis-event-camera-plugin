"""Bounded synthetic activation x budget diagnosis. No real data or formal queue."""
import argparse,copy,gc,hashlib,importlib.util,json,math,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from recon_short_diagnostic import setup,seed_all,state_sha,sha,save

METHODS=('eventgenesis','v2e');ARMS=('original','sigmoid');POINTS=(0,300,600)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
def rng_digest(r):
    import pickle
    return hashlib.sha256(pickle.dumps((r['python'],r['numpy'],r['cpu'].numpy().tobytes(),[x.cpu().numpy().tobytes() for x in r['cuda']]))).hexdigest()
def digest_sequence(sequence,keys):
    h=hashlib.sha256()
    for item in sequence:
        for k in keys:h.update(item[k].numpy().tobytes())
        h.update(str(item['sample_id']).encode())
    return h.hexdigest()
def counts(optimizer):return sorted(set(int(v['step'].item()) for v in optimizer.state.values()))

def metric_setup(campaign):
    binding=json.loads((Path(__file__).parent/'metric_source_binding.json').read_text())
    path=Path(binding['path'])
    assert path==campaign.parent/'20261107-stage9-hqf15-v1/repo/benchmark/sim2real/scripts/evaluate_synthetic_reconstruction_fixture.py'
    assert sha(path)==binding['sha256'] and path.stat().st_size==binding['size_bytes']
    spec=importlib.util.spec_from_file_location('synthetic_same_metric_functions',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    alex,*assets=m.load_lpips_alex(campaign.parents[2]/'external/sources/downstream/event_cnn_minimal',torch.device('cuda:0'))
    return SimpleNamespace(alex=alex,functions=m,source=dict(path=str(path),sha256=sha(path)),assets=[str(x) for x in assets])

def evaluate(context,metrics,model,loader,activation,rollout_count=8,sequence_length=200):
    saved=rng_state();model.eval();rollouts=[];logits=[]
    assert model.unetrecurrent.pred.norm==context.cfg['model_args']['norm'] and model.unetrecurrent.pred.norm not in ('BN','IN')
    hook=model.unetrecurrent.pred.conv2d.register_forward_hook(lambda module,inputs,out:logits.append(out.detach()))
    try:
        with torch.inference_mode():
            for ri in range(rollout_count):
                sequence=default_collate([loader.dataset[ri]]);assert len(sequence)==200;sequence=sequence[:sequence_length]
                model.reset_states();frames=[]
                for i,item in enumerate(sequence):
                    logits.clear();raw=model(item['events'].float().cuda())['image'];target=item['frame'].float().cuda();pred=raw.clamp(0,1)
                    assert len(logits)==1 and torch.isfinite(raw).all()
                    z=logits[0];sig=torch.sigmoid(z)
                    if activation=='sigmoid':assert torch.equal(raw,sig)
                    vgg=float(context.perceptual(pred,target,normalize=True))
                    outside=bool(((raw<0)|(raw>1)).any())
                    raw_vgg=float(context.perceptual(raw,target,normalize=True)) if outside else vgg
                    pa=pred[0,0].cpu().numpy();ta=target[0,0].cpu().numpy()
                    classical=metrics.functions.classical_metrics(pa,ta)
                    a=metrics.functions.lpips_input(pa,torch.device('cuda:0'));b=metrics.functions.lpips_input(ta,torch.device('cuda:0'))
                    alex=float(metrics.alex.forward(a,b,normalize=False).reshape(-1)[0])
                    pc=pred-pred.mean();tc=target-target.mean();den=(pc.square().mean()*tc.square().mean()).sqrt().clamp_min(1e-10)
                    edge=lambda x:.5*((x[...,1:,:]-x[...,:-1,:]).abs().mean()+(x[...,:,1:]-x[...,:,:-1]).abs().mean())
                    frames.append(dict(index=i,lpips_vgg=vgg,raw_lpips_vgg=raw_vgg,lpips_alex=alex,
                        mse=classical['mse'],ssim=classical['ssim'],psnr_db=classical['psnr_db'],
                        clipped_std=float(pred.std(unbiased=False)),target_std=float(target.std(unbiased=False)),
                        correlation=float((pc*tc).mean()/den),edge_mean=float(edge(pred)),target_edge_mean=float(edge(target)),
                        raw_mean=float(raw.mean()),raw_std=float(raw.std(unbiased=False)),
                        negative_fraction=float((raw<0).float().mean()),above_one_fraction=float((raw>1).float().mean()),
                        near_boundary_fraction=float(((raw<.02)|(raw>.98)).float().mean()),
                        sigmoid_derivative_mean=float((sig*(1-sig)).mean()) if activation=='sigmoid' else 1.,
                        logit_std=float(z.std(unbiased=False)),flat_fraction=float(pred.std(unbiased=False)<.001)))
                def aggregate(rows):
                    ans={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='index'}
                    ans['contrast_ratio']=ans['clipped_std']/max(ans['target_std'],1e-12)
                    ans['edge_ratio']=ans['edge_mean']/max(ans['target_edge_mean'],1e-12)
                    return ans
                rollouts.append(dict(rollout=sequence[-1]['rollout_id'][0],all200=aggregate(frames),tail195=aggregate(frames[5:]),frame_count=sequence_length))
        result=dict(rollouts=rollouts,real_data_used=False,diagnostic_fixture=sequence_length!=200,frames_per_rollout=sequence_length)
        for subset in ['all200','tail195']:
            result[subset]={k:float(np.mean([r[subset][k] for r in rollouts])) for k in rollouts[0][subset]}
        return result
    finally:
        hook.remove();logits.clear();model.train();restore_rng(saved)
        assert rng_digest(rng_state())==rng_digest(saved),'Validation changed subsequent training RNG'

def fit(context,metrics,args,arm):
    run=args.out/arm;run.mkdir(exist_ok=False);train=context.loader_for(args.method,True);val=context.loader_for(args.method)
    model=context.model_for(0,arm);initial=state_sha(model);c=context.cfg['optimizer']
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],betas=tuple(c['betas']),weight_decay=c['weight_decay'],amsgrad=c['amsgrad'])
    shim=context.shim_for(model,arm);tc=shim.loss_ftns[1]
    seed_all(70000);start=time.monotonic();stream=(run/'training_metrics.jsonl').open('x');samples=[];event_samples=[]
    results={0:evaluate(context,metrics,model,val,arm)};save(run/'validation_000000.json',results[0])
    try:
        for update in range(600):
            epoch,batch=divmod(update,60);order=np.random.default_rng(epoch).permutation(120).tolist();indices=order[2*batch:2*batch+2]
            optimizer.zero_grad(set_to_none=True);losses=[];labels=[];events=[];tc_grad_probes=[];before_calls=tc.calls
            for index in indices:
                seq=default_collate([train.dataset[index]])
                labels.append(digest_sequence(seq,['frame','flow','flow_valid']));events.append(digest_sequence(seq,['events']))
                loss=shim.forward_sequence(seq);assert torch.isfinite(loss['loss'])
                if update==0:
                    temporal=next(v for k,v in loss.items() if k.startswith('temporal_consistency_loss/'))
                    g=torch.autograd.grad(temporal,tuple(model.parameters()),retain_graph=True,allow_unused=True)
                    norm=float(torch.stack([x.square().sum() for x in g if x is not None]).sum().sqrt());assert norm>0;tc_grad_probes.append(norm)
                (loss['loss']/2).backward();losses.append({k:float(v.detach()) for k,v in loss.items()})
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            assert len(gradients)==30 and all(torch.isfinite(g).all() for g in gradients)
            gradnorm=float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt());assert gradnorm>0
            assert tc.calls-before_calls==76
            optimizer.step();samples.append(labels);event_samples.append(events)
            row=dict(update=update+1,indices=indices,label_augmentation_sha256=labels,event_sha256=events,
                losses={k:float(np.mean([x[k] for x in losses])) for k in losses[0]},gradient_l2=gradnorm,
                finite_gradient_tensors=30,tc_calls=76,first_update_tc_gradient_probes=tc_grad_probes,elapsed_seconds=time.monotonic()-start)
            stream.write(json.dumps(row)+'\n');stream.flush()
            if (update+1)%20==0:print('UPDATE',args.method,arm,update+1,flush=True)
            if update+1 in [20,300,600]:
                payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng_state=rng_state(),
                    optimizer_update=update+1,next_epoch=(update+1)//60,next_batch=(update+1)%60,
                    method=args.method,seed=0,arm=arm,initial_state_sha256=initial,model_sha256=state_sha(model),
                    model_args={**context.cfg['model_args'],'final_activation':'sigmoid' if arm=='sigmoid' else ''},
                    data_manifest_sha256=sha(args.campaign/'adapters'/args.method/'reconstruction/adapter_manifest.json'),
                    config_sha256=sha(args.campaign/'sources/training_recipe.json'),script_sha256=sha(Path(__file__)),real_data_used=False)
                path=run/f'update_{update+1:06d}.pt';torch.save(payload,path)
                restored=torch.load(path,map_location='cpu',weights_only=False)
                h=hashlib.sha256()
                for k,v in sorted(restored['model'].items()):h.update(k.encode());h.update(v.numpy().tobytes())
                assert h.hexdigest()==payload['model_sha256'] and counts(optimizer)==[update+1]
                assert rng_digest(restored['rng_state'])==rng_digest(rng_state())
                save(run/f'checkpoint_{update+1:06d}.json',dict(path=str(path),sha256=sha(path),update=update+1,restore_model_hash_passed=True,optimizer_counts=counts(optimizer),rng_roundtrip_passed=True))
                del restored,payload
            if update+1 in [300,600]:
                results[update+1]=evaluate(context,metrics,model,val,arm);save(run/f'validation_{update+1:06d}.json',results[update+1])
        result=dict(method=args.method,arm=arm,seed=0,updates=600,initial_state_sha256=initial,
            label_sequence_sha256=hashlib.sha256(json.dumps(samples).encode()).hexdigest(),
            event_sequence_sha256=hashlib.sha256(json.dumps(event_samples).encode()).hexdigest(),
            validation=results,real_data_used=False,elapsed_seconds=time.monotonic()-start)
        save(run/'completion.json',result);return result
    finally:stream.close()

def references(context,metrics,args):
    formal=args.campaign.parent/'20261107-recon-sigmoid-formal-v2/training/seed0'/args.method
    paths=[('sigmoid_4800',formal/'synthetic_best.pt','sigmoid'),('sigmoid_stop',formal/'last.pt','sigmoid'),
        ('linear_historical',args.campaign/'training/reconstruction/seed0'/args.method/'synthetic_best.pt','original')]
    results=[]
    for name,path,arm in paths:
        meta=torch.load(path,map_location='cpu',weights_only=False)
        assert meta['generator']==args.method and meta['seed']==0
        if name=='sigmoid_4800':assert meta['optimizer_update']==4800
        model=context.model_for(0,arm);model.load_state_dict(meta['model'],strict=True)
        result=dict(name=name,method=args.method,checkpoint=str(path),checkpoint_sha256=sha(path),
            updates=meta['optimizer_update'],activation=arm,interpretation='existing continuous formal trajectory; historical reference, not same-RNG600-step cohort',
            validation=evaluate(context,metrics,model,context.loader_for(args.method),arm))
        results.append(result);save(args.out/'references_partial.json',results)
        del meta,model;gc.collect();torch.cuda.empty_cache()
    return results

def smoke(context,metrics,args):
    results=[]
    for arm in ARMS:
        model=context.model_for(0,arm);shim=context.shim_for(model,arm);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,amsgrad=True)
        seed_all(70000);loader=context.loader_for(args.method,True);labels=[]
        for index in np.random.default_rng(0).permutation(120).tolist()[:2]:
            seq=default_collate([loader.dataset[index]]);labels.append(digest_sequence(seq,['frame','flow','flow_valid']))
            loss=shim.forward_sequence(seq);(loss['loss']/2).backward()
        assert shim.loss_ftns[1].calls==76
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();assert counts(optimizer)==[1]
        x=np.zeros((64,64),dtype=np.float32);classical=metrics.functions.classical_metrics(x,x)
        assert classical['mse']==0 and classical['ssim']==1 and math.isinf(classical['psnr_db'])
        t=metrics.functions.lpips_input(x,torch.device('cuda:0'))
        with torch.inference_mode():assert abs(float(metrics.alex.forward(t,t,normalize=False).reshape(-1)[0]))<1e-7
        fixture=evaluate(context,metrics,model,context.loader_for(args.method),arm,rollout_count=1,sequence_length=6)
        assert fixture['diagnostic_fixture'] and len(fixture['rollouts'])==1
        results.append(dict(arm=arm,optimizer_updates=1,tc_calls=76,labels=labels,optimizer_counts=counts(optimizer),validation_fixture_frames=6,validation_rng_restored=True))
        del model,shim,optimizer,loss;gc.collect();torch.cuda.empty_cache()
    assert results[0]['labels']==results[1]['labels']
    save(args.out/'completion.json',dict(status='complete',results=results,real_data_used=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    assert torch.cuda.device_count()==1;torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.out.mkdir(exist_ok=False);context=setup(a.campaign)
    assert all(context.cfg['loss'][k]==v for k,v in dict(LPIPS_VGG_weight=1.,temporal_weight=2.,temporal_L0=2).items())
    original_call=context.temporal_class.__call__
    def count_calls(self,*args,**kwargs):
        value=original_call(self,*args,**kwargs)
        if value is not None and not kwargs.get('output_images',False):self.calls=getattr(self,'calls',0)+1
        return value
    context.temporal_class.__call__=count_calls
    # Newly constructed native temporal objects have no calls attribute until first use.
    old_shim=context.shim_for
    def shim_for(*args,**kwargs):
        shim=old_shim(*args,**kwargs);shim.loss_ftns[1].calls=0;return shim
    context.shim_for=shim_for
    metrics=metric_setup(a.campaign)
    if a.smoke_only:return smoke(context,metrics,a)
    save(a.out/'protocol.json',dict(method=a.method,seed=0,arms=list(ARMS),maximum_updates_per_arm=600,
        validation_points=list(POINTS),validation='all8 complete200-frame rollouts; all200 primary,tail195 secondary',
        learning_rate=context.cfg['optimizer']['lr'],loss=context.cfg['loss'],sampling_rng_seed=70000,
        real_data_used=False,real_scoring_enabled=False,source_sha256=sha(Path(__file__)),metrics_source=metrics.source,
        limitation='early activation effects and local late-checkpoint trend; not a proof of final capacity or HQF causality'))
    ref=references(context,metrics,a)
    arms=[]
    for arm in ARMS:
        r=fit(context,metrics,a,arm);arms.append(r);save(a.out/'partial_results.json',arms);gc.collect();torch.cuda.empty_cache()
    assert len({r['initial_state_sha256'] for r in arms})==1 and len({r['label_sequence_sha256'] for r in arms})==1 and len({r['event_sequence_sha256'] for r in arms})==1
    save(a.out/'completion.json',dict(status='complete',arms=arms,references=ref,pairing_passed=True,real_data_used=False))
if __name__=='__main__':main()
