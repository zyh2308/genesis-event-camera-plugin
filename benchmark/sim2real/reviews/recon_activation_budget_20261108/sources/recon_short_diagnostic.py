"""Synthetic-only paired E2VID range/loss diagnosis. Never imports real data."""
import argparse,ast,copy,hashlib,json,random,sys,time,types
from pathlib import Path
from types import SimpleNamespace
import h5py,numpy as np,torch
from torch.utils.data._utils.collate import default_collate

METHODS=['eventgenesis','v2e','vid2e','film_esim','rife_esim']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def setup(campaign):
    R=campaign;P=R.parents[2];root=P/'external/sources/downstream/event_cnn_minimal'
    sys.path[:0]=[str(root),str(root/'PerceptualSimilarity'),str(R/'sources/benchmark/sim2real/compat')]
    import skimage.measure
    from skimage.metrics import structural_similarity
    if not hasattr(skimage.measure,'compare_ssim'):skimage.measure.compare_ssim=structural_similarity
    try:import IPython
    except ImportError:
        stub=types.ModuleType('IPython');stub.embed=lambda *a,**k:None;sys.modules['IPython']=stub
    from data_loader import dataset as dm,data_loaders as dl
    from data_loader.dataset import DynamicH5Dataset,SequenceDataset
    from utils.data import data_sources
    from model.model import E2VIDRecurrent
    from model.loss import perceptual_loss,temporal_consistency_loss,l2_loss
    from trainer.trainer import Trainer
    from e2vid_coordinate_compat import patch_temporal_sampling
    from utils import loss as warp
    coordinate=patch_temporal_sampling(warp,root)
    source=R/'sources/benchmark/sim2real/scripts/train_e2vid_stage9.py'
    main=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='main')
    classes=[x for x in main.body if isinstance(x,ast.ClassDef) and x.name in ['Stage4SequenceDataset','Stage4E2VIDDataset']]
    assert len(classes)==2
    scope=dict(DynamicH5Dataset=DynamicH5Dataset,SequenceDataset=SequenceDataset,random=random,h5py=h5py,np=np,torch=torch,data_sources=data_sources)
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),scope)
    dl.SequenceDataset=scope['Stage4SequenceDataset'];dm.Stage4E2VIDDataset=scope['Stage4E2VIDDataset']
    cfg=json.loads((R/'sources/training_recipe.json').read_text())['reconstruction']
    perceptual=perceptual_loss(weight=1,net='vgg',use_gpu=True);perceptual.model.eval()
    for p in perceptual.model.parameters():p.requires_grad_(False)
    def model_for(seed,arm):
        seed_all(seed);kwargs=copy.deepcopy(cfg['model_args'])
        if arm=='sigmoid':kwargs['final_activation']='sigmoid'
        m=E2VIDRecurrent(kwargs).cuda()
        expected=json.loads((R/f'training/reconstruction/seed{seed}/eventgenesis/completion.json').read_text())['initial_state_sha256']
        assert state_sha(m)==expected,'Activation must not change parameter initialization'
        return m
    def loader_for(method,train=False):
        data=R/'adapters'/method/'reconstruction';manifest=json.loads((data/'adapter_manifest.json').read_text())
        assert not manifest['real_data_used'] and not manifest['real_scoring_enabled']
        kwargs=dict(dataset_type='Stage4E2VIDDataset',proba_pause_when_running=0.,proba_pause_when_paused=0.,
            normalize_image=False,noise_kwargs={},hot_pixel_kwargs={},sequence_length=40 if train else 200,step_size=40 if train else 200,
            dataset_kwargs=dict(num_bins=5,combined_voxel_channels=True,filter_hot_events=False,
                transforms={'RandomCrop':{'size':[112,112]},'RandomFlip':{'p_hflip':.5,'p_vflip':.5}} if train else {'CenterCrop':{'size':[160,160]}}))
        loader=dl.HDF5DataLoader(str(data/(method+('_train_files.txt' if train else '_validation_files.txt'))),
            batch_size=1,shuffle=False,num_workers=0,pin_memory=True,sequence_kwargs=kwargs)
        assert len(loader.dataset)==(120 if train else 8)
        return loader
    def shim_for(model,arm,tc_weight=2.):
        temporal=temporal_consistency_loss(weight=tc_weight,L0=2)
        losses=[perceptual,temporal]+([l2_loss(weight=1)] if arm=='linear_l2' else [])
        shim=SimpleNamespace(model=model,loss_ftns=losses,device=torch.device('cuda:0'))
        shim.to_device=lambda item:Trainer.to_device(shim,item)
        shim.forward_sequence=lambda seq:Trainer.forward_sequence(shim,seq)
        return shim
    return SimpleNamespace(cfg=cfg,root=root,source=source,coordinate=coordinate,perceptual=perceptual,
        temporal_class=temporal_consistency_loss,model_for=model_for,loader_for=loader_for,shim_for=shim_for)

def evaluate(context,model,loader,length=40):
    model.eval();rows=[]
    with torch.inference_mode():
        for index in range(8):
            sequence=default_collate([loader.dataset[index]])[:length];model.reset_states();frames=[]
            for i,item in enumerate(sequence):
                pred=model(item['events'].float().cuda())['image'];target=item['frame'].float().cuda();clipped=pred.clamp(0,1)
                # Initialization is included in diagnostics; the gate separately checks post-warm-up frames.
                lpips=float(context.perceptual(clipped,target,normalize=True))
                zero=torch.zeros_like(target);gray=torch.full_like(target,.5)
                centered=clipped-clipped.mean();gtcenter=target-target.mean()
                denom=(centered.square().mean()*gtcenter.square().mean()).sqrt().clamp_min(1e-10)
                frames.append(dict(i=i,raw_mean=float(pred.mean()),raw_min=float(pred.min()),raw_max=float(pred.max()),
                    negative_fraction=float((pred<0).float().mean()),clipped_std=float(clipped.std(unbiased=False)),
                    target_std=float(target.std(unbiased=False)),clipped_lpips=lpips,raw_lpips=float(context.perceptual(pred,target,normalize=True)),
                    zero_lpips=float(context.perceptual(zero,target,normalize=True)),gray_lpips=float(context.perceptual(gray,target,normalize=True)),
                    mse=float((clipped-target).square().mean()),zero_mse=float(target.square().mean()),gray_mse=float((gray-target).square().mean()),
                    correlation=float((centered*gtcenter).mean()/denom)))
            tail=frames[5:];assert tail
            row=dict(rollout=sequence[-1]['rollout_id'][0],frames=len(frames),
                mean={k:float(np.mean([f[k] for f in tail])) for k in tail[0] if k!='i'},
                degenerate_frame_fraction=float(np.mean([f['clipped_std']<.001 for f in tail])),
                frame_probes=[frames[i] for i in sorted(set([0,len(frames)//2,len(frames)-1]))])
            rows.append(row)
    result=dict(rollouts=rows,equal_rollout_mean={k:float(np.mean([r['mean'][k] for r in rows])) for k in rows[0]['mean']},
        degenerate_frame_fraction=float(np.mean([r['degenerate_frame_fraction'] for r in rows])),sequence_length=length,real_data_used=False)
    model.train();return result

def objective_probe(context,R,out):
    results=[]
    for method in ['eventgenesis','v2e']:
        loader=context.loader_for(method,True);seed_all(91919);sequence=default_collate([loader.dataset[0]])
        for seed in [0,1,2]:
            model=context.model_for(seed,'original');model.eval();model.reset_states()
            with torch.no_grad():
                predictions=[model(item['events'].float().cuda())['image'].detach() for item in sequence]
            pred=predictions[-1];target=sequence[-1]['frame'].float().cuda();flow=sequence[-1]['flow'].float().cuda()
            row=dict(method=method,seed=seed,initial_state_sha256=state_sha(model),
                tail_bias=float(model.unetrecurrent.pred.conv2d.bias.detach()[0]),prediction_mean=float(pred.mean()),
                negative_fraction=float((pred<0).float().mean()),perturbations=[])
            for offset in [-.5,-.1,0,.1,.5]:
                offset_tensor=torch.tensor(float(offset),device='cuda',requires_grad=True)
                candidate=pred+offset_tensor;loss=context.perceptual(candidate,target,normalize=True)
                grad=float(torch.autograd.grad(loss,offset_tensor)[0])
                row['perturbations'].append(dict(offset=offset,perceptual=float(loss.detach()),offset_gradient=grad,
                    clipped_perceptual=float(context.perceptual(candidate.detach().clamp(0,1),target,normalize=True)),
                    mse=float((candidate.detach()-target).square().mean())))
            from utils.loss import temporal_consistency_loss as tc
            prev_target=sequence[-2]['frame'].float().cuda()
            positive=tc(prev_target,target,predictions[-2],pred,-flow)
            negative=tc(prev_target,target,-predictions[-2],-pred,-flow)
            row.update(tc_raw=float(positive),tc_sign_reversed=float(negative),tc_sign_symmetry_error=float((positive-negative).abs()))
            # Model-parameter gradients separate a temporal effect from a perceptual one.
            model.train();shim=context.shim_for(model,'original');values=shim.forward_sequence(sequence)
            bias=model.unetrecurrent.pred.conv2d.bias
            row['initial_head_bias_gradients']={k:float(torch.autograd.grad(v,bias,retain_graph=True)[0][0]) for k,v in values.items() if k.startswith(('perceptual_loss/','temporal_consistency_loss/'))}
            results.append(row);save(out/'objective_probe.json',dict(real_data_used=False,optimizer_updates=0,results=results))
            print('OBJECTIVE',method,seed,row['prediction_mean'],row['initial_head_bias_gradients'],flush=True)
            del shim,values,model,predictions;torch.cuda.empty_cache()
    return results

def short_fit(context,R,out,method,seed,arm,updates,fixed,length):
    run=out/f'{method}_seed{seed}_{arm}';run.mkdir(exist_ok=False)
    train=context.loader_for(method,True);validation=context.loader_for(method)
    model=context.model_for(seed,arm);initial_hash=state_sha(model)
    optcfg=context.cfg['optimizer'];optimizer=torch.optim.Adam(model.parameters(),lr=optcfg['lr'],betas=tuple(optcfg['betas']),
        weight_decay=optcfg['weight_decay'],amsgrad=optcfg['amsgrad']);shim=context.shim_for(model,arm)
    before=evaluate(context,model,validation,length);save(run/'before.json',before)
    seed_all(seed+70000)
    initial_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    fixed_sequences=None;stream=(run/'training_metrics.jsonl').open('x');start=time.monotonic()
    sample_records=[]
    try:
        for update in range(updates):
            epoch,batch=divmod(update,60);order=np.random.default_rng(seed+epoch).permutation(120).tolist()
            indices=order[batch*2:batch*2+2] if not fixed else order[:2]
            if fixed and fixed_sequences is not None:sequences=fixed_sequences
            else:
                sequences=[default_collate([train.dataset[i]]) for i in indices]
                if fixed:fixed_sequences=sequences
            optimizer.zero_grad(set_to_none=True);losses=[];samples=[]
            for sequence in sequences:
                digest=hashlib.sha256()
                for item in sequence:
                    for k in ['frame','flow','flow_valid']:digest.update(item[k].numpy().tobytes())
                    digest.update(str(item['sample_id']).encode())
                samples.append(digest.hexdigest())
                values=shim.forward_sequence(sequence);assert torch.isfinite(values['loss'])
                (values['loss']/2).backward();losses.append({k:float(v.detach()) for k,v in values.items()})
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
            gradient=float(torch.stack([g.square().sum() for g in grads]).sum().sqrt());assert gradient>0
            optimizer.step();row=dict(update=update+1,indices=indices,sample_label_hashes=samples,
                losses={k:float(np.mean([v[k] for v in losses])) for k in losses[0]},finite_gradient_tensors=len(grads),gradient_l2=gradient)
            stream.write(json.dumps(row)+'\n');stream.flush();sample_records.append(samples)
            if (update+1)%10==0:print('SHORT_UPDATE',method,seed,arm,update+1,round(time.monotonic()-start,2),flush=True)
        after=evaluate(context,model,validation,length)
        path=run/'last.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),method=method,seed=seed,
            arm=arm,optimizer_update=updates,initial_state_sha256=initial_hash,real_data_used=False,
            model_args={**context.cfg['model_args'],'final_activation':'sigmoid' if arm=='sigmoid' else ''}),path)
        restored=context.model_for(seed,arm);payload=torch.load(path,map_location='cpu',weights_only=False);restored.load_state_dict(payload['model'])
        assert state_sha(restored)==state_sha(model)
        restored_opt=torch.optim.Adam(restored.parameters(),lr=optcfg['lr'],betas=tuple(optcfg['betas']),amsgrad=optcfg['amsgrad'])
        restored_opt.load_state_dict(payload['optimizer']);counts=[int(v['step']) for v in restored_opt.state.values()]
        assert counts and all(v==updates for v in counts)
        result=dict(method=method,seed=seed,arm=arm,updates=updates,fixed_train_batch=fixed,before=before,after=after,
            sample_label_hashes_sha256=hashlib.sha256(json.dumps(sample_records).encode()).hexdigest(),
            initial_state_sha256=initial_hash,checkpoint_sha256=sha(path),checkpoint=str(path),
            recovery_model_hash_match=True,optimizer_counters=sorted(set(counts)),wall_s=time.monotonic()-start,
            real_data_used=False)
        save(run/'completion.json',result);return result
    finally:stream.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=['probe','fixed','mixed'],required=True);parser.add_argument('--arms',nargs='+',choices=['original','sigmoid','linear_l2'],default=['original','sigmoid','linear_l2'])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS[:2]);parser.add_argument('--seeds',nargs='+',type=int,default=[2])
    parser.add_argument('--updates',type=int,default=20);parser.add_argument('--validation-length',type=int,choices=[40,200],default=40)
    args=parser.parse_args();assert torch.cuda.device_count()==1;torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=False);context=setup(args.campaign)
    protocol=dict(real_data_used=False,phase=args.phase,arms=args.arms,methods=args.methods,seeds=args.seeds,updates=args.updates,
        validation_length=args.validation_length,trainer_source_sha256=sha(context.source),coordinate_preflight=context.coordinate,
        recipe_sha256=sha(args.campaign/'sources/training_recipe.json'),script_sha256=sha(Path(__file__)),
        model_source_sha256={str(p.relative_to(context.root)):sha(p) for p in (context.root/'model').glob('*.py')},
        changes_to_simulator=False,loss_changes={'original':'LPIPS-VGG1 + TC2','sigmoid':'same losses; native final_activation=sigmoid',
            'linear_l2':'original linear output and losses + native l2_loss weight1'},
        gate_predeclared='all methods/seeds: after LPIPS < own init and zero and gray; MSE < zero; contrast ratio>0.1; degenerate fraction<0.05; correlation>0.1; no real data')
    save(args.out/'protocol.json',protocol)
    if args.phase=='probe':results=objective_probe(context,args.campaign,args.out)
    else:
        results=[]
        for method in args.methods:
            for seed in args.seeds:
                for arm in args.arms:
                    result=short_fit(context,args.campaign,args.out,method,seed,arm,args.updates,args.phase=='fixed',args.validation_length)
                    results.append(result);save(args.out/'partial_results.json',results);torch.cuda.empty_cache()
    save(args.out/'completion.json',dict(status='complete',real_data_used=False,results=results))
if __name__=='__main__':main()
