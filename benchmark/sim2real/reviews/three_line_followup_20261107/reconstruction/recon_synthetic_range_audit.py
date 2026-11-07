"""Read-only synthetic checkpoint audit; no optimizer and no real inputs."""
import argparse,ast,hashlib,json,random,sys,types
from pathlib import Path
import h5py,numpy as np,torch
from torch.utils.data._utils.collate import default_collate

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();R=args.campaign;P=R.parents[2]
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    modelroot=P/'external/sources/downstream/event_cnn_minimal'
    sys.path.insert(0,str(modelroot));sys.path.insert(0,str(modelroot/'PerceptualSimilarity'))
    try:import IPython
    except ImportError:
        stub=types.ModuleType('IPython');stub.embed=lambda *a,**kw:None;sys.modules['IPython']=stub
    import skimage.measure
    from skimage.metrics import structural_similarity
    if not hasattr(skimage.measure,'compare_ssim'):skimage.measure.compare_ssim=structural_similarity
    from data_loader import dataset as dataset_module
    from data_loader import data_loaders as loaders
    from data_loader.dataset import DynamicH5Dataset,SequenceDataset
    from model.model import E2VIDRecurrent
    from model.loss import perceptual_loss
    from utils.data import data_sources
    trainer=R/'sources/benchmark/sim2real/scripts/train_e2vid_stage9.py'
    tree=ast.parse(trainer.read_text());mainnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    classes=[n for n in mainnode.body if isinstance(n,ast.ClassDef) and n.name in ['Stage4SequenceDataset','Stage4E2VIDDataset']]
    assert len(classes)==2
    scope=dict(DynamicH5Dataset=DynamicH5Dataset,SequenceDataset=SequenceDataset,random=random,
        h5py=h5py,np=np,torch=torch,data_sources=data_sources)
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(trainer),'exec'),scope)
    loaders.SequenceDataset=scope['Stage4SequenceDataset'];dataset_module.Stage4E2VIDDataset=scope['Stage4E2VIDDataset']
    config=json.loads((R/'sources/training_recipe.json').read_text())['reconstruction']
    args.out.mkdir(parents=True,exist_ok=False)
    write(args.out/'protocol.json',dict(real_data_used=False,optimizer_updates=0,scope='one fixed first synthetic validation rollout; first 40 consecutive windows',
        methods=['eventgenesis','v2e','vid2e','film_esim','rife_esim'],seeds=[0,1,2],
        additional='eventgenesis/V2E seed2 initial and update20 states',perceptual_probe_frames=[0,19,39],
        loader_class_source=str(trainer),loader_class_source_sha256=sha(trainer),training_recipe_sha256=sha(R/'sources/training_recipe.json'),
        changes_to_training=False,real_metrics_protocol_changed=False))
    metric=perceptual_loss(weight=1.,net='vgg',use_gpu=True)
    metric.model.eval()
    for parameter in metric.model.parameters():parameter.requires_grad_(False)
    results=[];reference_target_hash=None
    for method in ['eventgenesis','v2e','vid2e','film_esim','rife_esim']:
        data=R/'adapters'/method/'reconstruction';manifest=json.loads((data/'adapter_manifest.json').read_text())
        assert manifest['real_data_used'] is False and manifest['real_scoring_enabled'] is False
        kwargs=dict(dataset_type='Stage4E2VIDDataset',proba_pause_when_running=0.,proba_pause_when_paused=0.,
            normalize_image=False,noise_kwargs={},hot_pixel_kwargs={},sequence_length=200,step_size=200,
            dataset_kwargs=dict(num_bins=5,combined_voxel_channels=True,filter_hot_events=False,transforms={'CenterCrop':{'size':[160,160]}}))
        loader=loaders.HDF5DataLoader(str(data/(method+'_validation_files.txt')),batch_size=1,shuffle=False,num_workers=0,pin_memory=True,sequence_kwargs=kwargs)
        assert len(loader.dataset)==8
        sequence=default_collate([loader.dataset[0]])[:40]
        target_hash=hashlib.sha256(b''.join(x['frame'].numpy().tobytes() for x in sequence)).hexdigest()
        reference_target_hash=reference_target_hash or target_hash;assert target_hash==reference_target_hash
        candidates=[(seed,'selected',R/'training/reconstruction'/('seed'+str(seed))/method/'synthetic_best.pt') for seed in range(3)]
        if method in ['eventgenesis','v2e']:
            candidates.extend([(2,'initial',None),(2,'update20',R/'training/reconstruction/seed2'/method/'update_000020.pt')])
        for seed,kind,path in candidates:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=E2VIDRecurrent(dict(config['model_args'])).cuda().eval();model.requires_grad_(False)
            record=dict(method=method,seed=seed,checkpoint_kind=kind,path=str(path) if path else None,
                checkpoint_sha256=sha(path) if path else None,target_sha256=target_hash,frames=[],validation_rollout=sequence[0]['rollout_id'][0])
            if path:
                payload=torch.load(path,map_location='cpu',weights_only=False)
                assert payload['generator']==method and payload['seed']==seed
                model.load_state_dict(payload['model'],strict=True);record['optimizer_update']=payload['optimizer_update']
                if kind=='selected':
                    selection=json.loads(path.with_suffix('.json').read_text());assert selection['sha256']==record['checkpoint_sha256']
                    record['selected_by_synthetic_only']=selection
            else:
                state_hash=hashlib.sha256()
                for key,value in sorted(model.state_dict().items()):state_hash.update(key.encode());state_hash.update(value.cpu().numpy().tobytes())
                expected=json.loads((R/'training/reconstruction'/('seed'+str(seed))/method/'completion.json').read_text())['initial_state_sha256']
                assert state_hash.hexdigest()==expected;record['initial_state_sha256']=expected
            model.reset_states()
            with torch.inference_mode():
                for i,item in enumerate(sequence):
                    pred=model(item['events'].float().cuda())['image'];target=item['frame'].float().cuda();clipped=pred.clamp(0,1)
                    row=dict(index=i,raw_min=float(pred.min()),raw_max=float(pred.max()),raw_mean=float(pred.mean()),raw_std=float(pred.std(unbiased=False)),
                        raw_negative_fraction=float((pred<0).float().mean()),raw_above_one_fraction=float((pred>1).float().mean()),
                        clipped_mean=float(clipped.mean()),clipped_std=float(clipped.std(unbiased=False)),
                        target_mean=float(target.mean()),target_std=float(target.std(unbiased=False)),
                        raw_mse=float((pred-target).square().mean()),clipped_mse=float((clipped-target).square().mean()),zero_mse=float(target.square().mean()))
                    if i in [0,19,39]:
                        row.update(raw_lpips_vgg=float(metric(pred,target,normalize=True)),clipped_lpips_vgg=float(metric(clipped,target,normalize=True)),
                            zero_lpips_vgg=float(metric(torch.zeros_like(pred),target,normalize=True)))
                    record['frames'].append(row)
            write(args.out/(method+'_seed'+str(seed)+'_'+kind+'.json'),record)
            print('SYNTHETIC_RANGE_AUDIT',method,seed,kind,'last_raw_mean',record['frames'][-1]['raw_mean'],
                'last_clipped_std',record['frames'][-1]['clipped_std'],flush=True)
            results.append(record);del model
    write(args.out/'completion.json',dict(status='complete',real_data_used=False,optimizer_updates=0,results=results,
        interpretation='Range audit only, not a new model or generator ranking; no seed excluded'))

if __name__=='__main__':main()
