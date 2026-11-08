"""Fixed seed0 native STE photometric alignment diagnostics, zero updates."""
import argparse,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import flow_synthetic_diagnostic as D
import photometric_math as M
import strict_support as S

ROOT=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-photometric-alignment-v2')
UUID='GPU-253ab9b8-a0e5-4f12-0579-261818cf31dd'
def need(x,message):
    if not x:raise ValueError(message)

def validate(c):
    need(c['schema']=='flow_photometric_alignment_v2' and c['root']==str(ROOT),'Wrong registered diagnostic')
    need(c['checkpoint_source']=='complete8910_last' and c['seed']==0 and c['methods']==['eventgenesis','v2e'],'Wrong frozen CP cohort')
    need(c['optimizer_updates']==0 and not c['real_data_used'] and not c['real_scores_used'] and not c['automatic_training'],'No training/real access allowed')
    need(c['validation_indices']==[0,22,43,65] and c['training_sample_indices']==[11,44] and c['augmentation_seed']==20261108,'Fixed original samples required')
    need(c['finite_difference_epsilons']==list(M.EPSILONS) and c['texture_threshold']==M.TEXTURE_THRESHOLD,'Predefined diagnostic constants changed')
    need(c['deadline_seconds']==900 and c['GPU_UUID']==UUID,'Fixed resource budget/device required')
    return D.validate_contract(c['frozen_inputs'])

def model_pin(root,pin):
    rows=[]
    for f in pin['files']:
        rel=Path(f['path']);need(not rel.is_absolute() and '..' not in rel.parts,'Unsafe model source')
        p=root/rel;need(D.sha(p)==f['sha256'] and p.stat().st_size==f['size_bytes'],'Frozen downstream source changed')
        rows.append([f['path'],f['size_bytes'],f['sha256']])
    tree=hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    need(pin['commit']=='eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6' and len(rows)==14 and tree==pin['included_tree_sha256'],'Model source tree differs')
    return dict(tree_sha256=tree,files_verified=14,commit=pin['commit'])

def labels_digest(gray,flow,valid,sensor):
    return {k:D.array_sha(v.detach().cpu().numpy()) for k,v in [('gray',gray),('flow',flow),('valid',valid),('sensor',sensor)]}

def diagnose(trainer,loss_module,output,gray,event,flow,valid,common,sensor,meta,strict_raw_support=None):
    need(isinstance(output,(tuple,list)) and len(output)==4,'Original four training pyramid outputs required')
    photo=trainer.objective_dt1(output,gray[:,:1],gray[:,1:2],event,loss_module)[1]
    components=[];diags=[]
    for index,pred in enumerate(output):
        need(pred.shape==((1,2,gray.shape[-2]//(2**index),gray.shape[-1]//(2**index))),'Unexpected native pyramid scale')
        prev=M.gray_at_scale(gray[:,:1],pred.shape[-2:]);nxt=M.gray_at_scale(gray[:,1:2],pred.shape[-2:])
        components.append(loss_module.charbonnier_loss(loss_module.warp(nxt,pred)-prev))
        target,strict,nearest,active,sensor_scaled=M.target_at_scale(flow,valid,common,sensor,pred.shape[-2:])
        if strict_raw_support is not None:
            strict=strict&(torch.nn.functional.interpolate(strict_raw_support.float(),size=pred.shape[-2:],mode='bilinear',align_corners=False)[:,0]>=.9999)
        diags.append(M.analyze_scale(prev,nxt,pred,target,strict,nearest,active,sensor_scaled,loss_module.warp,input_hw=gray.shape[-2:]))
    independent=sum(components)/4
    need(torch.equal(photo,independent),'Native4-scale photometric source parity failed; do not relax')
    return dict(**meta,input_hw=list(gray.shape[-2:]),labels_sha256=labels_digest(gray,flow,valid,sensor),
        events_sha256=D.array_sha(event.detach().cpu().numpy()),native_photo=float(photo.detach()),native_photo_components=[float(x.detach()) for x in components],
        independent_formula_photo=float(independent.detach()),native_formula_exact=True,scales=diags,optimizer_updates=0,
        strict_raw_support_sha256=D.array_sha(strict_raw_support.detach().cpu().numpy()) if strict_raw_support is not None else None,
        GT_role='diagnostic direction/residual oracle only; strict support excludes raw-valid interpolation pollution but does not guarantee physical GT; no optimizer, network input or new GT-mask recipe')

def gradient_agreement(left,right,names):
    dot=0.;aa=0.;bb=0.;per_parameter=[]
    for name,a,b in zip(names,left,right):
        if a is None or b is None:
            an=float(a.double().square().sum()) if a is not None else 0.
            bn=float(b.double().square().sum()) if b is not None else 0.
            aa+=an;bb+=bn
            per_parameter.append(dict(parameter=name,photo_missing=a is None,GT_missing=b is None,photo_gradient_l2=an**.5,GT_gradient_l2=bn**.5,cosine=None));continue
        x=a.double();y=b.double();d=float((x*y).sum());an=float(x.square().sum());bn=float(y.square().sum())
        dot+=d;aa+=an;bb+=bn
        per_parameter.append(dict(parameter=name,cosine=d/(an*bn)**.5 if an and bn else None,photo_gradient_l2=an**.5,GT_gradient_l2=bn**.5))
    return dict(dot=dot,cosine=dot/(aa*bb)**.5 if aa and bb else None,photo_gradient_l2=aa**.5,GT_gradient_l2=bb**.5,
        GT_directional_derivative_for_negative_photo_gradient=-dot,per_parameter=per_parameter,
        interpretation='Positive cosine means infinitesimal negative-photo parameter direction lowers the diagnostic GT EPE, to first order; no optimizer update is performed.')

def parameter_oracle(trainer,loss_module,model,inputs,deadline):
    need(len(inputs)==8 and time.monotonic()<deadline,'Exactly the original fixed8 train samples required')
    event,gray,flow,valid,common,sensor=[torch.cat([row[i] for row in inputs]) for i in range(6)]
    inventory=[dict(name=n,shape=list(p.shape),requires_grad=bool(p.requires_grad)) for n,p in model.named_parameters()]
    inventory_sha=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    need(len(inventory)==82 and len(named)==82,'Pinned architecture must retain82 named/trainable parameters')
    output=model(event,256,use_dt4=False,flow_seq=False)
    photo=trainer.objective_dt1(output,gray[:,:1],gray[:,1:2],event,loss_module)[1]
    rho=M.native_map(gray[:,:1],gray[:,1:2],output[0],loss_module.warp)
    mask=valid[:,0]>.5;inside=sensor[:,0].bool();need(bool(mask.any()),'No valid GT in parameter oracle')
    diagnostic_epe=((output[0]-flow).square().sum(dim=1)+1e-12).sqrt()[mask].mean()
    losses=[photo,(rho*mask).sum()/4,(rho*(inside&~mask)).sum()/4,diagnostic_epe]
    gradients=[];gradient_records=[]
    for i,loss in enumerate(losses):
        need(time.monotonic()<deadline and torch.isfinite(loss),'Budget/nonfinite parameter oracle loss')
        gs=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True,retain_graph=i<3)
        need(all(x is None or torch.isfinite(x).all() for x in gs),'Nonfinite parameter oracle gradient')
        nonnone=[name for (name,_),g in zip(named,gs) if g is not None]
        if i==0:need(len(nonnone)==70,'Original dt1 native-photo must involve70 parameter tensors out of82')
        gradient_records.append(dict(loss_index=i,nonNone_count=len(nonnone),nonNone_names=nonnone,
            unused_names=[name for (name,_),g in zip(named,gs) if g is None],finite_nonNone=True,
            exactly_zero_gradient_tensors=sum(g is not None and not bool(torch.count_nonzero(g)) for g in gs)))
        gradients.append([None if x is None else x.detach().cpu() for x in gs]);del gs
    result=dict(status='complete',batch=8,named_parameter_tensors=len(inventory),trainable_parameter_tensors=len(named),parameter_inventory=inventory,parameter_inventory_sha256=inventory_sha,gradient_records=gradient_records,model_forward='original training-mode4scale, no BN/dropout, no optimizer',
        oracle='finest original256-grid valid-only GT EPE withsqrt floor1e-12; diagnostic only',native_photo=float(photo.detach()),GT_epe=float(diagnostic_epe.detach()),
        comparisons={name:gradient_agreement(gradients[i],gradients[3],[n for n,_ in named]) for i,name in enumerate(('native_all4scale_photo','finest_physical_valid_photo','finest_physical_invalid_photo'))},optimizer_updates=0)
    del output,gradients,losses,photo,rho,diagnostic_epe;gc.collect();torch.cuda.empty_cache();return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    started=time.monotonic();deadline=started+900;c=json.loads(a.contract.read_text());R=validate(c);f=c['frozen_inputs']
    need(a.contract.resolve()==ROOT/'sources/contract.json' and a.out.resolve()==ROOT/'diagnostic' and not a.out.exists(),'Independent fixed output directory required')
    manifest=json.loads((ROOT/'sources/source_manifest.json').read_text())
    for name,h in manifest.items():need(Path(name).name==name and D.sha(ROOT/'sources'/name)==h,'Changed registered source: '+name)
    for label,source in c['sources'].items():need(D.sha(source['path'])==source['sha256'],'Native source changed: '+label)
    prior=Path(c['preserved_failure']['root'])
    need(D.sha(prior/'sources/source_manifest.json')==c['preserved_failure']['source_manifest_sha256'],'Preserved151 source manifest changed')
    prior_failure=json.loads((prior/'diagnostic/failure.json').read_text())
    need(prior_failure['status']=='failed' and 'Expected70 unchanged model parameter tensors' in prior_failure['error'],'Expected preserved151 parameter-count failure receipt missing')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==UUID and torch.cuda.is_available() and torch.cuda.device_count()==1,'Only existing queue leasedGPU3 may run')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        trainer=D.load_module(f['training_source'],'photometric_native_frozen_trainer');sys.path.insert(0,f['model_root'])
        import models,models.corr,multiscaleloss
        modelsource=model_pin(Path(f['model_root']),c['model_source_pin'])
        models.corr.spatial_correlation_sample=trainer.pure_torch_spatial_correlation
        compatibility=trainer.patch_ste(models,multiscaleloss,Path(f['model_root']))
        need(multiscaleloss.warp.__name__=='loss_warp' and D.sha(Path(sys.modules[multiscaleloss.warp.__module__].__file__))==f['compat_sha256'],'Actual warp must be the original frozen align_corners patch')
        controls=M.controls(multiscaleloss.warp,torch.device('cuda:0'));D.write(a.out/'controlled_interventions.json',controls)
        val=D.entries(f,'validation');train=D.entries(f,'train');summaries=[];pairing={}
        for method in f['methods']:
            need(time.monotonic()<deadline,'Fifteen-minute diagnostic budget reached')
            b=f['methods_binding'][method];path=Path(b['last_checkpoint']);expected=b['last_checkpoint_sha256']
            need(D.sha(path)==expected and path.name=='last.pt','Only fixed complete last checkpoint accepted')
            cp=torch.load(path,map_location='cpu',weights_only=False)
            for key,value in [('generator',method),('seed',0),('optimizer_update',8910),('resume_phase','complete'),('resume_allowed',False),('config_sha256',f['recipe_sha256']),('data_manifest_sha256',b['data_manifest_sha256']),('initial_state_sha256',b['initial_state_sha256']),('real_data_used',False)]:need(cp.get(key)==value,'CP source identity differs: '+key)
            model=models.steflow().cuda();model.load_state_dict(cp['state_dict'],strict=True);expected_state=cp['current_state_sha256'];del cp
            need(not any(isinstance(x,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout,torch.nn.Dropout2d)) for x in model.modules()),'No BN/dropout model state/randomness allowed')
            before=trainer.state_sha256(model);need(before==expected_state,'Independent model reconstruction SHA differs')
            model.train();rows=[];counter=0;oracle_inputs=[];strict_inputs=[];strict_statistics=[]
            for split,lookup in [('train',train),('validation',val)]:
                chosen=([sorted(r for r in lookup[method] if r.startswith(f's{s}_'))[0] for s in range(4)] if split=='train' else sorted(lookup[method]))
                indices=c['training_sample_indices'] if split=='train' else c['validation_indices']
                for rid in chosen:
                    aa,bb=(D.load_episode(lookup[m][rid]) for m in f['methods']);D.pair_equal(aa,bb)
                    for i in indices:
                        need(time.monotonic()<deadline,'Fifteen-minute diagnostic budget reached');transformed=[]
                        if split=='train':
                            st=int(rid.split('_')[0][1:]);seed=c['augmentation_seed']+st*1000+i
                            for data in (aa,bb):
                                sample={k:data[k][i:i+1] for k in ('events','gray_prev','gray_next','flow_gt','flow_valid')}
                                transformed.append(trainer.apply_joint_transform(sample,np.random.default_rng(seed),torch.device('cuda:0')))
                            need(all(torch.equal(transformed[0][j],transformed[1][j]) for j in (1,2,3)),'Physical augmentation differs')
                            event,gray,flow,valid=transformed[f['methods'].index(method)]
                            common=(transformed[0][0].sum(dim=(1,4))>0)[:,None]&(transformed[1][0].sum(dim=(1,4))>0)[:,None]
                            raw_grid,in_crop,_=trainer.transform_matrix_and_grid(260,346,np.random.default_rng(seed));sensor=in_crop.to('cuda:0')
                            raw_valid=torch.from_numpy(aa['flow_valid'][i:i+1,None]).float()
                            strict_raw,raw_support=S.strict_support(raw_valid,raw_grid,in_crop,valid.cpu());strict_raw=strict_raw.cuda();raw_support=raw_support.cuda()
                        else:
                            data=aa if method=='eventgenesis' else bb
                            event,g0,g1=trainer.val_input(data,i,torch.device('cuda:0'));gray=torch.cat((g0,g1),1)
                            flow=torch.nn.functional.pad(torch.from_numpy(data['flow_gt'][i:i+1]).float(),(3,3,46,46)).cuda()
                            valid=torch.nn.functional.pad(torch.from_numpy(data['flow_valid'][i:i+1,None]).float(),(3,3,46,46)).cuda()
                            common=torch.nn.functional.pad(torch.from_numpy((D.active(aa['events'][i])&D.active(bb['events'][i]))[None,None]),(3,3,46,46)).cuda()
                            sensor=torch.nn.functional.pad(torch.ones((1,1,260,346),dtype=torch.bool),(3,3,46,46)).cuda();seed=None
                            strict_raw=valid>.5
                        with torch.no_grad():output=model(event,gray.shape[-1],use_dt4=False,flow_seq=False)
                        meta=dict(method=method,seed=0,split=split,rollout=rid,stratum=lookup[method][rid]['stratum'],index=i,sample_id=str(aa['sample_ids'][i]),augmentation_seed=seed,
                            source_npz_sha256={m:lookup[m][rid]['sha256'] for m in f['methods']})
                        if split=='validation' and counter==8:
                            model.eval()
                            with torch.no_grad():check=model(event,352,use_dt4=False,flow_seq=False)
                            err=float((check-output[0]).abs().max());need(err<=1e-6,'Training-return/eval-return finest forward differs');model.train()
                            D.write(a.out/f'{method}_return_mode_check.json',dict(passed=True,max_abs=err,threshold=1e-6,no_BN_dropout=True))
                        result=diagnose(trainer,multiscaleloss,output,gray,event,flow,valid,common,sensor,meta,strict_raw)
                        if split=='train':
                            oracle_inputs.append(tuple(x.detach() for x in (event,gray,flow,valid,common,sensor)))
                            strict_inputs.append(tuple(x.detach() for x in (event,gray,flow,strict_raw.float(),common,sensor)))
                            ff=output[0].detach().clone().requires_grad_(True);rr=M.native_map(gray[:,:1],gray[:,1:2],ff,multiscaleloss.warp)
                            gg=torch.autograd.grad(rr.sum()/4,ff)[0]
                            stats=dict(**meta,statistics=S.removal_summary(valid,strict_raw,flow,gg,rr.detach(),raw_support),
                                raw_valid_sha256=D.array_sha(aa['flow_valid'][i]),grid_sha256=D.array_sha(raw_grid.numpy()),strict_mask_sha256=D.array_sha(strict_raw.cpu().numpy()))
                            strict_statistics.append(stats);result['raw_strict_support_statistics']=stats;del ff,rr,gg
                        name=f'{method}_{split}_{rid}_{i:02d}.json';D.write(a.out/name,result)
                        physical_key=f'{split}/{rid}/{i}';digest=dict(labels=result['labels_sha256'],strict_raw_support=result['strict_raw_support_sha256'])
                        if method=='eventgenesis':pairing[physical_key]=digest
                        else:need(pairing[physical_key]==digest,'Cross-generator diagnostic physical labels differ')
                        rows.append(dict(split=split,rollout=rid,index=i,path=str(a.out/name),sha256=D.sha(a.out/name),native_photo=result['native_photo']))
                        counter+=1;print('PHOTOMETRIC_ALIGNMENT',method,split,rid,i,flush=True)
                        del event,gray,flow,valid,common,sensor,output,result;gc.collect()
                    del aa,bb;gc.collect()
                if split=='train':
                    nn_oracle=parameter_oracle(trainer,multiscaleloss,model,oracle_inputs,deadline)
                    strict_oracle=parameter_oracle(trainer,multiscaleloss,model,strict_inputs,deadline)
                    comparison=[]
                    for name in nn_oracle['comparisons']:
                        x=nn_oracle['comparisons'][name]['cosine'];y=strict_oracle['comparisons'][name]['cosine']
                        comparison.append(dict(comparison=name,NN_cosine=x,strict_cosine=y,sign_changed=(x*y<0) if x is not None and y is not None else None))
                    parameter_result=dict(status='complete',NN_oracle=nn_oracle,strict_oracle=strict_oracle,comparison_summary=comparison,strict_statistics=strict_statistics,
                        note='Strict invalid complement includes interpolation/border support, not pure physical occlusion; no guaranteed physical GT. CUDA backward not bitwise deterministic.',optimizer_updates=0)
                    D.write(a.out/f'{method}_parameter_gradient_oracle.json',parameter_result)
                    oracle_inputs.clear();strict_inputs.clear();gc.collect();torch.cuda.empty_cache()
            after=trainer.state_sha256(model);need(after==before and D.sha(path)==expected,'Read-only probe changed model/source state')
            need(sum(r['split']=='train' for r in rows)==8 and sum(r['split']=='validation' for r in rows)==32,'Fixed sample count differs')
            summary=dict(method=method,checkpoint_path=str(path),checkpoint_sha256=expected,model_state_before_sha256=before,model_state_after_sha256=after,model_state_unchanged=True,source_checkpoint_unchanged=True,training_windows=8,validation_windows=32,examples=rows,optimizer_updates=0)
            D.write(a.out/f'{method}_summary.json',summary);summaries.append(summary);del model;gc.collect();torch.cuda.empty_cache()
        need(time.monotonic()<deadline,'Fifteen-minute diagnostic budget reached')
        D.write(a.out/'completion.json',dict(schema='flow_photometric_alignment_result_v2',status='complete',methods=summaries,controlled_interventions_passed=controls['clean_required_controls_passed'],strict_raw_support_completed=True,NN_and_strict_parameter_oracles_complete=True,source_manifest_sha256=D.sha(ROOT/'sources/source_manifest.json'),contract_sha256=D.sha(a.contract),preserved151_failure_sha256=D.sha(prior/'diagnostic/failure.json'),model_source=modelsource,compatibility=compatibility,paired_physical_labels=True,paired_label_digest=hashlib.sha256(json.dumps(pairing,sort_keys=True,separators=(',',':')).encode()).hexdigest(),optimizer_updates=0,real_data_used=False,real_scores_used=False,training_recipe_published=False,automatic_training=False,elapsed_seconds=time.monotonic()-started))
    except BaseException:
        D.write(a.out/'failure.json',dict(status='failed',error=traceback.format_exc(),optimizer_updates=0,real_data_used=False));raise

if __name__=='__main__':main()
