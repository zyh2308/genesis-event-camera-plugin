"""One training-only visible-mask head; unchanged e188 C1 center objective."""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pose_priority_policy import state_sha,draw_auxiliary,center_supervision,center_retention


def need(ok,message):
    if not ok:raise ValueError(message)


def balanced_mask_bce(logits,target):
    """Per sample: .5 foreground mean + .5 background mean; absent class=0."""
    need(logits.shape==target.shape and logits.ndim==4 and logits.shape[1]==1,'Mask logits/target shape mismatch')
    need(bool(torch.isfinite(logits).all() and torch.isfinite(target).all()) and bool(((target>=0)&(target<=1)).all()),'Invalid mask logits/occupancy')
    dims=(1,2,3);positive=target.sum(dims);negative=(1-target).sum(dims)
    pd=torch.where(positive>0,positive,torch.ones_like(positive));nd=torch.where(negative>0,negative,torch.ones_like(negative))
    fore=(target*F.softplus(-logits)).sum(dims)/pd
    back=((1-target)*F.softplus(logits)).sum(dims)/nd
    loss=.5*(fore+back)
    return loss.mean(),dict(foreground_mass=positive.detach().cpu().tolist(),background_mass=negative.detach().cpu().tolist(),
        empty_foreground_samples=int((positive==0).sum()),empty_background_samples=int((negative==0).sum()),
        foreground_loss=float(fore.mean().detach()),background_loss=float(back.mean().detach()))


def paired_mask_affine(shared,images,target,masks,rng):
    """Exactly one official RNG draw; cloned-state label transform is loss-only."""
    need(masks.shape==(len(images),1,128,128),'Mask128 inventory/shape differs')
    twin=np.random.default_rng();twin.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
    x,y=shared.affine(images,target,rng);mask,yy=shared.affine(masks,target,twin)
    need(torch.equal(y,yy) and rng.bit_generator.state==twin.bit_generator.state,'Mask/image affine tape differs')
    return x,y,F.interpolate(mask,(16,16),mode='area')


def auxiliary_head(seed,channels,device):
    # All randomness is CPU-local in fork_rng. No CUDA RNG is consumed.
    cpu_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(892100+seed);head=nn.Conv2d(channels,1,1)
    head=head.to(device)
    need(torch.equal(cpu_before,torch.get_rng_state()),'Auxiliary initialization advanced original CPU RNG')
    if cuda_before is not None:need(all(torch.equal(a,b) for a,b in zip(cuda_before,torch.cuda.get_rng_state_all())),'Auxiliary initialization advanced original CUDA RNG')
    return head


def native_loss(model,clean,predicted,reference,y,b):
    supervised=.5*(center_supervision(model,clean,y)+center_supervision(model,predicted,b))
    retention=center_retention(clean,reference).mean()
    return supervised+2*retention,supervised,retention


def fit_center(native,scale,state,normal,target,aux,aux_target,aux_clean,normal_mask,aux_mask,seed,smoke,out,method,old_log):
    """Same initialization/tape/C1 terms; only mask loss adds student gradients."""
    shared=native.shared;out=Path(out);directory=out/'training_evidence';directory.mkdir(exist_ok=True)
    need(len(normal)==len(target)==len(normal_mask) and len(aux)==len(aux_target)==len(aux_clean)==len(aux_mask),'Input/mask inventory differs')
    half_count=len(aux)//2;need(len(aux)%2==0 and torch.equal(aux_target[:half_count],aux_target[half_count:]),'Original paired auxiliary labels differ')
    torch.manual_seed(90710+seed)
    model=scale.CenterScale().to(normal.device);model.load_state_dict(state,strict=True);model.size.requires_grad_(False)
    teacher=scale.CenterScale().to(normal.device).eval();teacher.load_state_dict(state,strict=True);teacher.requires_grad_(False)
    source_hash=state_sha(state);need(state_sha(model.state_dict())==state_sha(teacher.state_dict())==source_hash,'Source init/teacher mismatch')
    channels=int(model.center.in_channels);head=auxiliary_head(seed,channels,normal.device)
    original_parameters=tuple(p for p in model.parameters() if p.requires_grad);parameters=original_parameters+tuple(head.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.00015,weight_decay=.01);rng=np.random.default_rng(71070+seed)
    history=[];updates=0;labels=hashlib.sha256();exposure=dict(normal=0,half=0,eighth=0);features=[]
    hook=model.features.register_forward_hook(lambda module,args,output:features.append(output))
    tag=f'{method}_seed{seed}_aux_visible_mask_center';log=directory/(tag+'.jsonl');first_oracle=None
    try:
        with log.open('x') as stream:
            for epoch in range(1 if smoke else 6):
                order=rng.permutation(len(normal));order=order[:32] if smoke else order;totals=[]
                for offset in range(0,len(order),32):
                    indices=order[offset:offset+32];ai,half=draw_auxiliary(rng,half_count,len(indices));idx=torch.as_tensor(indices,device=normal.device);aid=torch.as_tensor(ai,device=normal.device)
                    x,y,mask=paired_mask_affine(shared,normal[idx],target[idx],normal_mask[idx],rng)
                    # Preserve native matched_affine's sole official auxiliary RNG draw.
                    twin=np.random.default_rng();twin.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
                    a,paired,b=native.matched_affine(aux[aid],aux_clean[aid],aux_target[aid],rng)
                    _,bb,amask=paired_mask_affine(shared,aux[aid],aux_target[aid],aux_mask[aid],twin)
                    need(torch.equal(b,bb) and rng.bit_generator.state==twin.bit_generator.state,'Auxiliary image/mask RNG differs')
                    features.clear();clean,predicted=model(x),model(a);need(len(features)==2,'Student feature hook inventory differs')
                    with torch.no_grad():reference=teacher(x)
                    base,supervised,retention=native_loss(model,clean,predicted,reference,y,b)
                    m0,stats0=balanced_mask_bce(head(features[0]),mask);m1,stats1=balanced_mask_bce(head(features[1]),amask);auxloss=.5*(m0+m1);loss=base+auxloss
                    if updates==0:
                        old=old_log[0];deltas={name:abs(float(value.detach())-old[key]) for name,value,key in [('supervised',supervised,'supervised'),('clean_retention',retention,'clean_retention'),('native_scalar_loss',base,'loss')]}
                        need(all(np.isfinite(v) and v<=2e-6+1e-6*abs(old['loss']) for v in deltas.values()),'First native-loss oracle differs from original C1')
                        first_oracle=dict(passed=True,absolute_errors=deltas,original_update=1,actual_native_loss=float(base.detach()),original_native_loss=old['loss'])
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    need(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),'Missing/nonfinite native/auxiliary gradient')
                    need(all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for p in model.size.parameters()),'Frozen teacher/size received gradient')
                    norm=nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True);need(bool(torch.isfinite(loss)) and norm>0,'Invalid objective/gradient');optimizer.step();updates+=1
                    need(sorted({int(v['step']) for v in optimizer.state.values()})==[updates],'Adam counters differ')
                    labels.update(y.detach().cpu().numpy().tobytes());labels.update(b.detach().cpu().numpy().tobytes());exposure['normal']+=len(indices);exposure['half']+=int(half.sum());exposure['eighth']+=int((~half).sum())
                    row=dict(update=updates,seed=seed,arm='aux_visible_mask_center',branch='center',normal_indices=indices.tolist(),auxiliary_indices=ai.tolist(),half_mask=half.tolist(),
                        supervised=float(supervised.detach()),clean_retention=float(retention.detach()),native_scalar_loss=float(base.detach()),auxiliary_mask_loss=float(auxloss.detach()),loss=float(loss.detach()),gradient_l2=float(norm),
                        finite_gradient_tensors=len(parameters),native_finite_gradient_tensors=len(original_parameters),auxiliary_finite_gradient_tensors=len(tuple(head.parameters())),normal_mask=stats0,auxiliary_mask=stats1)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();totals.append(row['loss'])
                history.append(float(np.mean(totals)));print('AUX_MASK_CENTER_EPOCH',method,seed,epoch+1,updates,history[-1],flush=True)
    finally:hook.remove()
    need(updates==(1 if smoke else 288),'Incomplete fixed update budget')
    need(state_sha(teacher.state_dict())==source_hash and all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items() if k.startswith('size.')),'Frozen teacher/size changed')
    checkpoint=directory/(tag+'_training_state.pt')
    torch.save(dict(model=model.state_dict(),auxiliary=head.state_dict(),optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_cpu_rng=torch.get_rng_state(),
        torch_cuda_rng=torch.cuda.get_rng_state_all() if normal.is_cuda else [],update=updates,source_state_sha256=source_hash,real_data_used=False),checkpoint)
    restored=torch.load(checkpoint,map_location='cpu',weights_only=False);cpu_rng=torch.get_rng_state()
    try:
        rebuilt=scale.CenterScale();rebuilt.load_state_dict(restored['model'],strict=True);rebuilt.size.requires_grad_(False)
        rebuilt_head=auxiliary_head(seed,channels,'cpu');rebuilt_head.load_state_dict(restored['auxiliary'],strict=True)
        rebuilt_optimizer=torch.optim.AdamW([p for p in rebuilt.parameters() if p.requires_grad]+list(rebuilt_head.parameters()),lr=.00015,weight_decay=.01);rebuilt_optimizer.load_state_dict(restored['optimizer'])
        need(state_sha(rebuilt.state_dict())==state_sha(model.state_dict()) and state_sha(rebuilt_head.state_dict())==state_sha(head.state_dict()),'Independent model/auxiliary readback differs')
        need(len(restored['optimizer']['state'])==len(rebuilt_optimizer.state)==len(parameters),'Rebuilt optimizer inventory differs')
        for before,after in zip(restored['optimizer']['state'].values(),rebuilt_optimizer.state.values()):
            need(before.keys()==after.keys() and all(torch.equal(before[k].cpu(),after[k].cpu()) for k in before),'Independent Adam state readback differs')
    finally:torch.set_rng_state(cpu_rng)
    receipt=dict(method=method,seed=seed,arm='aux_visible_mask_center',branch='center',updates=updates,new_optimizer_updates=updates,source_state_sha256=source_hash,final_state_sha256=state_sha(model.state_dict()),
        auxiliary_state_sha256=state_sha(head.state_dict()),teacher_unchanged=True,size_unchanged=True,labels_sha256=labels.hexdigest(),exposure=exposure,independent_model_optimizer_rebuild=True,optimizer_state_equal=True,
        first_native_loss_oracle=first_oracle,checkpoint_path=str(checkpoint),checkpoint_sha256=shared.sha(checkpoint),checkpoint_verification_scope='model_auxiliary_optimizer_readback_only_no_resume_claim',resume_supported=False)
    shared.write(directory/(tag+'_receipt.json'),receipt)
    return model.eval(),head.eval(),history,receipt
