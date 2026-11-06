"""Paired synthetic-only adaptation of the event center and scale branch."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F

METHODS = ('eventgenesis', 'v2e')
ARMS = ('clean', 'augmented', 'consistency')
CONDITIONS = ('clean', 'moderate', 'strong')
EPOCHS = 6
BATCH = 32


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def tensors(rows, device):
    return torch.cat([F.interpolate(torch.from_numpy(np.stack([r['image'] for r in rows[i:i+32]])).to(device),
                                    (128,128), mode='area') for i in range(0,len(rows),32)])


def labels(scale, rows, vertices, device):
    data=[]
    for r in rows:
        T,K=r['T'],r['K']
        pts=vertices@T[:3,:3].T+T[:3,3]
        assert (pts[:,2]>0).all()
        q=pts@K.T
        uv=q[:,:2]/q[:,2:]/[346,260]
        z=(T[:3,:3]@r['keys'][0]+T[:3,3])[2]
        data.append([*r['uv'][0],float(np.log(np.ptp(uv,axis=0)).mean()),scale.focal(K),float(np.log(z))])
    return torch.as_tensor(data,dtype=torch.float32,device=device)


def distort(x, strength, seed, validation=False):
    """Count-space stress applied to resized frames, with shared paired draws."""
    g=torch.Generator(device=x.device).manual_seed(seed)
    b,_,h,w=x.shape
    size=(b,1,1,1)
    level=torch.as_tensor(strength,dtype=x.dtype,device=x.device).reshape(-1,1,1,1)
    def rand(shape):return torch.rand(shape,device=x.device,generator=g)
    def rnd(shape):return rand(shape)*2-1
    signal=torch.expm1(x[:,:2]*np.log(9.0))
    gain=torch.exp(level*(.65*rnd(size)+.25*rnd((b,2,1,1))))
    if validation:
        field=F.interpolate(rand((b,2,9,9)),(h,w),mode='bilinear',align_corners=False)
        burst=(rand((b,2,h,w))<(.025+.09*level))*(.35+.65*field)*level
    else:
        field=F.interpolate(rand((b,2,7,7)),(h,w),mode='bilinear',align_corners=False)
        burst=(rand((b,2,h,w))<(.03+.12*level))*(.25+.85*field)*level
    keep=rand((b,1,h,w))>(.22*level)
    noisy=signal*gain*keep+burst
    total=noisy.sum(dim=1,keepdim=True)
    return (torch.log1p(torch.cat((noisy,total),dim=1))/np.log(9.0)).clamp(0,1)


def affine(x, labels, rng):
    t=torch.as_tensor(rng.uniform(-1,1,(len(x),3)),dtype=x.dtype,device=x.device)
    zoom,shift=1+.15*t[:,0],.06*t[:,1:]
    theta=torch.zeros((len(x),2,3),device=x.device)
    theta[:,0,0]=theta[:,1,1]=1/zoom
    theta[:,:,2]=-2*shift/zoom[:,None]
    y=labels.clone()
    y[:,:2]=(y[:,:2]-.5)*zoom[:,None]+.5+shift
    y[:,2:4]+=zoom.log()[:,None]
    return F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),align_corners=False),y


def smooth(a,b):
    return F.smooth_l1_loss(a,b,beta=.1)


def supervised(model, predictions, target):
    logits,uv,size,shape=predictions
    dist=(model.grid[None]-target[:,None,:2]).square().sum(-1)
    heat=(-.5*dist*16**2).softmax(1)
    center=-(heat*logits.log_softmax(1)).sum(1).mean()
    z=shape+target[:,3]-size
    return (center+4*smooth(uv,target[:,:2])+2*smooth(size,target[:,2])
            +smooth(shape,target[:,4]-target[:,3]+target[:,2])+2*smooth(z,target[:,4]))


def fit(scale,state,x,y,seed,arm,smoke):
    model=scale.CenterScale().to(x.device)
    model.load_state_dict(state,strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=0.00035,weight_decay=.01)
    rng=np.random.default_rng(seed+38190)
    history=[]
    for epoch in range(1 if smoke else EPOCHS):
        order=rng.permutation(len(x))
        if smoke:order=order[:BATCH]
        losses=[]
        for offset in range(0,len(order),BATCH):
            ids=torch.from_numpy(order[offset:offset+BATCH]).to(x.device)
            clean,targets=affine(x[ids],y[ids],rng)
            strengths=rng.uniform(.2,1.,len(ids))
            second=clean if arm=='clean' else distort(clean,strengths,seed*200000+epoch*1000+offset+7163)
            p,q=model(clean),model(second)
            loss=.5*(supervised(model,p,targets)+supervised(model,q,targets))
            if arm=='consistency':
                loss=loss+2*smooth(p[1].detach(),q[1])+smooth(p[2].detach(),q[2])+2*smooth((p[3]-p[2]).detach(),q[3]-q[2])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            opt.step()
            assert torch.isfinite(loss)
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model.eval(),history


@torch.inference_mode()
def frozen_rotation(rotation,head,images,centers,sizes,K):
    rays,axes=rotation.ray_frame(centers,K)
    intrinsics=np.asarray(K)
    aspect=np.sqrt(intrinsics[:,0,0]*260/(intrinsics[:,1,1]*346))
    args=[torch.as_tensor(v,dtype=torch.float32,device=images.device) for v in (centers,sizes,aspect)]
    results=[]
    for i in range(0,len(images),BATCH):
        sl=slice(i,i+BATCH)
        frames,_=rotation.crop(images[sl],*[v[sl] for v in args],torch.zeros(len(images[sl]),device=images.device))
        results.append(head(frames).cpu().numpy())
    return axes@np.concatenate(results)


def score(rotation,rows,centers,depths,sizes,gt_log_size,rotations,vertices,tree,threshold):
    grouped={}
    rays,_=rotation.ray_frame(centers,[r['K'] for r in rows])
    for sample,uv,z,size,target_size,R,ray in zip(rows,centers,depths,sizes,gt_log_size,rotations,rays,strict=True):
        T,keys=sample['T'],sample['keys']
        z_gt=(T[:3,:3]@keys[0]+T[:3,3])[2]
        truth=vertices@T[:3,:3].T+T[:3,3]
        t=ray*z-R@keys[0]
        err=float(tree.query((truth-t)@R,workers=1)[0].mean()) if ((keys@R.T+t)[:,2]>0).all() else None
        log_size_error=float(np.log(size)-target_size)
        grouped.setdefault(sample['rid'],[]).append([abs(z-z_gt),z-z_gt,abs(log_size_error),log_size_error,
                                                    np.linalg.norm((uv-sample['uv'][0])*[346,260]),
                                                    float(err is not None and err<threshold),max(1-err/.1,0) if err is not None else 0])
    names=('depth_mae_m','depth_bias_m','log_size_mae','log_size_bias','center_px','success','auc')
    by={rid:dict(zip(names,np.mean(x,axis=0).tolist(),strict=True)) for rid,x in sorted(grouped.items())}
    return dict(summary={key:float(np.mean([v[key] for v in by.values()])) for key in names},by_rollout=by)


def gate(results):
    checks=[]
    rng=np.random.default_rng(22671)
    for method in METHODS:
        paired=[r for r in results if r['method']==method]
        for condition in CONDITIONS[1:]:
            a=np.asarray([[run['arms']['clean'][condition]['by_rollout'][rid]['depth_mae_m']-
                           run['arms']['consistency'][condition]['by_rollout'][rid]['depth_mae_m']
                           for rid in run['arms']['clean'][condition]['by_rollout']] for run in paired])
            assert a.shape==(3,96)
            draws=rng.integers(96,size=(4000,96))
            ci=np.quantile(a.mean(0)[draws].mean(1),[.025,.975]).tolist()
            delta=a.mean(1).tolist()
            checks.append(dict(method=method,condition=condition,depth_improvement_m=delta,ci95_m=ci,
                               passed=all(v>0 for v in delta) and ci[0]>0))
        for seed,run in enumerate(paired):
            a=run['arms']
            d=a['consistency']['clean']['summary']['success']-a['clean']['clean']['summary']['success']
            checks.append(dict(method=method,seed=seed,condition='clean',success_delta=d,passed=d>=-.02))
    return dict(passed=all(x['passed'] for x in checks),checks=checks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    device=torch.device(args.device)
    if device.type=='cuda':assert torch.cuda.device_count()==1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    cv2.setNumThreads(1)
    R=args.campaign
    A=R/'recovery_20261105_mainline'
    D=A/'jobs/119_pose_fresh_confirm_paired_20261106/results'
    scale,rotation,pilot,fresh=(load(k,A/f) for k,f in [('scale','pose_scale_pilot_20261106.py'),
        ('rotation','pose_rotation_pilot_20261106.py'),('pilot','pose_depth_pilot_20261106.py'),
        ('fresh','pose_fresh_confirm_20261106_v3.py')])
    helper=R.parents[2]/'runs/sim2real/20261106-pose-synthetic-diagnostic-v1/causal_tests_v2.py'
    load_original=pilot.helpers(helper,R)['load']
    sys.path.insert(0,str(R/'sources/benchmark/sim2real/scripts'))
    from run_pose_stage2_pilot import encode_event_window
    protocol=json.loads((A/'pose_prepared_full_v5/protocol.json').read_text())
    asset=protocol['assets']['vertices']
    assert sha(asset['path'])==asset['sha256']
    vertices=np.load(asset['path'],allow_pickle=False)
    tree=cKDTree(vertices)
    threshold=.1*protocol['assets']['diameter_m']
    root=D/'pose_bank'
    confirmation=json.loads((D/'completion.json').read_text())
    assert confirmation['gate']['passed'] and not confirmation['real_data_used']
    files=sorted((root/'physical').glob('*.h5'))
    assert len(files)==96
    args.out.mkdir(parents=True,exist_ok=False)
    write(args.out/'protocol.json',dict(smoke=args.smoke,real_data_used=False,device=args.device,arms=ARMS,
          conditions=CONDITIONS,epochs=EPOCHS,windows_per_trajectory=4,source_sha256=sha(__file__)))
    results=[]
    pair_ids=pair_train=None
    for method in METHODS:
        train=load_original(method,'train')
        assert len(train)==1529
        binding=hashlib.sha256()
        for row in train:
            binding.update(row['id'].encode()+row['rid'].encode())
            for key in ('keys','K','T','uv'):
                binding.update(row[key].tobytes())
        pair_train=pair_train or binding.hexdigest()
        assert pair_train==binding.hexdigest()
        if args.smoke:train=train[:64]
        x,y=tensors(train,device),labels(scale,train,vertices,device)
        del train
        validation=fresh.adapter(root,files[:2] if args.smoke else files,method,encode_event_window)
        by={}
        for sample in validation:by.setdefault(sample['rid'],[]).append(sample)
        val=[g[i] for g in by.values() for i in np.linspace(0,len(g)-1,4,dtype=int)]
        del by,validation
        ids=[r['id'] for r in val]
        pair_ids=ids if pair_ids is None else pair_ids
        assert pair_ids==ids
        clean=tensors(val,device)
        views={'clean':clean,'moderate':distort(clean,.35,20261107,True),'strong':distort(clean,.75,20261108,True)}
        K=[r['K'] for r in val]
        gt_log_size=labels(scale,val,vertices,device)[:,2].cpu().numpy()
        for seed in ((0,) if args.smoke else (0,1,2)):
            tag=f'{method}_seed{seed}'
            record=next(r for r in confirmation['results'] if (r['method'],r['seed'])==(method,seed))
            sp=A/f'jobs/115_pose_scale_paired_20261106/results/{tag}_scale.pt'
            rp=A/f'jobs/117_pose_rotation_paired_20261106/results/{tag}_rotation.pt'
            assert sha(sp)==record['scale_sha256'] and sha(rp)==record['rotation_sha256']
            state=torch.load(sp,map_location='cpu',weights_only=True)['model']
            fixed=scale.CenterScale().to(device).eval()
            fixed.load_state_dict(state,strict=True)
            head=rotation.RotationHead().to(device).eval()
            head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
            reference={}
            for cond,v in views.items():
                center,z,size=scale.estimate(fixed,v,K)
                r=frozen_rotation(rotation,head,v,center,size,K)
                reference[cond]=(center,z,size,r)
            run=dict(method=method,seed=seed,arms={},train_loss={})
            for arm in ('original',*ARMS):
                run['arms'][arm]={}
                if arm!='original':
                    model,history=fit(scale,state,x,y,seed,arm,args.smoke)
                    run['train_loss'][arm]=history
                    torch.save(model.state_dict(),args.out/f'{tag}_{arm}.pt')
                for cond,v in views.items():
                    center,z,size,r=reference[cond]
                    if arm!='original':center,z,size=scale.estimate(model,v,K)
                    run['arms'][arm][cond]=score(rotation,val,center,z,size,gt_log_size,r,vertices,tree,threshold)
                print('ROBUST_ARM',tag,arm,json.dumps({k:v['summary'] for k,v in run['arms'][arm].items()}),flush=True)
            write(args.out/f'{tag}.json',run)
            results.append(run)
    write(args.out/'completion.json',dict(status='complete',results=results,real_data_used=False,paired_train_label_sha256=pair_train,
          gate=None if args.smoke else gate(results)))


if __name__=='__main__':main()
