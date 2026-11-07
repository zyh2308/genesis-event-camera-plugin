"""Paired short adaptation with physical relighting and a matched clean control."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

import pose_light_robustness as shared


def identity(rows):
    return [(r['id'],*(r[k].tobytes() for k in ('K','T','keys','uv'))) for r in rows]


def fit(scale,state,normal,target,aux,aux_target,seed,smoke):
    model=scale.CenterScale().to(normal.device)
    model.load_state_dict(state,strict=True)
    teacher=scale.CenterScale().to(normal.device).eval()
    teacher.load_state_dict(state,strict=True);teacher.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.01)
    rng=np.random.default_rng(71070+seed)
    history=[]
    for epoch in range(1 if smoke else 6):
        order=rng.permutation(len(normal))
        if smoke:order=order[:32]
        totals=[]
        for start in range(0,len(order),32):
            ids=torch.as_tensor(order[start:start+32],device=normal.device)
            aux_ids=torch.as_tensor(rng.integers(len(aux),size=len(ids)),device=normal.device)
            x,y=shared.affine(normal[ids],target[ids],rng)
            a,b=shared.affine(aux[aux_ids],aux_target[aux_ids],rng)
            clean,pred=model(x),model(a)
            with torch.no_grad():reference=teacher(x)
            loss=.5*(shared.supervised(model,clean,y)+shared.supervised(model,pred,b))
            loss=loss+2*(4*shared.smooth(clean[1],reference[1])+shared.smooth(clean[2],reference[2])
                +2*shared.smooth(clean[3]-clean[2],reference[3]-reference[2]))
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            assert torch.isfinite(loss) and norm>0
            opt.step();totals.append(float(loss.detach()))
        history.append(float(np.mean(totals)))
    return model.eval(),history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('campaign','radiance-train','radiance-validation','out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda');R=args.campaign;A=R/'recovery_20261105_mainline'
    D=A/'jobs/119_pose_fresh_confirm_paired_20261106/results'
    prep=json.loads((args.radiance_train/'completion.json').read_text())
    assert prep['status']=='complete' and not prep['real_data_used']
    selected=prep['selected'];assert len(selected)==8 and all(s.startswith('train_') for s in selected)
    scale=shared.load('scale',A/'pose_scale_pilot_20261106.py')
    rotation=shared.load('rotation',A/'pose_rotation_pilot_20261106.py')
    pilot=shared.load('pilot',A/'pose_depth_pilot_20261106.py')
    fresh=shared.load('fresh',A/'pose_fresh_confirm_20261106_v3.py')
    loader=pilot.helpers(R.parents[2]/'runs/sim2real/20261106-pose-synthetic-diagnostic-v1/causal_tests_v2.py',R)['load']
    sys.path.insert(0,str(R/'sources/benchmark/sim2real/scripts'))
    from run_pose_stage2_pilot import encode_event_window
    asset=json.loads((A/'pose_prepared_full_v5/protocol.json').read_text())['assets']
    assert shared.sha(asset['vertices']['path'])==asset['vertices']['sha256']
    vertices=np.load(asset['vertices']['path'],allow_pickle=False)
    tree=cKDTree(vertices);threshold=.1*asset['diameter_m']
    parents=json.loads((D/'completion.json').read_text())['results']
    args.out.mkdir(parents=True,exist_ok=False)
    shared.write(args.out/'protocol.json',dict(real_data_used=False,smoke=args.smoke,seeds=[0] if args.smoke else [0,1,2],
        epochs=1 if args.smoke else 6,lr=.00015,teacher_clean_weight=2,arms=['clean_control','physical_augmented'],
        training='all original train windows plus one matched auxiliary batch per step; eight training trajectories at half/eighth radiance',
        source_sha256=shared.sha(__file__),real_evaluation='disabled for this study',
        validation='96 previously inspected synthetic trajectories and four disjoint relit validation trajectories',
        gate='each seed clean ADD-S success decline <=2pp; each seed improves depth at half and eighth physical radiance; both methods'))
    results=[];paired=None
    for method in shared.METHODS:
        rows=loader(method,'train');assert len(rows)==1529
        if args.smoke:rows=rows[:64]
        x,y=shared.tensors(rows,device),shared.labels(scale,rows,vertices,device)
        base_files=[R/'pose_bank/physical'/(s+'.h5') for s in selected]
        base=fresh.adapter(R/'pose_bank',base_files,method,encode_event_window)
        relit=[];controls=[]
        for condition in ('half','eighth'):
            bank=args.radiance_train/condition/'pose_bank'
            current=fresh.adapter(bank,[bank/'physical'/(s+'.h5') for s in selected],method,encode_event_window)
            assert identity(current)==identity(base)
            relit+=current;controls+=base
        paired=identity(relit) if paired is None else paired
        assert identity(relit)==paired
        auxiliary={'clean_control':shared.tensors(controls,device),'physical_augmented':shared.tensors(relit,device)}
        aux_y=shared.labels(scale,relit,vertices,device)
        valfiles=sorted((D/'pose_bank/physical').glob('*.h5'));assert len(valfiles)==96
        validation=fresh.adapter(D/'pose_bank',valfiles[:2] if args.smoke else valfiles,method,encode_event_window)
        grouped={}
        for row in validation:grouped.setdefault(row['rid'],[]).append(row)
        val=[g[i] for g in grouped.values() for i in np.linspace(0,len(g)-1,4,dtype=int)]
        conditions={'clean':val}
        for condition in ('half','eighth'):
            bank=args.radiance_validation/condition/'pose_bank'
            rows=fresh.adapter(bank,sorted((bank/'physical').glob('*.h5')),method,encode_event_window)
            assert not ({r['rid'] for r in rows}&set(selected))
            conditions[condition]=rows
        views={c:shared.tensors(rows,device) for c,rows in conditions.items()}
        for seed in ([0] if args.smoke else range(3)):
            tag=f'{method}_seed{seed}';parent=next(r for r in parents if (r['method'],r['seed'])==(method,seed))
            sp=A/f'jobs/115_pose_scale_paired_20261106/results/{tag}_scale.pt'
            rp=A/f'jobs/117_pose_rotation_paired_20261106/results/{tag}_rotation.pt'
            assert shared.sha(sp)==parent['scale_sha256'] and shared.sha(rp)==parent['rotation_sha256']
            state=torch.load(sp,map_location='cpu',weights_only=True)['model']
            frozen=scale.CenterScale().to(device).eval();frozen.load_state_dict(state,strict=True)
            head=rotation.RotationHead().to(device).eval()
            head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
            orientations={}
            for c,frames in views.items():
                K=[r['K'] for r in conditions[c]]
                uv,_,size=scale.estimate(frozen,frames,K)
                orientations[c]=shared.frozen_rotation(rotation,head,frames,uv,size,K)
            result=dict(method=method,seed=seed,arms={},train_loss={},parents=parent)
            for arm in auxiliary:
                model,history=fit(scale,state,x,y,auxiliary[arm],aux_y,seed,args.smoke)
                result['train_loss'][arm]=history;result['arms'][arm]={}
                torch.save(model.state_dict(),args.out/f'{tag}_{arm}.pt')
                for c,frames in views.items():
                    rows=conditions[c];K=[r['K'] for r in rows]
                    uv,z,size=scale.estimate(model,frames,K)
                    target_size=shared.labels(scale,rows,vertices,device)[:,2].cpu().numpy()
                    result['arms'][arm][c]=shared.score(rotation,rows,uv,z,size,target_size,orientations[c],vertices,tree,threshold)
                print('PHYSICAL_ADAPTATION',tag,arm,json.dumps({k:v['summary'] for k,v in result['arms'][arm].items()}),flush=True)
            shared.write(args.out/(tag+'.json'),result);results.append(result)
    checks=[]
    if not args.smoke:
        for run in results:
            a,b=run['arms']['clean_control'],run['arms']['physical_augmented']
            checks.append(dict(method=run['method'],seed=run['seed'],condition='clean',
                passed=b['clean']['summary']['success']-a['clean']['summary']['success']>=-.02))
            for c in ('half','eighth'):
                checks.append(dict(method=run['method'],seed=run['seed'],condition=c,
                    passed=b[c]['summary']['depth_mae_m']<a[c]['summary']['depth_mae_m']))
    shared.write(args.out/'completion.json',dict(status='complete',real_data_used=False,results=results,
        gate=None if args.smoke else dict(passed=all(c['passed'] for c in checks),checks=checks),
        limitation='four radiance validation trajectories, exploratory development only; no population significance claim'))


if __name__=='__main__':main()
