"""Select one shared weight interpolation on synthetic development trajectories."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

import pose_light_robustness as robust

ALPHAS=(0.,.03125,.0625,.09375,.125,.25,.5,.75,1.)


def interpolate(clean,augmented,alpha):
    assert clean.keys()==augmented.keys()
    out={}
    for key,original in clean.items():
        if original.is_floating_point():
            out[key]=(1-alpha)*original+alpha*augmented[key]
        else:
            assert torch.equal(original,augmented[key])
            out[key]=original.clone()
    return out


def make_views(rows,device):
    x=robust.tensors(rows,device)
    return {'clean':x,'moderate':robust.distort(x,.35,20261107,True),'strong':robust.distort(x,.75,20261108,True)}


def evaluate(scale,rotation,head,states,samples,views,vertices,tree,threshold,baseline):
    K=[x['K'] for x in samples]
    device=views['clean'].device
    target_size=robust.labels(scale,samples,vertices,device)[:,2].cpu().numpy()
    reference={}
    original=scale.CenterScale().to(device).eval()
    original.load_state_dict(baseline,strict=True)
    for condition,frames in views.items():
        uv,_,size=scale.estimate(original,frames,K)
        reference[condition]=robust.frozen_rotation(rotation,head,frames,uv,size,K)
    results={}
    for alpha,state in states.items():
        model=scale.CenterScale().to(device).eval()
        model.load_state_dict(state,strict=True)
        scores={}
        for condition,frames in views.items():
            uv,z,size=scale.estimate(model,frames,K)
            scores[condition]=robust.score(rotation,samples,uv,z,size,target_size,reference[condition],vertices,tree,threshold)
        results[alpha]=scores
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('campaign','from-study','out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--development-cache',type=Path,required=True)
    args=p.parse_args()
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    cv2.setNumThreads(1)
    device=torch.device('cuda')
    R=args.campaign;A=R/'recovery_20261105_mainline'
    D=A/'jobs/119_pose_fresh_confirm_paired_20261106/results'
    prior=json.loads((args.from_study/'robustness/completion.json').read_text())
    assert prior['status']=='complete' and not prior['real_data_used'] and len(prior['results'])==6
    cached=json.loads(args.development_cache.read_text())
    assert cached['selected_alpha']==0 and not cached['real_data_used']
    scale=robust.load('scale',A/'pose_scale_pilot_20261106.py')
    rotation=robust.load('rotation',A/'pose_rotation_pilot_20261106.py')
    pilot=robust.load('pilot',A/'pose_depth_pilot_20261106.py')
    fresh=robust.load('fresh',A/'pose_fresh_confirm_20261106_v3.py')
    helper=R.parents[2]/'runs/sim2real/20261106-pose-synthetic-diagnostic-v1/causal_tests_v2.py'
    loader=pilot.helpers(helper,R)['load']
    sys.path.insert(0,str(R/'sources/benchmark/sim2real/scripts'))
    from run_pose_stage2_pilot import encode_event_window
    frozen=json.loads((D/'completion.json').read_text())['results']
    asset=json.loads((A/'pose_prepared_full_v5/protocol.json').read_text())['assets']
    assert robust.sha(asset['vertices']['path'])==asset['vertices']['sha256']
    vertices=np.load(asset['vertices']['path'],allow_pickle=False)
    tree=cKDTree(vertices);threshold=.1*asset['diameter_m']
    args.out.mkdir(parents=True,exist_ok=False)
    robust.write(args.out/'protocol.json',dict(real_data_used=False,primary='interpolated',alphas=ALPHAS,
         development='original 32 synthetic validation trajectories; all510 windows',
         evaluation='96 separate synthetic trajectories; four windows per trajectory',
         selection='one shared alpha; clean success loss at most2pp for each method andseed; minimize pooled stress depth MAE',
         refinement='coarse development grid admitted only alpha zero; refine below the smallest positive alpha without changing gate',
         development_cache_sha256=robust.sha(args.development_cache),
         source_sha256=robust.sha(__file__)))
    dev,weights,base,heads,pins={}, {}, {}, {}, {}
    for method in robust.METHODS:
        rows=loader(method,'validation')
        assert len(rows)==510 and len({r['rid'] for r in rows})==32
        views=make_views(rows,device)
        for seed in range(3):
            tag=f'{method}_seed{seed}'
            c,a=(args.from_study/'robustness'/f'{tag}_{arm}.pt' for arm in ('clean','augmented'))
            clean,aug=(torch.load(path,map_location='cpu',weights_only=True) for path in (c,a))
            weights[tag]=(clean,aug)
            record=next(x for x in frozen if (x['method'],x['seed'])==(method,seed))
            sp=A/f'jobs/115_pose_scale_paired_20261106/results/{tag}_scale.pt'
            rp=A/f'jobs/117_pose_rotation_paired_20261106/results/{tag}_rotation.pt'
            assert robust.sha(sp)==record['scale_sha256'] and robust.sha(rp)==record['rotation_sha256']
            pins[tag]=dict(clean_sha256=robust.sha(c),augmented_sha256=robust.sha(a),
                           original_sha256=record['scale_sha256'],rotation_sha256=record['rotation_sha256'])
            assert pins[tag]==cached['parent_sha256'][tag]
            base[tag]=torch.load(sp,map_location='cpu',weights_only=True)['model']
            head=rotation.RotationHead().to(device).eval()
            head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
            heads[tag]=head
            dev[tag]=dict(cached['development'][tag])
            additions={str(alpha):interpolate(clean,aug,alpha) for alpha in ALPHAS if str(alpha) not in dev[tag]}
            assert set(additions)=={'0.03125','0.0625','0.09375'}
            dev[tag].update(evaluate(scale,rotation,head,additions,rows,views,vertices,tree,threshold,base[tag]))
            print('INTERPOLATION_DEVELOPMENT',tag,flush=True)
        del views,rows
    selection=[]
    for alpha in ALPHAS:
        candidate=str(alpha)
        eligible=all(dev[tag][candidate]['clean']['summary']['success']-
                     dev[tag]['0.0']['clean']['summary']['success']>=-.02 for tag in dev)
        loss=float(np.mean([d[candidate][c]['summary']['depth_mae_m'] for d in dev.values() for c in ('moderate','strong')]))
        selection.append(dict(alpha=alpha,eligible=eligible,depth_mae_m=loss))
    best=min((s for s in selection if s['eligible']),key=lambda s:(s['depth_mae_m'],s['alpha']))['alpha']
    robust.write(args.out/'selection.json',dict(selected_alpha=best,candidates=selection,development=dev,
                 parent_sha256=pins,real_data_used=False))
    print('INTERPOLATION_SELECTED',best,flush=True)
    results=[]
    bank=D/'pose_bank'
    files=sorted((bank/'physical').glob('*.h5'))
    assert len(files)==96
    for method in robust.METHODS:
        allrows=fresh.adapter(bank,files,method,encode_event_window)
        grouped={}
        for row in allrows:grouped.setdefault(row['rid'],[]).append(row)
        rows=[g[i] for g in grouped.values() for i in np.linspace(0,len(g)-1,4,dtype=int)]
        del grouped,allrows
        views=make_views(rows,device)
        for seed in range(3):
            tag=f'{method}_seed{seed}'
            c,a=weights[tag]
            state=interpolate(c,a,best)
            run=evaluate(scale,rotation,heads[tag],{'clean':c,'interpolated':state},rows,views,
                         vertices,tree,threshold,base[tag])
            result=dict(method=method,seed=seed,alpha=best,arms=run)
            torch.save(state,args.out/f'{tag}_interpolated.pt')
            robust.write(args.out/(tag+'.json'),result)
            results.append(result)
            print('INTERPOLATION_CONFIRMED',tag,flush=True)
        del rows,views
    robust.write(args.out/'completion.json',dict(status='complete',results=results,real_data_used=False,
          selected_alpha=best,gate=robust.gate(results,'interpolated')))


if __name__=='__main__':main()
