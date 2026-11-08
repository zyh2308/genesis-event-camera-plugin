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


def matched_affine(a,normal,target,rng):
    import copy
    paired=np.random.default_rng()
    paired.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
    first,y=shared.affine(a,target,rng)
    second,yy=shared.affine(normal,target,paired)
    assert torch.equal(y,yy)
    return first,second,y


def fit(scale,state,normal,target,aux,aux_target,seed,smoke,aux_clean,pair_weight):
    model=scale.CenterScale().to(normal.device)
    model.load_state_dict(state,strict=True)
    teacher=scale.CenterScale().to(normal.device).eval()
    teacher.load_state_dict(state,strict=True);teacher.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.01)
    rng=np.random.default_rng(71070+seed)
    history=[];updates=0
    for epoch in range(1 if smoke else 6):
        order=rng.permutation(len(normal))
        if smoke:order=order[:32]
        totals=[]
        for start in range(0,len(order),32):
            ids=torch.as_tensor(order[start:start+32],device=normal.device)
            aux_ids=torch.as_tensor(rng.integers(len(aux),size=len(ids)),device=normal.device)
            x,y=shared.affine(normal[ids],target[ids],rng)
            a,paired,b=matched_affine(aux[aux_ids],aux_clean[aux_ids],aux_target[aux_ids],rng)
            clean,pred=model(x),model(a)
            with torch.no_grad():reference=teacher(x);paired_reference=teacher(paired)
            loss=.5*(shared.supervised(model,clean,y)+shared.supervised(model,pred,b))
            loss=loss+2*(4*shared.smooth(clean[1],reference[1])+shared.smooth(clean[2],reference[2])
                +2*shared.smooth(clean[3]-clean[2],reference[3]-reference[2]))
            loss=loss+pair_weight*(shared.smooth(pred[2],paired_reference[2])
                +2*shared.smooth(pred[3]-pred[2],paired_reference[3]-paired_reference[2]))
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            assert torch.isfinite(loss) and norm>0 and all(p.grad is None for p in teacher.parameters())
            opt.step();updates+=1;totals.append(float(loss.detach()))
        history.append(float(np.mean(totals)))
        print('DEPTH_EPOCH',seed,pair_weight,epoch+1,updates,history[-1],flush=True)
    assert updates==(1 if smoke else 288)
    assert all(torch.equal(v.cpu(),state[k]) for k,v in teacher.state_dict().items())
    return model.eval(),history


@torch.inference_mode()
def estimate_centers(model,images):
    return np.concatenate([model(images[i:i+32])[1].cpu().numpy() for i in range(0,len(images),32)])


def fit_center(scale,state,normal,target,aux,aux_target,seed,smoke,aux_clean,pair_weight):
    model=scale.CenterScale().to(normal.device);model.load_state_dict(state,strict=True)
    model.size.requires_grad_(False)
    teacher=scale.CenterScale().to(normal.device).eval();teacher.load_state_dict(state,strict=True);teacher.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00015,weight_decay=.01)
    rng=np.random.default_rng(71070+seed);history=[];updates=0
    def supervised(pred,y):
        dist=(model.grid[None]-y[:,None,:2]).square().sum(-1)
        heat=(-.5*dist*16**2).softmax(1)
        return -(heat*pred[0].log_softmax(1)).sum(1).mean()+4*shared.smooth(pred[1],y[:,:2])
    def consistency(pred,ref):
        return 4*shared.smooth(pred[1],ref[1])+torch.nn.functional.kl_div(pred[0].log_softmax(1),ref[0].softmax(1),reduction='batchmean')
    for epoch in range(1 if smoke else 6):
        order=rng.permutation(len(normal))
        if smoke:order=order[:32]
        losses=[]
        for start in range(0,len(order),32):
            ids=torch.as_tensor(order[start:start+32],device=normal.device)
            ai=torch.as_tensor(rng.integers(len(aux),size=len(ids)),device=normal.device)
            x,y=shared.affine(normal[ids],target[ids],rng)
            a,paired,b=matched_affine(aux[ai],aux_clean[ai],aux_target[ai],rng)
            clean,pred=model(x),model(a)
            with torch.no_grad():reference=teacher(x);paired_reference=teacher(paired)
            loss=.5*(supervised(clean,y)+supervised(pred,b))+2*consistency(clean,reference)+pair_weight*consistency(pred,paired_reference)
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5,error_if_nonfinite=True)
            assert torch.isfinite(loss) and norm>0 and all(p.grad is None for p in teacher.parameters())
            assert all(p.grad is None for p in model.size.parameters())
            opt.step();updates+=1;losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        print('LOCALIZER_EPOCH',seed,pair_weight,epoch+1,updates,history[-1],flush=True)
    assert updates==(1 if smoke else 288)
    assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items() if k.startswith('size.'))
    assert all(torch.equal(v.cpu(),state[k]) for k,v in teacher.state_dict().items())
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
    selected=prep['selected'];assert len(selected)==96 and all(s.startswith('train_') for s in selected)
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
        epochs=1 if args.smoke else 6,lr=.00015,teacher_clean_weight=2,paired_teacher_weight=2,final_center='independent trained localizer',localizer_epochs=(1 if args.smoke else 6),total_optimizer_updates_per_fit=(2 if args.smoke else 576),arms=['clean_control','physical_supervised','paired_consistency'],
        training='all original train windows plus one matched auxiliary batch per step; all 96 original training trajectories at half/eighth radiance',
        source_sha256=shared.sha(__file__),real_evaluation='disabled for this study',
        validation='96 previously inspected synthetic trajectories and four disjoint relit validation trajectories',
        gate='each seed clean ADD-S success decline <=2pp; each seed improves depth at half and eighth physical radiance and never lowers relit pose success; both methods'))
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
        control_tensor=shared.tensors(controls,device);relit_tensor=shared.tensors(relit,device)
        auxiliary={'clean_control':control_tensor,'physical_supervised':relit_tensor,'paired_consistency':relit_tensor}
        aux_y=shared.labels(scale,relit,vertices,device)
        valfiles=sorted((D/'pose_bank/physical').glob('*.h5'));assert len(valfiles)==96
        validation=fresh.adapter(D/'pose_bank',valfiles[:2] if args.smoke else valfiles,method,encode_event_window)
        grouped={}
        for row in validation:grouped.setdefault(row['rid'],[]).append(row)
        val=[g[i] for g in grouped.values() for i in np.linspace(0,len(g)-1,4,dtype=int)]
        conditions={'clean':val,'clean_full':validation}
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
            control=R.parent/'20261107-pose-physical-light-train-v2/physical_adaptation'/f'{tag}_clean_control.pt'
            state=torch.load(control,map_location='cpu',weights_only=True)
            localizer=scale.CenterScale().to(device).eval();localizer.load_state_dict(state,strict=True)
            localizer.requires_grad_(False)
            fixed_centers={c:scale.estimate(localizer,frames,[r['K'] for r in conditions[c]])[0] for c,frames in views.items()}
            result=dict(method=method,seed=seed,arms={},train_loss={},parents=parent,
                localization_path=str(control),localization_sha256=shared.sha(control),final_center_source='independent learned localizer per arm',
                rotation_source='frozen rotation weights; final crop and ray frame use predicted localizer and depth-size outputs',localization_paths={},localization_sha256s={},localization_train_loss={},depth_only_arms={},translation_only_arms={})
            for arm in auxiliary:
                model,history=fit(scale,state,x,y,auxiliary[arm],aux_y,seed,args.smoke,control_tensor,2 if arm=='paired_consistency' else 0)
                result['train_loss'][arm]=history;result['arms'][arm]={}
                torch.save(model.state_dict(),args.out/f'{tag}_{arm}.pt')
                learned_localizer,loc_history=fit_center(scale,state,x,y,auxiliary[arm],aux_y,seed,args.smoke,control_tensor,2 if arm=='paired_consistency' else 0)
                locpath=args.out/f'{tag}_{arm}_localizer.pt';torch.save(learned_localizer.state_dict(),locpath)
                result['localization_paths'][arm]=str(locpath);result['localization_sha256s'][arm]=shared.sha(locpath)
                result['localization_train_loss'][arm]=loc_history
                result['depth_only_arms'][arm]={};result['translation_only_arms'][arm]={}
                for c,frames in views.items():
                    rows=conditions[c];K=[r['K'] for r in rows]
                    _,z,size=scale.estimate(model,frames,K)
                    uv=fixed_centers[c]
                    assert np.array_equal(uv,scale.estimate(localizer,frames,K)[0])
                    target_size=shared.labels(scale,rows,vertices,device)[:,2].cpu().numpy()
                    result['depth_only_arms'][arm][c]=shared.score(rotation,rows,uv,z,size,target_size,orientations[c],vertices,tree,threshold)
                    uv=estimate_centers(learned_localizer,frames)
                    result['translation_only_arms'][arm][c]=shared.score(rotation,rows,uv,z,size,target_size,orientations[c],vertices,tree,threshold)
                    final_orientations=shared.frozen_rotation(rotation,head,frames,uv,size,K)
                    result['arms'][arm][c]=shared.score(rotation,rows,uv,z,size,target_size,final_orientations,vertices,tree,threshold)
                print('INDEPENDENT_LOCALIZER_AND_DEPTH',tag,arm,json.dumps({k:v['summary'] for k,v in result['arms'][arm].items()}),flush=True)
            shared.write(args.out/(tag+'.json'),result);results.append(result)
    checks=[]
    if not args.smoke:
        for run in results:
            a,b=run['arms']['clean_control'],run['arms']['paired_consistency']
            checks.append(dict(method=run['method'],seed=run['seed'],condition='clean',
                passed=b['clean']['summary']['success']-a['clean']['summary']['success']>=-.02))
            for c in ('half','eighth'):
                checks.append(dict(method=run['method'],seed=run['seed'],condition=c,
                    passed=b[c]['summary']['depth_mae_m']<a[c]['summary']['depth_mae_m']))
                checks.append(dict(method=run['method'],seed=run['seed'],condition=c+'_pose',
                    passed=b[c]['summary']['success']>=a[c]['summary']['success']))
    from pose_coverage_real import original_confirmation
    confirmation=None if args.smoke else original_confirmation(results,R)
    shared.write(args.out/'completion.json',dict(status='complete',real_data_used=False,results=results,
        gate=None if args.smoke else dict(passed=all(c['passed'] for c in checks),checks=checks),original_confirmation=confirmation,
        limitation='four radiance validation trajectories, exploratory development only; no population significance claim'))


if __name__=='__main__':main()
