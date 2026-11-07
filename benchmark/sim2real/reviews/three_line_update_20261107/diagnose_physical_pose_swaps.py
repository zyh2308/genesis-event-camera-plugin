"""Separate center/depth contributions on fixed synthetic relighting validation."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('campaign','study','radiance','out'):
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.study/'sources'))
    import pose_light_robustness as shared
    source=json.loads((args.study/'physical_adaptation/completion.json').read_text())
    assert source['status']=='complete' and not source['real_data_used']
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    device=torch.device('cpu');R=args.campaign;A=R/'recovery_20261105_mainline'
    scale=shared.load('scale',A/'pose_scale_pilot_20261106.py')
    rotation=shared.load('rotation',A/'pose_rotation_pilot_20261106.py')
    fresh=shared.load('fresh',A/'pose_fresh_confirm_20261106_v3.py')
    sys.path.insert(0,str(R/'sources/benchmark/sim2real/scripts'))
    from run_pose_stage2_pilot import encode_event_window
    assets=json.loads((A/'pose_prepared_full_v5/protocol.json').read_text())['assets']
    assert shared.sha(assets['vertices']['path'])==assets['vertices']['sha256']
    vertices=np.load(assets['vertices']['path'],allow_pickle=False)
    tree=cKDTree(vertices);threshold=.1*assets['diameter_m']
    args.out.mkdir(exist_ok=False)
    shared.write(args.out/'protocol.json',dict(real_data_used=False,device='cpu',
        scope='fixed four synthetic validation trajectories; no training or real evaluation',
        rotations='original frozen branch, same reference predictions as the paired training report',
        swaps='two by two factorial of clean-control and physical-augmented predicted center and depth',
        source_sha256=shared.sha(__file__)))
    records=[]
    for method in shared.METHODS:
        for condition in ('half','eighth'):
            bank=args.radiance/condition/'pose_bank'
            rows=fresh.adapter(bank,sorted((bank/'physical').glob('*.h5')),method,encode_event_window)
            assert len({r['rid'] for r in rows})==4
            images=shared.tensors(rows,device);K=[r['K'] for r in rows]
            targets=shared.labels(scale,rows,vertices,device)[:,2].numpy()
            for seed in range(3):
                tag=f'{method}_seed{seed}'
                before=next(r for r in source['results'] if r['method']==method and r['seed']==seed)
                original=scale.CenterScale().eval()
                original_path=A/f'jobs/115_pose_scale_paired_20261106/results/{tag}_scale.pt'
                assert shared.sha(original_path)==before['parents']['scale_sha256']
                original.load_state_dict(torch.load(original_path,map_location='cpu',weights_only=True)['model'],strict=True)
                head=rotation.RotationHead().eval()
                head_path=A/f'jobs/117_pose_rotation_paired_20261106/results/{tag}_rotation.pt'
                assert shared.sha(head_path)==before['parents']['rotation_sha256']
                head.load_state_dict(torch.load(head_path,map_location='cpu',weights_only=True)['model'],strict=True)
                uv,_,size=scale.estimate(original,images,K)
                orientations=shared.frozen_rotation(rotation,head,images,uv,size,K)
                predictions={};pins={}
                for arm in ('clean_control','physical_augmented'):
                    path=args.study/'physical_adaptation'/f'{tag}_{arm}.pt'
                    pins[arm]=shared.sha(path)
                    model=scale.CenterScale().eval()
                    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
                    predictions[arm]=scale.estimate(model,images,K)
                cu,cz,cs=predictions['clean_control'];au,az,ass=predictions['physical_augmented']
                scores={}
                for name,(uv,z,size) in dict(control=(cu,cz,cs),center_only=(au,cz,cs),
                                            depth_only=(cu,az,ass),both=(au,az,ass)).items():
                    scores[name]=shared.score(rotation,rows,uv,z,size,targets,orientations,vertices,tree,threshold)
                for arm,name in [('clean_control','control'),('physical_augmented','both')]:
                    old=before['arms'][arm][condition]['summary'];new=scores[name]['summary']
                    assert abs(old['depth_mae_m']-new['depth_mae_m'])<1e-5
                    assert abs(old['center_px']-new['center_px'])<1e-3
                    assert old['success']==new['success']
                s={k:v['summary']['success'] for k,v in scores.items()}
                result=dict(method=method,seed=seed,condition=condition,scores=scores,weights_sha256=pins,
                    center_contribution_pp=50*((s['center_only']-s['control'])+(s['both']-s['depth_only'])),
                    depth_contribution_pp=50*((s['depth_only']-s['control'])+(s['both']-s['center_only'])))
                shared.write(args.out/f'{tag}_{condition}.json',result);records.append(result)
                print('SYNTHETIC_POSE_SWAP',tag,condition,flush=True)
    shared.write(args.out/'completion.json',dict(status='complete',real_data_used=False,results=records))


if __name__=='__main__':main()
