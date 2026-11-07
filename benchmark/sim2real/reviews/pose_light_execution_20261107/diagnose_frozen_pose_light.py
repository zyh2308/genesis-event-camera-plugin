"""One frozen candidate evaluation on previously registered E-POSE diagnostic windows."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

import pose_light_robustness as robust


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','study','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    C=json.loads((args.study/'robustness/completion.json').read_text())
    assert C['status']=='complete' and not C['real_data_used'] and C['gate']['passed']
    protocol=json.loads((args.study/'robustness/protocol.json').read_text())
    arm=protocol['primary']
    R=args.campaign;A=R/'recovery_20261105_mainline'
    D=A/'jobs/119_pose_fresh_confirm_paired_20261106/results'
    scale=robust.load('scale',A/'pose_scale_pilot_20261106.py')
    rotation=robust.load('rotation',A/'pose_rotation_pilot_20261106.py')
    native=robust.load('native',A/'stage9_epose_v5.py')
    prepared=A/'pose_prepared_full_v5/protocol.json'
    metadata=json.loads(prepared.read_text())
    authorization=json.loads((D/'epose/registration.json').read_text())
    assert robust.sha(prepared)==authorization['hashes']['protocol']
    for key in ('vertices','keypoints'):
        x=metadata['assets'][key]
        assert robust.sha(x['path'])==x['sha256']
    vertices=np.load(metadata['assets']['vertices']['path'],allow_pickle=False)
    keys=np.load(metadata['assets']['keypoints']['path'],allow_pickle=False)
    tree=cKDTree(vertices)
    threshold=.1*metadata['assets']['diameter_m']
    fresh=json.loads((D/'completion.json').read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    models,pins={},{}
    for run in fresh['results']:
        method,seed=run['method'],run['seed'];tag=f'{method}_seed{seed}'
        sp=A/f'jobs/115_pose_scale_paired_20261106/results/{tag}_scale.pt'
        rp=A/f'jobs/117_pose_rotation_paired_20261106/results/{tag}_rotation.pt'
        candidate=args.study/'robustness'/f'{tag}_{arm}.pt'
        assert robust.sha(sp)==run['scale_sha256'] and robust.sha(rp)==run['rotation_sha256']
        pins[tag]=dict(candidate_sha256=robust.sha(candidate),original_scale_sha256=run['scale_sha256'],rotation_sha256=run['rotation_sha256'])
        old,new,head=scale.CenterScale().eval(),scale.CenterScale().eval(),rotation.RotationHead().eval()
        old.load_state_dict(torch.load(sp,map_location='cpu',weights_only=True)['model'],strict=True)
        head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True)
        new.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=True),strict=True)
        models[tag]=(old,new,head)
    robust.write(args.out/'registration.json',dict(scope='posthoc development diagnostic, previously inspected woodblock sequences',
          candidate_selected_on_synthetic_only=True,protocol_sha256=robust.sha(prepared),weights=pins,
          synthetic_completion_sha256=robust.sha(args.study/'robustness/completion.json'),samples_per_sequence=64,
          sample_policy='same evenly spaced ordinal positions as 20261107 condition diagnostic',source_sha256=robust.sha(__file__)))
    output=[]
    for seq in metadata['sequences']:
        path=Path(seq['sample_manifest'])
        assert robust.sha(path)==seq['sample_manifest_sha256']
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        chosen=[rows[i] for i in np.linspace(0,len(rows)-1,64,dtype=int)]
        K=[np.asarray(r['K']) for r in chosen]
        lut=native.rectification_lut(K[0],np.asarray(metadata['distortion_k1_k2_p1_p2_k3']))
        with h5py.File(seq['raw_h5'],'r') as source:
            frames=robust.tensors([dict(image=native.frame_from_raw(source[seq['raw_dataset']],r,lut)[0]) for r in chosen],torch.device('cpu'))
        samples=[dict(rid=seq['sequence'],id=r['index'],K=intrinsics,T=np.asarray(r['target_to_camera']),keys=keys,
                      uv=np.asarray(r['gt_uv_ideal'])/[346,260]) for r,intrinsics in zip(chosen,K,strict=True)]
        target_size=robust.labels(scale,samples,vertices,torch.device('cpu'))[:,2].numpy()
        for tag,(old,new,head) in models.items():
            center,z,size=scale.estimate(old,frames,K)
            orientations=robust.frozen_rotation(rotation,head,frames,center,size,K)
            control=robust.score(rotation,samples,center,z,size,target_size,orientations,vertices,tree,threshold)
            center,z,size=scale.estimate(new,frames,K)
            result=robust.score(rotation,samples,center,z,size,target_size,orientations,vertices,tree,threshold)
            output.append(dict(tag=tag,sequence=seq['sequence'],group=seq['group'],original=control,candidate=result))
        print('FROZEN_LIGHT_DIAGNOSTIC',seq['sequence'],len(chosen),flush=True)
    robust.write(args.out/'completion.json',dict(status='complete',results=output,diagnostic_only=True,real_fit=False))


if __name__=='__main__':main()
