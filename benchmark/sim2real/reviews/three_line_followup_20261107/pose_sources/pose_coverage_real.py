"""Gate-bound E-POSE development evaluation of frozen paired model bundles."""
import argparse,csv,hashlib,importlib.util,json,sys
from pathlib import Path
import cv2,h5py,numpy as np,torch
from scipy.spatial import cKDTree

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def gate_rows(results):
    checks=[]
    for r in results:
        a,b=r['arms']['clean_control'],r['arms']['paired_consistency']
        checks.append(dict(method=r['method'],seed=r['seed'],condition='clean',passed=b['clean']['summary']['success']-a['clean']['summary']['success']>=-.02))
        for c in ['half','eighth']:
            checks.append(dict(method=r['method'],seed=r['seed'],condition=c,passed=b[c]['summary']['depth_mae_m']<a[c]['summary']['depth_mae_m']))
            checks.append(dict(method=r['method'],seed=r['seed'],condition=c+'_pose',passed=b[c]['summary']['success']>=a[c]['summary']['success']))
    return checks

def original_confirmation(results,campaign):
    fresh=load('original_fresh_confirmation',campaign/'recovery_20261105_mainline/pose_fresh_confirm_20261106_v3.py')
    converted=[]
    for r in results:
        values=r['arms']['paired_consistency']['clean_full']['by_rollout'];parent=r['parents']['by_rollout']
        assert len(values)==96 and set(values)==set(parent)
        by={rid:dict(original=parent[rid]['original'],median=parent[rid]['median'],
            learned=dict(success=v['success'],depth_mae=v['depth_mae_m'])) for rid,v in sorted(values.items())}
        converted.append(dict(method=r['method'],seed=r['seed'],by_rollout=by))
    result=fresh.gate(converted)
    assert len(result['comparisons'])==8
    result['source_sha256']=sha(campaign/'recovery_20261105_mainline/pose_fresh_confirm_20261106_v3.py')
    result['sample_contract']='all original 16 candidate windows per trajectory, after the unchanged adapter visibility rule; not the four-window light-pilot subset'
    return result

@torch.inference_mode()
def predict_bundle(scale,shared,rotation,localizer,depths,head,images,intrinsics):
    # Whole event frames and public K only; no GT/sample-manifest argument.
    result={}
    for arm,depth_model in depths.items():
        centers=scale.estimate(localizer[arm],images,intrinsics)[0]
        _,z,boxes=scale.estimate(depth_model,images,intrinsics)
        orientations=shared.frozen_rotation(rotation,head,images,centers,boxes,intrinsics)
        result[arm]=dict(centers=centers,depth=z,rotation=orientations)
    return result


def measures(row,predicted,keys,vertices,tree,diameter,rotation):
    uv,z,R=predicted;T=np.asarray(row['target_to_camera']);K=np.asarray(row['K'])
    assert np.isfinite(uv).all() and np.isfinite(z) and z>0 and np.isfinite(R).all()
    ray=rotation.ray_frame(np.asarray([uv]),[K])[0][0];t=ray*z-R@keys[0]
    truth=vertices@T[:3,:3].T+T[:3,3];valid=bool(((keys@R.T+t)[:,2]>0).all())
    adds=float(tree.query((truth-t)@R,workers=1)[0].mean()) if valid else None
    gt_z=float(T[2,:3]@keys[0]+T[2,3]);target_uv=np.asarray(row['gt_uv_ideal'])[0]/[346,260]
    return dict(valid=valid,adds_m=adds,success=float(valid and adds<.1*diameter),
        auc=float(max(1-adds/.1,0)) if valid else 0.,depth_mae_m=abs(float(z)-gt_z),
        center_px=float(np.linalg.norm((uv-target_uv)*[346,260])),translation_error_m=float(np.linalg.norm(t-T[:3,3])),
        rotation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(R.T@T[:3,:3])-1)/2,-1,1)))),
        prediction=dict(center_normalized=uv.tolist(),depth_m=float(z),R=R.tolist(),t_m=t.tolist()))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['campaign','study','out']:p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();R=args.campaign;A=R/'recovery_20261105_mainline';study=args.study
    done=json.loads((study/'physical_adaptation/completion.json').read_text())
    assert done['status']=='complete' and not done['real_data_used']
    training_protocol=json.loads((study/'physical_adaptation/protocol.json').read_text())
    assert not training_protocol['smoke'] and training_protocol['seeds']==[0,1,2] and training_protocol['epochs']==6 and training_protocol['localizer_epochs']==6 and training_protocol['total_optimizer_updates_per_fit']==576
    checks=gate_rows(done['results']);assert len(checks)==30 and checks==done['gate']['checks'] and all(c['passed'] for c in checks)
    original_gate=original_confirmation(done['results'],R)
    assert original_gate==done['original_confirmation'] and original_gate['passed']
    assert {(r['method'],r['seed']) for r in done['results']}=={(m,s) for m in ['eventgenesis','v2e'] for s in range(3)}
    assert json.loads((study/'overlap_audit.json').read_text())['status']=='passed'
    for name,digest in json.loads((study/'sources/source_manifest.json').read_text()).items():assert sha(study/'sources'/name)==digest
    assert torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    shared=load('coverage_shared',study/'sources/pose_light_robustness.py')
    scale=load('coverage_scale',A/'pose_scale_pilot_20261106.py');rotation=load('coverage_rotation',A/'pose_rotation_pilot_20261106.py')
    native=load('coverage_native',A/'stage9_epose_v5.py')
    protocol_path=A/'pose_prepared_full_v5/protocol.json';protocol=json.loads(protocol_path.read_text());assets=protocol['assets']
    assert protocol['success_operator']=='lt' and len(protocol['sequences'])==18
    assert sum(s['sample_count'] for s in protocol['sequences'])==20916
    for k in ['vertices','keypoints']:assert sha(assets[k]['path'])==assets[k]['sha256']
    for seq in protocol['sequences']:
        assert sha(seq['sample_manifest'])==seq['sample_manifest_sha256'] and sha(seq['raw_h5'])==seq['raw_h5_sha256']
    vertices=np.load(assets['vertices']['path'],allow_pickle=False);keys=np.load(assets['keypoints']['path'],allow_pickle=False)
    tree=cKDTree(vertices);args.out.mkdir(parents=True,exist_ok=False)
    models={};bindings={}
    def center_model(path,nested=False):
        state=torch.load(path,map_location='cpu',weights_only=True);state=state['model'] if nested else state
        model=scale.CenterScale().cuda().eval();model.load_state_dict(state,strict=True);model.requires_grad_(False);return model
    for r in done['results']:
        tag=f"{r['method']}_seed{r['seed']}";sp=A/f'jobs/115_pose_scale_paired_20261106/results/{tag}_scale.pt'
        rp=A/f'jobs/117_pose_rotation_paired_20261106/results/{tag}_rotation.pt';lp=Path(r['localization_path'])
        assert sha(lp)==r['localization_sha256'] and sha(sp)==r['parents']['scale_sha256'] and sha(rp)==r['parents']['rotation_sha256']
        head=rotation.RotationHead().cuda().eval();head.load_state_dict(torch.load(rp,map_location='cpu',weights_only=True)['model'],strict=True);head.requires_grad_(False)
        paths={arm:study/'physical_adaptation'/f'{tag}_{arm}.pt' for arm in ['clean_control','paired_consistency']}
        localpaths={arm:Path(r['localization_paths'][arm]) for arm in paths}
        for arm,path in localpaths.items():assert sha(path)==r['localization_sha256s'][arm]
        models[tag]=({arm:center_model(path) for arm,path in localpaths.items()},{arm:center_model(path) for arm,path in paths.items()},head)
        bindings[tag]={name:dict(path=str(path),sha256=sha(path)) for name,path in dict(rotation=rp,reference_scale=sp,localizer_parent=lp,**{arm+'_localizer':path for arm,path in localpaths.items()},**paths).items()}
    write(args.out/'registration.json',dict(protocol_sha256=sha(protocol_path),synthetic_completion_sha256=sha(study/'physical_adaptation/completion.json'),
        source_manifest_sha256=sha(study/'sources/source_manifest.json'),bundles=bindings,checks=checks,original_confirmation=original_gate,
        primary='paired_consistency',control='clean_control',checkpoint_selection='fixed last synthetic checkpoints',
        scope='Previously inspected woodblock E-POSE; development diagnostic, not independent blind final test',
        real_data_used_in_training=False,GT_inference_inputs=False))
    streams={(tag,arm):(args.out/(tag+'_'+arm+'.jsonl')).open('x') for tag in models for arm in ['clean_control','paired_consistency']}
    summaries=[];metric_keys=['success','auc','depth_mae_m','center_px','translation_error_m','rotation_error_deg']
    try:
        for seq in protocol['sequences']:
            name=seq['sequence'];rows=[json.loads(l) for l in Path(seq['sample_manifest']).read_text().splitlines()]
            assert len(rows)==seq['sample_count']
            lut=native.rectification_lut(np.asarray(rows[0]['K']),np.asarray(protocol['distortion_k1_k2_p1_p2_k3']))
            sums={key:np.zeros(len(metric_keys)) for key in streams};valid={key:0 for key in streams}
            with h5py.File(seq['raw_h5'],'r') as f:
                data=f[seq['raw_dataset']]
                for i in range(0,len(rows),32):
                    block=rows[i:i+32]
                    images=shared.tensors([dict(image=native.frame_from_raw(data,row,lut)[0]) for row in block],torch.device('cuda'))
                    K=[np.asarray(row['K']) for row in block]
                    for tag,(localizer,depths,head) in models.items():
                        predictions=predict_bundle(scale,shared,rotation,localizer,depths,head,images,K)
                        for arm,pred in predictions.items():
                            for row,uv,z,orient in zip(block,pred['centers'],pred['depth'],pred['rotation'],strict=True):
                                result=measures(row,(uv,z,orient),keys,vertices,tree,assets['diameter_m'],rotation)
                                sums[tag,arm]+=np.asarray([result[k] for k in metric_keys]);valid[tag,arm]+=int(result['valid'])
                                streams[tag,arm].write(json.dumps(dict(sequence=name,group=seq['group'],index=row['index'],**result),allow_nan=False)+'\n')
                    print('POSE_COVERAGE_REAL_PROGRESS',name,i+len(block),len(rows),flush=True)
            for (tag,arm),total in sums.items():
                summaries.append(dict(tag=tag,arm=arm,sequence=name,group=seq['group'],samples=len(rows),valid=valid[tag,arm],
                    metrics=dict(zip(metric_keys,(total/len(rows)).tolist(),strict=True))))
            write(args.out/(name.replace('/','__')+'_summary.json'),summaries[-len(streams):])
        table=[]
        for row in summaries:
            method,seed=row['tag'].rsplit('_seed',1)
            table.append(dict(method=method,seed=int(seed),arm=row['arm'],sequence=row['sequence'],group=row['group'],
                samples=row['samples'],valid_fraction=row['valid']/row['samples'],**row['metrics']))
        with (args.out/'per_sequence_metrics.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
        write(args.out/'completion.json',dict(status='complete',real_data_used=True,real_data_used_in_training=False,
            real_scores_computed=True,results=summaries,scope='development diagnostic only; no real checkpoint/model selection'))
    finally:
        for f in streams.values():f.close()

if __name__=='__main__':main()
