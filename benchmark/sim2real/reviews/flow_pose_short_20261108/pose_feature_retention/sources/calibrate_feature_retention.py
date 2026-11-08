"""Zero-update coefficient calibration using TRAIN data and fixed C1 weights only."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from feature_retention import protected_feature_loss, feature_gradient_l2, common_coefficient
from pose_feature_retention_policy import (state_sha,draw_auxiliary,center_supervision,
                                          center_retention,depth_retention)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def one_calibration(scale,shared,initial_state,C1_state,x,y,a,b,half_mask,branch):
    if branch not in ('depth','center'): raise ValueError('Only existing two branches supported')
    teacher=scale.CenterScale().to(x.device).eval();teacher.load_state_dict(initial_state,strict=True);teacher.requires_grad_(False)
    model=scale.CenterScale().to(x.device).eval();model.load_state_dict(C1_state,strict=True)
    if branch=='center': model.size.requires_grad_(False)
    with torch.no_grad():
        reference=teacher(x)
        initial_feature_loss=protected_feature_loss(teacher,teacher,x,a,half_mask)
    assert float(initial_feature_loss)==0
    before=state_sha(model.state_dict());initial_before=state_sha(teacher.state_dict())
    predicted,auxiliary=model(x),model(a)
    if branch=='depth':
        supervised=.5*(shared.supervised(model,predicted,y)+shared.supervised(model,auxiliary,b))
        normal_retention=depth_retention(predicted,reference,include_center=True).mean()
    else:
        supervised=.5*(center_supervision(model,predicted,y)+center_supervision(model,auxiliary,b))
        normal_retention=center_retention(predicted,reference).mean()
    objective=supervised+2*normal_retention
    feature=protected_feature_loss(model,teacher,x,a,half_mask)
    g=feature_gradient_l2(objective,model); f=feature_gradient_l2(feature,model)
    assert torch.isfinite(objective) and torch.isfinite(feature) and feature>0
    assert state_sha(model.state_dict())==before and state_sha(teacher.state_dict())==initial_before
    assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in teacher.parameters())
    return dict(C1_objective_feature_gradient_l2=g,feature_retention_gradient_l2=f,
                C1_objective=float(objective.detach()),feature_retention=float(feature.detach()),
                source_initial_feature_retention=float(initial_feature_loss),
                source_and_C1_weights_unchanged=True,optimizer_updates=0)


def calibrate(native,args,own,sha):
    """No validation loader, validation metric, real dataset or optimizer is invoked."""
    R=args.campaign; A=R/'recovery_20261105_mainline'
    registration_path=own/'TRAINING_REGISTRATION.json'
    registration=json.loads(registration_path.read_text())
    assert registration['training_only'] and not registration['real_data_used']
    expected={(r['training_record']['method'],r['training_record']['seed'],r['training_record']['branch']):r
              for r in registration['fixed_C1_training_records']}
    assert len(expected)==12
    initializers={Path(r['path']).name:r for r in registration['source_initializers']}
    prep=json.loads((args.radiance_train/'completion.json').read_text())
    assert prep['status']=='complete' and not prep['real_data_used'] and len(prep['selected'])==96
    selected=prep['selected'];assert all(s.startswith('train_') for s in selected)
    shared=native.shared
    dependency_paths=dict(scale=A/'pose_scale_pilot_20261106.py',pilot=A/'pose_depth_pilot_20261106.py',
        fresh=A/'pose_fresh_confirm_20261106_v3.py',rotation=A/'pose_rotation_pilot_20261106.py',
        causal=R.parents[2]/'runs/sim2real/20261106-pose-synthetic-diagnostic-v1/causal_tests_v2.py',
        encoder=R/'sources/benchmark/sim2real/scripts/run_pose_stage2_pilot.py')
    dependency_sources={name:dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p)) for name,p in dependency_paths.items()}
    scale=shared.load('feature_scale',A/'pose_scale_pilot_20261106.py')
    pilot=shared.load('feature_pilot',A/'pose_depth_pilot_20261106.py')
    fresh=shared.load('feature_fresh',A/'pose_fresh_confirm_20261106_v3.py')
    loader=pilot.helpers(R.parents[2]/'runs/sim2real/20261106-pose-synthetic-diagnostic-v1/causal_tests_v2.py',R)['load']
    sys.path.insert(0,str(R/'sources/benchmark/sim2real/scripts'))
    from run_pose_stage2_pilot import encode_event_window
    assets=json.loads((A/'pose_prepared_full_v5/protocol.json').read_text())['assets']
    assert sha(assets['vertices']['path'])==assets['vertices']['sha256']
    vertices=np.load(assets['vertices']['path'],allow_pickle=False)
    rows_out=[];source_records=[];pairing={}
    for method in shared.METHODS:
        rows=loader(method,'train');assert len(rows)==1529
        normal,target=shared.tensors(rows,'cuda'),shared.labels(scale,rows,vertices,'cuda')
        base_files=[R/'pose_bank/physical'/(s+'.h5') for s in selected]
        base=fresh.adapter(R/'pose_bank',base_files,method,encode_event_window)
        relit=[];controls=[]
        for condition in ('half','eighth'):
            bank=args.radiance_train/condition/'pose_bank'
            current=fresh.adapter(bank,[bank/'physical'/(s+'.h5') for s in selected],method,encode_event_window)
            assert native.identity(current)==native.identity(base)
            relit+=current;controls+=base
        clean_tensor=shared.tensors(controls,'cuda');auxiliary=shared.tensors(relit,'cuda')
        aux_target=shared.labels(scale,relit,vertices,'cuda')
        half_count=len(auxiliary)//2;assert torch.equal(aux_target[:half_count],aux_target[half_count:])
        for seed in range(3):
            rng=np.random.default_rng(71070+seed);indices=rng.permutation(len(normal))[:32]
            ai,half=draw_auxiliary(rng,half_count,len(indices))
            idx=torch.as_tensor(indices,device='cuda');aid=torch.as_tensor(ai,device='cuda')
            x,y=shared.affine(normal[idx],target[idx],rng)
            a,_,b=native.matched_affine(auxiliary[aid],clean_tensor[aid],aux_target[aid],rng)
            mask=torch.as_tensor(half,device='cuda')
            label_hash=hashlib.sha256(y.cpu().numpy().tobytes()+b.cpu().numpy().tobytes()).hexdigest()
            if seed in pairing: assert pairing[seed]==label_hash
            pairing[seed]=label_hash
            initial=initializers[f'{method}_seed{seed}_clean_control.pt']
            assert sha(initial['path'])==initial['sha256']
            initial_state=torch.load(initial['path'],map_location='cpu',weights_only=True)
            assert state_sha(initial_state)==initial['state_sha256']
            for branch in ('depth','center'):
                item=expected[(method,seed,branch)];rec=item['training_record']
                receipt_path=Path(item['receipt']['path'])
                assert sha(receipt_path)==item['receipt']['sha256']
                assert json.loads(receipt_path.read_text())==rec
                checkpoint=Path(rec['checkpoint_path']);assert sha(checkpoint)==rec['checkpoint_sha256']
                value=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert value['update']==rec['updates']==288
                assert state_sha(value['model'])==rec['final_state_sha256']
                assert state_sha(initial_state)==rec['source_state_sha256']
                log_path=checkpoint.with_name(checkpoint.name.replace('_training_state.pt','.jsonl'))
                with log_path.open() as stream: first=json.loads(next(stream))
                assert first['update']==1 and first['normal_indices']==indices.tolist()
                assert first['auxiliary_indices']==ai.tolist() and first['half_mask']==half.tolist()
                result=one_calibration(scale,shared,initial_state,value['model'],x,y,a,b,mask,branch)
                result.update(method=method,seed=seed,branch=branch,labels_sha256=label_hash,
                              normal_indices=indices.tolist(),auxiliary_indices=ai.tolist(),half_mask=half.tolist(),
                              normal_input_sha256=tensor_sha(x),auxiliary_input_sha256=tensor_sha(a),
                              source_state_sha256=state_sha(initial_state),C1_final_state_sha256=rec['final_state_sha256'],
                              C1_checkpoint_sha256=rec['checkpoint_sha256'])
                rows_out.append(result)
                source_records.append(dict(checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                                           receipt_path=str(receipt_path),receipt_sha256=sha(receipt_path),
                                           log_path=str(log_path),log_sha256=sha(log_path)))
                print('FEATURE_CALIBRATION',method,seed,branch,result['C1_objective_feature_gradient_l2'],result['feature_retention_gradient_l2'],flush=True)
            del initial_state
        del normal,target,auxiliary,aux_target,clean_tensor
    coefficient,ratios=common_coefficient(rows_out)
    for row,ratio in zip(rows_out,ratios): row['weighted_feature_to_C1_gradient_ratio']=ratio
    for item in source_records:
        for kind in ('checkpoint','receipt','log'): assert sha(item[kind+'_path'])==item[kind+'_sha256']
    assert all(sha(item['path'])==item['sha256'] for item in initializers.values())
    assert all(sha(item['path'])==item['sha256'] for item in dependency_sources.values())
    args.out.mkdir(parents=False,exist_ok=False)
    shared.write(args.out/'calibration.json',dict(schema='pose_feature_retention_calibration_v1',status='passed',
        source_manifest_sha256=sha(own/'source_manifest.json'),training_registration_sha256=sha(registration_path),
        frozen_simulator_core='62da4f35ce3c5c96f04268bd1687a27657af9238',
        coefficient=coefficient,target_median_gradient_ratio=.1,calibration_location='fixed 288-update C1, never zero-KD initialization',
        feature_term='0.5*(mean normal relative feature MSE + mean half-masked auxiliary relative feature MSE)',
        normalization='per-sample detached teacher mean feature-square, clamped minimum 1e-6',
        records=rows_out,source_records=source_records,dependency_sources=dependency_sources,optimizer_updates=0,real_data_used=False,
        validation_data_used=False,validation_metrics_used=False,initializers_unchanged=True,
        train_only=True,methods=['eventgenesis','v2e'],seeds=[0,1,2],branches=['depth','center']))
