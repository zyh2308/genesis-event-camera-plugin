#!/usr/bin/env python3
"""Independent arithmetic/receipt audit only; no model, GPU, SSH or experiment code."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np

def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def record(path):return dict(path=str(path),size_bytes=path.stat().st_size,sha256=sha(path))
def hash_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.review_root.resolve(strict=True);e=root/'evidence/round2/census';v=root/'flow_census/sources'
    paths={n:e/r for n,r in dict(phase0='phase0/completion.json',lock='phase0/calibration_lock.json',replay='phase0/control_replay.json',
        cpu='cpu/completion.json',smoke='smoke/completion.json',eg_signature='phase0/eventgenesis_parameter_signature.json',
        v2e_signature='phase0/v2e_parameter_signature.json').items()}
    d={n:read(path) for n,path in paths.items()};phase=d['phase0'];cpu=d['cpu'];smoke=d['smoke'];lock=d['lock'];replay=d['replay']
    manifest='45343592cc7eff1c86f64bafe0d91362f96371f84071daf19ae32cff9060ff04'
    need(sha(v/'source_manifest.json')==manifest,'Archived source manifest differs')
    for name,digest in read(v/'source_manifest.json').items():
        need(Path(name).name==name and sha(v/name)==digest,'Archived source bytes changed: '+name)
    frozen=read(v/'contract.json')['frozen_inputs']
    for obj in (phase,cpu,smoke):need(obj['source_manifest_sha256']==manifest,'Execution uses different source')
    need(cpu['status']=='passed' and cpu['optimizer_updates']==0 and cpu['real_data_used'] is False and cpu['CUDA_initialized'] is False,'CPU admission scope differs')
    need(phase['status']=='calibrated_train_only' and phase['optimizer_updates']==0 and phase['old_checkpoints_unchanged'] is True,'Phase0 not immutable zero-step')
    need(all(phase[k] is False for k in ('GT_in_lambda','validation_in_lambda','real_in_lambda','real_data_used','real_scores_used','GT_flow_optimizer_supervision')),'Calibration isolation flags differ')
    need(phase['GT_direction_oracle'] is None,'No GT oracle expected')
    need(lock['status']=='frozen_before_optimizer' and lock['optimizer_updates']==0 and
        lock['completion']['sha256']==sha(paths['phase0']) and lock['completion']['size_bytes']==paths['phase0'].stat().st_size,
        'Calibration lock does not bind raw phase0 bytes')
    rows=phase['records'];methods=('eventgenesis','v2e')
    need(len(rows)==16 and {(r['method'],r['batch']) for r in rows}=={(m,i) for m in methods for i in range(8)},'Wrong16-batch inventory')
    inventory=cpu['parameter_inventory'];signature=hash_json(inventory);names={r['name'] for r in inventory}
    need(len(inventory)==len(names)==cpu['total_named_parameters']==82 and
         sum(r['requires_grad'] for r in inventory)==cpu['total_requires_grad_parameters']==82,'Actual CPU inventory is not82')
    need(signature==cpu['parameter_signature_sha256'],'CPU raw parameter signature SHA differs')
    for n in ('eg_signature','v2e_signature'):
        sig=d[n];need(sig['parameters']==inventory and sig['sha256']==signature and sig['total_requires_grad_parameters']==82,'GPU/CPU raw signature differs')
    ratios=[];details=[];unused=None;states={};labels={}
    for r in rows:
        need(r['split']=='train' and r['optimizer_updates']==0 and r['GT_in_lambda'] is False,'Non-TRAIN calibration row')
        need(r['gradient_space']=='all_requires_grad_parameters_zero_unused' and r['total_requires_grad_parameters']==82 and
             r['native_finite_gradient_tensors']==r['census_finite_gradient_tensors']==70 and r['parameter_signature_sha256']==signature,'82/70 parameter convention differs')
        need(r['native_unused_parameter_names']==r['census_unused_parameter_names'],'Native and Census unused names differ')
        nn=r['native_unused_parameter_names']
        need(len(nn)==len(set(nn))==12 and set(nn)<=names,'Wrong actual unused inventory')
        if unused is None:unused=nn
        need(nn==unused,'Unused parameter inventory changed between batches/methods')
        m=r['method'];b=frozen['methods_binding'][m];source=r['source_checkpoint']
        need(source['path']==b['last_checkpoint'] and source['sha256']==b['last_checkpoint_sha256'] and
             source['model_state_sha256']==r['model_state_sha256'],'Original model/CP binding differs')
        if m in states:need(states[m]==r['model_state_sha256'],'Model changed during calibration')
        states[m]=r['model_state_sha256']
        if r['batch'] in labels:need(labels[r['batch']]==r['labels_and_augmentation_sha256'],'Unpaired label/augmentation hash')
        labels[r['batch']]=r['labels_and_augmentation_sha256']
        need(r['replay_sha256']==phase['replay_sha256'],'Different calibration replay')
        x,y=r['native_photo_gradient_l2'],r['census_photo_gradient_l2']
        need(all(math.isfinite(z) and z>0 for z in (x,y)),'Invalid gradient norm')
        ratio=y/x;need(ratio==r['census_to_native_ratio'],'Recorded ratio does not equal its two raw norms')
        ratios.append(ratio);details.append(dict(method=m,batch=r['batch'],native_norm=x,census_norm=y,recomputed_ratio=ratio,
            recorded_ratio=r['census_to_native_ratio'],ratio_absolute_difference=0.))
    median=statistics.median(ratios);weight=1/median
    need(median==phase['median_ratio'] and weight==phase['shared_census_weight']==lock['shared_census_weight']==0.00018879906228896475,
         'Independent common lambda arithmetic differs')
    need(replay['updates']==200 and replay['batch']==8 and replay['arms']['native_control']==replay['arms']['census'],'Arms do not share control schedule')
    sample_ids=replay['samples'];rids=sorted({r[0] for r in sample_ids})
    need(len(rids)==24 and sample_ids==[[rid,i] for rid in rids for i in range(66)],'Original24x66 sample grid differs')
    indices=[];aug=[]
    for epoch in (45,46):
        perm=np.random.default_rng(1000003*epoch).permutation(1584)
        for batch in range(198):
            if len(indices)>=1600:break
            indices.extend(perm[8*batch:8*batch+8].tolist())
            aug.extend([[0,epoch,batch,micro] for micro in range(8)])
    need(indices==replay['arms']['native_control'] and aug==replay['augmentation_seed_sequences'],'Not the original control permutation/affine seeds')
    replay_body=dict(replay);given=replay_body.pop('replay_sha256')
    need(hash_json(replay_body)==given==phase['replay_sha256'],'Replay content SHA differs')
    per_method={}
    for m in methods:
        rr=[r for r in details if r['method']==m]
        weighted=[weight*r['recomputed_ratio'] for r in rr]
        per_method[m]=dict(batch_count=8,raw_ratio_min=min(r['recomputed_ratio'] for r in rr),
            raw_ratio_median=statistics.median(r['recomputed_ratio'] for r in rr),raw_ratio_max=max(r['recomputed_ratio'] for r in rr),
            shared_weight_applied=weight,weighted_gradient_ratio_min=min(weighted),
            weighted_gradient_ratio_median=statistics.median(weighted),weighted_gradient_ratio_max=max(weighted))
    need(smoke['status']=='complete' and smoke['mode']=='smoke' and smoke['pairing_passed'] is True and
         len(smoke['results'])==4 and {(r['method'],r['arm']) for r in smoke['results']}=={(m,a) for m in methods for a in ('native_control','census')},
         'Four-arm actual smoke incomplete')
    need(all(smoke[k] is False for k in ('formal_weights_modified','GT_flow_optimizer_supervision','real_data_used','real_scores_used')),'Smoke isolation differs')
    need(smoke['shared_calibration']['sha256']==sha(paths['phase0']) and smoke['shared_census_weight']==weight,'Smoke uses different common coefficient')
    for r in smoke['results']:
        need(r['additional_updates']==1 and r['optimizer_update']==8911 and r['finite_gradient_tensors']==70 and
             r['independent_model_and_Adam_restore_passed'] is True and r['checkpoint']['model_rebuild_passed'] is True and
             r['checkpoint']['optimizer_state_equal'] is True and r['scheduler_and_RNG_saved'] is True and
             r['shared_calibration']['sha256']==sha(paths['phase0']) and r['shared_census_weight']==weight,'Smoke update/restore/weight binding differs')
    for k in ('label_sequence_sha256','sample_sequence_sha256','augmentation_sequence_sha256'):
        need(len({r[k] for r in smoke['results']})==1,'Smoke cross-method/arm pairing differs: '+k)
    for m in methods:
        rr=[r for r in smoke['results'] if r['method']==m]
        need(rr[0]['initial_rng_fingerprint']==rr[1]['initial_rng_fingerprint'],'Smoke arm initial RNG differs')
    need(cpu['model_source_verification']==phase['model_source_verification']==smoke['model_source_verification'],'Pinned source verification changed')
    result=dict(schema='independent_census_calibration_audit_v1',status='passed',inputs={n:record(path) for n,path in paths.items()},
        source_manifest_sha256=manifest,independent_of_experiment_helpers=True,source_read_only=True,
        recomputed_ratio_count=16,records=details,recomputed_median_ratio=median,recomputed_shared_lambda=weight,
        recorded_shared_lambda=phase['shared_census_weight'],lambda_absolute_difference=abs(weight-phase['shared_census_weight']),
        no_per_generator_coefficient_selection=True,per_method_weighted_gradient_magnitude_ratios=per_method,
        pooled_weighted_ratio_median=statistics.median([weight*r for r in ratios]),
        actual_CPU_and_GPU_parameter_inventory=dict(named=82,requires_grad=82,dt1_native_connected=70,dt1_census_connected=70,
            unused_count=12,unused_names=unused,parameter_signature_sha256=signature,source='actual archived CPU inventory and both GPU phase0 signature files'),
        pairing=dict(first8TRAIN_batches_each_method=True,label_augmentation_hashes_equal_for_all8=True,
            original_control_replay_reconstructed=True,phase0_replay_sha256=given,smoke_all4_label_sample_augmentation_hashes_equal=True,
            same_method_smoke_initial_RNG_equal=True),
        isolation=dict(phase0_optimizer_steps=0,GT_in_lambda=False,validation_in_lambda=False,real_in_lambda=False,GT_direction_oracle=None,
            old_checkpoints_unchanged_per_actual_receipt=True,source_tree_consistent=True),
        actual_smoke=dict(arms=4,updates_each=1,optimizer_update_after=8911,independent_model_and_Adam_restore=True),
        limits=['Recomputed arithmetic from actual archived norms, not re-executed gradients or model quality tests.',
            'Large checkpoints remain server-side: their recorded hashes/model-state bindings were cross-checked, not reread via SSH.',
            'Weighted ratios match pooled median magnitude, not every batch/generator, gradient directions or downstream accuracy.',
            'GT label hashes support pairing only; no GT or validation/real scores enter lambda. CPU validation-file encoding preflight is not lambda calibration.',
            'Unused12 names are the actual dt1 observation; their architectural purpose is not inferred here.'],
        new_experiments_started=False,SSH_connected=False,git_commands_executed=False)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():raise FileExistsError('Refusing existing audit JSON')
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='passed',out=str(a.out),lambda_value=weight,lambda_difference=0.,per_method=per_method,
        parameter_signature_sha256=signature,unused_parameters=12),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
