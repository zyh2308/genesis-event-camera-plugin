"""Local static source bundle; no SSH, GPU, experiment update or deployment."""
import ast,base64,gzip,hashlib,json,zipfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    here=Path(__file__).resolve().parent
    inherited=here.parent/'pose_feature_retention_20261108/TRAINING_REGISTRATION.json'
    old=json.loads(inherited.read_text())
    registration=dict(schema='pose_slow_features_registration_v1',status='prepared_only_hold_until_input_contract_probe_resolved',
        training_only=True,real_data_used=False,methods=['eventgenesis','v2e'],seeds=[0,1,2],
        canonical_candidate='slow_features',legacy_evaluator_slot='paired_consistency',
        physical_C1_loss_unchanged=True,normal_teacher_weight=2,half_teacher_weight=0,feature_KD_weight=0.,
        feature_learning_rate=1.5e-5,head_learning_rate=1.5e-4,weight_decay=.01,features_frozen=False,
        batch=32,epochs=6,updates_per_branch=288,new_study_branches=12,reused_immutable_control_branches=24,
        auxiliary_half_probability=.75,original_thirty_checks=30,original_confirmation_checks=8,
        fixed_C1_training_records=old['fixed_C1_training_records'],source_initializers=old['source_initializers'],
        prior_study_completion_sha256=old['prior_study_completion_sha256'],prior_execution_receipt_sha256=old['prior_execution_receipt_sha256'],
        source_training_registration_sha256=sha(inherited),new_calibration_required=False,
        actual_server_smoke_performed=False,actual_study_performed=False,automatic_deployment_or_real_scoring=False)
    target=here/'TRAINING_REGISTRATION.json'
    if target.exists():raise FileExistsError('Refusing existing frozen registration')
    target.write_text(json.dumps(registration,indent=2)+'\n')
    names=['slow_feature_optimizer.py','pose_slow_features_policy.py','reuse_controls.py','run_pose_slow_features.py','smoke_contract.py',
           'pose_priority_policy.py','test_pose_priority_policy.py','test_slow_features.py','derive_policy.py','derive_worker.py',
           'README.md','PROTOCOL.md','DERIVATION.md','TRAINING_REGISTRATION.json','CPU_TEST_RESULT.json','build_bundle.py']
    for name in names:
        if name.endswith('.py'):ast.parse((here/name).read_text())
    manifest={n:sha(here/n) for n in names}
    (here/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(here/'pose_slow_features_source_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in names+['source_manifest.json']:z.write(here/name,name)
    files={n:dict(base64=base64.b64encode((here/n).read_bytes()).decode(),size_bytes=(here/n).stat().st_size,sha256=sha(here/n)) for n in names+['source_manifest.json']}
    raw=json.dumps(dict(schema='pose_slow_features_rpc_bundle_v1',files=files),separators=(',',':')).encode()
    (here/'upload.b64').write_text(base64.b64encode(gzip.compress(raw,mtime=0)).decode())
    print(json.dumps(dict(files=len(files),source_manifest_sha256=sha(here/'source_manifest.json'),decoded_bundle_sha256=hashlib.sha256(raw).hexdigest(),
                         upload_chars=(here/'upload.b64').stat().st_size,prepared_only=True),indent=2))

if __name__=='__main__':main()
