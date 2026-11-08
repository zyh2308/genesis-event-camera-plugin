"""Freeze local preparation only; never connects, imports CUDA or starts a study."""
import ast,base64,gzip,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root=Path(__file__).resolve().parent;old=root.parents[1]/'flow_paired_short_20261108'
    f=json.loads((old/'contract.json').read_text())['frozen_inputs']
    contract=dict(schema='flow_census_paired_short_v1',root='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-census-paired-short-v1',
        methods=['eventgenesis','v2e'],seed=0,arms=['native_control','census'],updates_per_arm=200,effective_batch=8,warm_update=8910,
        smoke_updates_total=4,main_updates_total=800,real_data_used=False,real_scores_used=False,GT_flow_optimizer_supervision=False,
        sampling='original_control_schedule_only',original_smoothness_coefficient=10.,patch_size=7,
        calibration=dict(batches_per_method=8,gradient_space='all_requires_grad_parameters_zero_unused',native_non_none_expected=70,
                         formula='1/median(norm_census/norm_native_photo)',train_only=True,GT_in_lambda=False),
        source_input_descriptor_scope='frozen_inputs describes old read-only source bindings, not the scope of this new declared short optimizer study',
        quality_gate=dict(primary_all_valid_improves=True,common_active_improves=True,maximum_relative_stratum_epe_increase=.05,both_generators_required=True),
        downstream_source_bundle_sha256='828bccdaf0b046b56ee2b8c8230ebc3651b8ccfc16652ae7ea9a3f739dbabdb9',
        original_pilot_sha256='167a88520162d06a7b1973eb4fcca64cf73680d50a8f583a60f953f27a0dd48a',
        original_helper_sha256='5340990638eecf0865d2a322e45cb903f990a9d4077cf047f7cb40b5de3f4319',
        original_soft_census_v1_sha256='7ceae6b8774322a12fac69ee8c2defff2368233eae024e0106c1e4e276f471e7',
        raw_endpoint_domain='strict finite [0,1] in NPZ',known_interpolated_endpoint_and_warp_domain='8*dtypeeps check only, no clamp/no value or gradient changes',
        warm_before_reuse=dict(eventgenesis=None,v2e=None),phase0_deadline_seconds=1800,study_deadline_seconds=7200,
        automatic_long_training_or_real_scoring=False,GT_direction_oracle=None,
        frozen_inputs=f,protocol_sha256=sha(root/'PROTOCOL.md'),
        references=dict(UFlow_robust='https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py#L573-L575',
                        UFlow_census='https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py#L1089-L1144'),
        preparation_status='CPU fixtures only; no actual server CPU admission,phase0,GPU smoke,study,lambda or validation result yet')
    target=root/'contract.json'
    if target.exists():raise FileExistsError('Refusing to replace frozen candidate contract')
    target.write_text(json.dumps(contract,indent=2)+'\n')
    names=['flow_sampling_short.py','flow_synthetic_diagnostic.py','sampling.py','soft_census.py','soft_census_v2.py','census_contract.py',
           'run_census_paired_short.py','test_census_contract.py','build_bundle.py','contract.json','PROTOCOL.md','README.md','cpu_preflight.json']
    for n in names:
        if n.endswith('.py'):ast.parse((root/n).read_text())
    manifest={n:sha(root/n) for n in names};(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={n:dict(base64=base64.b64encode((root/n).read_bytes()).decode(),size_bytes=(root/n).stat().st_size,sha256=sha(root/n)) for n in names+['source_manifest.json']}
    raw=json.dumps(dict(schema='flow_census_paired_short_rpc_bundle_v1',files=files),separators=(',',':')).encode()
    (root/'upload.b64').write_text(base64.b64encode(gzip.compress(raw,mtime=0)).decode())
    print(json.dumps(dict(source_manifest_sha256=sha(root/'source_manifest.json'),decoded_bundle_sha256=hashlib.sha256(raw).hexdigest(),
        files=len(files),bundle_chars=(root/'upload.b64').stat().st_size,no_training_or_deployment_started=True),indent=2))

if __name__=='__main__':main()
