"""Freeze a new sampler-only trial bundle; no training, SSH or real scoring."""
import ast,base64,gzip,hashlib,json,shutil
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=Path(__file__).resolve().parent;old=p.parent/'flow_priority_20261108'
    shutil.copyfile(old/'flow_synthetic_diagnostic.py',p/'flow_synthetic_diagnostic.py')
    original=json.loads((old/'diagnostic_contract.json').read_text())
    diagnostic=json.loads((old/'compact_diagnostic.json').read_text())
    assert diagnostic['real_data_used'] is False and diagnostic['optimizer_updates']==0
    contract=dict(schema='flow_motion_sampler_short_v1',root='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-motion-sampling-paired-v1',
        methods=['eventgenesis','v2e'],seed=0,arms=['control','motion'],updates_per_arm=200,effective_batch=8,smoke_updates_total=4,main_updates_total=800,
        warm_source='complete45epoch last.pt; explicit diagnostic fork, not original formal resume',warm_update=8910,
        sampler_target_counts=[480,160,480,480],sampler_target_probabilities=[.3,.1,.3,.3],
        candidate_changes_only='replacement of excess low-motion-stratum slots; all other control slots and augmentation sequences unchanged',
        no_model_loss_noise_threshold_label_changes=True,GT_flow_optimizer_supervision=False,real_data_used=False,real_scores_used=False,
        deadline_seconds=7200,automatic_long_training_or_real_scoring=False,
        quality_gate=dict(primary_all_valid_improves=True,common_active_improves=True,maximum_relative_stratum_epe_increase=.05,both_generators_required=True),
        diagnostic_source=dict(path=str(old/'compact_diagnostic.json'),sha256=sha(old/'compact_diagnostic.json'),
                               full_bank_source_sha256=diagnostic['bank_source_sha256'],fixed_probe_source_sha256=diagnostic['probe_source_sha256']),
        protocol_sha256=sha(p/'PROTOCOL.md'),frozen_inputs=original)
    (p/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    names=['sampling.py','flow_sampling_short.py','flow_synthetic_diagnostic.py','contract.json','PROTOCOL.md','test_sampling_contract.py','build_bundle.py']
    for n in names:
        if n.endswith('.py'):ast.parse((p/n).read_text())
    manifest={n:sha(p/n) for n in names};(p/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={n:base64.b64encode((p/n).read_bytes()).decode() for n in names+['source_manifest.json']}
    b64=base64.b64encode(gzip.compress(json.dumps(files).encode())).decode();(p/'upload.b64').write_text(b64)
    decoded=json.loads(gzip.decompress(base64.b64decode(b64)))
    for n,digest in manifest.items():assert hashlib.sha256(base64.b64decode(decoded[n])).hexdigest()==digest
    print(json.dumps(dict(files=len(files),bundle_chars=len(b64),source_manifest_sha256=sha(p/'source_manifest.json'),no_training_started=True),indent=2))
if __name__=='__main__':main()
