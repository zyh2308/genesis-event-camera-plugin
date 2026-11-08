"""Freeze complete synthetic-only diagnostics; never SSH or execute CUDA."""
import ast,base64,gzip,hashlib,json,shutil
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=Path(__file__).resolve().parent
    base=json.loads((p.parent/'flow_priority_20261108/diagnostic_contract.json').read_text())
    payload=json.loads((p.parent/'mvsec_final_preparation_20261108/research_review_payload.json').read_text())
    probe=json.loads((p.parent/'mvsec_final_preparation_20261108/runtime_v2_export/runtime_v2/gpu2_probe/numerical_preflight.json').read_text())
    shutil.copyfile(p.parent/'flow_priority_20261108/flow_synthetic_diagnostic.py',p/'flow_synthetic_diagnostic.py')
    # UUID map supplied and checked by the resource parent; immutable in this bundle.
    devices=json.loads((p/'registered_devices.json').read_text())
    c=dict(schema='flow_full_synthetic_readonly_v1',root='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-full-synthetic-readonly-v1',campaign=base['campaign'],recipe_sha256=base['recipe_sha256'],frozen_simulator_core=base['frozen_simulator_core'],methods=list(payload['protocol']['methods']),seeds=[0,1,2],windows_per_checkpoint=528,optimizer_updates=0,real_data_used=False,real_scores_used=False,checkpoint_selection_changed=False,deadline_seconds=3600,minimum_ram_available_bytes=8*1024**3,model_root=base['model_root'],model_source=payload['model_source'],checkpoint_binding=payload['manifest']['checkpoint_binding'],sources=dict(trainer=dict(path=base['training_source'],sha256=base['training_source_sha256']),reference=probe['sources']['reference'],compat=dict(path=base['compat_source'],sha256=base['compat_sha256'])),shards=devices,protocol_sha256=sha(p/'PROTOCOL.md'),flow_unit='px_per_30ms',common_active='fixed intersection of EventGenesis and V2E, shared across all five methods',aggregation='window equal within rollout; then rollout equal; no pooled pixel weighting; missing support null')
    (p/'contract.json').write_text(json.dumps(c,indent=2)+'\n')
    c['binding_provenance']=dict(full15_binding_payload_sha256=sha(p.parent/'mvsec_final_preparation_20261108/research_review_payload.json'),original_seed0_diagnostic_contract_sha256=sha(p.parent/'flow_priority_20261108/diagnostic_contract.json'),scope='new independent full15 synthetic-only contract; original seed0 diagnostic scope is not modified or impersonated')
    (p/'contract.json').write_text(json.dumps(c,indent=2)+'\n')
    names=['evaluate_frozen_synthetic.py','flow_synthetic_diagnostic.py','contract.json','registered_devices.json','PROTOCOL.md','test_readonly_contract.py','build_bundle.py']
    for n in names:
        if n.endswith('.py'):ast.parse((p/n).read_text())
    manifest={n:sha(p/n) for n in names};(p/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={n:base64.b64encode((p/n).read_bytes()).decode() for n in names+['source_manifest.json']}
    text=base64.b64encode(gzip.compress(json.dumps(files).encode())).decode();(p/'upload.b64').write_text(text)
    restored=json.loads(gzip.decompress(base64.b64decode(text)))
    for n,h in manifest.items():assert hashlib.sha256(base64.b64decode(restored[n])).hexdigest()==h
    print(json.dumps(dict(files=len(files),bundle_chars=len(text),source_manifest_sha256=sha(p/'source_manifest.json'),no_training_or_real_scoring=True),indent=2))
if __name__=='__main__':main()
