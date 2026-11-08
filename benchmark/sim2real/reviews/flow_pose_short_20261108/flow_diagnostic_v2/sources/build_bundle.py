"""Freeze a new diagnostic package. Local only; never SSH or run CUDA."""
import ast,base64,gzip,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=Path(__file__).resolve().parent;old=p.parent/'flow_priority_20261108';f=json.loads((old/'diagnostic_contract.json').read_text())
    needsha='ff401317e49ce37ae0f8e9bfd6b82f133549b4a0271be35095f74bdc50a439c0';assert sha(p/'pinned_multiscaleloss.py')==needsha
    pins=json.loads((p.parent/'mvsec_final_preparation_20261108/source_snapshot/manifests/downstream_source_bundle.json').read_text())
    pin=next(r for r in pins['repositories'] if r['repo']=='STE-FlowNet')
    c=dict(schema='flow_photometric_alignment_v2',root='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-photometric-alignment-v2',checkpoint_source='complete8910_last',seed=0,methods=['eventgenesis','v2e'],optimizer_updates=0,real_data_used=False,real_scores_used=False,automatic_training=False,validation_indices=f['validation_indices'],training_sample_indices=f['training_sample_indices'],augmentation_seed=f['augmentation_seed'],finite_difference_epsilons=[.01,.05],texture_threshold=.01,deadline_seconds=900,GPU_UUID=f['GPU_UUID'],protocol_sha256=sha(p/'PROTOCOL.md'),frozen_inputs=f,model_source_pin=pin,
        sources=dict(trainer=dict(path=f['training_source'],sha256=f['training_source_sha256']),reference=dict(path=f['campaign']+'/sources/benchmark/sim2real/scripts/run_steflow_stage2_pilot.py',sha256='167a88520162d06a7b1973eb4fcca64cf73680d50a8f583a60f953f27a0dd48a'),compat=dict(path=f['compat_source'],sha256=f['compat_sha256']),native_loss=dict(path=f['model_root']+'/multiscaleloss.py',sha256=needsha)),
        parameter_oracle=dict(batch=8,named_parameter_tensors=82,trainable_parameter_tensors=82,expected_native_photo_nonNone=70,allow_unused=True,losses=['native_photo_all4scales','finest_physical_valid_photo','finest_physical_invalid_photo','finest_valid_GT_EPE_diagnostic'],oracle_masks=['native_nearest','raw_bilinear_strict'],optimizer_updates=0,on_OOM='fail_new_root; preserve evidence; no automatic retry'),
        preserved_failure=dict(root='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-photometric-alignment-v1',source_manifest_sha256=sha(p.parent/'flow_photometric_alignment_20261108/source_manifest.json'),cause='v1 incorrectly asserted total named parameters70; actual82trainable,70involved in dt1 gradient'),
        scope='new diagnostic; original seed0 contract/source files preserved; GT never an optimizer or network input',provenance=dict(original_seed0_contract_sha256=sha(old/'diagnostic_contract.json')))
    (p/'contract.json').write_text(json.dumps(c,indent=2)+'\n')
    names=['run_photometric_alignment.py','photometric_math.py','flow_synthetic_diagnostic.py','pinned_compat.py','pinned_multiscaleloss.py','strict_support.py','PROTOCOL.md','contract.json','test_photometric_alignment.py','test_strict_support.py','build_bundle.py']
    for n in names:
        if n.endswith('.py'):ast.parse((p/n).read_text())
    manifest={n:sha(p/n) for n in names};(p/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={n:base64.b64encode((p/n).read_bytes()).decode() for n in names+['source_manifest.json']};text=base64.b64encode(gzip.compress(json.dumps(files).encode())).decode();(p/'upload.b64').write_text(text)
    restored=json.loads(gzip.decompress(base64.b64decode(text)))
    for n,h in manifest.items():assert hashlib.sha256(base64.b64decode(restored[n])).hexdigest()==h
    print(json.dumps(dict(source_manifest_sha256=sha(p/'source_manifest.json'),files=len(files),bundle_chars=len(text),optimizer_updates=0,real_data_used=False),indent=2))
if __name__=='__main__':main()
