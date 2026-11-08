"""Create only the GPU2 device variant; preserve verified GPU5 source bytes."""
import ast
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here=Path(__file__).resolve().parent
    worker5=here/'probe_synthetic_forward_equivalence.py'
    parent5=here/'run_forward_probe_gpu5.py'
    worker5_sha=sha(worker5);parent5_sha=sha(parent5)
    assert worker5_sha=='12a166df66f902e571d079ae2ace119a08276dda63d52568cdf7553affdaf7d7'
    quarter=here.parent/'recon_l1_quarter_gpu2_20261108/run_paired_gpu2.py'
    tree=ast.parse(quarter.read_text())
    uuid=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='UUID' for t in n.targets))
    # Historical same-UUID raw nvidia-smi receipt; runtime rechecks exactly.
    evidence=here.parent/'efficiency_comparison_results_20261105/evidence/preflight.json'
    gpu_lines=json.loads(evidence.read_text())['gpu'].splitlines()
    fields=next(line.split(',') for line in gpu_lines if uuid in line)
    assert fields[0].strip()=='2' and fields[2].strip()==uuid
    model=fields[1].strip();assert model=='Tesla V100S-PCIE-32GB'
    old_uuid="GPU-06ee62ba-c171-40d4-b4ae-ea63391dc885"
    original=worker5.read_text()
    worker=original.replace(old_uuid,uuid).replace('GPU5','GPU2')
    guard="        result.update(GPU_used=True,gpu_uuid=UUID,gpu_name=torch.cuda.get_device_name(0),tf32=False,eval_mode=True,inference_mode=True)"
    assert worker.count(guard)==1
    worker=worker.replace(guard,guard+"\n        require(torch.cuda.get_device_name(0)=="+repr(model)+",'GPU2 exact model differs')")
    ast.parse(worker)
    (here/'probe_synthetic_forward_equivalence_gpu2.py').write_text(worker)
    parent=parent5.read_text().replace(old_uuid,uuid).replace('GPU5','GPU2').replace('gpu5','gpu2')
    parent=parent.replace("worker=ROOT/'probe_synthetic_forward_equivalence.py'", "worker=ROOT/'probe_synthetic_forward_equivalence_gpu2.py'")
    parent=parent.replace("'-i','5'", "'-i','2'").replace('gpu_index=5,','gpu_index=2,')
    old="        assert UUID in observed and 'TITAN RTX' in observed"
    new="        gpu=observed.strip().split(',',1)\n        assert len(gpu)==2 and gpu[0].strip()==UUID and gpu[1].strip()=="+repr(model)
    assert parent.count(old)==1;parent=parent.replace(old,new)
    parent=parent.replace("    os.killpg(child.pid,signal.SIGTERM)","    assert os.getpgid(child.pid)==child.pid, 'Own child process-group identity changed'\n    os.killpg(child.pid,signal.SIGTERM)")
    ast.parse(parent)
    (here/'run_forward_probe_gpu2.py').write_text(parent)
    # Device changes must not alter any numerical function/threshold/fixture.
    original_tree=ast.parse(original);variant_tree=ast.parse(worker)
    by_name=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    first,second=by_name(original_tree),by_name(variant_tree)
    for name in first:
        if name!='main':assert first[name]==second[name], ('Numerical helper changed',name)
    assert sha(worker5)==worker5_sha and sha(parent5)==parent5_sha
    manifest=dict(gpu5_worker_sha256=worker5_sha,gpu5_parent_sha256=parent5_sha,gpu5_sources_unchanged=True,
                  gpu2_uuid=uuid,gpu2_model=model,gpu2_model_identity_source=str(evidence),gpu2_model_identity_source_sha256=sha(evidence),
                  quarter_uuid_source_sha256=sha(quarter),gpu2_worker_sha256=sha(here/'probe_synthetic_forward_equivalence_gpu2.py'),
                  gpu2_parent_sha256=sha(here/'run_forward_probe_gpu2.py'),numerical_helper_asts_unchanged=True,
                  thresholds=dict(max_abs_px=.001,rms_px=.0001),complete_checkpoint_cases=15*2,
                  schema='mvsec_synthetic_forward_equivalence_v1',no_training_or_GPU_execution=True)
    (here/'probe_gpu2_variant_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
