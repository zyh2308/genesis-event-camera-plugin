"""Validate original frozen inputs and the separate reconstruction repair gate."""
import copy,hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
from stage9_launch_contract import validate_launch as parent_validate,verify_initialization as parent_init
from recon_repair_gate import decide

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_launch(args,task):
    assert task=='reconstruction'
    R=Path(os.environ['STAGE9_CAMPAIGN']).resolve();F=Path(os.environ['STAGE9_RECON_REPAIR_ROOT']).resolve()
    assert F.parent==R.parent and F!=R
    source=F/'sources';manifest=json.loads((source/'source_manifest.json').read_text())
    for name,digest in manifest.items():assert sha(source/name)==digest,name
    assert Path(args.config).resolve()==source/'training_recipe.json'
    cfg=json.loads(Path(args.config).read_text());original=json.loads((R/'sources/training_recipe.json').read_text())
    metadata=cfg.pop('reconstruction_repair')
    assert metadata['parent_recipe_sha256']==sha(R/'sources/training_recipe.json')
    assert metadata['native_output_activation']=='sigmoid' and metadata['real_data_used_for_choice'] is False
    expected=copy.deepcopy(original);expected['reconstruction']['model_args']['final_activation']='sigmoid'
    assert cfg==expected,'Only the declared native output activation may change'
    assert args.max_updates in (0,20,21),'Formal repair launches allow only update20/21 checks or the full budget'
    expected_output=F/'training'/f'seed{args.seed}'/args.generator
    assert Path(args.output_root).resolve()==expected_output
    # Reuse all original source/core/noise/data/label checks. max_updates=20
    # deliberately requests input-only validation here; the full new gate below
    # is mandatory for EVERY repaired launch, including its smoke/resume steps.
    parent_args=SimpleNamespace(**vars(args));parent_args.config=R/'sources/training_recipe.json';parent_args.max_updates=20
    parent_validate(parent_args,task)
    study=Path(metadata['synthetic_adoption_study']).resolve()
    assert study.parent==R.parent and study.name=='20261107-recon-sigmoid-mixed-v2'
    study_manifest=json.loads((study/'sources/source_manifest.json').read_text())
    assert study_manifest['recon_repair_gate.py']==sha(source/'recon_repair_gate.py')
    assert sha(study/'sources/recon_short_diagnostic.py')==study_manifest['recon_short_diagnostic.py']
    completion=study/'mixed/completion.json';gate=json.loads((study/'synthetic_gate.json').read_text())
    assert gate['completion_sha256']==sha(completion) and gate['gate_source_sha256']==sha(source/'recon_repair_gate.py')
    results=json.loads(completion.read_text())['results'];recomputed=decide(results)
    assert recomputed['passed'] and gate['passed'] and recomputed['checks']==gate['checks']
    for row in results:assert sha(row['checkpoint'])==row['checkpoint_sha256']
    assert json.loads((study/'resource_release.json').read_text())['status'].startswith('released')
    for method in ['eventgenesis','v2e']:
        assert len([r for r in results if r['method']==method])==3
    receipt=dict(passed=True,parent_frozen_input_checks=True,synthetic_adoption_checks=recomputed,
        recipe_sha256=sha(args.config),source_manifest_sha256=sha(source/'source_manifest.json'),
        real_data_used_in_training=False,real_scoring_enabled=False)
    expected_output.mkdir(parents=True,exist_ok=True)
    (expected_output/'repaired_launch_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')

def verify_initialization(task,seed,digest):return parent_init(task,seed,digest)
