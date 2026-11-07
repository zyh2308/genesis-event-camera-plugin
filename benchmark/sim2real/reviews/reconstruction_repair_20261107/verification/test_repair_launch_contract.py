"""Procedural tamper tests for the repair boundary; no server/core/data/model execution."""
import ast,copy,hashlib,importlib.util,json,os,sys,tempfile,types
from pathlib import Path
from types import SimpleNamespace
from recon_repair_gate import decide

p=Path(__file__).parent
scope={};tree=ast.parse((p/'test_repair_gate.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixture')
exec(compile(ast.Module(body=[node],type_ignores=[]),'fixture_only','exec'),scope);fixture=scope['fixture']
calls=[];stub=types.ModuleType('stage9_launch_contract');stub.validate_launch=lambda args,task:calls.append((args.config,args.max_updates,task))
stub.verify_initialization=lambda *args:None;sys.modules['stage9_launch_contract']=stub
spec=importlib.util.spec_from_file_location('contract_under_test',p/'recon_repair_launch_contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
checks=[]
with tempfile.TemporaryDirectory(prefix='recon_contract_fixture_') as directory:
    root=Path(directory);R=root/'original';F=root/'repair';study=root/'20261107-recon-sigmoid-mixed-v2'
    for base in [R,F,study]:(base/'sources').mkdir(parents=True)
    (study/'mixed').mkdir();original=dict(reconstruction=dict(model_args=dict(final_activation=''),optimizer=dict(lr=.0001)),real_scoring_enabled=False)
    (R/'sources/training_recipe.json').write_text(json.dumps(original));sha=contract.sha
    metadata=dict(parent_recipe_sha256=sha(R/'sources/training_recipe.json'),native_output_activation='sigmoid',real_data_used_for_choice=False,synthetic_adoption_study=str(study))
    config=copy.deepcopy(original);config['reconstruction']['model_args']['final_activation']='sigmoid';config['reconstruction_repair']=metadata
    configpath=F/'sources/training_recipe.json';configpath.write_text(json.dumps(config))
    (F/'sources/recon_repair_gate.py').write_bytes((p/'recon_repair_gate.py').read_bytes())
    (study/'sources/recon_short_diagnostic.py').write_text('PROCEDURAL_FIXTURE_ONLY')
    (study/'sources/source_manifest.json').write_text(json.dumps({'recon_repair_gate.py':sha(F/'sources/recon_repair_gate.py'),'recon_short_diagnostic.py':sha(study/'sources/recon_short_diagnostic.py')}))
    rows=fixture()
    for i,row in enumerate(rows):
        cp=root/f'fixture_checkpoint_{i}.bin';cp.write_bytes(('not a model: fixture'+str(i)).encode());row.update(checkpoint=str(cp),checkpoint_sha256=sha(cp))
    completion=study/'mixed/completion.json';completion.write_text(json.dumps(dict(results=rows)))
    gate=decide(rows);gate.update(completion_sha256=sha(completion),gate_source_sha256=sha(F/'sources/recon_repair_gate.py'))
    (study/'synthetic_gate.json').write_text(json.dumps(gate));(study/'resource_release.json').write_text(json.dumps(dict(status='released')))
    def manifest():
        files={f.name:sha(f) for f in (F/'sources').glob('*') if f.name!='source_manifest.json'}
        (F/'sources/source_manifest.json').write_text(json.dumps(files))
    manifest();os.environ.update(STAGE9_CAMPAIGN=str(R),STAGE9_RECON_REPAIR_ROOT=str(F))
    args=SimpleNamespace(config=configpath,output_root=F/'training/seed2/eventgenesis',seed=2,generator='eventgenesis',max_updates=0)
    contract.validate_launch(args,'reconstruction');assert calls==[(R/'sources/training_recipe.json',20,'reconstruction')];checks.append('new_gate_plus_parent_input_validation_required')
    def reject(label,apply,undo):
        apply()
        try:contract.validate_launch(args,'reconstruction');raise RuntimeError('Invalid launch passed: '+label)
        except (AssertionError,ValueError,FileNotFoundError):checks.append(label)
        finally:undo()
    original_bytes=configpath.read_bytes()
    def bad_lr():
        bad=copy.deepcopy(config);bad['reconstruction']['optimizer']['lr']=.001;configpath.write_text(json.dumps(bad));manifest()
    reject('unregistered_optimizer_change_rejected',bad_lr,lambda:(configpath.write_bytes(original_bytes),manifest()))
    reject('wrong_output_root_rejected',lambda:setattr(args,'output_root',F/'wrong'),lambda:setattr(args,'output_root',F/'training/seed2/eventgenesis'))
    reject('arbitrary_training_budget_rejected',lambda:setattr(args,'max_updates',999),lambda:setattr(args,'max_updates',0))
    cp=Path(rows[0]['checkpoint']);cpbytes=cp.read_bytes()
    reject('changed_adoption_checkpoint_rejected',lambda:cp.write_bytes(b'changed'),lambda:cp.write_bytes(cpbytes))
    gatepath=study/'synthetic_gate.json';gatebytes=gatepath.read_bytes()
    reject('claimed_gate_flag_cannot_override_failed_check',lambda:gatepath.write_text(json.dumps({**gate,'checks':[]})),lambda:gatepath.write_bytes(gatebytes))
    helper=F/'sources/recon_repair_gate.py';helperbytes=helper.read_bytes()
    reject('changed_helper_source_rejected',lambda:helper.write_bytes(helperbytes+b'\n'),lambda:helper.write_bytes(helperbytes))
    cpayload=completion.read_bytes()
    reject('changed_study_results_rejected',lambda:completion.write_bytes(cpayload+b' '),lambda:completion.write_bytes(cpayload))
(p/'repair_launch_contract_preflight.json').write_text(json.dumps(dict(passed=True,fixture_only=True,parent_core_checks_stubbed_only_in_procedural_test=True,checks=checks),indent=2)+'\n')
print('launch preflight',len(checks),'passed')
