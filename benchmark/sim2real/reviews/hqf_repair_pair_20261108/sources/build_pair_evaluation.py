"""Prepare an isolated, explicitly authorized paired development evaluation."""
import ast, difflib, hashlib, json, pathlib, textwrap
out=pathlib.Path(__file__).parent
repo=pathlib.Path('/home/科研/Eventbased_WAM/repositories/gec-mainline-resume-20261105')
parent=repo/'benchmark/sim2real/reviews/stage9_mainline_20261105/reconstruction/stage9_hqf_bridge.py'
source=parent.read_text()
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='0cda3a62c88d7d33bc6398d4f14e33f5d86f05d92a9f2ecd01c01e5e21db3c55'
new_collect='''def validate_checkpoint_metadata(cp, recipe, method, expected_sha, actual_sha):
    require(actual_sha == expected_sha, 'Checkpoint SHA differs from preselected synthetic checkpoint')
    require(cp.get('generator') == method and cp.get('seed') == 0, 'Checkpoint identity differs')
    require(cp.get('optimizer_update') == 4800, 'Both checkpoints must be update4800')
    require(cp.get('model_args') == recipe['reconstruction']['model_args'], 'Checkpoint model configuration differs')
    require(cp['model_args']['final_activation'] == 'sigmoid', 'Native sigmoid is required')
    require(cp.get('real_data_used') is False, 'Checkpoint must be synthetic-only')
    require(cp.get('initial_state_sha256') == '48426d94f4159592155eda873d4e71e2f06439c10eade990de5979231ed07a0f', 'Wrong registered initialization')
    require(cp.get('trainer_script_sha256') == 'e7d9db6632930c8319ddaee174414d1984ae1b7a380b26c026a795b110b77b28', 'Wrong repair trainer')

def collect_training(campaign, recipe_path):
    campaign=Path(campaign).resolve(strict=True)
    require(campaign.name=='20261107-recon-sigmoid-formal-v2', 'Only the safely stopped repair campaign is eligible')
    recipe=read(recipe_path)
    require(recipe['frozen_simulator_core']==CORE, 'Frozen core changed')
    require(sha(recipe_path)=='cf04fa50cd833a21b2a1cae7127acff8678317b9535e0adc99fc7c0eb5d0013a', 'Repair recipe changed')
    scope=campaign/'scope_reduction_20261108/final_receipt.json'
    receipt=read(scope)
    require(not receipt['processes_remaining'], 'Repair trainers have not safely stopped')
    expected={'eventgenesis':'6ecce87e995bfd214b564584470927cc8fde0ac7e5a12aae8578bddea94c8192',
              'v2e':'904a887ce46cddb1fe590aa48fe714cb1d5c4fbc3d50df24b168de7559783102'}
    require(all(len(h)==64 for h in expected.values()), 'Malformed checkpoint hash pin')
    import torch
    rows=[]
    for method in METHODS:
        run=campaign/'training/seed0'/method
        selected=read(run/'synthetic_best.json');cp=run/'synthetic_best.pt'
        require(selected['optimizer_update']==4800 and selected['epoch']==80, 'Frozen synthetic selection differs')
        require(selected['selection_source']=='synthetic validation only', 'Real checkpoint selection forbidden')
        require(selected['sha256']==expected[method], 'Synthetic checkpoint changed after user request')
        meta=torch.load(cp,map_location='cpu',weights_only=False)
        validate_checkpoint_metadata(meta,recipe,method,expected[method],sha(cp))
        require(set(meta['optimizer_step_counts'])=={4800}, 'Checkpoint optimizer counters differ')
        require(meta['config_sha256']==sha(recipe_path), 'Checkpoint recipe binding differs')
        summary=read(run/'run_summary.json')
        require(summary['status']=='interrupted_at_safe_boundary', 'Expected safely stopped validation run')
        rows.append(dict(run_id=f'{method}_seed0',method=method,seed=0,
            checkpoint=record(cp),selection_record=record(run/'synthetic_best.json'),
            safe_stop_summary=record(run/'run_summary.json'),scope_receipt=record(scope),
            initial_state_sha256=meta['initial_state_sha256'],checkpoint_updates=4800,
            training_stopped_update=summary['optimizer_update'],full_training_complete=False))
        del meta
    require(len(rows)==2 and len({r['initial_state_sha256'] for r in rows})==1, 'Paired initialization mismatch')
    return rows

'''
a=source.index('def collect_training(');b=source.index('def inspect_data(',a)
s=source[:a]+new_collect+source[b:]
s=s.replace("METHODS = ('eventgenesis','v2e','vid2e','film_esim','rife_esim')", "METHODS = ('eventgenesis','v2e')")
s=s.replace('SEEDS = (0,1,2)', 'SEEDS = (0,)')
s=s.replace("require(recipe['reconstruction']['model_args']==old_model,'Stage9 architecture differs from inherited evaluator')", "expected_model=copy.deepcopy(old_model);expected_model['final_activation']='sigmoid'\n    require(recipe['reconstruction']['model_args']==expected_model,'Only native sigmoid differs from the inherited model')")
s=s.replace("schema='stage9_hqf15_frozen_protocol_v1'", "schema='hqf_repair_pair_4800_development_protocol_v1'")
s=s.replace("p.get('schema')=='stage9_hqf15_frozen_protocol_v1'", "p.get('schema')=='hqf_repair_pair_4800_development_protocol_v1'")
s=s.replace('expected_runs=15','expected_runs=2').replace('expected_sequence_runs=210','expected_sequence_runs=28')
s=s.replace("p.get('expected_runs')==15", "p.get('expected_runs')==2").replace("p.get('expected_sequence_runs')==210", "p.get('expected_sequence_runs')==28")
s=s.replace("len(p['runs'])==15", "len(p['runs'])==2")
s=s.replace("'stage9_final_real_evaluation_authorization_v1'", "'hqf_repair_development_authorization_v1'")
s=s.replace("schema='stage9_hqf15_run_v1'", "schema='hqf_repair_pair_4800_run_v1'")
s=s.replace('all_210_inferences_complete_unscored','all_28_inferences_complete_unscored')
s=s.replace("selection='per-run synthetic_best.pt only; no real selection or fallback to last/preflight'", "selection='both pre-existing update4800 synthetic_best.pt checkpoints, fixed before HQF; no real selection'")
s=s.replace("aggregation='frames within sequence; equal14 sequence means; then3seed mean/sampleSD; paired sequence bootstrap'", "aggregation='frame means within sequence; equal14 sequence means; single seed development evaluation, no seed variance estimate'")
s=s.replace("real_test_exposure='HQF14 previously seen benchmark re-evaluation; not a blind test'", "real_test_exposure='HQF14 previously seen development evaluation; not a blind final test', training_budget_note='Both selected checkpoints update4800; safely stopped parent training is incomplete'")
s=s.replace("'hqf15_protocol.json'", "'hqf_repair_pair_protocol.json'")
s=s.replace("'runs':15,'sequence_runs':210", "'runs':2,'sequence_runs':28")
s=s.replace("'Stage9 HQF15", "'HQF repair pair").replace('Stage9 HQF15 bridge.', 'HQF repair pair development bridge.')
s=s.replace('All15 formal receipts', 'Paired repair checkpoints')
s=s.replace("'run matrix", "'paired run matrix")
# Keep inference and metric arithmetic unchanged. Only run bindings and protocol
# validation differ from the original 15-run final-experiment entry.
ast.parse(s)
(out/'hqf_repair_pair_bridge.py').write_text(s)
old_ast={n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
new_ast={n.name:n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
checks={name:ast.dump(old_ast[name],include_attributes=False)==ast.dump(new_ast[name],include_attributes=False)
    for name in ['inspect_data','verify_downstream_sources','load_scorer','validate_inference_contract','preflight_metric_assets']}
assert all(checks.values())
class NormalizeStatus(ast.NodeTransformer):
    def visit_Constant(self,n):
        if n.value=='all_28_inferences_complete_unscored':n.value='all_210_inferences_complete_unscored'
        return n
checks['score_existing_unchanged_except_inventory_status']=ast.dump(old_ast['score_existing'],include_attributes=False)==ast.dump(NormalizeStatus().visit(new_ast['score_existing']),include_attributes=False)
assert all(checks.values())
(out/'bridge_preservation_checks.json').write_text(json.dumps(dict(parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),checks=checks),indent=2)+'\n')
(out/'bridge_delta.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='original/stage9_hqf_bridge.py',tofile='repair/hqf_repair_pair_bridge.py',n=0)))
# Reuse the existing safe GPU manager with a small local support module.
wrapper=(repo/'benchmark/sim2real/reviews/three_line_update_20261107/run_hqf15_reserved.py').read_text()
a=wrapper.index('    manifest=json.loads((args.lease_helpers.parent');b=wrapper.index("    protocol=root/'protocol",a)
wrapper=wrapper[:a]+'''    spec=importlib.util.spec_from_file_location('hqf_local_helpers',root/'sources/evaluation_support.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    manifest=json.loads((root/'sources/source_manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((root/'sources'/name).read_bytes()).hexdigest()==digest,name
'''+wrapper[b:]
wrapper=wrapper.replace("root/'protocol/hqf15_protocol.json'", "root/'protocol/hqf_repair_pair_protocol.json'")
wrapper=wrapper.replace("p.add_argument('--lease-helpers',type=Path,required=True)", "p.add_argument('--lease-helpers',type=Path)")
(out/'run_hqf_pair_reserved.py').write_text(wrapper)
(out/'evaluation_support.py').write_text('''import datetime,json,os,signal,subprocess
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);t=path.with_name(path.name+'.partial');t.write_text(json.dumps(value,indent=2)+'\\n');t.replace(path)
def stop_child(child):
    if child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
''')
# Actual synthetic inference with repaired weights; no real GT is opened here.
fixture=(repo/'benchmark/sim2real/reviews/three_line_update_20261107/hqf_stage9_runtime_preflight.py').read_text()
fixture=fixture.replace("receipt=json.loads((args.campaign/'training/reconstruction/seed0/eventgenesis/completion.json').read_text())", "receipt=json.loads((args.campaign/'training/seed0/eventgenesis/synthetic_best.json').read_text())")
fixture=fixture.replace("receipt['selected_checkpoint']", "receipt['checkpoint']").replace("receipt['selected_checkpoint_sha256']", "receipt['sha256']")
(out/'hqf_repair_runtime_preflight.py').write_text(fixture)
for f in out.glob('*.py'):ast.parse(f.read_text())
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.py') if f.name!='build_pair_evaluation.py'}
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(sources=manifest,preservation_checks=checks),indent=2))
