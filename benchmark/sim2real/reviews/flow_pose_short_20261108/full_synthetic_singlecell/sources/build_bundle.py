"""Build a conditional, isolated1cell fallback. No SSH/GPU/parent modification."""
import ast,base64,gzip,hashlib,json,shutil
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def change(text,old,new):
    assert old in text,old[:80]
    return text.replace(old,new)
def main():
    p=Path(__file__).resolve().parent;orig=p.parent/'flow_multiseed_short_20261108';rt=p.parent/'flow_full_synthetic_readonly_20261108/runtime'
    oldroot='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-full-synthetic-readonly-v1'
    root='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-flow-full-synthetic-readonly-rife-seed1-v1'
    source=(orig/'evaluate_frozen_synthetic.py').read_text();text=change(source,oldroot,root)
    text=change(text,"def validate(c,seeds,out,visible):","def validate(c,seeds,methods,out,visible):")
    text=change(text,"c['schema']=='flow_full_synthetic_readonly_v1'","c['schema']=='flow_full_synthetic_readonly_cell_v1'")
    text=change(text,"shard='seed01' if seeds==[0,1] else 'seed2' if seeds==[2] else None","shard='rife_seed1' if seeds==[1] and methods==['rife_esim'] else None")
    text=change(text,"p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--out'","p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--methods',nargs='+',required=True);p.add_argument('--out'")
    text=change(text,"validate(c,a.seeds,a.out.resolve(),","validate(c,a.seeds,a.methods,a.out.resolve(),")
    text=change(text,"need(not a.out.exists(),'Refusing to overwrite evaluation')","need(not a.out.exists(),'Refusing to overwrite evaluation')\n    from singlecell_contract import fallback_admission\n    fallback_proof=fallback_admission(c)")
    text=change(text,'deadline=started+3600','deadline=started+1200')
    text=change(text,'            for method in METHODS:','            for method in a.methods:')
    text=change(text,"schema='flow_full_synthetic_readonly_result_v1'","schema='flow_full_synthetic_readonly_cell_result_v1',fallback_admission=fallback_proof")
    text=text.replace('One-hour shard budget exceeded','Twenty-minute singlecell budget exceeded')
    (p/'evaluate_single_cell.py').write_text(text)
    shutil.copyfile(orig/'flow_synthetic_diagnostic.py',p/'flow_synthetic_diagnostic.py')
    c=json.loads((orig/'contract.json').read_text());c.update(schema='flow_full_synthetic_readonly_cell_v1',root=root,deadline_seconds=1200,
        shards={'rife_seed1':dict(gpu_index=1,gpu_uuid='GPU-0262d545-e9bd-7fcb-66e5-2ef81465c460',gpu_model='NVIDIA GeForce RTX 3090',seeds=[1],methods=['rife_esim'])},
        selected_seeds=[1],selected_methods=['rife_esim'],origin_root=oldroot,origin_source_manifest_sha256=sha(orig/'source_manifest.json'),
        origin_worker_sha256=sha(orig/'evaluate_frozen_synthetic.py'),origin_checkpoint_binding_sha256=hashlib.sha256(json.dumps(c['checkpoint_binding'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
    c['protocol_sha256']=sha(p/'PROTOCOL.md');(p/'contract.json').write_text(json.dumps(c,indent=2)+'\n')
    parent=(rt/'lease_flow_eval.py').read_text()
    parent=change(parent,'MAX_SECONDS = 3600','MAX_SECONDS = 1200')
    parent=change(parent,"c['schema'] == 'flow_full_synthetic_worker_command_v1'","c['schema'] == 'flow_full_synthetic_cell_command_v1'")
    parent=parent.replace("c['task'] == 'flow_full_synthetic_readonly'","c['task'] == 'flow_full_synthetic_readonly_singlecell' and gpu_index == 1")
    parent=change(parent,"out.name in ('seed01', 'seed2')","out.name == 'rife_seed1'")
    parent=change(parent,"out.name == ('seed01' if gpu_index == 0 else 'seed2')","out.name == 'rife_seed1' and gpu_index == 1")
    parent=change(parent,"expected_seeds = ['0', '1'] if out.name == 'seed01' else ['2']","expected_seeds = ['1']")
    parent=change(parent,"*expected_seeds, '--out', str(out)","*expected_seeds, '--methods', 'rife_esim', '--out', str(out)")
    parent=change(parent,"ready['schema'] == 'flow_full_synthetic_readonly_ready_v1'","ready['schema'] == 'flow_full_synthetic_cell_ready_v1'")
    parent=change(parent,"choices=tuple(GPUS)","choices=(1,)")
    parent=parent.replace('Own worker exceeded 60 minute bound','Own worker exceeded20 minute bound')
    (p/'lease_single_cell.py').write_text(parent)
    command=(rt/'build_command.py').read_text().replace('import lease_flow_eval as r','import lease_single_cell as r')
    command=change(command,'choices=[0,1]','choices=[1]')
    command=command.replace("sources/evaluate_frozen_synthetic.py","sources/evaluate_single_cell.py")
    command=change(command,"shard='seed01' if a.gpu_index==0 else 'seed2';out=root/shard","shard='rife_seed1';out=root/shard")
    command=change(command,"seeds=['0','1'] if a.gpu_index==0 else ['2']","seeds=['1']")
    command=command.replace("schema='flow_full_synthetic_worker_command_v1',task='flow_full_synthetic_readonly'","schema='flow_full_synthetic_cell_command_v1',task='flow_full_synthetic_readonly_singlecell'")
    command=change(command,'timeout_seconds=3600','timeout_seconds=1200')
    command=change(command,"*seeds,'--out',str(out)","*seeds,'--methods','rife_esim','--out',str(out)")
    (p/'build_command.py').write_text(command)
    tests=(rt/'test_lease_flow_eval.py').read_text().replace('import lease_flow_eval as r','import lease_single_cell as r')
    tests=tests.replace("'evaluate_frozen_synthetic.py'","'evaluate_single_cell.py'").replace("'lease_flow_eval.py'","'lease_single_cell.py'")
    tests=tests.replace("'flow_full_synthetic_readonly_ready_v1'","'flow_full_synthetic_cell_ready_v1'").replace("'flow_full_synthetic_worker_command_v1'","'flow_full_synthetic_cell_command_v1'")
    tests=tests.replace("task='flow_full_synthetic_readonly'","task='flow_full_synthetic_readonly_singlecell'").replace("timeout_seconds=3600","timeout_seconds=1200")
    tests=tests.replace("root/'seed01'","root/'rife_seed1'")
    tests=tests.replace("gpu_index=0,gpu_uuid=r.GPUS[0]","gpu_index=1,gpu_uuid=r.GPUS[1]").replace("verify_contract(c,root,0,parent)","verify_contract(c,root,1,parent)").replace("verify_contract(d,root,0,parent)","verify_contract(d,root,1,parent)")
    tests=tests.replace("'--seeds','0','1','--out'","'--seeds','1','--methods','rife_esim','--out'")
    tests=tests.replace("d.__setitem__('gpu_uuid',r.GPUS[1])","d.__setitem__('gpu_uuid',r.GPUS[0])")
    (p/'test_runtime.py').write_text(tests)
    original_functions={n.name:hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest() for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name not in ('main','validate')}
    (p/'origin_function_pins.json').write_text(json.dumps(original_functions,indent=2)+'\n')
    names=['evaluate_single_cell.py','flow_synthetic_diagnostic.py','singlecell_contract.py','merge_matrix.py','lease_single_cell.py','build_command.py','test_runtime.py','test_scientific.py','origin_function_pins.json','contract.json','PROTOCOL.md','build_bundle.py']
    for n in names:
        if n.endswith('.py'):ast.parse((p/n).read_text())
    manifest={n:sha(p/n) for n in names};(p/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={n:base64.b64encode((p/n).read_bytes()).decode() for n in names+['source_manifest.json']};encoded=base64.b64encode(gzip.compress(json.dumps(files).encode())).decode();(p/'upload.b64').write_text(encoded)
    for n,v in json.loads(gzip.decompress(base64.b64decode(encoded))).items():assert base64.b64decode(v)==(p/n).read_bytes()
    print(json.dumps(dict(files=len(files),bundle_chars=len(encoded),manifest_sha256=sha(p/'source_manifest.json'),no_execution=True),indent=2))
if __name__=='__main__':main()
