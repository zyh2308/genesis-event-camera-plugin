"""CPU-only source/data/old-checkpoint admission; never publishes DEPLOYMENT_READY."""
import argparse,copy,gc,os,sys
from pathlib import Path
from types import SimpleNamespace
from selected_common import *

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    F=a.root.resolve();s=sources(F);plan=plan_check(read(F/'sources/registered_plan.json'));R=Path(plan['campaign'])
    need(str(F)==plan['root'] and not (F/'cpu_admission.json').exists(),'Wrong root or do not overwrite admission')
    need(sha(F/'sources/train_e2vid_repaired.py')==plan['trainer_sha256'] and sha(F/'sources/training_recipe.json')==plan['recipe_sha256'],'Native stable source/recipe changed')
    sys.path.insert(0,str(R/'sources/benchmark/sim2real/scripts'))
    from stage9_launch_contract import validate_launch
    previous=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['STAGE9_CAMPAIGN']=str(R)
    os.environ['CUDA_VISIBLE_DEVICES']=plan['lanes'][0]['gpu_uuid']
    data=[]
    for method in METHODS:
        args=SimpleNamespace(config=R/'sources/training_recipe.json',data_root=R/'adapters'/method/'reconstruction',generator=method,seed=0,max_updates=20)
        validate_launch(args,'reconstruction');data.append(record(args.data_root/'adapter_manifest.json'))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    recipe=read(F/'sources/training_recipe.json');resume_checks=[];history=[]
    initial=read(R/'preflight/initialization_reference.json')['tasks']['reconstruction']['hashes']
    for key,b in plan['resume_sources'].items():
        method,seed=key.split('/seed');seed=int(seed)
        summary=read(check(b['summary']))
        need(summary['status']=='interrupted_at_safe_boundary' and summary['optimizer_update']==b['last']['updates'] and summary['final_checkpoint_sha256']==b['last']['checkpoint_sha256'],'Old safe-stop summary differs')
        oldlog=Path(b['last']['checkpoint']).parent/'training_metrics.jsonl'
        rows=[__import__('json').loads(line) for line in oldlog.open()]
        updates=[r['optimizer_update'] for r in rows if r.get('event')=='optimizer_update']
        need(updates==list(range(1,b['last']['updates']+1)),'Old log update inventory differs')
        history.append(dict(key=key,log=record(oldlog)))
        for name in ('last','best'):
            item=b[name];path=Path(item['checkpoint']);need(sha(path)==item['checkpoint_sha256'],'Old checkpoint changed')
            cp=torch.load(path,map_location='cpu',weights_only=False);cursor_check(cp,item['updates'])
            need(cp['generator']==method and cp['seed']==seed and cp['model_args']==recipe['reconstruction']['model_args'],'Resume identity/model differs')
            need(cp['config_sha256']==plan['recipe_sha256'] and cp['trainer_script_sha256']==plan['trainer_sha256'],'Resume source fingerprint differs')
            need(cp['data_manifest_sha256']==sha(R/'adapters'/method/'reconstruction/adapter_manifest.json'),'Resume data differs')
            need(cp['initial_state_sha256']==initial[str(seed)] and cp['real_data_used'] is False,'Resume init/isolation differs')
            state=__import__('hashlib').sha256()
            for n,v in sorted(cp['model'].items()):state.update(n.encode());state.update(v.cpu().numpy().tobytes())
            need(state.hexdigest()==cp['model_sha256'],'Model tensor fingerprint differs')
            need(cp['rng_state']['torch_cpu'].device.type=='cpu','RNG device differs')
            resume_checks.append(dict(key=key,kind=name,record=record(path),update=item['updates'],next_epoch=cp['next_epoch'],next_batch=cp['next_batch_in_epoch']))
            del cp;gc.collect()
    # Recompute all six inherited stability cases and bind every actual CP.
    inherited=read(F/'sources/inherited_gate_results.json')
    study=Path(plan['inherited_gate']['root']);declared=plan['inherited_gate']['source_manifest']
    need(read(study/'sources/source_manifest.json')==declared,'Inherited study source manifest changed')
    for name,digest in declared.items():need(sha(study/'sources'/name)==digest,'Inherited gate source changed: '+name)
    need(declared['recon_short_diagnostic.py']==sha(F/'sources/recon_short_diagnostic.py'),'Gate worker implementation changed')
    need(sha(study/'mixed/completion.json')==plan['inherited_gate']['completion_sha256'],'Inherited gate completion changed')
    from inherited_gate import decide
    actual_inherited=read(study/'mixed/completion.json')['results']
    gate=decide(inherited);actual_gate=decide(actual_inherited)
    need(gate['passed'] and actual_gate['passed'] and actual_gate['checks']==gate['checks'],'Inherited paired three-seed gate changed')
    for row in inherited:
        need(sha(Path(row['checkpoint']))==row['checkpoint_sha256'],'Inherited gate CP changed')
        need(sha(Path(row['source_result_path']))==row['source_result_sha256'],'Inherited per-case result changed')
    # This deployment is train-only. Future selected3 scoring has a separate
    # source/manifest authorization; no real images or metric assets are read.
    if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
    else:os.environ['CUDA_VISIBLE_DEVICES']=previous
    save(F/'cpu_admission.json',dict(passed=True,source_manifest_sha256=s,registered_plan_sha256=sha(F/'sources/registered_plan.json'),
        resume_checks=resume_checks,resume_history_logs=history,adapter_manifests=data,inherited_six_gate=gate,CPU_only=True,GPU_used=False,real_scores_computed=False,
        inherited_gate_results_sha256=sha(F/'sources/inherited_gate_results.json'),initial_state_reference=record(R/'preflight/initialization_reference.json')))
    print('CPU_ADMISSION_PASSED_READY_NOT_PUBLISHED',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        if '--root' in sys.argv:
            target=Path(sys.argv[sys.argv.index('--root')+1])
            if target.is_dir() and not (target/'GLOBAL_FAILURE.json').exists():save(target/'GLOBAL_FAILURE.json',dict(stage='cpu_admission',error=traceback.format_exc()))
        raise
