"""Operational replacement only; byte-identical repaired trainer and recipe."""
import copy,os
from pathlib import Path
from types import SimpleNamespace
from selected_common import need,read,sha,sources,plan_check,METHODS
from stage9_launch_contract import validate_launch as parent_validate,verify_initialization as parent_init

def validate_launch(args,task):
    need(task=='reconstruction','Only reconstruction is registered')
    R=Path(os.environ['STAGE9_CAMPAIGN']).resolve();F=Path(os.environ['STAGE9_RECON_REPAIR_ROOT']).resolve()
    p=plan_check(read(F/'sources/registered_plan.json'));sources(F)
    need(str(F)==p['root'] and str(R)==p['campaign'],'Wrong campaign/root')
    need(Path(args.config).resolve()==F/'sources/training_recipe.json' and sha(Path(args.config))==p['recipe_sha256'],'Recipe changed')
    cfg=read(args.config);parent=read(R/'sources/training_recipe.json');meta=cfg.pop('reconstruction_repair')
    expected=copy.deepcopy(parent);expected['reconstruction']['model_args']['final_activation']='sigmoid'
    need(cfg==expected and meta['parent_recipe_sha256']==sha(R/'sources/training_recipe.json'),'Only native sigmoid may differ')
    need(args.generator in METHODS and args.seed in (0,1,2),'Unregistered identity')
    output=F/'training'/('seed'+str(args.seed))/args.generator
    need(Path(args.output_root).resolve()==output,'Wrong isolated output')
    gate=read(F/'ALL15_SYNTHETIC_GATE.json')
    need(gate['passed'] and gate['source_manifest_sha256']==sources(F),'Full15 synthetic stability gate not passed')
    need(read(F/'cpu_admission.json')['passed'],'CPU admission missing')
    if args.resume:
        resume=Path(args.resume).resolve();key=args.generator+'/seed'+str(args.seed)
        if key in p['resume_sources'] and resume==Path(p['resume_sources'][key]['last']['checkpoint']):
            binding=p['resume_sources'][key]['last']
            need(sha(resume)==binding['checkpoint_sha256'] and args.max_updates==binding['updates']+1,'Old CP must first pass next-update restore')
        else:
            need(resume==output/'last.pt','Unregistered resume source')
            need(args.max_updates in (0,21),'Unregistered resume boundary')
            if args.max_updates==0:need(read(output/'boundary_preflight.json')['passed'],'Process-boundary restore not verified')
    else:need(args.max_updates==20,'Fresh full run must first complete update20 check')
    parent_args=SimpleNamespace(**vars(args));parent_args.config=R/'sources/training_recipe.json';parent_args.max_updates=20
    parent_validate(parent_args,task)
    output.mkdir(parents=True,exist_ok=True)
    (output/'repaired_launch_preflight.json').write_text(__import__('json').dumps(dict(passed=True,source_manifest_sha256=sources(F),
        stable_recipe_sha256=p['recipe_sha256'],real_data_used_in_training=False,real_scoring_enabled=False)))

def verify_initialization(task,seed,digest):return parent_init(task,seed,digest)
