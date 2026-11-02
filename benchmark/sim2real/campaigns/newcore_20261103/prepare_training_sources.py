"""Derive Stage-9 runners without changing pinned models or objectives."""
import ast,json,sys
from pathlib import Path
R=Path(sys.argv[1]);s=R/'sources/benchmark/sim2real/scripts'
methods='("eventgenesis", "v2e", "vid2e", "film_esim", "rife_esim")'
for name in ['steflow','e2vid']:
 old=s/('train_'+name+'_stage4.py');text=old.read_text()
 text=text.replace('choices=("eventgenesis", "v2e")','choices='+methods)
 text=text.replace('    if not os.environ.get("STAGE4_LAUNCH_GATE_VERIFIED"):\n        raise SystemExit("refusing direct trainer invocation; use scripts/stage4_launch_gate.py")\n','')
 text=text.replace('    args = p.parse_args()\n','    args = p.parse_args()\n    from stage9_launch_contract import validate_launch\n    validate_launch(args, "flow")\n') if name=='steflow' else text.replace('    args = parser.parse_args()\n','    args = parser.parse_args()\n    from stage9_launch_contract import validate_launch\n    validate_launch(args, "reconstruction")\n')
 if name=='steflow':
  text=text.replace('    initial_hash = state_sha256(model)\n','    initial_hash = state_sha256(model)\n    from stage9_launch_contract import verify_initialization\n    verify_initialization("flow", args.seed, initial_hash)\n')
  begin=text.index('    viz_root = run / "fixed_validation_visuals"');end=text.index('    result = {',begin)
  text=text[:begin]+'    # Numerical results only in this campaign.\n'+text[end:]
  text=text.replace('"fixed_visualization_samples": sorted(value[0] + "/sample000" for value in fixed_visualizations.values()),','"fixed_numeric_samples": sorted(value[0] + "/sample000" for value in fixed_visualizations.values()),')
 else:
  text=text.replace('    initial_hash = state_sha256(model)\n','    initial_hash = state_sha256(model)\n    from stage9_launch_contract import verify_initialization\n    verify_initialization("reconstruction", args.seed, initial_hash)\n')
  text=text.replace('"model": model.state_dict(), "model_sha256": state_sha256(model),','"model": model.state_dict(), "model_sha256": state_sha256(model), "initial_state_sha256": initial_hash,')
  text=text.replace('    from model.model import E2VIDRecurrent\n','    from model.model import E2VIDRecurrent\n    sys.path.insert(0, str(HERE.parents[0] / "compat"))\n    from e2vid_coordinate_compat import patch_temporal_sampling\n    from utils import loss as temporal_warp_module\n    atomic_json(run_root / "coordinate_preflight.json", patch_temporal_sampling(temporal_warp_module, model_root))\n')
 ast.parse(text);(s/('train_'+name+'_stage9.py')).write_text(text)
config=json.loads((R/'sources/benchmark/sim2real/configs/stage4_corrected_v2_flowdt1_seed0.json').read_text())
config.update(schema='stage9_newcore_training_recipe_v1',date='2026-11-03',status='frozen_before_optimizer_updates',frozen_simulator_core='62da4f35ce3c5c96f04268bd1687a27657af9238',training_entrypoint='run_training_job.py')
config['shared_bank'].update(version='stage9_replicad_8_layouts_whole_layout_holdout',split_seed=20261103)
config['runtime']['GPU_allocation']='UUID leases on authorized g1 GPUs 3/4/5 only'
config['flow_dt1']['microbatch']=8;config['reconstruction']['microbatch']=1
config['generators']['eventgenesis'].update(warp_composite='coverage_time_log_linear',sampler_max_samples=512,task_profile='stage9_sensor_profile.json')
config['reconstruction']['temporal_coordinate_compatibility']='two upstream W-1/H-1 grids use explicit align_corners=True, equal across every method; pinned source files unchanged'
config['reconstruction']['temporal_flow_semantics']='retain native negative-forward-flow proxy and image-difference visibility mask; do not claim exact target-grid backward GT or explicit flow_valid masking'
config['pose']={'model':'pinned_SingleShotPose_YOLO6D','initialization':'random common seed; no warmstart','source_commit':'2c282619101f9f31666dd18b42393375bea1600f','input':'full-frame ON/OFF/total log-count, fixed log(9) scale, 416x416','batch_size':8,'updates':30000,'learning_rate':.0001,'weight_decay':0.,'gradient_clip':100.,'loss':'official coordinate MSE per batch plus center-grid cross entropy (declared shared adapter from R10)','lr_schedule':'30 update warmup then cosine to 10% at fixed budget','validate_every_updates':1000,'checkpoint_selection':'minimum rollout-equal mean 9-keypoint Euclidean error in native sensor pixels; tie earlier update','augmentation':'joint affine scale .85-1.2, translation +/-24 network pixels; keep centroid in bounds; same random draws per method and seed','event_window_s':.01,'warmup_s':.04,'train_windows':1536,'validation_windows':512,'cad_scope':'YCB 036 wood block','inference':'ordinary eval, highest predicted center logit, predicted corners plus fixed EPNP; no GT crop/mask/cell'}
(R/'sources/training_recipe.json').write_text(json.dumps(config,indent=2)+'\n')
print('TRAINING_SOURCES_PREPARED')
