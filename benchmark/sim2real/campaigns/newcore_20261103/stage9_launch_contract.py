"""Fail closed on stale data, changed recipe, real training, or missing preflight."""
import hashlib,json,os
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate_launch(args,task):
 R=Path(os.environ['STAGE9_CAMPAIGN']).resolve();cfg=Path(args.config).resolve();data=Path(args.data_root).resolve()
 assert cfg==R/'sources/training_recipe.json' and data.is_relative_to(R/'adapters')
 c=json.loads(cfg.read_text());release='62da4f35ce3c5c96f04268bd1687a27657af9238';assert c['frozen_simulator_core']==release
 sources=json.loads((R/'sources/training_source_manifest.json').read_text())
 for path,h in sources['files'].items():assert sha(R/'sources'/path)==h,path
 assert args.generator in ('eventgenesis','v2e','vid2e','film_esim','rife_esim') and args.seed in (0,1,2)
 assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
 mpath=data/('training_bank_manifest.json' if task=='flow' else 'adapter_manifest.json');m=json.loads(mpath.read_text());assert m['real_data_used'] is False and m.get('real_scoring_enabled',False) is False
 assert m['generator']==args.generator
 if task=='pose':assert json.loads((R/'pose_physical_geometry_audit.json').read_text())['passed']
 if task=='reconstruction':assert json.loads((R/'preflight/early_runtime/reconstruction/result.json').read_text())['passed']
 kind='pose' if task=='pose' else 'rich';physical=R/(kind+'_bank')/'bank_manifest.json';assert m['physical_manifest_sha256']==sha(physical)
 rows=m.get('entries',m.get('rollout_rows',m.get('rows',[])));rows=[r for r in rows if 'path' in r]
 assert len(rows)==(128 if task=='pose' else 32)
 for row in rows:assert sha(Path(row['path']))==row['sha256'],row['path']
 reference=R/'adapters/eventgenesis'/task/mpath.name
 if args.generator!='eventgenesis' and reference.exists():
  ref=json.loads(reference.read_text());ref_rows=ref.get('entries',ref.get('rollout_rows',ref.get('rows',[])))
  left={r['rollout_id']:r.get('label_sha256',r.get('flow_label_sha256')) for r in rows}
  right={r['rollout_id']:r.get('label_sha256',r.get('flow_label_sha256')) for r in ref_rows if 'path' in r}
  assert left==right and all(left.values()),'Shared labels differ across event generators'
 pointer=json.loads((Path('/home/zhaoyuhan/Eventbased_WAM')/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text());assert pointer['release_commit']==release
 for name,h in pointer['descriptor']['core_canonical_LF_sha256'].items():assert hashlib.sha256((Path(pointer['core_path'])/'genesis_event_plugin'/name).read_text().encode()).hexdigest()==h
 from verify_pinned_source import verify_pinned_source
 name={'flow':'STE-FlowNet','reconstruction':'event_cnn_minimal','pose':'SingleShotPose'}[task]
 bundle=R/'sources/downstream_source_bundle.json';record=next(r for r in json.loads(bundle.read_text())['repositories'] if r['repo']==name)
 verify_pinned_source(Path('/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream')/name,name,record['commit'],bundle)
 limit=args.max_updates
 if limit not in (20,21):
  gate=R/'preflight/tasks'/task/'paired_gate.json';g=json.loads(gate.read_text());assert g['passed'] and g['recipe_sha256']==sha(cfg)
  assert {'eventgenesis','v2e'}<=set(g['methods'])
 assert not c['real_scoring_enabled']

def verify_initialization(task,seed,digest):
 R=Path(os.environ['STAGE9_CAMPAIGN']);reference=json.loads((R/'preflight/initialization_reference.json').read_text());assert reference['passed']
 assert reference['recipe_sha256']==sha(R/'sources/training_recipe.json')
 assert digest==reference['tasks'][task]['hashes'][str(seed)],(task,seed,'Initial weights differ from independent CPU reference')
