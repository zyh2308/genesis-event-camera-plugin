from pathlib import Path
import subprocess,shlex,base64,json
CODE=r'''from pathlib import Path
import json,datetime,hashlib,h5py
b=Path('/home/zhaoyuhan/Eventbased_WAM')
bank=b/'data/sim2real/stage5/20261021-stage5-flow-coverage-v3-eight-rollouts'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(4*1024*1024),b''):h.update(z)
 return h.hexdigest()
m=json.loads((bank/'bank_manifest.json').read_text())
p=bank/'shared_trajectory.h5';actual=sha(p)
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physical_bank_bytes':p.stat().st_size,'actual_sha256':actual,'manifest_sha256':m.get('bank_sha256'),'bank_hash_matches':actual==m.get('bank_sha256'),'manifest':m,'progress':json.loads((bank/'generation_progress.json').read_text()),'hdf5':{}}
with h5py.File(p,'r') as h:
 out['hdf5']['root_keys']=list(h.keys());out['hdf5']['episode_count']=len(h['episodes'])
 out['hdf5']['episode_schema']={}
 for k in h['episodes']:
  g=h['episodes'][k]
  out['hdf5']['episode_schema'][k]={key:{'shape':list(v.shape),'dtype':str(v.dtype)} if isinstance(v,h5py.Dataset) else {'keys':list(v.keys())} for key,v in g.items()}
out['related_stage5_runs']=[p.name for p in (b/'runs/sim2real').iterdir() if any(x in p.name for x in ('stage5','coverage'))]
out['related_stage5_data']=[p.name for p in (b/'data/sim2real/stage5').iterdir()]
out['reconstruction_verified']={}
for arm in ('eventgenesis','v2e'):
 root=b/'runs/sim2real/20261019-stage5-e2vid-tc-aligntrue-seed0'/arm
 s=json.loads((root/'run_summary.json').read_text());best=json.loads((root/'synthetic_best.json').read_text())
 log=root/'training_metrics.jsonl';pt=root/'synthetic_best.pt'
 out['reconstruction_verified'][arm]={'summary_sha256':sha(root/'run_summary.json'),'complete':s['status']=='complete','updates':s['optimizer_update'],'epochs':s['completed_epochs'],'best_epoch':s['best_epoch'],'synthetic_validation_lpips_vgg':s['best_synthetic_validation_lpips_vgg'],'best_checkpoint_sha256_actual':sha(pt),'best_metadata':best,'log_sha256_actual':sha(log),'log_records':sum(1 for _ in log.open())}
print(json.dumps(out))
'''
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','Hostname=10.129.161.238','-o','HostKeyAlias=g1.pkuml.org','zhaoyuhan@g1.pkuml.org']
cmd=shlex.join(['/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(ssh+[cmd],input=base64.b64encode(CODE.encode()),capture_output=True,timeout=100)
if r.returncode:print(r.stderr.decode());r.check_returncode()
d=json.loads(r.stdout);dst=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real/benchmark/sim2real/evidence/20261024-research-audit')
(dst/'bank_and_reconstruction_verification.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
print(json.dumps({k:d[k] for k in ('observed_utc','bank_hash_matches','actual_sha256','progress','related_stage5_runs','related_stage5_data','reconstruction_verified')},indent=2))
print(json.dumps({'first_episode_schema':next(iter(d['hdf5']['episode_schema'].values()))},indent=2))
