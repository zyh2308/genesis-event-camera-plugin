from pathlib import Path
import torch,hashlib,json,datetime
B=Path('/home/zhaoyuhan/Eventbased_WAM');run=B/'runs/sim2real/20261029-stage5-hightexture-fixed-gtfit-g3-v2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device':'CPU map_location; no model forward, optimizer update or CUDA context','arms':{}}
for arm in ('eventgenesis','v2e'):
 p=run/'fixed_sample_fit'/f'{arm}_last.pt';cp=torch.load(p,map_location='cpu',weights_only=False);state=hashlib.sha256()
 for name,value in sorted(cp['state_dict'].items()):state.update(name.encode());state.update(value.detach().contiguous().numpy().tobytes())
 optim=cp['optimizer']['state'];steps=[int(v['step'].item()) for v in optim.values() if 'step' in v]
 out['arms'][arm]={'checkpoint_path':str(p),'checkpoint_sha256_actual':sha(p),'state_sha256_actual':state.hexdigest(),'state_sha256_expected':cp['state_sha256'],'state_matches':state.hexdigest()==cp['state_sha256'],'optimizer_states':len(optim),'optimizer_unique_steps':sorted(set(steps)),'updates':cp['updates'],'generator':cp['generator'],'sample_id':cp['sample_id'],'data_manifest_sha256':cp['data_manifest_sha256'],'config_sha256':cp['config_sha256'],'no_one_update_resume_claimed':True}
 assert out['arms'][arm]['state_matches'] and set(steps)=={200}
 del cp
print(json.dumps(out))
