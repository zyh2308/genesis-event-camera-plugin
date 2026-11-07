import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--campaign',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
cache=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261107-pose-physical-light-train-v2/radiance_train')
completion=json.loads((cache/'completion.json').read_text())
assert completion['status']=='complete' and not completion['real_data_used']
assert len(completion['selected'])==8 and all(x.startswith('train_') for x in completion['selected'])
assert a.out.parent.resolve()==Path(__file__).resolve().parents[1] and not a.out.exists()
a.out.symlink_to(cache,target_is_directory=True)
(a.out.parent/'reused_training_data.json').write_text(json.dumps(dict(source=str(cache),completion_sha256=hashlib.sha256((cache/'completion.json').read_bytes()).hexdigest(),read_only_reuse=True,real_data_used=False),indent=2))
print('REUSED_COMPLETE_PAIRED_SYNTHETIC_LIGHT_BANK',flush=True)
