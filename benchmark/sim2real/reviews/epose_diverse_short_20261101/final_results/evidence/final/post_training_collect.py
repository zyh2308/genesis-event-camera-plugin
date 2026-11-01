import os,sys,time,json,subprocess
from pathlib import Path
R=Path(os.environ['EPOSE_REPAIR_RUN'])
while not (R/'completion.json').exists():time.sleep(5)
while subprocess.run(['tmux','has-session','-t','research_epose_r10_training_20261101'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:time.sleep(2)
with (R/'collector.stdout').open('w') as a,(R/'collector.stderr').open('w') as b:code=subprocess.run([sys.executable,str(R/'sources/collect_diverse_evidence.py')],stdout=a,stderr=b).returncode
assert code==0
print('COLLECTION_COMPLETE',flush=True)
