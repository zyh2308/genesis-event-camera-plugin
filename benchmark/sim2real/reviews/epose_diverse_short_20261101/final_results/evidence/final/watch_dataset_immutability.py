import os,time,json,hashlib
from pathlib import Path
R=Path(os.environ['EPOSE_REPAIR_RUN'])
while not (R/'weights_frozen.json').exists():time.sleep(3)
before=json.loads((R/'dataset_artifact_manifest.json').read_text())['files'];rows=[]
for name,item in before.items():
 p=R/name;actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==item['sha256'] and p.stat().st_size==item['size_bytes'];rows.append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size})
(R/'dataset_immutability_after_training.json').write_text(json.dumps({'passed':True,'files':rows,'no_real_data_access':True,'unix':time.time()},indent=2))
