import hashlib, json, zipfile
from pathlib import Path
root=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-pose-baf-raw-teacher-short-v1')
R=root.parent/'20261103-stage9-direct-v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
files={};external=[]
for phase in ['smoke','study']:
    for p in (root/phase).rglob('*'):
        if not p.is_file():continue
        if p.suffix in ['.json','.jsonl']:files[str(p.relative_to(root))]=p.read_bytes()
        elif p.suffix=='.pt' and '128' not in p.name:external.append(dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p)))
for p in root.glob('*'):
    if p.is_file() and p.suffix in ['.json','.log','.py']:files[p.name]=p.read_bytes()
for job in ['164_pose_baf_raw_teacher_smoke_20261108','165_pose_baf_raw_teacher_study_20261108']:
    p=R/'gpu4'/(job+'.done.json');files['queue/'+p.name]=p.read_bytes()
for phase in ['smoke','study']:
    p=R/'recovery_20261108_pose_baf_raw_teacher_queue'/phase/'stdout.log';files['queue/'+phase+'_stdout.log']=p.read_bytes()
for name in ['rt_summary','rt_rid_summary']:
    if name in globals():files['derived/'+name+'.json']=(json.dumps(globals()[name],indent=2)+'\n').encode()
c=json.loads((root/'study/completion.json').read_text())
index=dict(schema='actual_RAW_teacher_BAF_center_short_final_v1',files={name:dict(size_bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for name,b in files.items()},
           external_checkpoints=external,external_input_caches=c['input_caches']+c['raw_teacher_caches'],
           external_scope='Checkpoint bytes hashed during collection. Input caches use hashes verified by actual study before training and again at completion; not rehashed by this collector.',
           actual_source_manifest_sha256=sha(root/'sources/source_manifest.json'),real_scores_computed=False)
files['RAW_TEACHER_FINAL_INDEX.json']=(json.dumps(index,indent=2)+'\n').encode()
archive=root/'RAW_TEACHER_SHORT_FINAL.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,b in files.items():z.writestr(name,b)
print(json.dumps(dict(archive=str(archive),size_bytes=archive.stat().st_size,sha256=sha(archive),files=len(files)),indent=2))
