"""Byte-preserving lightweight archive plus external large-artifact hashes."""
import datetime,hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent;P=Path('/home/zhaoyuhan/Eventbased_WAM');prior=P/'runs/simulator_efficiency_20261105a'
assert json.loads((root/'final_audit.json').read_text())['scope_passed']
lease=P/'.gpu_reservations/simulator_gpu_resident_20261105b_gpu2.json';assert json.loads(lease.read_text())['status']=='released_complete'
meta=root/'delivery_metadata';meta.mkdir(exist_ok=False)
for name,path in [('gpu_lease_final.json',lease),('frozen_baseline.json',P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json'),('sensor_profile.json',P/'runs/sim2real/20261103-stage9-direct-v1/sources/stage9_sensor_profile.json')]:
    (meta/name).write_bytes(path.read_bytes())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
mapping=[];external=[]
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.is_symlink():continue
    rel=path.relative_to(root);parts=rel.parts
    if parts[0] in ['delivery_metadata','queue','build']:continue
    if path.name=='GPU_RESIDENT_REPORT.md':mapping.append((path,'report/'+path.name))
    elif path.suffix in ['.py','.md'] and len(parts)==1:mapping.append((path,'source/'+path.name))
    elif path.name=='trace.json' or path.suffix in ['.npz','.npy','.pt','.h5','.so']:
        external.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    elif path.suffix in ['.json','.log','.csv']:mapping.append((path,'evidence/'+str(rel)))
binary=root/'build/frozen_sensor_fused.so';binary_sha=sha(binary)
old=json.loads((prior/'delivery_metadata/external_artifacts.json').read_text())['artifacts']
assert any(item['path']==str(binary.resolve()) and item['sha256']==binary_sha for item in old)
external.append(dict(path=str(binary.resolve()),alias=str(binary),bytes=binary.stat().st_size,sha256=binary_sha,origin_commit='c1d4bb8eb47d5e89611e99010a282880cc4f84f3'))
(meta/'external_artifacts.json').write_text(json.dumps(dict(server_root=str(root),artifacts=external),indent=2)+'\n')
mapping += [(p,'evidence/'+p.name) for p in sorted(meta.iterdir()) if p.is_file()]
index=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),server_root=str(root),files=[dict(path=destination,server_source=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path,destination in mapping],self_excluded=True)
(meta/'artifact_index.json').write_text(json.dumps(index,indent=2)+'\n');mapping.append((meta/'artifact_index.json','evidence/artifact_index.json'))
archive=root/'delivery.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as stream:
    for path,destination in mapping:stream.add(path,arcname=destination,recursive=False)
digest=sha(archive);print('ARCHIVE_FILES',len(mapping),'EXTERNAL_FILES',len(external));print('ARCHIVE_BYTES',archive.stat().st_size);print('ARCHIVE_SHA_A',digest[:32]);print('ARCHIVE_SHA_B',digest[32:])
