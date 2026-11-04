"""Archive lightweight evidence and index large immutable server artifacts."""
import datetime,hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM')
assert json.loads((root/'final_audit.json').read_text())['passed']
assert json.loads((P/'.gpu_reservations/simulator_efficiency_20261105a_gpu2.json').read_text())['status']=='released_complete'
meta=root/'delivery_metadata';meta.mkdir(exist_ok=False)
(meta/'gpu_lease_final.json').write_bytes((P/'.gpu_reservations/simulator_efficiency_20261105a_gpu2.json').read_bytes())
(meta/'frozen_baseline.json').write_bytes((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_bytes())
(meta/'sensor_profile.json').write_bytes((P/'runs/sim2real/20261103-stage9-direct-v1/sources/stage9_sensor_profile.json').read_bytes())
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
mapping=[];external=[]
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.is_symlink():continue
    rel=path.relative_to(root);parts=rel.parts
    if parts[0] in ['delivery_metadata','queue']:continue
    if path.suffix in ['.py','.cpp','.cu'] and 'build' not in parts:
        if len(parts)==1:dest='source/v1/'+path.name
        elif parts[0]=='final_v2':dest='source/v2/'+path.name
        elif parts[0] in ['pre_geometry_sources','pre_early_sources']:dest='source/history/'+str(rel)
        else:continue
        mapping.append((path,dest))
    elif path.name=='BENCHMARK_REPORT.md':mapping.append((path,'report/BENCHMARK_REPORT.md'))
    elif path.name in ['README.md','IMPLEMENTATION_PLAN.md'] and len(parts)==1:mapping.append((path,'source/'+path.name))
    elif path.suffix in ['.json','.log','.csv'] and 'build' not in parts:
        mapping.append((path,'evidence/'+str(rel)))
    elif path.suffix in ['.npz','.npy','.pt','.so','.o','.prof']:
        external.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
(meta/'external_artifacts.json').write_text(json.dumps(dict(server_root=str(root),artifacts=external),indent=2)+'\n')
mapping += [(p,'evidence/'+p.name) for p in sorted(meta.iterdir()) if p.is_file()]
index=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),server_root=str(root),files=[dict(path=dest,server_source=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path,dest in mapping],self_excluded=True)
(meta/'artifact_index.json').write_text(json.dumps(index,indent=2)+'\n')
mapping.append((meta/'artifact_index.json','evidence/artifact_index.json'))
archive=root/'delivery.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as stream:
    for path,dest in mapping:stream.add(path,arcname=dest,recursive=False)
digest=sha(archive)
print('ARCHIVE_BYTES',archive.stat().st_size)
print('ARCHIVE_FILES',len(mapping),'EXTERNAL_FILES',len(external))
print('ARCHIVE_SHA_A',digest[:32]);print('ARCHIVE_SHA_B',digest[32:])
