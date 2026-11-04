"""Byte-preserving lightweight release/evidence bundle; large artifacts stay on g1."""
import hashlib,io,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert (ROOT/'FROZEN_RELEASE.json').exists() and (ROOT/'EFFICIENCY_COMPARISON_REPORT.md').exists()
assert json.loads((ROOT/'final_metrology.json').read_text())['passed']
assert json.loads((ROOT/'native_acceptance.json').read_text())['passed']
P=Path('/home/zhaoyuhan/Eventbased_WAM')
assert json.loads((P/'.gpu_reservations/simulator_efficiency_comparison_20261105c_gpu2.json').read_text())['status']=='released_complete'
payload={};external=[]
def add(path,name):
    assert name not in payload;payload[name]=path.read_bytes()
def ext(path,role='run_artifact'):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    external.append(dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest(),role=role))
for f in sorted(ROOT.rglob('*')):
    if not f.is_file() or f.is_symlink():continue
    rel=f.relative_to(ROOT);parts=rel.parts
    if '__pycache__' in parts or any(x.endswith('_cache') for x in parts) or parts[0] in ['render_cache','accepted_render_cache']:continue
    if parts[0]=='release_candidate':continue # initial prepared-only draft, not used in accepted results
    if parts[0]=='release_candidate_static':
        if f.name=='PREPARATION_MANIFEST.json':add(f,'evidence/initial_preparation_manifest.json')
        else:add(f,'release/'+str(f.relative_to(ROOT/'release_candidate_static')))
        continue
    if f.name in ['delivery.tar.gz'] or f.name.startswith('delivery_'):continue
    if f==ROOT/'EFFICIENCY_COMPARISON_REPORT.md':add(f,'report/'+f.name);continue
    if f.suffix=='.py':add(f,'source/'+str(rel));continue
    if len(parts)==1 and f.name in ['PLAN.md','RELEASE_README.md','REPRODUCE.md']:add(f,'source/'+f.name);continue
    if f.suffix in ['.json','.jsonl','.csv','.log','.running'] or f.name in ['build.ninja','.ninja_log']:
        if f.stat().st_size<=3*1024*1024:add(f,'evidence/'+str(rel))
        else:ext(f)
    elif f.suffix in ['.npy','.npz','.pt','.h5','.png','.so']:ext(f)
old=ROOT.parent/'simulator_gpu_resident_20261105b'
for f in [old/'long_baseline/trial0.npz',old/'long_baseline/summary.json',old/'strict_raw_gate.json']:
    ext(f,'prior_frozen_oracle')
for f,role in [(P/'external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt','baseline_checkpoint'),
               (P/'runs/robot_r1_5_sparse_time_repair_20261101a/fit/spatial_event_state.pt','fixed_regression_checkpoint'),
               (P/'runs/robot_r1_5_sparse_time_repair_20261101a/source/benchmark/robot_closed_loop/event_state_model.py','fixed_regression_source')]:ext(f,role)
add(ROOT/'FROZEN_RELEASE.json','release/RELEASE.json')
payload['evidence/external_artifacts.json']=(json.dumps(dict(artifacts=external,cache_directories_excluded_as_rebuildable=True),indent=2)+'\n').encode()
index=dict(server_root=str(ROOT),files=[dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(payload.items())])
payload['evidence/artifact_index.json']=(json.dumps(index,indent=2)+'\n').encode()
archive=ROOT/'delivery.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for name,data in sorted(payload.items()):
        item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;tar.addfile(item,io.BytesIO(data))
info=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(payload),external=len(external))
(ROOT/'delivery_transfer.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))
