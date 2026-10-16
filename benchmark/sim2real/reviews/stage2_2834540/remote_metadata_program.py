from pathlib import Path
import hashlib,json,subprocess
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real');B=R/'benchmark/sim2real'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
bundle=json.loads((B/'manifests/downstream_source_bundle.json').read_text()); checks=[]
for repo in bundle['repositories']:
    bad=[f['path'] for f in repo['files'] if not (Path(repo['server_path'])/f['path']).is_file() or sha(Path(repo['server_path'])/f['path'])!=f['sha256']]
    checks.append({'repo':repo['repo'],'files':len(repo['files']),'mismatches':bad})
out={'downstream_source_integrity':checks,'code_sha256':{p.name:sha(p) for p in (B/'scripts').glob('*stage2*.py')},'result_jsons':{},'real_directory_inventory':{}}
ext=json.loads((B/'evidence/S2/20261016-stage2-pilot/external_artifacts.json').read_text())
for d in ext['server_run_directories']:
    for p in Path(d).rglob('result.json'):out['result_jsons'][str(p)]=sha(p)
for name in ['HQF','MVSEC','E-POSE']:
    p=Path('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/real')/name
    out['real_directory_inventory'][name]={'exists':p.exists(),'files':sum(1 for x in p.rglob('*') if x.is_file()) if p.exists() else 0}
print(json.dumps(out))
