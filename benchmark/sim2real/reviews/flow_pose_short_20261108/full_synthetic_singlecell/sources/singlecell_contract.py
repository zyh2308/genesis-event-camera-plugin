"""Read-only eligibility and exact fixed-matrix merge checks."""
import hashlib,json
from pathlib import Path
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def valid_cell(v,b):
    return (v.get('method')==b['method'] and v.get('seed')==b['seed'] and v.get('checkpoint_sha256')==b['selected_checkpoint_sha256'] and
        v.get('window_count')==528 and v.get('rollout_count')==8 and v.get('optimizer_updates')==0 and v.get('real_data_used') is False and
        len(v.get('rollouts',[]))==8 and all(len(r.get('windows',[]))==66 for r in v['rollouts']) and
        v.get('parity',{}).get('passed') is True and abs(v['parity']['current_epe_px']-v['parity']['original_epe_px'])<=1e-5)
def fallback_admission(c):
    root=Path(c['origin_root']);need(sha(root/'sources/source_manifest.json')==c['origin_source_manifest_sha256'],'Original frozen sources changed')
    release=read(root/'parent_gpu0/resource_release.json')
    need(release['status']=='released' and release['stopped_all_own_group_members'] is True,'OriginalGPU0 parent must be fully stopped/released first')
    target=root/'seed01/rife_esim_seed1.json';complete=False
    if target.exists():
        try:complete=valid_cell(read(target),c['checkpoint_binding']['rife_esim/seed1'])
        except (ValueError,KeyError,TypeError):pass
    need(not complete,'Original target cell already complete; conditional fallback must not run')
    digest=hashlib.sha256(json.dumps(c['checkpoint_binding'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    need(digest==c['origin_checkpoint_binding_sha256'],'Original15 frozen checkpoint bindings differ')
    return dict(original_release_sha256=sha(root/'parent_gpu0/resource_release.json'),original_target_complete=False,original_target_existing_sha256=sha(target) if target.exists() else None)
