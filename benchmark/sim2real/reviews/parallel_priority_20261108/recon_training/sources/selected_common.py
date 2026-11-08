"""Shared fail-closed contracts; no CUDA initialization or scoring on import."""
import hashlib,json,os
from pathlib import Path
METHODS=('eventgenesis','v2e','vid2e','film_esim','rife_esim');SEEDS=(0,1,2)
SEQUENCES=('boxes','reflective_materials','still_life')
CORE='62da4f35ce3c5c96f04268bd1687a27657af9238'
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.'+str(os.getpid())+'.partial')
    q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def record(p):
    p=Path(p).resolve(strict=True);return dict(path=str(p),size_bytes=p.stat().st_size,sha256=sha(p))
def check(r):
    p=Path(r['path']);need(p.is_file() and p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],'Bound file changed: '+str(p));return p
def sources(root):
    m=read(root/'sources/source_manifest.json')
    for n,h in m.items():need(Path(n).name==n and sha(root/'sources'/n)==h,'Changed source '+n)
    return sha(root/'sources/source_manifest.json')
def plan_check(p):
    need(p['schema']=='recon_selected_full_plan_v1' and p['methods']==list(METHODS) and p['seeds']==list(SEEDS),'Changed matrix')
    need(p['training_budget_updates']==12000 and p['epochs']==200 and p['gate_updates']==300,'Changed budgets')
    need(p['real_sequences']==list(SEQUENCES) and p['real_result_scope']=='posthoc_selected_hqf3_development','Changed real scope')
    need(p['loss']==dict(lpips_vgg=1.,temporal=2.,auxiliary_l1=0.) and p['final_activation']=='sigmoid','Unapproved loss/model')
    need(p['real_scoring_enabled'] is False,'Training bundle cannot authorize real scoring')
    return p
def cursor_check(cp,update):
    need(cp['schema']=='e2vid_stage4_resume_v1' and cp['optimizer_update']==update,'Wrong resume schema/update')
    e,b=cp['next_epoch'],cp['next_batch_in_epoch']
    need(isinstance(e,int) and isinstance(b,int) and 0<=e<=200 and 0<=b<60 and e*60+b==update,'Wrong epoch/batch cursor')
    need(len(cp['optimizer_step_counts'])==30 and set(cp['optimizer_step_counts'])=={update},'Wrong Adam counters')
    need(set(cp['rng_state'])=={'python','numpy','torch_cpu','torch_cuda'} and len(cp['rng_state']['torch_cuda'])==1,'Incomplete RNG')
