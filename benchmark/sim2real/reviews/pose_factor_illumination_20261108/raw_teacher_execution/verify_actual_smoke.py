"""Independent CPU model/Adam readback; no optimizer update and no resume claim."""
import hashlib, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode=True
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
root=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-pose-baf-raw-teacher-short-v1')
out=root/'actual_smoke_readback.json'
assert not out.exists()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
sources=root/'sources'
assert sha(sources/'source_manifest.json')=='bdd07e6d176374f9f37dad34731e894a33b17307cfaaa4e0b1e93991e46eddc3'
for name,digest in json.loads((sources/'source_manifest.json').read_text()).items(): assert sha(sources/name)==digest
sys.path.insert(0,str(sources))
import run_raw_teacher as worker
from pose_priority_policy import state_sha
s=json.loads((root/'smoke/completion.json').read_text())
worker.verify_smoke(s,sha(sources/'source_manifest.json'))
c=json.loads((sources/'contract.json').read_text())
spec=importlib.util.spec_from_file_location('readback_export',Path(c['diagnostic_source_dir'])/'export_pose_factors.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
ctx=E.load_context(json.loads(Path(c['base_contract_path']).read_text()),check_gpu=False)
checks=[]
for row in s['records']:
    m=row['method'];seed=row['seed']
    raw=torch.load(row['final_model']['path'],map_location='cpu',weights_only=True)
    cp=Path(row['checkpoint_path']);assert sha(cp)==row['checkpoint_sha256']
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    assert saved['update']==1 and state_sha(raw)==state_sha(saved['model'])==row['final_state_sha256']
    assert sorted({int(v['step']) for v in saved['optimizer']['state'].values()})==[1]
    models=[]
    for state in [raw,saved['model']]:
        model=ctx.scale.CenterScale().eval();model.load_state_dict(state,strict=True);model.size.requires_grad_(False);models.append(model)
    optimizer=torch.optim.AdamW([p for p in models[1].parameters() if p.requires_grad],lr=.00015,weight_decay=.01)
    optimizer.load_state_dict(saved['optimizer'])
    for a,b in zip(saved['optimizer']['state'].values(),optimizer.state.values()):
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    admission=next(r for r in s['full288_admissions'] if (r['method'],r['seed'])==(m,seed))
    initial=torch.load(admission['source_model']['path'],map_location='cpu',weights_only=True)
    assert state_sha(initial)==row['source_state_sha256']
    assert all(torch.equal(v,initial[k]) for k,v in raw.items() if k.startswith('size.'))
    # No data-dependent selection: fixed first four prepared TRAIN windows.
    cache=next(r for r in s['input_caches'] if Path(r['path']).name==f'{m}_inputs128.pt'); assert sha(cache['path'])==cache['sha256']; cached=torch.load(cache['path'],map_location='cpu',weights_only=False)
    x=cached['data']['normal'][:4]
    with torch.inference_mode(): a,b=[ctx.native.estimate_centers(model,x) for model in models]
    assert __import__('numpy').isfinite(a).all() and __import__('numpy').array_equal(a,b)
    checks.append(dict(method=m,seed=seed,model_state_sha256=state_sha(raw),checkpoint_sha256=sha(cp),Adam_steps=[1],frozen_size_exact=True,CPU_forward_max_abs=float(abs(a-b).max())))
result=dict(status='passed',schema='independent_RAW_teacher_BAF_smoke_readback_v1',checks=checks,new_optimizer_updates=0,GPU_used=False,
            resume_supported=False,scope='model and Adam reconstruction; fixed TRAIN forward only; no resumed next step or whole-trajectory claim',
            receipt_sha256=sha(root/'smoke/completion.json'),source_manifest_sha256=sha(sources/'source_manifest.json'),verifier_sha256=sha(__file__))
out.write_text(json.dumps(result,indent=2)+'\n')
print('ACTUAL_BAF_SMOKE_READBACK_PASSED',len(checks),flush=True)
