"""Seal a benchmark-only fork; reuse the frozen server core, models and inputs."""
from pathlib import Path
import ast,base64,difflib,hashlib,json,subprocess

HERE=Path(__file__).resolve().parent;LOCAL=HERE.parent
W=Path('//wsl.localhost/Ubuntu-22.04/home/科研/Eventbased_WAM/repositories/gec-robot-catch-fair-frontend-20261108')
ROOT='/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/fair_frontend_blind_20261108b'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(b):return hashlib.sha256(b).hexdigest()

base=read(LOCAL/'FAIR_RUNTIME.json');runtime=read(LOCAL/'BASE_RUNTIME.json')
source=W/'benchmark/mature_catch';files={p.name:p.read_bytes() for p in source.glob('*.py')}
hashes={n:digest(b) for n,b in files.items()}
changed={n for n,h in base['runtime_files'].items() if hashes.get(n)!=h}
assert changed=={'run_original.py','sparse_event_sensor.py'}
assert set(hashes)-set(base['runtime_files'])=={'endpoint_event_worker.py'}
assert not set(base['runtime_files'])-set(hashes)
old={name:(HERE/'base_source_snapshot'/name).read_bytes() for name in changed}
for name,data in old.items():assert digest(data)==base['runtime_files'][name]
control=b'                while True:'
assert old['run_original.py'].count(control)==files['run_original.py'].count(control)==1
assert old['run_original.py'].split(control,1)[1]==files['run_original.py'].split(control,1)[1]
for name,data in files.items():ast.parse(data,filename=name)
commit=subprocess.check_output(['wsl.exe','-d','Ubuntu-22.04','--','git','-C',
    '/home/科研/Eventbased_WAM/repositories/gec-robot-catch-fair-frontend-20261108','rev-parse','HEAD'],text=True).strip()
fingerprint=digest(json.dumps(hashes,sort_keys=True).encode());remote_source=ROOT+'/source_endpoint_only_'+fingerprint[:12]
patch=''
for name in sorted(changed|{'endpoint_event_worker.py'}):
    before=old.get(name,b'').decode();after=files[name].decode()
    patch+=''.join(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
        fromfile='a/'+name if name in old else '/dev/null',tofile='b/'+name))
receipt=dict(changed_files=sorted(changed),added_files=['endpoint_event_worker.py'],
    files={n:dict(old_sha256=base['runtime_files'].get(n),new_sha256=hashes[n]) for n in changed|{'endpoint_event_worker.py'}},
    base_source=base['source'],source=remote_source,base_code_commit=base['code_commit'],code_commit=commit,
    unified_diff_file=ROOT+'/ENDPOINT_SOURCE_DELTA.patch',unified_diff_sha256=digest(patch.encode()),
    review=dict(allowed_hunks_only=True,control_policy_scoring_unchanged=True,
        actual_control_loop_tail_byte_identical=True,frozen_17_core_files_unchanged=True,
        scope='Benchmark selector and protocol evidence only; endpoint worker directly uses original frozen radiometry and pixel response. Every other runtime file remains byte-identical.',
        per_packet_checks='mode endpoint_only; zero intermediate frames; endpoint_reads=i+1 and DVS updates=i; original end+10ms availability.'))
receipt_bytes=json.dumps(receipt,indent=2).encode()
runtime.update(root=ROOT,source=remote_source,runtime_files=hashes,code_commit=commit,
    base_source=base['source'],base_code_commit=base['code_commit'],event_reconstruction_mode='endpoint_only',
    endpoint_source_delta_receipt=ROOT+'/ENDPOINT_SOURCE_DELTA.json',endpoint_source_delta_sha256=digest(receipt_bytes))
runtime['template']=[v.replace(base['source'],remote_source) if isinstance(v,str) else v for v in runtime['template']]
toolnames=['make_endpoint_protocol.py','advance_endpoint_ablation.py','collect_endpoint_ablation.py','plot_endpoint_ablation.py']
tools={n:base64.b64encode((HERE/n).read_bytes()).decode() for n in toolnames}
toolhash={n:digest((HERE/n).read_bytes()) for n in toolnames}
overlays={n:base64.b64encode(files[n]).decode() for n in changed|{'endpoint_event_worker.py'}}
for name,data in [('ENDPOINT_RUNTIME.json',json.dumps(runtime,indent=2).encode()),('ENDPOINT_SOURCE_DELTA.json',receipt_bytes),('ENDPOINT_SOURCE_DELTA.patch',patch.encode())]:
    (HERE/name).write_bytes(data)
code='''from pathlib import Path
import base64,hashlib,json,shutil,socket,subprocess
assert socket.gethostname().split('.')[0]=='g1'
root=Path(ROOT_VALUE);meta=META_VALUE;base=BASE_VALUE
source=Path(meta['source']);assert not source.exists();source.mkdir()
for name,want in base['runtime_files'].items():
 old=Path(base['source'])/name;assert hashlib.sha256(old.read_bytes()).hexdigest()==want,name
 shutil.copyfile(old,source/name)
for name,data in OVERLAYS_VALUE.items():(source/name).write_bytes(base64.b64decode(data))
for name,want in meta['runtime_files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==want,name
for name,want in meta['core_files'].items():assert hashlib.sha256((Path(meta['core_root'])/'genesis_event_plugin'/name).read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest()==want,name
for field,key in [('checkpoint','checkpoint_sha256'),('profile_file','profile_sha256')]:assert hashlib.sha256(Path(meta[field]).read_bytes()).hexdigest()==meta[key]
for name,data in ARTIFACTS_VALUE.items():
 with (root/name).open('xb') as f:f.write(base64.b64decode(data))
tools=root/'endpoint_ablation_tools_v1';tools.mkdir(exist_ok=False)
for name,data in TOOLS_VALUE.items():
 p=tools/name;p.write_bytes(base64.b64decode(data));assert hashlib.sha256(p.read_bytes()).hexdigest()==TOOLHASH_VALUE[name]
python='/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python'
command=[python,'-B',str(tools/'make_endpoint_protocol.py'),'--main-protocol',str(root/'blind32/EVALUATION_PROTOCOL.json'),'--runtime',str(root/'ENDPOINT_RUNTIME.json'),'--output',str(root/'endpoint_only32_20261109a')]
subprocess.run(command,check=True,timeout=30)
print(json.dumps({'status':'SEALED_AND_REGISTERED_NOT_EXECUTED','source':str(source),'runtime_files':len(meta['runtime_files']),'code_commit':meta['code_commit'],'protocol':str(root/'endpoint_only32_20261109a/EVALUATION_PROTOCOL.json'),'tools':TOOLHASH_VALUE}))
'''
artifacts={n:base64.b64encode((HERE/n).read_bytes()).decode() for n in ['ENDPOINT_RUNTIME.json','ENDPOINT_SOURCE_DELTA.json','ENDPOINT_SOURCE_DELTA.patch']}
for key,value in [('ROOT_VALUE',ROOT),('META_VALUE',runtime),('BASE_VALUE',base),('OVERLAYS_VALUE',overlays),('ARTIFACTS_VALUE',artifacts),('TOOLS_VALUE',tools),('TOOLHASH_VALUE',toolhash)]:
    code=code.replace(key,repr(value))
ast.parse(code)
q=LOCAL/'remote_queue_g1';target=q/'060_seal_endpoint_source.request.py';assert not target.exists()
draft=target.with_suffix('.draft');draft.write_text(code,encoding='utf-8');draft.rename(target)
print(json.dumps(dict(request=str(target),source=remote_source,commit=commit,runtime_files=len(hashes))))
