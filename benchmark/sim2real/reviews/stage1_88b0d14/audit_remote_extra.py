import base64,json,pathlib,shlex,subprocess
out=pathlib.Path(__file__).resolve().parent
pin=json.loads(pathlib.Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real/benchmark/sim2real/manifests/v2e_source_pin.json').read_text())
program=r'''
import base64,hashlib,json,pathlib
pin=json.loads(base64.b64decode('__PIN__'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
root=pathlib.Path('/home/zhaoyuhan/Eventbased_WAM')
src=root/'external/sources/v2e-cac6384445/v2e-pinned'
bad=[i['path'] for i in pin['files'] if not (src/i['path']).is_file() or digest(src/i['path'])!=i['sha256']]
slomo=root/'external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt'
runtime=root/'runtime/sim2real/genesis_hdr_v2/runtime_manifest.json'
d={'V2E_files_checked':len(pin['files']),'V2E_mismatches':bad,'SuperSloMo_weight_sha_matches':digest(slomo)==pin['slomo_checkpoint']['sha256'],'runtime_manifest_exists':runtime.is_file()}
if runtime.is_file():
    m=json.loads(runtime.read_text());d['runtime_manifest_keys']=list(m);d['runtime_manifest_small_fields']={k:v for k,v in m.items() if isinstance(v,(str,int,float,bool))};d['runtime_manifest_file_entry_examples']={k:v[:1] for k,v in m.items() if isinstance(v,list)}
print(json.dumps(d))
'''.replace('__PIN__',base64.b64encode(json.dumps(pin).encode()).decode())
code='import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'
cmd=shlex.join(['/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c',code])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode(program.encode()).decode(),capture_output=True,text=True,timeout=60)
(out/'remote_extra.json').write_text(r.stdout)
print(r.stdout or r.stderr)
