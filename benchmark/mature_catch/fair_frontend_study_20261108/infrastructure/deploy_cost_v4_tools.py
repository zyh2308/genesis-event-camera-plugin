"""Deploy sealed cost tools and checkpoint helper through the existing relay."""
from pathlib import Path
import ast, base64, hashlib, json

local = Path(__file__).resolve().parent
root = '/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/fair_frontend_blind_20261108b'
tools = local/'cost_design/v4'
manifest = json.loads((tools/'SOURCE_SHA256.json').read_text(encoding='utf-8'))
for name, digest in manifest['files'].items():
    assert hashlib.sha256((tools/name).read_bytes()).hexdigest() == digest
files = {'cost_tools_v4/'+p.name:p.read_bytes() for p in tools.iterdir() if p.is_file()}
for name in ['checkpoint_blind_for_cost_v4.py','CHECKPOINT_COST_V4_INDEX.json','CHECKPOINT_COST_V4_HANDOFF.md']:
    files['tools/'+name] = (local/name).read_bytes()
expected = {name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
payload = {name:base64.b64encode(data).decode() for name,data in files.items()}
code = '''from pathlib import Path
import base64,hashlib,json,socket
assert socket.gethostname().split('.')[0]=='g1'
root=Path(ROOT_VALUE)
assert (root/'FAIR_RUNTIME.json').exists() and not (root/'cost_tools_v4').exists()
payload=PAYLOAD_VALUE;expected=EXPECTED_VALUE
for name,data in payload.items():
 path=root/name;assert not path.exists(),str(path)
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(base64.b64decode(data))
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[name]
print(json.dumps({'status':'DEPLOYED_NOT_STARTED','files':expected}))
'''.replace('ROOT_VALUE',repr(root)).replace('PAYLOAD_VALUE',repr(payload)).replace('EXPECTED_VALUE',repr(expected))
ast.parse(code)
queue=local/'remote_queue_g1'
target=queue/'049_deploy_cost_v4_tools.request.py'
assert not target.exists()
draft=target.with_suffix('.draft');draft.write_text(code,encoding='utf-8');draft.rename(target)
print(target)
