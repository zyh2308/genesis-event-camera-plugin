from pathlib import Path
import subprocess,base64,shlex,os
code='''from pathlib import Path
import urllib.request,tarfile,hashlib,json,subprocess
b=Path('/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1')
url='https://www.rarlab.com/rar/rarlinux-x64-723.tar.gz'
p=b/'tools/rarlinux-x64-723.tar.gz'
if not p.exists():
 urllib.request.urlretrieve(url,str(p))
out=b/'tools/rar-official-7.23'
out.mkdir(exist_ok=True)
with tarfile.open(p) as t:
 for m in t.getmembers():
  if m.name=='rar/unrar':
   f=t.extractfile(m); binary=out/'unrar';binary.write_bytes(f.read());binary.chmod(0o755)
r=subprocess.run([str(binary)],capture_output=True,text=True)
receipt={'source_url':url,'package_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'version_output':r.stdout[:800]}
(b/'manifests/unrar_tool.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''
remote=shlex.join(['/usr/bin/python3','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run((['wsl','-d','Ubuntu-22.04','--'] if os.name=='nt' else [])+['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','Hostname=10.129.161.238','-o','HostKeyAlias=g1.pkuml.org','zhaoyuhan@g1.pkuml.org',remote],input=base64.b64encode(code.encode()),capture_output=True,timeout=100)
print(r.stdout.decode());print(r.stderr.decode());r.check_returncode()
