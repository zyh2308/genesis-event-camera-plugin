"""Export official Git-tracked baseline files and exact commits."""
import hashlib,json,subprocess,tarfile,sys
from pathlib import Path
base=Path(sys.argv[1]);output=Path(sys.argv[2]);rows=[]
with tarfile.open(output,'w:gz') as tar:
 for name in ['vid2e_ssm','rife','film']:
  root=base/name;commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True);assert not dirty.strip()
  files=[]
  for item in subprocess.check_output(['git','-C',str(root),'ls-files','-z']).split(b'\0'):
   if not item:continue
   rel=item.decode();path=root/rel
   if not path.is_file():continue
   files.append({'path':rel,'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});tar.add(path,arcname=name+'/'+rel,recursive=False)
  rows.append({'name':name,'commit':commit,'files':files})
manifest=output.with_suffix('.manifest.json');manifest.write_text(json.dumps({'schema':'stage9_official_baseline_sources_v1','repositories':rows,'archive_sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2)+'\n');print(output.stat().st_size,manifest)
