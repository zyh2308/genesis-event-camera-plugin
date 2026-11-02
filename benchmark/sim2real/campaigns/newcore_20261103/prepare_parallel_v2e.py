"""Create an isolated, disjoint two-worker V2E producer for the larger bank."""
from pathlib import Path
import sys
R=Path(sys.argv[1]);s=R/'sources';text=(s/'replay_v2e_events.py').read_text()
text=text.replace("a=p.parse_args();", "p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=2)\na=p.parse_args();")
begin=text.index("for f in sorted((bank/'physical').glob('*.h5')):")
text=text[:begin]+'''if a.worker is None:
 processes=[]
 for slot in range(a.workers):
  log=(out/('worker%d.log'%slot)).open('wb')
  processes.append(subprocess.Popen([sys.executable,'-u',__file__,'--campaign',str(R),'--kind',a.kind,'--worker',str(slot),'--workers',str(a.workers)],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy()));log.close()
 codes=[p.wait() for p in processes];assert all(c==0 for c in codes),codes
 rows=[json.loads((out/(f.stem+'.json')).read_text()) for f in sorted((bank/'physical').glob('*.h5'))]
 for row in rows:assert row['source_physical_manifest_sha256']==physical_hash and hashlib.sha256((out/(row['rollout_id']+'.npy')).read_bytes()).hexdigest()==row['events_sha256']
 (out/'manifest.json').write_text(json.dumps({'schema':'stage9_v2e_official_ssm_v1','generator':'v2e','source_commit':pin['commit'],'physical_manifest_sha256':physical_hash,'rollouts':rows,'real_data_used':False},indent=2)+'\\n')
 raise SystemExit(0)
''' +text[begin:]
text=text.replace("for f in sorted((bank/'physical').glob('*.h5')):","for index,f in enumerate(sorted((bank/'physical').glob('*.h5'))):\n if index%a.workers!=a.worker:continue")
text=text.replace("(out/'progress.json').write_text", "(out/('worker%d_progress.json'%a.worker)).write_text")
text=text.replace("(out/'manifest.json').write_text(json.dumps({'schema':'stage9_v2e_official_ssm_v1','generator':'v2e','source_commit':'cac638444557a224ba6ee5c29260916f152c00bd'", "(out/('worker%d_manifest.json'%a.worker)).write_text(json.dumps({'schema':'stage9_v2e_official_ssm_v1','generator':'v2e','source_commit':'cac638444557a224ba6ee5c29260916f152c00bd'")
import ast;ast.parse(text);(s/'replay_v2e_parallel.py').write_text(text)
