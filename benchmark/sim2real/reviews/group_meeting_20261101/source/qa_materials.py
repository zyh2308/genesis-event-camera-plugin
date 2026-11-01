import json,subprocess,hashlib,zipfile,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/Sim2Real_20261101';BUILD=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'videos':{},'data_checks':{},'pptx':{},'local_references':{}}
m=json.loads((OUT/'provenance/export_manifest.json').read_text());assert not m['errors']
for p in OUT.glob('*.mp4'):
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)]))
 v=next(s for s in probe['streams'] if s['codec_type']=='video')
 assert (v['width'],v['height'],v['codec_name'],v['pix_fmt'])==(1280,720,'h264','yuv420p')
 checked=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],capture_output=True,text=True)
 assert checked.returncode==0 and not checked.stderr,checked.stderr
 assert sha(p)==m['artifact_hashes'][p.name]['sha256']
 report['videos'][p.name]={'duration':float(probe['format']['duration']),'frames':int(v['nb_frames']),'width':v['width'],'height':v['height'],'decode_ok':True,'server_sha256_match':True}
d=json.loads((OUT/'results.json').read_text());f=np.load(OUT/'provenance/flow_predictions_and_gt.npz')
for i,mask in enumerate([f['valid'].astype(bool),f['common_active'].astype(bool)]):
 for arm in ['eventgenesis','v2e']:
  value=float(np.hypot(*(f[arm].astype(float)-f['gt']))[mask].mean());expected=d['flow'][arm][i];assert abs(value-expected)<1e-5
  report['data_checks'][f'flow_{arm}_mask{i}']={'recomputed':value,'reference':expected,'passed':True}
report['data_checks']['flow_common_mask_pixels']=int(f['common_active'].sum())
assert d['reconstruction']['frames_per_arm']==15498 and len(d['reconstruction']['sequences'])==14
ppt=OUT/'EventGenesis_Sim2Real_组会汇报.pptx'
with zipfile.ZipFile(ppt) as z:
 names=z.namelist();report['pptx']={'slides':len([n for n in names if re.fullmatch(r'ppt/slides/slide\d+.xml',n)]),'native_charts':len([n for n in names if re.fullmatch(r'.*/chart\d+.xml',n)]),'embedded_workbooks':len([n for n in names if n.endswith('.xlsx')]),'native_tables':sum(z.read(n).count(b'<a:tbl>') for n in names if re.fullmatch(r'ppt/slides/slide\d+.xml',n)),'sha256':sha(ppt)}
 assert report['pptx']['slides']==8 and report['pptx']['native_charts']==3 and report['pptx']['embedded_workbooks']==3 and report['pptx']['native_tables']==3
 refs=[]
 for n in names:
  if n.endswith('.rels'):
   for target in re.findall(rb'Target="([^"]+[.]mp4)"',z.read(n)):
    t=target.decode();assert (OUT/t).is_file();refs.append(t)
 report['pptx']['video_links']=refs;assert len(refs)==3
for ref in re.findall(r'(?:src|href)="([^"]+)"',(OUT/'index.html').read_text()):
 if not ref.startswith('#'):assert (OUT/ref).is_file(),ref
report['local_references']['html_passed']=True
(BUILD/'media_and_data_qa.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
