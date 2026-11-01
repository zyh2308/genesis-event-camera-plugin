"""Cross-check result values and rendered package, then copy to Windows desktop."""
import csv,json,hashlib,subprocess,shutil,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse,unquote
O=Path('/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/outputs/epose_diverse_short_20261101')
D=Path('/mnt/c/Users/zhaoyuhan/Desktop');A=D/'EventGenesis_组会素材_20261101';A.mkdir(exist_ok=True)
S=json.loads((O/'evidence/final/real_eval/summary.json').read_text());R=S['results']
assert json.loads((O/'evidence/final/real_eval/independent_verification.json').read_text())['passed']
assert json.loads((O/'evidence/final/independent_synthetic_verification.json').read_text())['passed']
P=O/'EventGenesis_组会汇报_EPOSE初测更新版.pptx'
N={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
video_links=[];chart_values=[]
with zipfile.ZipFile(P) as z:
 slides=[n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml') and n.count('/')==2]
 assert len(slides)==9
 first=ET.fromstring(z.read('ppt/slides/slide1.xml'));assert not any((x.text or '').strip() for x in first.findall('.//a:t',N))
 slide8=ET.fromstring(z.read('ppt/slides/slide8.xml'));table_text=[x.text or '' for x in slide8.findall('.//a:tbl//a:t',N)]
 for arm in S['eligible_new_arms']:
  value=100*R[arm]['adds_success_0_1_d_fraction_all_samples'];assert f'{value:.1f}%' in table_text
  adds=R[arm]['adds_mean_over_valid_pnp_m'];expected='—' if adds is None else (f'{100*adds:.1e}' if 100*adds>=1e4 else f'{100*adds:.2f}');equivalents=[expected,expected.replace('e+0','e+').replace('e-0','e-')];assert any(v in table_text for v in equivalents),(arm,expected,table_text)
 for name in z.namelist():
  if '/charts/' in name and name.endswith('.xml'):
   root=ET.fromstring(z.read(name))
   for series in root.findall('.//c:ser',N):
    categories=[x.text for x in series.findall('.//c:cat//c:pt/c:v',N)]
    values=[float(x.text) for x in series.findall('.//c:val//c:pt/c:v',N)]
    if categories==['EventGenesis','V2E']:chart_values.append(values)
  if name.endswith('.rels'):
   for relationship in ET.fromstring(z.read(name)):
    target=relationship.attrib.get('Target','')
    if target.lower().endswith('.mp4'):video_links.append(target)
 expected=[R[a]['corner_epe_all_samples_px'] for a in ['eventgenesis','v2e'] if a in R]
 assert any(len(v)==len(expected) and all(abs(a-b)<1e-5 for a,b in zip(v,expected)) for v in chart_values),(chart_values,expected)
assert len(video_links)==3,video_links
for link in video_links:
 parsed=urlparse(link);p=unquote(parsed.path)
 assert parsed.scheme=='file' and p.startswith('/C:/'),link
 local=Path('/mnt/c')/p[4:];assert local.name in ['01_reconstruction_HQF.mp4','02_flow_ReplicaCAD.mp4','EPOSE_真实预测_配对短训.mp4']
 # The new E-POSE clip is copied below; existing clips were staged earlier.
 if local.name!='EPOSE_真实预测_配对短训.mp4':assert local.is_file(),local
with (O/'EPOSE_真实初测_指标.csv').open(encoding='utf-8-sig',newline='') as f:
 rows=list(csv.DictReader(f))
assert len(rows)==len(R)
for row in rows:
 arm=('eventgenesis' if row['模型']=='EventGenesis' else 'v2e')+('_v1_control' if row['版本']=='初次短训' else '')
 assert abs(float(row['角点误差_px'])-R[arm]['corner_epe_all_samples_px'])<1e-10
 assert int(row['ADD-S成功数_分母200'])==R[arm]['adds_success_0_1_d_count']
from PIL import Image
figures=['合成场景_扩充后.png','合成事件_配对样例.png','合成定位_新留出集.png','真实位姿初测_主指标.png','真实预测_固定第51窗.png']
for name in figures:assert Image.open(O/name).size==(1280,720)
files=figures+['EPOSE_真实初测_指标.csv','EPOSE_服务器短训_核验报告.md','EPOSE_真实预测_配对短训.mp4','组会讲稿_EPOSE更新.txt','posttest_domain_gap.json']
for name in files:shutil.copy2(O/name,A/name)
shutil.copy2(P,D/P.name)
videos={}
for name in ['01_reconstruction_HQF.mp4','02_flow_ReplicaCAD.mp4','EPOSE_真实预测_配对短训.mp4']:
 result=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_read_frames:format=duration','-of','json',str(A/name)],text=True));videos[name]=result
stream=videos['EPOSE_真实预测_配对短训.mp4']['streams'][0];assert int(stream['nb_read_frames'])==200 and (stream['width'],stream['height'])==(1280,720) and stream['codec_name']=='h264'
manifest={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [D/P.name]+[A/name for name in files+['01_reconstruction_HQF.mp4','02_flow_ReplicaCAD.mp4']]}
receipt={'passed':True,'first_slide_blank':True,'slides':9,'editable_pose_chart_matches_raw_result':True,'editable_pose_table_matches_raw_result':True,'csv_matches_raw_result':True,'all_video_link_targets_exist':True,'videos':videos,'files':manifest,'PowerPoint_application_tested':False}
(O/'final_delivery_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));(A/'文件核验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps({'passed':True,'ppt':str(D/P.name),'assets':str(A),'results':{a:R[a] for a in S['eligible_new_arms']}},ensure_ascii=False,indent=2))
