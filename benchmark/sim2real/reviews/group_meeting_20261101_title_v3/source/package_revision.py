from pathlib import Path
import shutil,zipfile,json,hashlib,re
from lxml import etree as E
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BUILD=Path(__file__).parent;OLD=ROOT/'outputs/Sim2Real_20261101_模板精简版';OUT=ROOT/'outputs/Sim2Real_20261101_标题预留版'
ppt=OUT/'EventGenesis_组会汇报_标题预留版.pptx';assert ppt.exists()
for p in OLD.rglob('*'):
 if p.is_file() and p.suffix!='.pptx' and p.name not in ('README.txt','SHA256SUMS.txt'):
  q=OUT/p.relative_to(OLD);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
index=OUT/'index.html';index.write_text(index.read_text(encoding='utf8').replace('EventGenesis_组会汇报_模板精简版.pptx',ppt.name),encoding='utf8')
(OUT/'README.txt').write_text('共9页：第1页为空白标题页，供自行填写；第2页是三项任务概览。\n正文沿用新模板，按01—08编号；最后一页补充三项实验的后续计划。\n图表、数值、三段视频未变；PPT图表仍可编辑。\n播放视频请保留同目录文件，也可打开index.html。\n',encoding='utf8')
for n in ['revise.mjs','preserve_chart_workbooks.py','package_revision.py']:shutil.copyfile(BUILD/n,OUT/'source'/n)
qa={}
for i in range(2,8):
 old=np.array(Image.open(ROOT/f'artifacts/meeting_template_v2/final_renders/slide_{i}.png'))
 new=np.array(Image.open(BUILD/f'renders/slide_{i+1}.png'))
 qa[f'unchanged_slide_{i+1}']=bool(np.array_equal(old,new))
assert all(qa.values())
first=np.array(Image.open(BUILD/'renders/slide_1.png'));assert np.all(first==255)
qa['blank_first_slide']=True
with zipfile.ZipFile(ppt) as z,zipfile.ZipFile(OLD/'EventGenesis_组会汇报_模板精简版.pptx') as old:
 slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)];assert len(slides)==9
 assert not E.fromstring(z.read('ppt/slides/slide1.xml')).xpath('//*[local-name()="t"]/text()')
 for n in old.namelist():
  if n.endswith('.xlsx') or (n.startswith('ppt/slides/charts/') and n.endswith('.xml')):assert z.read(n)==old.read(n),n
 qa['native_chart_parts_and_workbooks_unchanged']=True
 qa['slide_count']=9
(BUILD/'content_qa.json').write_text(json.dumps(qa,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt')+'\n')
archive=ROOT/'outputs/EventGenesis_组会材料_标题预留版.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in OUT.rglob('*'):
  if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'qa':qa},ensure_ascii=False))
