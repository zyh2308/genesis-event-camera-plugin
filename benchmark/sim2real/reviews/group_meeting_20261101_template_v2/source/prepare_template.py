"""Clone the supplied slide's real furniture/master, remove former experiment content."""
import zipfile,copy,re,json,hashlib
from pathlib import Path
from lxml import etree as E
src=Path(r'C:\Users\zhaoyuhan\Desktop\Genesis事件插件_双线验证工作汇报_2026-09-14.pptx')
dst=Path(__file__).parent/'template_base.pptx'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
REL='http://schemas.openxmlformats.org/package/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types'
def b(r):return E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
with zipfile.ZipFile(src) as z,zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as out:
 slide=E.fromstring(z.read('ppt/slides/slide3.xml'))
 tree=slide.find('p:cSld/p:spTree',NS)
 for n in list(tree):
  if E.QName(n).localname not in ('nvGrpSpPr','grpSpPr','sp'):tree.remove(n)
 for n in list(slide):
  if E.QName(n).localname in ('timing','transition','extLst'):slide.remove(n)
 rels=E.fromstring(z.read('ppt/slides/_rels/slide3.xml.rels'))
 for n in list(rels):
  if not n.get('Type','').endswith('/slideLayout'):rels.remove(n)
 pres=E.fromstring(z.read('ppt/presentation.xml'));lst=pres.find('p:sldIdLst',NS);lst.clear()
 pr=E.fromstring(z.read('ppt/_rels/presentation.xml.rels'))
 for n in list(pr):
  if n.get('Type','').endswith('/slide') or n.get('Type','').endswith('/notesMaster'):pr.remove(n)
 for n in list(pres):
  if E.QName(n).localname in ('notesMasterIdLst','custShowLst','extLst'):pres.remove(n)
 ct=E.fromstring(z.read('[Content_Types].xml'))
 for n in list(ct):
  part=n.get('PartName','')
  if part.startswith('/ppt/slides/') or part.startswith('/ppt/notes') or n.get('Extension')=='mp4':ct.remove(n)
 for i in range(1,9):
  rid=f'rIdMeeting{i}'
  E.SubElement(lst,'{'+NS['p']+'}sldId',id=str(255+i),attrib={'{'+NS['r']+'}id':rid})
  E.SubElement(pr,'{'+REL+'}Relationship',Id=rid,Type=NS['r']+'/slide',Target=f'slides/slide{i}.xml')
  E.SubElement(ct,'{'+CT+'}Override',PartName=f'/ppt/slides/slide{i}.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
  out.writestr(f'ppt/slides/slide{i}.xml',b(slide));out.writestr(f'ppt/slides/_rels/slide{i}.xml.rels',b(rels))
 replacements={'[Content_Types].xml':b(ct),'ppt/presentation.xml':b(pres),'ppt/_rels/presentation.xml.rels':b(pr)}
 for n in z.namelist():
  if n in replacements:out.writestr(n,replacements[n]);continue
  if n.startswith('ppt/slides/') or n.startswith('ppt/notes') or n.endswith('.mp4'):continue
  out.writestr(n,z.read(n))
Path(__file__).with_name('template_reference.json').write_text(json.dumps({'path':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'source_slide':3,'preserved':'actual master, layout, header number/title/lines/page furniture','old_experiment_content_used':False},ensure_ascii=False,indent=2),encoding='utf8')
print(dst)
