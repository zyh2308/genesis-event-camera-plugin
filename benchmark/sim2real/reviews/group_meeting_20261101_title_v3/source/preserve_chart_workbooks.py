"""Preserve unchanged source charts and their original editable workbooks."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];BUILD=Path(__file__).parent
src=ROOT/'outputs/Sim2Real_20261101_模板精简版/EventGenesis_组会汇报_模板精简版.pptx'
candidate=BUILD/'candidate.pptx';temp=BUILD/'candidate_preserved.pptx'
with ZipFile(src) as z,ZipFile(candidate) as c:
 names=[n for n in z.namelist() if n.startswith('ppt/slides/charts/') or n.endswith('.xlsx')]
 charts=[n for n in names if n.endswith('.xml') and '/_rels/' not in n]
 assert len(charts)==3
 for n in charts:
  assert n in c.namelist()
  # Source data formulas must still match after the unrelated page edits.
  get=lambda a:E.fromstring(a).xpath('//*[local-name()="f"]/text()')
  assert get(z.read(n))==get(c.read(n))
 parts={n:z.read(n) for n in names}
 ct=E.fromstring(c.read('[Content_Types].xml'));source_ct=E.fromstring(z.read('[Content_Types].xml'))
 for n in source_ct:
  if n.get('Extension')=='xlsx' and not any(x.get('Extension')=='xlsx' for x in ct):ct.append(n)
  elif n.get('PartName','').endswith('.xlsx') and not any(x.get('PartName')==n.get('PartName') for x in ct):ct.append(n)
 parts['[Content_Types].xml']=E.tostring(ct,xml_declaration=True,encoding='UTF-8',standalone=True)
 with ZipFile(temp,'w',ZIP_DEFLATED) as out:
  for n in c.namelist():
   if n not in parts:out.writestr(n,c.read(n))
  for n,b in parts.items():out.writestr(n,b)
candidate.write_bytes(temp.read_bytes())
(BUILD/'chart_preservation.json').write_text(json.dumps({n:hashlib.sha256(parts[n]).hexdigest() for n in names},indent=2))
