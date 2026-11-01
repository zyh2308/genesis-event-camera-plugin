"""Restore authentic supplied XLSX bindings dropped by PPTX import/export.

No slide authoring occurs here. Charts were edited through artifact-tool. This
adapter preserves the original literal workbooks, updates their referenced
cells to the authored chart caches, and restores their original OPC bindings.
"""
import sys,io,json,copy,hashlib,posixpath,re,zipfile
from pathlib import Path
from lxml import etree as E
import openpyxl
from openpyxl.utils.cell import range_boundaries
source,candidate,output=map(Path,sys.argv[1:4]);C='http://schemas.openxmlformats.org/drawingml/2006/chart';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';P='http://schemas.openxmlformats.org/package/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types';N={'c':C}
with zipfile.ZipFile(source) as z:original={n:z.read(n) for n in z.namelist()}
with zipfile.ZipFile(candidate) as z:parts={n:z.read(n) for n in z.namelist()}
charts=sorted(n for n in original if '/charts/' in n and n.endswith('.xml') and E.fromstring(original[n]).tag==f'{{{C}}}chartSpace')
assert len(charts)==3;repairs=[]
for chart in charts:
 assert chart in parts;old=E.fromstring(original[chart]);new=E.fromstring(parts[chart]);assert [n.text for n in old.findall('.//c:f',N)]==[n.text for n in new.findall('.//c:f',N)]
 assert not new.findall('./c:externalData',N),'Preserve existing bindings rather than replacing them.'
 external=old.findall('./c:externalData',N);assert len(external)==1;relpart=posixpath.join(posixpath.dirname(chart),'_rels',posixpath.basename(chart)+'.rels');rels=E.fromstring(original[relpart]);rid=external[0].get(f'{{{R}}}id');relation=next(x for x in rels if x.get('Id')==rid);assert relation.get('Type')==R+'/package' and relation.get('TargetMode','Internal')=='Internal'
 workbook=posixpath.normpath(posixpath.join(posixpath.dirname(chart),relation.get('Target')));assert workbook.startswith('ppt/embeddings/') and workbook in original
 wb=openpyxl.load_workbook(io.BytesIO(original[workbook]),data_only=False);assert not any(cell.data_type=='f' for sheet in wb for row in sheet for cell in row),'Formula-bearing source requires a formula-preserving adapter.'
 edits=[];bindings=[]
 for node in new.findall('.//c:strRef',N)+new.findall('.//c:numRef',N):
  formula=node.find('c:f',N).text;m=re.fullmatch(r"'([^']+)'!(\$?[A-Z]+\$?\d+:\$?[A-Z]+\$?\d+)",formula);assert m,formula;sheet,area=m.groups();assert sheet in wb;minc,minr,maxc,maxr=range_boundaries(area.replace('$',''));coords=[(rr,cc) for rr in range(minr,maxr+1) for cc in range(minc,maxc+1)];assert minc==maxc or minr==maxr
  cache=node.find('c:strCache',N) if node.tag.endswith('strRef') else node.find('c:numCache',N);assert cache is not None;count=int(cache.find('c:ptCount',N).get('val'));assert count==len(coords);pts={int(pt.get('idx')):pt.find('c:v',N).text for pt in cache.findall('c:pt',N)};assert set(pts)==set(range(count))
  for i,(rr,cc) in enumerate(coords):
   value=pts[i] if node.tag.endswith('strRef') else float(pts[i]);cell=wb[sheet].cell(rr,cc);prior=cell.value
   same=prior==value if isinstance(value,str) else isinstance(prior,(int,float)) and abs(prior-value)<1e-12
   if not same:edits.append({'sheet':sheet,'cell':cell.coordinate,'old':prior,'new':value});cell.value=value
   bindings.append((sheet,rr,cc,value))
 # Keep workbook column headers aligned with an explicitly changed series name.
 for series in new.findall('.//c:ser',N):
  tx=series.find('c:tx/c:v',N);ref=series.find('c:val/c:numRef/c:f',N)
  if tx is not None and ref is not None:
   m=re.fullmatch(r"'([^']+)'!\$?([A-Z]+)\$?(\d+):\$?[A-Z]+\$?\d+",ref.text);assert m;sheet,col,start=m.groups();cell=wb[sheet][col+str(int(start)-1)]
   if cell.value!=tx.text:edits.append({'sheet':sheet,'cell':cell.coordinate,'old':cell.value,'new':tx.text});cell.value=tx.text
 if edits:
  data=io.BytesIO();wb.save(data);workbook_bytes=data.getvalue()
 else:workbook_bytes=original[workbook]
 check=openpyxl.load_workbook(io.BytesIO(workbook_bytes),data_only=False)
 for sheet,rr,cc,value in bindings:
  actual=check[sheet].cell(rr,cc).value;assert actual==value if isinstance(value,str) else abs(actual-value)<1e-9
 assert not any(cell.data_type=='f' for sheet in check for row in sheet for cell in row)
 position=next((i for i,x in enumerate(new) if E.QName(x).localname in ['printSettings','userShapes','extLst']),len(new));new.insert(position,copy.deepcopy(external[0]));parts[chart]=E.tostring(new,xml_declaration=True,encoding='UTF-8',standalone=True)
 if relpart in parts:
  existing=E.fromstring(parts[relpart]);assert all(x.get('Id')==rid and x.get('Target')==relation.get('Target') for x in existing)
 parts[relpart]=original[relpart];parts[workbook]=workbook_bytes;repairs.append({'chart':chart,'source_workbook':workbook,'source_workbook_sha256':hashlib.sha256(original[workbook]).hexdigest(),'restored_workbook_sha256':hashlib.sha256(workbook_bytes).hexdigest(),'source_formula_cells':0,'intentional_literal_cell_edits':edits,'original_relationship_restored':True})
content=E.fromstring(parts['[Content_Types].xml']);xlsx=[x for x in content if x.get('Extension')=='xlsx'];assert len(xlsx)<=1
mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
if not xlsx:E.SubElement(content,f'{{{CT}}}Default',Extension='xlsx',ContentType=mime)
else:assert xlsx[0].get('ContentType')==mime
parts['[Content_Types].xml']=E.tostring(content,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in parts.items():z.writestr(name,data)
receipt={'passed':True,'scope':'restore actual supplied workbook parts and bindings after artifact-tool export; no invented formulas or source lineage','source':str(source),'candidate':str(candidate),'output':str(output),'charts':repairs};output.with_suffix('.workbook_restoration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False,indent=2))
