from pathlib import Path
import ast,json,hashlib,datetime
r=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real')
root=r/'benchmark/sim2real'
p=root/'evidence/S5/20261021-recon-flow-localization/continuation_evidence_index_20261022.json'
d=json.loads(p.read_text());rows=d['artifacts'];fail=[];checked=0;external=0
for row in rows:
 if not row.get('in_repository'):external+=1;continue
 f=r/row['path']
 if not f.is_file():fail.append({'path':row['path'],'issue':'missing'});continue
 data=f.read_bytes()
 expected_size=row.get('size_bytes',row.get('bytes'))
 if len(data)!=expected_size or hashlib.sha256(data).hexdigest()!=row['sha256']:fail.append({'path':row['path'],'issue':'size/hash mismatch'})
 checked+=1
diagnostics=[]
for name,fn in [('train_stage5_flow_selfsup_diagnostic.py','registered_code_fingerprints'),('register_stage5_flow_selfsup_diagnostic_config.py',None)]:
 f=root/'scripts'/name;t=ast.parse(f.read_text())
 bad=[]
 for n in ast.walk(t):
  if isinstance(n,ast.Dict):
   for k,v in zip(n.keys,n.values):
    if isinstance(k,ast.Constant) and isinstance(k.value,str) and k.value.endswith(name) and isinstance(v,ast.Name) and v.id=='HERE':bad.append({'line':v.lineno,'key':k.value,'value':'HERE','resolved':str(f.parent),'is_file':f.parent.is_file(),'is_directory':f.parent.is_dir()})
 diagnostics.append({'file':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'findings':bad})
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'index':str(p),'entries':len(rows),'repository_checked':checked,'external_entries':external,'failures':fail,'source_path_diagnostics':diagnostics}
dest=root/'evidence/20261024-research-audit';dest.mkdir(exist_ok=True)
(dest/'local_evidence_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
