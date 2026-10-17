from pathlib import Path
import hashlib,json,shutil,subprocess
O=Path(__file__).resolve().parent
R=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real');S=R/'benchmark/sim2real';D=S/'reviews/stage4_preflight_followup'
D.mkdir(parents=True,exist_ok=True)
for p in O.iterdir():
    if p.is_file() and p.name!='artifact_manifest.json':shutil.copy2(p,D/p.name)
name='STAGE4_CORRECTNESS_REPAIR_AND_PREFLIGHT.md'
shutil.copy2(O/name,S/name)
old=S/'STAGE4_TRAINING_START.md';text=old.read_text(encoding='utf-8')
notice='> **2026-10-17 研究验收修正：长训练放行条件已由 [STAGE4_CORRECTNESS_REPAIR_AND_PREFLIGHT.md](STAGE4_CORRECTNESS_REPAIR_AND_PREFLIGHT.md) 更新。旧 Stage 4 数据/代码组合未通过正确性复核；先完成该文件的前置检查，通过后按既有授权自主继续，不能直接复用旧组合重跑。**\n\n'
if notice not in text:
    title,rest=text.split('\n',1);old.write_text(title+'\n\n'+notice+rest.lstrip('\n'),encoding='utf-8')
files=[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(D.iterdir()) if p.is_file() and p.name!='artifact_manifest.json']
(D/'artifact_manifest.json').write_text(json.dumps({'files':files,'root_handoff':{'path':name,'sha256':hashlib.sha256((S/name).read_bytes()).hexdigest()}},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert (S/name).read_bytes()==(D/name).read_bytes()
assert all(hashlib.sha256((D/f['file']).read_bytes()).hexdigest()==f['sha256'] for f in files)
verdict=json.loads((D/'gate_verdict.json').read_text())
print(json.dumps({'root_handoff':str(S/name),'review':str(D/'REVIEW.md'),'captured_evidence_verdict':verdict['status'],'archived_files':len(files)+1,'git_status':subprocess.run(['git','-C',str(R),'status','--short'],text=True,capture_output=True,check=True).stdout},ensure_ascii=False))
