from pathlib import Path
import json,hashlib,statistics,subprocess,urllib.request,urllib.error,concurrent.futures
from datetime import datetime,timezone
R=Path('//wsl.localhost/Ubuntu-22.04/home/科研/Eventbased_WAM/repositories/gec-sim2real')
B=R/'benchmark/sim2real';D=B/'evidence/S5/20261019-parallel-diagnostics'
out={'observed_utc':datetime.now(timezone.utc).isoformat()}
idx=json.loads((D/'parallel_evidence_index.json').read_text())
out['source_hash_checks']={k:hashlib.sha256((R/k).read_bytes()).hexdigest()==v['sha256'] for k,v in idx['new_source_files'].items()}
out['json_structure']={}
paths=['flow_input_perturbation_results.json','pose_generic_mechanism_result.json','gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80/fit_run/summary.json','gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80/posthoc_active_eval/summary.json','gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80/scale_target_audit/summary.json']
for rel in paths:
 d=json.loads((D/rel).read_text())
 out['json_structure'][rel]={k:(list(v) if isinstance(v,dict) else f'list[{len(v)}]' if isinstance(v,list) else v) for k,v in d.items()}
 out['json_structure'][rel]['sha256']=hashlib.sha256((D/rel).read_bytes()).hexdigest()
 if isinstance(d.get('arms'),dict):
  out['json_structure'][rel]['arm_keys']={a:list(v) for a,v in d['arms'].items()}
 p=D/rel
 if '/fit_run/' in rel:
  out['fit_logs']={}
  for arm in ['eventgenesis','v2e']:
   rows=[json.loads(x) for x in (p.parent/f'{arm}_updates.jsonl').read_text().splitlines() if x.strip()]
   ids=sorted(set(r['sample_id'] for r in rows))
   out['fit_logs'][arm]={'n':len(rows),'ids':ids,'contiguous': [r['update'] for r in rows]==list(range(1,81)),
        'first':rows[0],'last':rows[-1],
        'first_cycle_loss':statistics.mean(r['loss'] for r in rows[:8]),'last_cycle_loss':statistics.mean(r['loss'] for r in rows[-8:])}
out['git_diff_core']=subprocess.run(['wsl','-d','Ubuntu-22.04','--','git','-C','/home/科研/Eventbased_WAM/repositories/gec-sim2real','diff','61554fb0031609200bc5ddbf2340ec8d20722725','HEAD','--name-only'],capture_output=True,text=True,encoding='utf-8').stdout.splitlines()
def probe(url):
 req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'},method='GET')
 try:
  with urllib.request.urlopen(req,timeout=18) as r:
   head=r.read(2048)
   return {'url':url,'method':'GET','status':r.status,'final_url':r.url,'content_type':r.headers.get('Content-Type'),'waf_action':r.headers.get('x-amzn-waf-action'),'read_bytes':len(head),'prefix':head[:180].decode(errors='replace')}
 except urllib.error.HTTPError as e:
  return {'url':url,'method':'GET','status':e.code,'content_type':e.headers.get('Content-Type'),'waf_action':e.headers.get('x-amzn-waf-action')}
 except Exception as e:return {'url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 out['windows_epose_get']=list(pool.map(probe,['https://api.figshare.com/v2/articles/28106717','https://springernature.figshare.com/articles/dataset/E-POSE_A_Large_Scale_Event_Camera_Dataset_for_Object_Pose_Estimation/28106717']))
Path(__file__).with_suffix('.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
