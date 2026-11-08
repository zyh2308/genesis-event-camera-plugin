from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',choices=['old','cadence'],required=True);a=p.parse_args();r=a.root;base=r.parent;recovery=base/'serial_recovery_20261108b';python='/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python';os.nice(10);os.sched_setaffinity(0,[120,121]);os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
if a.dataset=='old':
 while not (r/'DATA_COMPLETION_RECEIPT.json').exists():time.sleep(10)
else:
 while True:
  original=r/'catchit_cadence_pair_shard0_shared_g1_20261108a.result.json'
  if original.exists():assert json.loads(original.read_text())['exit_code']==0
  if original.exists() and (recovery/'RECOVERY_COMPLETE.json').exists():break
  time.sleep(15)
 spec=json.loads((r/'EVALUATION_PROTOCOL.json').read_text());module_path=Path(spec['source'])/'run_cadence_comparison.py';assert hashlib.sha256(module_path.read_bytes()).hexdigest()==spec['worker_sha256'];ms=importlib.util.spec_from_file_location('cadence_source_verifier',module_path);m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m);m.verify(spec)
 valid={}
 for path in r.glob('shard*_progress.json'):
  for x in json.loads(path.read_text()):
   if x['exit_code']==0:valid[x['case_id']+'|'+x['method']]=x['output']
 redirects=json.loads((r/'RECOVERY_MAP.json').read_text())
 for key,x in redirects.items():assert x['exit_code']==0;valid[key]=x['output']
 expected=[case['case_id']+'|'+method for case in spec['cases'] for method in case['method_order']];assert len(expected)==44 and all(key in valid for key in expected)
 for key in expected:assert (Path(valid[key])/'episodes.json').exists()
 (r/'DATA_COMPLETION_RECEIPT.json').write_text(json.dumps({'status':'PASS','expected_unique_episodes':44,'valid_completed_unique_episodes':44,'frozen_source_policy_profile_hashes_verified':True,'recovered_outputs':redirects,'original_failed_attempts_retained':True},indent=2))
names=['analyze_paired.py','make_paired_media_server.py','audit_paired_final.py'] if a.dataset=='old' else ['analyze_cadence.py','make_cadence_media_server.py']
for name in names:
 with (r/(name+'.final.log')).open('w') as log:subprocess.run([python,'-u',str(r/name),'--root',str(r)],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
if a.dataset=='cadence':
 redirects=json.loads((r/'RECOVERY_MAP.json').read_text())
 for shard in [0,1]:
  selection=json.loads((r/'world_demo'/('shard'+str(shard)+'_selection.json')).read_text())
  for method in ['ours','v2e']:
   key=selection['case_id']+'|'+method;run=Path(redirects.get(key,{}).get('output',str(r/'cases'/selection['case_id']/method)))
   if not (run/'episode_000/trace.json').exists():continue
   out=r/'world_demo'/(selection['case_id']+'_'+method)
   with (out/'event_panel.log').open('w') as log:subprocess.run([python,'-u',str(r/'render_arrived_event_packets.py'),'--run',str(run),'--output',str(out)],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
 with (r/'audit_cadence_final.py.final.log').open('w') as log:subprocess.run([python,'-u',str(r/'audit_cadence_final.py'),'--root',str(r)],cwd=r,stdout=log,stderr=subprocess.STDOUT,check=True)
(r/'POSTPROCESS_COMPLETE.json').write_text(json.dumps({'status':'PASS','dataset':a.dataset,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_aborted_attempts_preserved':True}));print(json.dumps({'status':'PASS','dataset':a.dataset}),flush=True)
