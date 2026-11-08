"""Verify every completed episode and archived raw event packet, once."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 spec=json.loads((r/'EVALUATION_PROTOCOL.json').read_text());data=json.loads((r/'PAIRED_PORTABLE.json').read_text());assert data['complete'] and not data['violations']
 records=[]
 for row in data['records']:
  run=Path(row['output']);sensor=run/'migration_audit/sensor_episode_000';method=row['mode'];entry={'case':row['case_id'],'method':method,'verified_catch':row['verified_catch'],'raw_time_limit_success_flag':row['result']['success'],'terminated':row['result'].get('terminated',False),'terminal_flag_conflict':row['terminal_flag_conflict'],'actual_contact_steps':row['result'].get('hand_contact_steps',0)}
  if method!='state_reference':
   packets=json.loads((sensor/'packet_audit.json').read_text());events=np.load(sensor/'raw_events.npz')['events'];assert events.shape==(sum(x['events'] for x in packets),4)
   if method=='init_only':assert len(events)==0 and len(packets)==0
   offset=0
   for packet in packets:
    chunk=events[offset:offset+packet['events']];offset+=packet['events'];assert abs(packet['available_s']-packet['time_s']-.01)<1e-8
    if len(chunk):
     assert np.isfinite(chunk).all();assert chunk[:,0].max()<=packet['time_s']+1e-6
     assert ((chunk[:,1]>=0)&(chunk[:,1]<640)&(chunk[:,2]>=0)&(chunk[:,2]<480)).all()
     assert np.isin(chunk[:,3],[-1,1]).all()
    if method=='v2e':assert packet['worker_input_keys']==['gray8','time_s']
    assert packet['dense_HDR_reads']==0 and packet['renderer_calls_in_reconstruction']==0
   entry.update(raw_events=len(events),packets=len(packets),causal_packet_contract='PASS')
  records.append(entry)
 assert len(records)==sum(len(c['method_order']) for c in spec['cases'])
 completion=r/'DATA_COMPLETION_RECEIPT.json'
 if completion.exists():
  receipt=json.loads(completion.read_text());assert receipt['status']=='PASS' and receipt['valid_completed_unique_episodes']==len(records) and receipt['frozen_source_policy_profile_hashes_verified']
 else:
  for shard in sorted({int(c['shard']) for c in spec['cases']}):assert json.loads((r/f'shard{shard}_complete.json').read_text())['postflight_hashes_pass']
 def file_sha(path,canonical=False):
  b=Path(path).read_bytes()
  return hashlib.sha256(b.replace(b'\r\n',b'\n') if canonical else b).hexdigest()
 for n,digest in spec['runtime_files'].items():assert file_sha(Path(spec['source'])/n)==digest,n
 for n,digest in spec['core_files'].items():assert file_sha(Path(spec['core_root'])/'genesis_event_plugin'/n,True)==digest,n
 for field in ['checkpoint','profile_file','v2e_checkpoint']:
  digest_field='profile_sha256' if field=='profile_file' else field+'_sha256';assert file_sha(spec[field])==spec[digest_field]

 receipt={'status':'PASS','episodes':len(records),'raw_event_contract_and_packet_counts_pass':True,'all_defined_successes_and_failures_retained':True,'records':records}
 (r/'FINAL_AUDIT.json').write_text(json.dumps(receipt,indent=2))
 # Exclude the manifest itself and transient logs; preserve hashes of every raw input,
 # trace, native recording, protocol and closed result. The experiment is finished.
 candidates=set(p for p in r.rglob('*') if p.is_file())
 for row in data['records']:
  folder=Path(row['output'])
  if not str(folder).startswith(str(r)+'/'):candidates.update(p for p in folder.rglob('*') if p.is_file())
 recovery=json.loads((r/'RECOVERY_MAP.json').read_text()) if (r/'RECOVERY_MAP.json').exists() else {}
 for entry in recovery.values():
  if entry.get('actor_log'):candidates.add(Path(entry['actor_log']))
 paths=[p for p in candidates if p.name!='SHA256SUMS' and not p.name.endswith('.final.log') and not p.name.endswith('.supervisor.log') and p.name not in ['consolidated_postprocess.log','postprocess_coordinator.log']]

 lines=[]
 for path in sorted(paths):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  try:label=str(path.relative_to(r))
  except ValueError:label=str(path)
  lines.append(h.hexdigest()+'  '+label)
 (r/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'PASS','episodes':len(records),'hashed_files':len(lines),'total_bytes':sum(p.stat().st_size for p in paths)}))

if __name__=='__main__':main()
