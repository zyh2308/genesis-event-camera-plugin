import os,time,json
from pathlib import Path
R=Path(os.environ['EPOSE_REPAIR_RUN']);pids=json.loads((R/'training_processes.json').read_text())['pids'];records={};starts={}
for pid in pids:
 txt=Path(f'/proc/{pid}/stat').read_text();tail=txt[txt.rfind(')')+2:].split();starts[pid]=tail[19]
 assert str(R) in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
while len(records)<len(pids):
 for pid in pids:
  if pid in records:continue
  path=Path(f'/proc/{pid}/stat')
  if not path.exists():records[pid]={'exit_code':None,'reason':'process disappeared before zombie status observed'};continue
  txt=path.read_text();tail=txt[txt.rfind(')')+2:].split();assert tail[19]==starts[pid]
  if tail[0]=='Z':
   status=int(tail[49]);records[pid]={'exit_code':os.waitstatus_to_exitcode(status),'raw_linux_wait_status':status,'start_ticks':starts[pid]}
 time.sleep(2)
(R/'training_process_exit_codes.json').write_text(json.dumps({'passed':all(v['exit_code']==0 for v in records.values()),'processes':records,'method':'Linux proc stat field52 raw wait status for own verified child PIDs, captured before parent reaps','unix':time.time()},indent=2))
