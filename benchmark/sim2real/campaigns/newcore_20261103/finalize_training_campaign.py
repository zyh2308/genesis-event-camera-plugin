"""Release temporary GPU5 after all runs; GPU3/4 queues retain their leases."""
import datetime,json,sys,time
from pathlib import Path
R=Path(sys.argv[1]);methods=['eventgenesis','v2e','vid2e','film_esim','rife_esim']
while True:
 files=[R/'training'/task/('seed'+str(seed))/m/'completion.json' for task in ['flow','reconstruction','pose'] for seed in range(3) for m in methods]
 if all(p.is_file() for p in files):
  rows=[json.loads(p.read_text()) for p in files];assert all(row['status']=='complete' for row in rows)
  result={'status':'all_45_training_runs_complete','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':rows,'retained_gpu_indices':[3,4],'temporary_gpu5_release_requested':True,'real_evaluation_completed':False}
  (R/'TRAINING_CAMPAIGN_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');(R/'campaign_finalized').write_text('Release temporary GPU5 when idle; retain GPU3/4.\n');break
 time.sleep(120)
