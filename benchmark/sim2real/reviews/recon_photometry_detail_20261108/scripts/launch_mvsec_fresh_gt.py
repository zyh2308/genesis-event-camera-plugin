import datetime,hashlib,json,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
script=p/'resume_mvsec_fresh_gt.py'
root=Path('/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-EINexus-left-resume-driver-20261108-v2')
root.mkdir(exist_ok=False)
command=[sys.executable,'-u','-B',str(script),'--execute','--writer-check-confirmed']
with (root/'driver.log').open('x') as log:
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=child.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=command,
    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),source_registration_unchanged=True,
    parent_preservation_receipt='/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-EINexus-left-20261107-v1/gt_mismatch_preserved_20261108-v1.json',
    GPU_used=False,real_scores_computed=False)
(root/'submitted.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
