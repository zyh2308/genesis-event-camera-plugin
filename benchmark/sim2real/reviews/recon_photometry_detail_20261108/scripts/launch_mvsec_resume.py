import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent;script=p/'resume_mvsec_outdoor.py'
expected='4b42262410087e73acdd6569dfdbf6bd09b76971a00850e9920a2cfe1b2ca05b'
assert hashlib.sha256(script.read_bytes()).hexdigest()==expected
root=Path('/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-EINexus-left-resume-driver-20261108-v1')
root.mkdir(exist_ok=False)
command=[sys.executable,'-u','-B',str(script),'--execute','--writer-check-confirmed']
with (root/'driver.log').open('x') as log:
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=child.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=command,script_sha256=expected,
    parent_remote_writer_checked_utc='2026-11-08T04:43:46.270698+00:00',no_GPU=True,no_training=True,no_real_scoring=True)
(root/'submitted.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
