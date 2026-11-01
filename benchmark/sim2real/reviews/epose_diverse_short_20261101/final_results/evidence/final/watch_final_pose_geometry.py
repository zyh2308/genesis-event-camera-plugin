import os,time,subprocess,sys
from pathlib import Path
R=Path(os.environ['EPOSE_REPAIR_RUN'])
while not (R/'weights_frozen.json').exists():time.sleep(3)
with (R/'synthetic_pose_geometry_final.stdout').open('w') as a,(R/'synthetic_pose_geometry_final.stderr').open('w') as b:code=subprocess.run([sys.executable,str(R/'audit_synthetic_pose_geometry.py')],stdout=a,stderr=b).returncode
assert code==0
