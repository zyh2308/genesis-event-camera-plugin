import json, hashlib, time
from pathlib import Path
import numpy as np, h5py, scipy.io as sio
from scipy.spatial.transform import Rotation, Slerp
BASE=Path("/home/zhaoyuhan/Eventbased_WAM")
D=BASE/"data/real/epose/figshare_28106717_v1/extracted"
MAT=D/"block/block/r_01_gl"; H5=D/"original_data/original_data/block/r_01_gl/data.h5"
OUT=BASE/"runs/sim2real/20261029-epose-second-sample-audit/camera_pose_trajectory_v3.json"
def m4(x):
 x=np.asarray(x,float); return x if x.shape==(4,4) else np.vstack([x,[0,0,0,1]])
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
with h5py.File(H5,"r") as f:
 cp=np.asarray(f["camera_pose"],float); e0=float(f["events_parallel"][3,0])
t=cp[7]
q=np.stack([cp[4],cp[5],cp[6],cp[3]],axis=1)
rot=Rotation.from_quat(q)
out={"schema":"epose_camera_pose_timestamp_alignment_v3","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"source_time_mapping":"source sets events t := t-0.100, then subtracts adjusted first event t0=events_parallel[0]-0.100 from both events and camera_pose; therefore raw_camera_pose_time_for_mat_endpoint = first_raw_event_time - 0.100 + MAT.time","first_raw_event_time_s":e0,"camera_pose_time_range_raw_absolute_s":[float(t.min()),float(t.max())],"inputs":{"h5":str(H5),"h5_sha256":sha(H5)},"samples":{}}
for i in (1,2):
 p=MAT/f"{i:04d}_data.mat"; d=sio.loadmat(p)
 tm=float(np.asarray(d["time"]).reshape(-1)[0]); target=e0-0.1+tm
 j=int(np.searchsorted(t,target,side="right"))
 if j==0 or j>=len(t): raise RuntimeError(f"sample {i} outside pose trajectory")
 a,b=j-1,j; alpha=(target-t[a])/(t[b]-t[a])
 pos=(1-alpha)*cp[:3,a]+alpha*cp[:3,b]
 r=Slerp(t[[a,b]],rot[[a,b]])([target])[0]
 expected=np.eye(4); expected[:3,:3]=r.as_matrix(); expected[:3,3]=pos
 saved=m4(d["cameraPose"]); rd=r.inv()*Rotation.from_matrix(saved[:3,:3])
 out["samples"][str(i)]={"mat_path":str(p),"mat_sha256":sha(p),"mat_label_time_s":tm,"expected_raw_camera_pose_time_s":float(target),"bracket_time_s":[float(t[a]),float(t[b])],"bracket_index":[a,b],"fraction":float(alpha),"expected_camera_translation_m":pos.tolist(),"saved_camera_translation_m":saved[:3,3].tolist(),"translation_error_m":float(np.linalg.norm(expected[:3,3]-saved[:3,3])),"rotation_error_deg":float(np.degrees(rd.magnitude())),"matrix_max_abs_error":float(np.max(np.abs(expected-saved)))}
out["sample_pair_cameraPose_equal_max_abs"]=float(np.max(np.abs(m4(sio.loadmat(MAT/"0001_data.mat")["cameraPose"])-m4(sio.loadmat(MAT/"0002_data.mat")["cameraPose"]))))
out["real_scoring_enabled"]=False
p=OUT; p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True)); print("result_sha256="+sha(p))
