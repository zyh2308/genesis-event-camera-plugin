import os,sys
from pathlib import Path
import h5py,numpy as np,cv2
R=Path(os.environ['EPOSE_REPAIR_RUN']);P=Path('/home/zhaoyuhan/Eventbased_WAM');sys.path.insert(0,str(P/'runs/sim2real/20261101-epose-wood-short-research-v1/sources/benchmark/sim2real/scripts'));import run_pose_stage2_pilot as pose
out=R/'input_previews';out.mkdir(exist_ok=True)
with h5py.File(R/'bank/shared_trajectory.h5','r') as h:
 for name in ['train_00','train_08','train_16','validation_00']:
  for arm in ['eventgenesis','v2e']:
   t,x,y,p,_=pose.read_event_table(arm,name,h,R/'bank/v2e_events');inside=(t>.08)&(t<=.09);im=np.full((260,346,3),245,np.uint8);im[y[inside],x[inside]]=(220,80,35);on=inside&(p>0);im[y[on],x[on]]=(40,55,210);cv2.imwrite(str(out/f'{name}_{arm}.png'),im)
