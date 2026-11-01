import sys,json,time
from pathlib import Path
import torch,numpy as np,h5py
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261101-epose-diverse-bank-v8';sys.path.insert(0,str(P/'external/sources/v2e-cac6384445/v2e-pinned'));from v2ecore.emulator import EventEmulator
torch.set_num_threads(8)
with h5py.File(P/'runs/sim2real/20261101-epose-wood-short-research-v1/bank/shared_trajectory.h5','r') as h:frames=np.asarray(h['episodes/train_00/v2e_display_luma_u8'])
e=EventEmulator(pos_thres=.2,neg_thres=.2,sigma_thres=.03,cutoff_hz=300.,leak_rate_hz=.01,leak_jitter_fraction=.1,noise_rate_cov_decades=.1,shot_noise_rate_hz=.001,refractory_period_s=.0005,seed=42,device='cpu',output_width=346,output_height=260);v=[]
for i,im in enumerate(frames):
 a=e.generate_events(im,i*.001)
 if a is not None:v.append(a)
e.cleanup();a=np.concatenate(v);assert len(a)>0 and np.isfinite(a).all() and a[:,0].max()<=.0040001
r={'passed':True,'device':'cpu','frames':len(frames),'synthetic_api_smoke_only':True,'timestamp_unit':'seconds','polarity_values':np.unique(a[:,3]).tolist(),'events':len(a)};(R/'v2e_api_preflight.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
