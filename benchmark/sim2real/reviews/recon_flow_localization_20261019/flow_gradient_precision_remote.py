from pathlib import Path
import sys,json
import torch
torch.set_num_threads(2)
R=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-stage5-coverage-exec-20261019')
sys.path[:0]=[str(R/'benchmark/sim2real/scripts'),str(R/'benchmark/sim2real/compat')]
from stage5_flow_supervised_pilot import multiscale_supervised_epe
gt=torch.zeros(1,2,32,48,dtype=torch.float64);gt[:,0]=.75;gt[:,1]=-.375
valid=torch.ones(1,1,32,48,dtype=torch.float64)
shapes=[(32,48),(16,24),(8,12),(4,6)]
pred=[torch.zeros(1,2,*s,dtype=torch.float64,requires_grad=True) for s in shapes]
loss=multiscale_supervised_epe(pred,gt,valid)[0];loss.backward()
rows=[]
for i,s in enumerate(shapes):
 for c in [0,1]:
  idx=(0,c,s[0]//2,s[1]//2);eps=1e-4
  a=[p.detach().clone() for p in pred];b=[p.detach().clone() for p in pred]
  a[i][idx]+=eps;b[i][idx]-=eps
  numeric=float((multiscale_supervised_epe(a,gt,valid)[0]-multiscale_supervised_epe(b,gt,valid)[0])/(2*eps))
  analytic=float(pred[i].grad[idx]);err=abs(numeric-analytic)
  rows.append({'scale':i,'component':c,'analytical':analytic,'central_difference':numeric,'abs_error':err,'passed':err<1e-8})
print(json.dumps({'schema':'research_flow_gradient_precision_v1','dtype':'float64','all_passed':all(r['passed'] for r in rows),'checks':rows,'real_data_used':False}))
