import csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/Sim2Real_20261101';OUT.mkdir(exist_ok=True,parents=True)
BUILD=Path(__file__).parent
rows=list(csv.DictReader((ROOT/'artifacts/stage7_supplement_20261030/hqf14_per_sequence.csv').open(encoding='utf-8-sig')))
seqs=[r['sequence'] for r in rows if r['arm']=='eventgenesis'];arms=['eventgenesis','v2e'];metrics=['lpips_alex','ssim','psnr_db','mse']
arr={a:{r['sequence']:r for r in rows if r['arm']==a} for a in arms}
recon={a:{k:float(np.mean([float(arr[a][s][k]) for s in seqs])) for k in metrics} for a in arms}
data={'reconstruction':{'aggregation':'unweighted mean of 14 per-sequence means','sequences':seqs,'frames_per_arm':sum(int(arr[arms[0]][s]['frames']) for s in seqs),'metrics':recon,'lpips_delta':[float(arr['eventgenesis'][s]['lpips_alex'])-float(arr['v2e'][s]['lpips_alex']) for s in seqs]},'flow':{'scope':'GT-supervised synthetic diagnostic; final update100; one training interval and one held-out layout interval; seed0','categories':['All valid pixels','Common event-active pixels'],'eventgenesis':[2.254430661786054,2.6376902877777373],'v2e':[2.608888744568604,2.599874090672216],'zero':[3.497897978891263,2.715453411599548],'active_pixels':7871,'valid_pixels':74496},'pose':{'scope':'generic synthetic cuboid pilot; four held-out windows in two rollouts; seed0; not E-POSE','eventgenesis':31.583652973175,'v2e':23.1761610507965,'real_epose':'not evaluated'}}
(OUT/'results.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
with (OUT/'reconstruction_all14_sequences.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
with (OUT/'result_tables.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['Task','Evidence scope','Metric','EventGenesis','V2E','Zero flow'])
 for k in metrics:w.writerow(['Reconstruction','Real HQF14, seed0, sequence macro mean',k,recon['eventgenesis'][k],recon['v2e'][k],''])
 for i,cat in enumerate(data['flow']['categories']):w.writerow(['Optical flow','Synthetic held-out layout, 1 interval, final update100','EPE px/30ms: '+cat,data['flow']['eventgenesis'][i],data['flow']['v2e'][i],data['flow']['zero'][i]])
 w.writerow(['Pose','Generic synthetic cuboid, 4 held-out windows','2D corner EPE px',data['pose']['eventgenesis'],data['pose']['v2e'],''])
 w.writerow(['Pose','Real E-POSE','ADD / ADD-S','NOT EVALUATED','NOT EVALUATED',''])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'white','axes.titleweight':'bold'})
blue='#2580cb';orange='#e5962e';gray='#8491a0'
fig,ax=plt.subplots(figsize=(10,6.4),layout='constrained')
y=np.arange(len(seqs));delta=data['reconstruction']['lpips_delta'];ax.barh(y,delta,color=[blue if d<0 else orange for d in delta],height=.65);ax.set_yticks(y,seqs);ax.invert_yaxis();ax.axvline(0,color='#29384b',lw=1);ax.set_xlabel('LPIPS-Alex: EventGenesis minus V2E (lower favors EventGenesis)');ax.set_title('Real HQF: paired differences across all 14 sequences',loc='left',pad=16);ax.text(0,-.16,'Seed 0 | same real inputs | fixed synthetic-selected checkpoints\n13/14 sequences favor EG in LPIPS; this is not a multi-seed significance test.',transform=ax.transAxes,fontsize=11,color='#526477');
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'figure_01_reconstruction_LPIPS.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5.6),layout='constrained');x=np.arange(2);bw=.24
for i,(k,label,color) in enumerate([('eventgenesis','EventGenesis',blue),('v2e','V2E',orange),('zero','Zero-flow baseline',gray)]):
 b=ax.bar(x+(i-1)*bw,data['flow'][k],bw,label=label,color=color);ax.bar_label(b,fmt='%.3f',padding=4,fontsize=12)
ax.set_ylim(0,4.15);ax.set_xticks(x,data['flow']['categories']);ax.set_ylabel('EPE (pixels / 30 ms), lower is better');ax.set_title('Flow: held-out ReplicaCAD layout, final update 100',loc='left',pad=16);ax.legend(frameon=False,ncol=3,loc='upper center');ax.text(0,-.19,'Synthetic diagnostic only | 1 train interval + 1 validation interval | seed 0\nEvent-active union: 7,871 / 74,496 valid pixels (10.57%). No MVSEC result.',transform=ax.transAxes,fontsize=11,color='#526477')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'figure_02_flow_EPE.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5.2),layout='constrained');b=ax.bar(['EventGenesis','V2E'],[data['pose'][a] for a in arms],color=[blue,orange],width=.5);ax.bar_label(b,fmt='%.2f',padding=6);ax.set_ylim(0,40);ax.set_ylabel('Mean 2D corner EPE (pixels), lower is better');ax.set_title('Pose: generic synthetic cuboid pilot',loc='left',pad=16);ax.text(0,-.18,'4 held-out windows / 2 rollouts | seed 0 | confidence-argmax inference\nThis is not a real E-POSE pose result. Real ADD / ADD-S: not evaluated.',transform=ax.transAxes,fontsize=10.5,color='#526477')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'figure_03_pose_pilot.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
print(json.dumps(data['reconstruction']['metrics'],indent=2))
