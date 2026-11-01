"""Measured scientific plot and full secondary-bank event-localization replay."""
from pathlib import Path
import csv,json,hashlib,os
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence';OUT.mkdir(exist_ok=False)
primary=json.loads((ROOT/'summary.json').read_text())
secondary=json.loads((ROOT/'secondary_confirmation/summary.json').read_text())
names=['old_ours','observed_candidate','v2e']
labels=['Old Ours','Observed-input candidate','V2E (fixed)']
colors=['#879CAA','#A00000','#3676A5']
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for ax,metric,title in zip(axes,['p50','p95'],['Median localization error','95th-percentile localization error']):
    for i,(name,label,color) in enumerate(zip(names,labels,colors)):
        values=[x['models'][name]['pixel'][metric] for x in [primary,secondary]]
        bars=ax.bar(np.arange(2)+(i-1)*.24,values,width=.23,color=color,label=label)
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=9)
    ax.set_xticks([0,1],['Primary: 9 episodes\n581 queries','Secondary: 3 archived episodes\n193 queries'])
    ax.set_ylabel('Error in original image pixels');ax.set_title(title,weight='bold')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_ylim(0,max(primary['models']['old_ours']['pixel'][metric],secondary['models']['old_ours']['pixel'][metric])*1.22)
handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='outside lower center',ncol=3,frameon=False)
fig.savefig(OUT/'pixel_localization_results.png',dpi=180);fig.savefig(OUT/'pixel_localization_results.svg');plt.close(fig)

with (ROOT/'secondary_confirmation/all_models_pixel_polls.csv').open(newline='') as f:rows=list(csv.DictReader(f))
lookups={name:{(int(r['episode_id']),round(float(r['poll_s']),9)):r for r in rows if r['method']==name} for name in ['old_ours','observed_candidate']}
assert lookups['old_ours'].keys()==lookups['observed_candidate'].keys()
ref=Path('/home/zhaoyuhan/Eventbased_WAM/repositories/genesis-event-camera-plugin-r0_7-heatmap-20261017/benchmark/robot_closed_loop/results/r0_7_heatmap_event_state_closed_loop_3seed_20261017')
events={}
for ep in range(3):
    a=np.load(ref/'episodes'/f'episode_{ep:03d}'/'direct_hdr_events_1khz_2khz.npz')['events_2000hz'];events[ep]=a[np.argsort(a[:,0],kind='stable')]
def font(n,bold=False):
    suffix='-Bold' if bold else ''
    for p in [f'/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,n)
    return ImageFont.load_default(size=n)
def text(d,xy,s,n=22,fill='#263746',bold=False):d.text(xy,s,font=font(n,bold),fill=fill)
frames=[];meta=[]
for ep,poll in sorted(lookups['old_ours']):
    old=lookups['old_ours'][ep,poll];new=lookups['observed_candidate'][ep,poll]
    cutoff=float(old['cutoff_s']);assert new['cutoff_s']==old['cutoff_s']
    ev=events[ep];a,b=np.searchsorted(ev[:,0],[cutoff-.025,cutoff],side='left');ev=ev[a:b]
    counts=np.zeros((2,240,320),np.int32)
    np.add.at(counts,(np.where(ev[:,3]>0,0,1),ev[:,2].astype(int),ev[:,1].astype(int)),1)
    strength=np.clip(np.log1p(counts)/np.log(5.),0,1)
    picture=np.full((240,320,3),255.,np.float32)
    for plane,color in zip(strength,[(180,25,40),(45,110,180)]):picture=picture*(1-plane[...,None])+np.asarray(color)[None,None,:]*plane[...,None]
    picture=Image.fromarray(picture.astype(np.uint8)).resize((560,420),Image.Resampling.NEAREST)
    frame=Image.new('RGB',(1280,720),'white');d=ImageDraw.Draw(frame)
    text(d,(40,28),'Same input events: target localization before contact',29,'#A00000',True)
    text(d,(40,80),'Old Ours',25,'#617787',True);text(d,(670,80),'Observed-input candidate',25,'#A00000',True)
    for x,row,color in [(40,old,'#617787'),(670,new,'#A00000')]:
        panel=picture.copy();pd=ImageDraw.Draw(panel)
        gt=np.asarray([float(row['gt_u_px']),float(row['gt_v_px'])])*np.asarray([560/320,420/240])
        pred=np.asarray([float(row['pred_u_px']),float(row['pred_v_px'])])*np.asarray([560/320,420/240])
        assert abs(np.linalg.norm(pred/np.asarray([560/320,420/240])-gt/np.asarray([560/320,420/240]))-float(row['pixel_euclidean_error_px']))<1e-9
        u,v=gt;pd.ellipse((u-8,v-8,u+8,v+8),outline='#00A660',width=3)
        u,v=pred;pd.line((u-10,v,u+10,v),fill=color,width=3);pd.line((u,v-10,u,v+10),fill=color,width=3)
        frame.paste(panel,(x,125));text(d,(x,563),f"Pixel error: {float(row['pixel_euclidean_error_px']):.2f} px",25,color,True)
    text(d,(40,610),f"Episode {ep:03d} | speed {float(old['speed_m_s']):.2f} m/s | y {float(old['launch_y_m'])*1000:+.0f} mm | observation {cutoff:.3f} s | available {poll:.3f} s",18)
    text(d,(40,646),'Green ring: target GT (overlay/scoring only). Cross: network prediction. Playback 0.2x.',17,'#617787')
    text(d,(40,679),'All 193 secondary-bank queries are shown; this is offline perception replay, not a new control rollout.',17,'#617787')
    if abs(cutoff-.4)<1e-7:frame.save(OUT/f'episode_{ep:03d}_at_0400s.png')
    frames.append(frame)
    meta.append({'episode_id':ep,'poll_s':poll,'cutoff_s':cutoff,'old_error_px':float(old['pixel_euclidean_error_px']),'new_error_px':float(new['pixel_euclidean_error_px'])})
assert len(frames)==193
frames[0].save(OUT/'pixel_localization_secondary_replay.gif',save_all=True,append_images=frames[1:],duration=50,loop=0,optimize=False)
metadata={'query_count':193,'all_secondary_queries_shown':True,'GT_used_only_for_scoring_and_overlay':True,
          'same_event_pixels_for_both_models':True,'playback_speed':.2,'width_height':[1280,720],
          'registration':secondary['registration'],'queries':meta}
(OUT/'replay_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
files=[p for p in OUT.iterdir() if p.is_file()]
(OUT/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(files)))
print(json.dumps({'status':'PASS','figure_files':['pixel_localization_results.png','pixel_localization_results.svg'],'replay_queries':len(frames),'gif_bytes':(OUT/'pixel_localization_secondary_replay.gif').stat().st_size}))
