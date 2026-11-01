"""Export existing evidence only. No training, selection or score-driven styling."""
import os,sys,json,hashlib,time,subprocess,traceback,gc
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import h5py,cv2
from scipy.io import loadmat
P=Path('/home/zhaoyuhan/Eventbased_WAM')
OUT=P/'runs/sim2real/20261101-research-meeting-export-v1'
OUT.mkdir(exist_ok=True)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BG=(13,23,39);WHITE=(238,242,248);MUTED=(159,176,195);BLUE=(70,164,255);ORANGE=(255,173,66);GREEN=(80,220,153)
manifest={'scope':'visualization and fixed-checkpoint inference only; no training','created_unix':time.time(),'exports':{},'errors':{}}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def txt(im,s,x,y,n=22,color=WHITE):ImageDraw.Draw(im).text((x,y),s,font=ImageFont.truetype(FONT,n),fill=color)
def base(title,sub):
 im=Image.new('RGB',(1280,720),BG);txt(im,title,32,20,32);txt(im,sub,32,66,19,MUTED);return im
def put(im,a,box,nearest=True):
 if not isinstance(a,Image.Image):
  if a.dtype!=np.uint8:a=np.uint8(np.clip(a,0,1)*255)
  a=Image.fromarray(a).convert('RGB')
 im.paste(a.resize((box[2],box[3]),Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR),(box[0],box[1]))
def writer(name,fps=30):
 return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/name)],stdin=subprocess.PIPE)
def evimg(x,y,p,h,w):
 a=np.full((h,w,3),245,np.uint8);x=np.asarray(x,dtype=int);y=np.asarray(y,dtype=int);p=np.asarray(p)>0
 ok=(x>=0)&(x<w)&(y>=0)&(y<h);a[y[ok&~p],x[ok&~p]]=(40,100,235);a[y[ok&p],x[ok&p]]=(220,50,60);return a
def savejson(): (OUT/'export_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))

def reconstruction():
 root=P/'runs/sim2real/20261025-hqf14-e2vid-seed0-actual-r2'
 data=P/'data/sim2real/stage5/20261021-hqf14-memmap-v3'
 score=json.loads((root/'paired_score.json').read_text())
 # preserve aggregate/sequence scores without a 38 MB per-frame replication
 compact={k:v for k,v in score.items() if k!='per_frame'}
 (OUT/'hqf_score_compact.json').write_text(json.dumps(compact,indent=2))
 vid=writer('01_reconstruction_HQF.mp4');records=[]
 for seq in ('boxes','poster_pillar_1'):
  sd=data/seq
  ims=np.load(sd/'images.npy',mmap_mode='r');ts=np.load(sd/'images_ts.npy').reshape(-1)
  ids=np.load(sd/'image_event_indices.npy').reshape(-1)
  xy=np.load(sd/'events_xy.npy',mmap_mode='r');ps=np.load(sd/'events_p.npy',mmap_mode='r')
  start=max(1,int((len(ts)-1)*.25));duration=min(8.,float(ts[-2]-ts[start]))
  indices=np.maximum(1,np.minimum(np.searchsorted(ts,ts[start]+np.arange(int(duration*30))/30,side='right')-1,len(ts)-1))
  rec={'sequence':seq,'selection':'time segment begins at floor(25% of final frame index), up to 8s; no score-driven selection','indices':indices.tolist(),'source_frame_count':len(ts),'predictions':[]}
  cache={}
  for j,idx in enumerate(indices):
   idx=int(idx)
   if idx not in cache:
    # bank intervals and official inference both use previous target -> current target.
    a,b=int(ids[idx-1]),int(ids[idx]);ev=evimg(xy[a:b,0],xy[a:b,1],ps[a:b],180,240)
    gt=ims[idx,:,:,0]
    preds=[]
    for arm in ('eventgenesis','v2e'):
     path=root/'inference'/arm/seq/f'reconstruction_{idx:06d}.npy'
     arr=np.load(path);assert arr.shape==(180,240) and np.isfinite(arr).all()
     preds.append(np.uint8(np.clip(arr,0,1)*255))
     rec['predictions'].append({'arm':arm,'index':idx,'path':str(path),'sha256':sha(path)})
    cache[idx]=(ev,gt,*preds)
   panels=cache[idx]
   im=base('L1 | Reconstruction on REAL HQF events',f'{seq}  |  seed 0, fixed synthetic-selected checkpoints  |  t={ts[idx]-ts[start]:.2f}s')
   for c,(label,arr) in enumerate(zip(('Real events','Reference image','EG-trained E2VID+','V2E-trained E2VID+'),panels)):
    x=24+c*312;txt(im,label,x,112,21,BLUE if c==2 else ORANGE if c==3 else WHITE)
    put(im,arr,(x,148,296,222));
    crop=arr[45:135,60:180];put(im,crop,(x,420,296,222))
   txt(im,'Fixed center crop (same region in every frame)',24,384,19,MUTED)
   txt(im,'Direct [0,1] display. No per-image brightness fitting. Reference images never enter the model.',24,676,17,MUTED)
   vid.stdin.write(np.asarray(im).tobytes())
   if j==0: im.save(OUT/f'reconstruction_{seq}_poster.png')
  records.append(rec)
 vid.stdin.close();assert vid.wait()==0
 manifest['exports']['reconstruction']={'records':records,'video':'01_reconstruction_HQF.mp4','score_source':str(root/'paired_score.json'),'score_sha256':sha(root/'paired_score.json'),'all_sequences':14,'scope':'real test; seed0; two deterministic demo segments; all14 sequences retained in aggregate metrics'}
 savejson();print('HQF finished',flush=True)

def flow():
 import torch
 torch.set_num_threads(1)
 repo=P/'execution_checkouts/sim2real-stage8-da019d2-20261101'
 sys.path.insert(0,str(repo/'benchmark/sim2real/scripts'));sys.path.insert(0,str(repo/'benchmark/sim2real/compat'))
 from run_stage8_replicad_dynamic_gtfit import load_rows,evaluate
 from stage5_flow_supervised_pilot import model_source,model_output_to_flow_hw
 from steflow_device_placement import install_cpu_cuda_placement_compat
 install_cpu_cuda_placement_compat(torch.device('cpu'))
 bank=P/'runs/sim2real/20261101-stage8-replicad-dynamic-v2/flow_bank'
 bm=json.loads((bank/'training_bank_manifest.json').read_text());rows=load_rows(bm);row=rows['validation'][0]
 run=P/'runs/sim2real/20261101-stage8-replicad-dynamic-v3-cpucompat/training_output'
 summary=json.loads((run/'summary.json').read_text());(OUT/'flow_summary.json').write_text(json.dumps(summary,indent=2))
 ev=row['arms'];active=np.any(ev['eventgenesis']['events']!=0,axis=(0,3))|np.any(ev['v2e']['events']!=0,axis=(0,3))
 arrays={};res={}
 for arm in ('eventgenesis','v2e'):
  ck=run/arm/'checkpoint_final.pt';assert sha(ck)==summary['arms'][arm]['checkpoint']['sha256']
  model,_=model_source(P/'external/sources/downstream/STE-FlowNet',torch.device('cpu'))
  checkpoint=torch.load(ck,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict'],strict=True)
  m=evaluate(model,ev[arm],active,torch.device('cpu'),{'model_output_to_flow_hw':model_output_to_flow_hw})
  expected=summary['arms'][arm]['validation_rollouts_final'][0]
  assert abs(m['full_valid_epe_px_per_30ms']-expected['full_valid_epe_px_per_30ms'])<1e-5
  assert abs(m['common_event_active_epe_px_per_30ms']-expected['common_event_active_epe_px_per_30ms'])<1e-5
  arrays[arm]=m.pop('prediction');gt=m.pop('gt');valid=m.pop('valid');common=m.pop('active');res[arm]=m
  print(arm,'flow inference matched',m['full_valid_epe_px_per_30ms'],flush=True)
  del model,checkpoint;gc.collect()
 np.savez_compressed(OUT/'flow_predictions_and_gt.npz',gt=gt,valid=valid,common_active=common,**arrays)
 info=bm['coverage_audit'][1];motion=Path(info['source_motion_dir']);episode=motion.parent
 rad=np.load(motion/'radiance_frames.npy');assert rad.shape==(4,240,320,3)
 # Linear -> sRGB display, same transform for all saved radiance endpoints.
 r=np.clip(rad,0,1);rgb=np.uint8(255*np.where(r<=.0031308,12.92*r,1.055*r**(1/2.4)-.055))
 rawEG=np.load(motion/'eventgenesis_events_seconds_xy_p.npy')
 with h5py.File(info['source_v2e_h5'],'r') as h:rawV=np.asarray(h['events'],dtype=np.float64)
 rawV[:,0]*=1e-6
 # The contact sheet V2E count was stale; use only the registered (0,30ms] input window.
 rawEG=rawEG[(rawEG[:,0]>0)&(rawEG[:,0]<=.03)]
 rawV=rawV[(rawV[:,0]>0)&(rawV[:,0]<=.03)]
 assert len(rawEG)==ev['eventgenesis']['event_count'] and len(rawV)==ev['v2e']['event_count'],(len(rawEG),len(rawV))
 maxmag=float(np.hypot(gt[0],gt[1])[valid].max())
 def color(f):
  mag=np.hypot(f[0],f[1]);h=(np.arctan2(f[1],f[0])/(2*np.pi)+1)%1
  hsv=np.uint8(np.stack([h*179,np.ones_like(h)*255,np.clip(mag/maxmag,0,1)*255],-1))
  a=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB);a[~valid]=[35,44,58];return a
 gtRGB=color(gt);preds={a:color(f) for a,f in arrays.items()}
 errs={a:cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(np.hypot(*(f-gt))/8,0,1)*255),cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB) for a,f in arrays.items()}
 for a in errs:errs[a][~valid]=[35,44,58]
 events=[evimg(rawEG[:,1],rawEG[:,2],rawEG[:,3],240,320),evimg(rawV[:,1],rawV[:,2],rawV[:,3],240,320)]
 im=base('L2 | Optical flow: held-out ReplicaCAD layout', 'GT-supervised pilot | train: 1 interval, test: 1 interval | 30ms | final update 100 | seed 0')
 for c,(lab,ar) in enumerate(zip(('Shared GT flow','EG prediction','V2E prediction'),(gtRGB,preds['eventgenesis'],preds['v2e']))):
  x=32+c*416;txt(im,lab,x,110,24,GREEN if c==0 else BLUE if c==1 else ORANGE);put(im,ar,(x,150,384,288))
 for c,(lab,ar) in enumerate(zip(('Scene (saved start frame)','EG EPE map: 0 - 8 px','V2E EPE map: 0 - 8 px'),(rgb[0],errs['eventgenesis'],errs['v2e']))):
  x=32+c*416;txt(im,lab,x,462,21);put(im,ar,(x,500,224,168))
 txt(im,f'Flow color brightness: 0 - {maxmag:.2f} px/30ms, shared for all predictions.',32,686,16,MUTED)
 im.save(OUT/'flow_result_poster.png')
 vid=writer('02_flow_ReplicaCAD.mp4')
 for j in range(360):
  if j<180:
   # Show actual four endpoints once, held 1s each. No interpolation or claim of continuous video.
   k=min(j//30,3)
   frame=base('L2 | Shared physical scene -> two event generators','Four saved endpoints over 30ms, held for inspection. This is a slowed diagnostic, not a continuous rollout.')
   panels=(rgb[k],events[0],events[1]);labs=(f'Genesis radiance: {k*10}ms',f'EventGenesis: {len(rawEG):,} events',f'V2E: {len(rawV):,} events')
   for c,(lab,ar) in enumerate(zip(labs,panels)):
    x=32+c*416;txt(frame,lab,x,140,23,BLUE if c==1 else ORANGE if c==2 else WHITE);put(frame,ar,(x,190,384,288))
   txt(frame,'Same 30ms camera / object trajectory and GT labels. Events shown over the full interval.',32,548,23)
   txt(frame,'Event count alone is not a quality metric.',32,602,22,MUTED)
  else:frame=im
  vid.stdin.write(np.asarray(frame).tobytes())
  if j==30:frame.save(OUT/'flow_scene_poster.png')
 vid.stdin.close();assert vid.wait()==0
 manifest['exports']['flow']={'run':str(run),'source_commit':'da019d2f85650267ca5ba50460cb14ca5b10b2bb','metrics_recomputed':res,'event_counts_reverified':{'eventgenesis':len(rawEG),'v2e':len(rawV)},'scope':'synthetic diagnostic only; one validation interval; not MVSEC','flow_display_max':maxmag,'error_display_max':8,'arrays_sha256':sha(OUT/'flow_predictions_and_gt.npz')}
 savejson();print('Flow finished',flush=True)

def pose():
 root=P/'data/real/epose/figshare_28106717_v1/extracted'
 sd=root/'block/block/r_01_gl'
 cad=P/'external/sim2real_assets/epose_wood_block_20261024/nontextured.ply'
 with cad.open() as f:
  while next(f).strip()!='end_header':pass
  verts=np.array([[float(v) for v in next(f).split()[:3]] for _ in range(8194)])
 lo,hi=verts.min(0),verts.max(0)
 corners=np.array([[hi[0],hi[1],lo[2]],[hi[0],lo[1],lo[2]],[lo[0],hi[1],lo[2]],[lo[0],lo[1],lo[2]],[hi[0],hi[1],hi[2]],[hi[0],lo[1],hi[2]],[lo[0],hi[1],hi[2]],[lo[0],lo[1],hi[2]]])
 edges=[(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]
 dist=np.array([-.362,.115,.0009576,.00000455,.130])
 vid=writer('03_pose_EPOSE_task.mp4');records=[]
 for n in range(3,603,3):
  mats=[loadmat(sd/f'{i:04d}_data.mat') for i in range(n-2,n+1)]
  m=mats[-1];ev=np.concatenate([v['events'] for v in mats]);raw=evimg(ev[:,0],ev[:,1],ev[:,3],260,346)
  K=np.asarray(m['intrinsicMatrix'],float);T=np.asarray(m['pose'],float)
  uv,_=cv2.projectPoints(corners,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],K,dist);uv=uv.reshape(-1,2)
  overlay=Image.fromarray(raw);d=ImageDraw.Draw(overlay)
  for i,j in edges:d.line([tuple(uv[i]),tuple(uv[j])],fill=GREEN,width=2)
  im=base('L3 | E-POSE object pose task: REAL event stream','GROUND-TRUTH overlay only. No trained E-POSE prediction or real pose score is available yet.')
  for x,lab,ar in [(50,'Raw full-frame events',raw),(680,'CAD box projected from GT 6D pose',overlay)]:
   txt(im,lab,x,128,24,GREEN if x==680 else WHITE);put(im,ar,(x,174,550,413))
  txt(im,f'block / r_01_gl | samples {n-2:04d}-{n:04d} | 30ms accumulated events | {float(np.asarray(m["time"]).squeeze()):.2f}s',50,621,22)
  txt(im,'Correct per-corner projection in the raw distorted pixel domain (K + distortion). GT is for display only.',50,672,18,MUTED)
  vid.stdin.write(np.asarray(im).tobytes())
  if n==153:im.save(OUT/'pose_real_task_poster.png')
  records.append({'sample_end':n,'mat_path':str(sd/f'{n:04d}_data.mat'),'mat_sha256':sha(sd/f'{n:04d}_data.mat'),'uv':uv.tolist(),'event_count':len(ev),'time':float(np.asarray(m['time']).squeeze())})
 vid.stdin.close();assert vid.wait()==0
 result=P/'runs/sim2real/20261101-stage8-pose-cpu-short-v2/training_output/result.json'
 pose_result=json.loads(result.read_text())
 (OUT/'pose_result.json').write_text(json.dumps(pose_result,indent=2))
 manifest['exports']['pose']={'scope':'real task illustration only; GT CAD projection; no model prediction','sequence':'block/r_01_gl','cad_path':str(cad),'cad_sha256':sha(cad),'distortion':dist.tolist(),'records':records,'synthetic_pilot_result':str(result),'synthetic_pilot_sha256':sha(result)}
 savejson();print('Pose finished',flush=True)

if (OUT/'export_manifest.json').is_file():
 manifest=json.loads((OUT/'export_manifest.json').read_text())
for name,fn in [('flow',flow)]:
 try:
  fn()
  manifest['errors'].pop(name,None)
 except Exception:
  manifest['errors'][name]=traceback.format_exc();savejson();print(traceback.format_exc(),flush=True)
manifest['completed_unix']=time.time()
for p in OUT.iterdir():
 if p.is_file() and p.suffix in ('.mp4','.png','.npz'):manifest.setdefault('artifact_hashes',{})[p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
savejson()
print(json.dumps({'directory':str(OUT),'errors':manifest['errors'],'files':list(manifest.get('artifact_hashes',{}))}),flush=True)
