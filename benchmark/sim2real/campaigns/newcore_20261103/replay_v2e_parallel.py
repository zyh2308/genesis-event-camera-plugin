"""Pinned official V2E+SuperSloMo on immutable shared endpoint images."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import cv2,h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--kind',choices=['rich','pose'],required=True)
p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=2)
a=p.parse_args();R=a.campaign.resolve();P=Path('/home/zhaoyuhan/Eventbased_WAM');bank=R/(a.kind+'_bank')
src=P/'external/sources/v2e-cac6384445/v2e-pinned';weight=P/'external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt'
pin=json.loads((R/'sources/v2e_source_pin.json').read_text())
assert pin['commit']=='cac638444557a224ba6ee5c29260916f152c00bd'
for row in pin['files']:
 f=src/row['path'];assert f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],str(f)
assert hashlib.sha256(weight.read_bytes()).hexdigest()==pin['slomo_checkpoint']['sha256']
assert weight.is_file() and os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
out=bank/'events/v2e';out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
physical_hash=hashlib.sha256((bank/'bank_manifest.json').read_bytes()).hexdigest();rows=[]
if a.worker is None:
 processes=[]
 for slot in range(a.workers):
  log=(out/('worker%d.log'%slot)).open('wb')
  processes.append(subprocess.Popen([sys.executable,'-u',__file__,'--campaign',str(R),'--kind',a.kind,'--worker',str(slot),'--workers',str(a.workers)],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy()));log.close()
 codes=[p.wait() for p in processes];assert all(c==0 for c in codes),codes
 rows=[json.loads((out/(f.stem+'.json')).read_text()) for f in sorted((bank/'physical').glob('*.h5'))]
 for row in rows:assert row['source_physical_manifest_sha256']==physical_hash and hashlib.sha256((out/(row['rollout_id']+'.npy')).read_bytes()).hexdigest()==row['events_sha256']
 (out/'manifest.json').write_text(json.dumps({'schema':'stage9_v2e_official_ssm_v1','generator':'v2e','source_commit':pin['commit'],'physical_manifest_sha256':physical_hash,'rollouts':rows,'real_data_used':False},indent=2)+'\n')
 raise SystemExit(0)
for index,f in enumerate(sorted((bank/'physical').glob('*.h5'))):
 if index%a.workers!=a.worker:continue
 rid=f.stem;record=out/(rid+'.json');assert not record.exists(), 'Completed artifact is immutable'
 inp=bank/'display_inputs'/rid;inp.mkdir(parents=True,exist_ok=True)
 with h5py.File(f,'r') as h:
  g=h['episodes'][rid];times=np.asarray(g['frame_timestamps_s']);frames=g['v2e_display_luma_u8'];split=str(g.attrs['split'])
  assert np.allclose(np.diff(times),.01,atol=1e-10)
  for i in range(len(times)):
   image=np.asarray(frames[i]);path=inp/('%06d.png'%i)
   assert not path.exists() or np.array_equal(cv2.imread(str(path),0),image)
   if not path.exists():assert cv2.imwrite(str(path),image)
 target=out/'official_raw'/(rid+'.h5');target.parent.mkdir(exist_ok=True);folder=out/'official_work'/rid
 assert not target.exists() and not folder.exists()
 cmd=[sys.executable,str(src/'v2e.py'),'--input',str(inp),'--input_frame_rate','100','--auto_timestamp_resolution','False','--timestamp_resolution','.000025','--slomo_model',str(weight),'--output_folder',str(folder),'--unique_output_folder','False','--skip_video_output','--no_preview','--dvs_h5',str(target),'--dvs_aedat2','None','--dvs_aedat4','None','--dvs_text','None','--pos_thres','.2','--neg_thres','.2','--sigma_thres','.03','--cutoff_hz','300','--leak_rate_hz','.01','--leak_jitter_fraction','.1','--noise_rate_cov_decades','.1','--shot_noise_rate_hz','.001','--refractory_period','.0005','--dvs_emulator_seed','42']
 started=time.monotonic()
 with (out/(rid+'.log')).open('wb') as log:result=subprocess.run(cmd,cwd=src,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,(rid,result.returncode)
 with h5py.File(target,'r') as h:
  raw=np.asarray(h['events']);assert raw.ndim==2 and raw.shape[1]==4
  events=raw.astype(np.float64);events[:,0]*=1e-6;events[:,3]=np.where(events[:,3]>0,1.,-1.)
 assert np.isfinite(events).all() and (len(events)==0 or (np.diff(events[:,0])>=0).all())
 assert len(events)==0 or (events[:,0].min()>=-1e-9 and events[:,0].max()<=times[-1]+1e-6)
 path=out/(rid+'.npy');np.save(path,events)
 row={'rollout_id':rid,'split':split,'event_count':len(events),'events_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_physical_manifest_sha256':physical_hash,'wall_s':time.monotonic()-started,'command':cmd,'weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'visible_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'real_data_used':False}
 record.write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
 (out/('worker%d_progress.json'%a.worker)).write_text(json.dumps({'complete':len(rows),'total':len(list((bank/'physical').glob('*.h5'))),'latest':row},indent=2)+'\n')
 print(json.dumps({'completed':rid,'count':len(events),'wall_s':row['wall_s']}),flush=True)
(out/('worker%d_manifest.json'%a.worker)).write_text(json.dumps({'schema':'stage9_v2e_official_ssm_v1','generator':'v2e','source_commit':'cac638444557a224ba6ee5c29260916f152c00bd','physical_manifest_sha256':physical_hash,'rollouts':rows,'real_data_used':False,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
