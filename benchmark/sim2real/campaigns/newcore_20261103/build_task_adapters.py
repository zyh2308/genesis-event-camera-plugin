"""Immutable per-method task adapters from the same frozen physical bank."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import cv2,h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--kind',choices=['rich','pose'],required=True);p.add_argument('--method',required=True)
a=p.parse_args();R=a.campaign.resolve();P=Path('/home/zhaoyuhan/Eventbased_WAM');bank=R/(a.kind+'_bank');events=bank/'events'/a.method
pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())
scripts=R/'sources/benchmark/sim2real/scripts';sys.path[:0]=[str(scripts),pointer['core_path']]
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
assert (bank/'bank_manifest.json').is_file()
started=time.monotonic()
while not (events/'manifest.json').exists():
 assert time.monotonic()-started<24*3600,'Raw event preparation exceeded 24 hours';time.sleep(30)
event_manifest=json.loads((events/'manifest.json').read_text());assert event_manifest['physical_manifest_sha256']==sha(bank/'bank_manifest.json')
event_rows={r['rollout_id']:r for r in event_manifest['rollouts']};files=sorted((bank/'physical').glob('*.h5'));assert len(files)==(32 if a.kind=='rich' else 128)
root=R/'adapters'/a.method
if a.kind=='rich':
 import prepare_stage4_steflow_bank as flow_adapter
 import build_e2vid_stage4_10ms_data as recon_adapter
 flow_rows=[];recon_rows=[]
 class Episode:
  def __init__(self,g,row):self.g=g;self.row=row;self.attrs=g.attrs;self.name=g.name
  def __getitem__(self,key):
   if key=='eventgenesis_interval_counts_10ms':return np.asarray(self.row['interval_event_counts'])
   return self.g[key]
 class Source:
  def __init__(self,g,row):self.g=g;self.row=row
  def __getitem__(self,key):return Episode(self.g,self.row)
 for file in files:
  rid=file.stem;raw_path=events/(rid+'.npy');assert sha(raw_path)==event_rows[rid]['events_sha256'];raw=np.load(raw_path)
  flow_adapter.load_raw_events=lambda *args:raw
  def load_events(*args):
   q=raw.copy();q[:,3]=(q[:,3]>0).astype(np.uint8);return q,{'source':str(raw_path),'source_sha256':sha(raw_path),'time_unit':'seconds','polarity':'OFF=0,ON=1'}
  recon_adapter.load_generator_events=load_events
  with h5py.File(file,'r') as physical:
   g=physical['episodes'][rid];split=str(g.attrs['split'])
   flow_rows.append(flow_adapter.convert_rollout(a.method,rid,physical,events,root/'flow'/a.method/split/(rid+'.npz'),{}))
   f10,v10,parity=recon_adapter.build_labels(g,verify_registered_30ms=True);assert parity==66
   row=recon_adapter.export_h5(Source(g,event_rows[rid]),{rid:(f10,v10)},a.method,rid,events,root/'reconstruction'/a.method/split/(rid+'.h5'))
   row['direct_30ms_parity_pairs']=parity;recon_rows.append(row)
  print(json.dumps({'adapter_complete':rid,'method':a.method,'events':len(raw)}),flush=True)
 base={'schema':'stage9_frozen_task_adapter_v1','generator':a.method,'core_release':pointer['release_commit'],'physical_manifest_sha256':sha(bank/'bank_manifest.json'),'raw_event_manifest_sha256':sha(events/'manifest.json'),'real_data_used':False,'real_scoring_enabled':False}
 for split,count in [('train',24),('validation',8)]:
  paths=sorted((root/'reconstruction'/a.method/split).glob('*.h5'));assert len(paths)==count
  (root/'reconstruction'/(a.method+'_'+split+'_files.txt')).write_text(''.join(str(x)+'\n' for x in paths))
 save(root/'flow/training_bank_manifest.json',{**base,'train_windows_per_generator':1584,'validation_windows_per_generator':528,'entries':flow_rows,'event_count_filtering':'none','paired_physical_label_contract':True})
 save(root/'reconstruction/adapter_manifest.json',{**base,'rollout_rows':recon_rows,'direct_flow_10ms':True,'label_derivation':'direct i-to-i+1 physical projection, not 30ms/3'})
else:
 import run_pose_stage2_pilot as pose
 rows=[];label_rows=[]
 for file in files:
  rid=file.stem;raw_path=events/(rid+'.npy');assert sha(raw_path)==event_rows[rid]['events_sha256'];raw=np.load(raw_path)
  with h5py.File(file,'r') as h:
   g=h['episodes'][rid];ts=np.asarray(g['frame_timestamps_s']);K=np.asarray(g['camera_intrinsics_K_px']);keys=np.asarray(g['cad_keypoints']);images=[];uvs=[];transforms=[];counts=[];sid=[]
   for i in range(4,20):
    image=pose.encode_event_window(raw[:,0],raw[:,1].astype(int),raw[:,2].astype(int),(raw[:,3]>0).astype(np.uint8),ts[i],ts[i+1],346,260,8.)
    C=np.asarray(g['camera_to_world_cv'][i+1]);T=np.asarray(g['target_pose'][i+1]);Tco=np.linalg.inv(C)@T;xyz=(Tco@np.c_[keys,np.ones(9)].T).T[:,:3];q=(K@xyz.T).T;uv=q[:,:2]/q[:,2:]
    cached=np.vstack([g['projected_centroid_uv'][i+1],g['projected_corners_uv'][i+1]])
    assert np.allclose(uv,cached,atol=1e-7) and np.isfinite(uv).all() and (xyz[:,2]>.03).all()
    assert (uv[0]>[0,0]).all() and (uv[0]<[346,260]).all(),'Synthetic centroid outside registered support'
    images.append(image);uvs.append(uv/np.array([346.,260.]));transforms.append(Tco);counts.append(int(np.searchsorted(raw[:,0],ts[i+1],side='right')-np.searchsorted(raw[:,0],ts[i],side='right')));sid.append(rid+'/'+str(i).zfill(3))
   split=str(g.attrs['split']);dest=root/'pose'/a.method/split/(rid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(dest,images=np.stack(images),uv_normalized=np.asarray(uvs,dtype=np.float32),target_to_camera=np.stack(transforms),K=K,cad_keypoints=keys,sample_ids=np.array(sid),event_counts=np.array(counts),label_times_s=ts[5:21])
   label_hash=hashlib.sha256(np.asarray(uvs,dtype=np.float32).tobytes()+np.stack(transforms).tobytes()).hexdigest();rows.append({'rollout_id':rid,'split':split,'path':str(dest),'sha256':sha(dest),'label_sha256':label_hash,'windows':16})
  print(json.dumps({'pose_adapter_complete':rid,'method':a.method}),flush=True)
 save(root/'pose/adapter_manifest.json',{'schema':'stage9_pose_fullframe_adapter_v1','generator':a.method,'physical_manifest_sha256':sha(bank/'bank_manifest.json'),'raw_event_manifest_sha256':sha(events/'manifest.json'),'rows':rows,'train_windows':1536,'validation_windows':512,'warmup_ms':40,'window_ms':10,'count_scale':8,'real_data_used':False,'real_scoring_enabled':False,'inference_input':'full sensor ON/OFF/total counts only','cad_scope':'YCB 036 wood block'})
print('ADAPTERS_COMPLETE',a.kind,a.method,flush=True)
