"""Replay immutable native physical endpoint caches through the frozen new core."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--kind',choices=['rich','pose'],required=True)
p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=2)
args=p.parse_args();R=args.campaign.resolve();P=Path('/home/zhaoyuhan/Eventbased_WAM')
bank=R/(args.kind+'_bank');manifest=bank/'bank_manifest.json';assert manifest.is_file()
out=bank/'events/eventgenesis';out.mkdir(parents=True,exist_ok=True)
profile_path=R/'sources/stage9_sensor_profile.json'
assert profile_path.is_file()
profile=json.loads(profile_path.read_text());profile_hash=hashlib.sha256(profile_path.read_bytes()).hexdigest()
physical_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
if args.worker is None:
    processes=[]
    for slot in range(args.workers):
        with (out/f'worker{slot}.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--campaign',str(R),'--kind',args.kind,'--worker',str(slot),'--workers',str(args.workers)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),close_fds=True)
        processes.append(child)
    (out/'processes.json').write_text(json.dumps({'pids':[p.pid for p in processes],'physical_manifest_sha256':physical_hash,'profile_sha256':profile_hash,'workers':args.workers},indent=2)+'\n')
    codes=[p.wait() for p in processes]
    (out/'worker_exits.json').write_text(json.dumps(codes)+'\n');assert all(c==0 for c in codes),codes
    rows=[]
    for file in sorted((bank/'physical').glob('*.h5')):
        rid=file.stem;row=json.loads((out/(rid+'.json')).read_text())
        assert row['source_physical_manifest_sha256']==physical_hash and row['profile_sha256']==profile_hash
        assert hashlib.sha256((out/(rid+'.npy')).read_bytes()).hexdigest()==row['events_sha256']
        rows.append(row)
    result={'schema':'stage9_newcore_raw_events_v1','generator':'eventgenesis','core_release':profile['release_commit'],
        'physical_manifest_sha256':physical_hash,'sensor_profile_sha256':profile_hash,'rollouts':rows,
        'real_data_used':False,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'complete','kind':args.kind,'rollouts':len(rows),'manifest':str(out/'manifest.json')}),flush=True)
    raise SystemExit(0)

import h5py
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text());core=Path(pointer['core_path'])
assert pointer['release_commit']==profile['release_commit']
for name,h in pointer['descriptor']['core_canonical_LF_sha256'].items():
    assert hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest()==h
sys.path.insert(0,str(core))
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig

files=sorted((bank/'physical').glob('*.h5'))
for index in range(args.worker,len(files),args.workers):
    file=files[index];rid=file.stem
    assert not (out/(rid+'.json')).exists(), 'Immutable completed rollout exists'
    begin=time.monotonic()
    with h5py.File(file,'r') as h:
        g=h['episodes'][rid];times=np.asarray(g['frame_timestamps_s'],dtype=np.float64)
        K=np.asarray(g['camera_intrinsics_K_px']);ids=np.asarray(g['object_segmentation_link_ids'])
        H,W=g['depth_m'].shape[1:];scene=SimpleNamespace(cur_t=float(times[0]));cam=SimpleNamespace(res=(W,H),intrinsics=K);cursor={'i':0}
        def frame(sc,ca):
            i=cursor['i'];return dict(radiance=np.asarray(g['radiance_linear_rgb'][i]),depth=np.asarray(g['depth_m'][i]),seg=np.asarray(g['segmentation_link_idx'][i]))
        def motion(sc,ca):
            i=cursor['i'];obj=np.asarray(g['object_to_world'][i]);return dict(camera_to_world=np.asarray(g['camera_to_world_cv'][i]),object_to_world={int(sid):obj[j] for j,sid in enumerate(ids)})
        kwargs=dict(profile['plugin_kwargs']);kwargs.update(device='cuda:0',warp_backend='torch')
        plugin=GenesisPhysicsEventPlugin(noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),
            sampler_config=AdaptiveSamplingConfig(**profile['sampler_kwargs']),frame_provider=frame,motion_state_provider=motion,**kwargs)
        plugin.attach(scene,cam).start_episode();assert len(plugin.capture())==0
        chunks=[];counts=[];unknown=[]
        for k in range(1,len(times)):
            cursor['i']=k;scene.cur_t=float(times[k]);events=np.asarray(plugin.capture(),dtype=np.float64)
            assert events.ndim==2 and events.shape[1]==4 and np.isfinite(events).all()
            if len(events):
                assert events[:,0].min()>=times[k-1]-1e-9 and events[:,0].max()<=times[k]+1e-9
                assert (events[:,1]>=0).all() and (events[:,1]<W).all() and (events[:,2]>=0).all() and (events[:,2]<H).all()
            counts.append(len(events));chunks.append(events);unknown.append(plugin.uncertainty['invalid_fraction'])
            if k%10==0:
                progress={'rollout':rid,'interval':k,'total_intervals':len(times)-1,'event_count':sum(counts),'wall_s':time.monotonic()-begin}
                (out/f'worker{args.worker}_progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
        raw=np.concatenate(chunks) if any(counts) else np.empty((0,4),np.float64)
        if len(raw):assert (np.diff(raw[:,0])>=0).all()
        np.save(out/(rid+'.npy'),raw)
        record={'rollout_id':rid,'split':str(g.attrs['split']),'source_path':str(file),'source_physical_manifest_sha256':physical_hash,
            'profile_sha256':profile_hash,'core_release':profile['release_commit'],'core_files_verified':17,
            'interval_event_counts':counts,'unknown_fraction_per_interval':unknown,'event_count':len(raw),
            'events_sha256':hashlib.sha256((out/(rid+'.npy')).read_bytes()).hexdigest(),
            'wall_s':time.monotonic()-begin,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
            'backend':'torch/cuda','real_data_used':False,'source_cadence_hz':100,'intermediate_real_renders':0}
        (out/(rid+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'rollout_complete':rid,'event_count':len(raw),'wall_s':record['wall_s']}),flush=True)
        plugin.close();del plugin;torch.cuda.empty_cache()
