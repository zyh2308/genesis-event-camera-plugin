"""Probe the frozen public entrypoint on cached physical endpoints; no training."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import h5py
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--campaign', type=Path, required=True)
p.add_argument('--mode', choices=['flow','reconstruction','pose'], required=True)
args = p.parse_args()
root = Path('/home/zhaoyuhan/Eventbased_WAM')
campaign = args.campaign.resolve()
out = campaign/'preflight'/args.mode
out.mkdir(parents=True, exist_ok=True)
pointer = json.loads((root/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())
core = Path(pointer['core_path'])
assert pointer['release_commit'] == '62da4f35ce3c5c96f04268bd1687a27657af9238'
for name, expected in pointer['descriptor']['core_canonical_LF_sha256'].items():
    assert hashlib.sha256((core/'genesis_event_plugin'/name).read_text().encode()).hexdigest() == expected
sys.path.insert(0, str(core))
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
profile = json.loads((root/'frozen_baselines/robot-sparse-cuda-r2-20261102/event_sparse_reconstruction_20261102.json').read_text())
noise = dict(profile['noise_config_kwargs'])
noise.update(sigma_thres=.03, cutoff_hz=300., leak_rate_hz=.01,
             leak_jitter_fraction=.1, noise_rate_cov_decades=.1,
             shot_noise_rate_hz=.001, refractory_period_s=.0005)
kwargs = dict(profile['plugin_kwargs'])
kwargs.update(device='cuda:0', warp_backend='torch')

if args.mode == 'reconstruction':
    source = root/'data/sim2real/stage4/training_bank_v3_geometry_corrected_20261017_32x2s/shared_trajectory.h5'
    with h5py.File(source,'r') as h:
        g = h['episodes'][sorted(h['episodes'])[0]]
        radiance = np.asarray(g['radiance_linear_rgb'][:2])
        depth = np.asarray(g['depth_m'][:2])
        seg = np.asarray(g['segmentation_link_idx'][:2])
        K = np.asarray(g['camera_intrinsics_K_px'])
        camera = np.asarray(g['camera_to_world_cv'][:2])
        objects = np.asarray(g['object_to_world'][:2])
        ids = np.asarray(g['object_segmentation_link_ids'])
        poses = [{int(sid): objects[i,j] for j,sid in enumerate(ids)} for i in range(2)]
    provenance = {'source':str(source),'rollout':'s0_00','kind':'existing physical 100Hz endpoints; legacy simple-bank compatibility probe only'}
elif args.mode == 'flow':
    candidates = []
    for folder in sorted((root/'runs/sim2real').glob('*replicad*')):
        for f in folder.rglob('radiance_frames.npy'):
            d = f.parent
            needed = ['depth_frames.npy','segmentation_physical_link_ids.npy','camera_poses_cv.npy','shared_intrinsics_pixel_index.npy','moving_object_poses_cv.npy','moving_link_id.txt']
            if all((d/n).is_file() for n in needed):
                candidates.append(d)
    assert candidates, 'No complete rich physical cache found'
    d = candidates[0]
    radiance = np.load(d/'radiance_frames.npy',mmap_mode='r')[:2].copy()
    depth = np.load(d/'depth_frames.npy',mmap_mode='r')[:2].copy()
    seg = np.load(d/'segmentation_physical_link_ids.npy',mmap_mode='r')[:2].copy()
    K = np.load(d/'shared_intrinsics_pixel_index.npy')
    camera = np.load(d/'camera_poses_cv.npy')[:2]
    moving = np.load(d/'moving_object_poses_cv.npy')[:2]
    sid = int((d/'moving_link_id.txt').read_text())
    poses = [{sid:moving[i]} for i in range(2)]
    provenance = {'source':str(d),'kind':'cached rich ReplicaCAD physical endpoint probe; not full training bank','complete_rich_caches_found':len(candidates),'unlisted_links':'static world geometry, identity delta'}
else:
    # Analytic public-entrypoint acceptance only, never a claimed pose scene.
    H,W = 8,10
    radiance = np.stack([np.ones((H,W),np.float32),np.full((H,W),np.exp(.9),np.float32)])
    depth = np.ones((2,H,W),np.float32)
    seg = np.zeros((2,H,W),np.int32)
    K = np.array([[20.,0,W/2],[0,20.,H/2],[0,0,1.]])
    camera = np.stack([np.eye(4),np.eye(4)])
    poses = [{},{}]
    provenance = {'kind':'analytic numerical acceptance for the pose lane; not CAD scene generation or model training'}

assert len(radiance)==len(depth)==len(seg)==len(camera)==2
assert np.isfinite(radiance).all() and np.isfinite(depth).all()
H,W = depth.shape[1:]
scene = SimpleNamespace(cur_t=0.)
cam = SimpleNamespace(res=(W,H),intrinsics=K)
cursor = {'i':0}
calls = []
def frame(sc,ca):
    i = cursor['i']; calls.append(float(sc.cur_t))
    return dict(radiance=radiance[i],depth=depth[i],seg=seg[i])
def motion(sc,ca):
    i=cursor['i'];return dict(camera_to_world=camera[i],object_to_world=poses[i])
plugin = GenesisPhysicsEventPlugin(noise_config=DvsNoiseConfig(**noise),
    sampler_config=AdaptiveSamplingConfig(max_samples=512),
    frame_provider=frame,motion_state_provider=motion,**kwargs)
plugin.attach(scene,cam)
assert plugin.interpolator.composite == 'coverage_time_log_linear'
assert len(plugin.capture())==0
start = time.monotonic();cursor['i']=1;scene.cur_t=.01
events = plugin.capture()
assert np.isfinite(events).all()
if len(events):
    assert (events[:,0]>=0).all() and (events[:,0]<=.01+1e-10).all()
    assert (events[:,1]>=0).all() and (events[:,1]<W).all()
    assert (events[:,2]>=0).all() and (events[:,2]<H).all()
assert calls==[0.,.01]
np.save(out/'new_events.npy', events)
result = {'status':'pass','phase':'public_entrypoint_preflight_not_formal_training',
    'mode':args.mode,'release':pointer['release_commit'],'core_files_verified':17,
    'visible_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'pid':os.getpid(),
    'resolution_wh':[W,H],'endpoint_interval_s':.01,'provider_calls':calls,
    'events':len(events),'generation_wall_s':time.monotonic()-start,
    'max_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
    'sensor_kwargs':noise,'plugin_kwargs':kwargs,'sampler_max_samples':512,
    'uncertainty':plugin.uncertainty,'provenance':provenance,
    'new_events_sha256':hashlib.sha256((out/'new_events.npy').read_bytes()).hexdigest(),
    'real_data_used':False,'optimizer_updates':0,
    'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')
print(json.dumps(result,default=str),flush=True)
