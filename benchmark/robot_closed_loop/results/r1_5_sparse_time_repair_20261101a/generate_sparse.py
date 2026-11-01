"""Endpoint-cache adapter invoking public capture, never dense observations."""
from pathlib import Path
from types import SimpleNamespace
import os,json,time,hashlib,csv,shutil,sys
import numpy as np
import h5py
import torch
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from check_sparse import response

BASE=Path('/home/zhaoyuhan/Eventbased_WAM')
CACHE=BASE/'repositories/genesis-event-camera-plugin-r1-ba137bd/benchmark/robot_closed_loop/results/r1_shared_ideal_small9_20261018a/trajectory_cache'
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def generate(index,max_frames=None):
    torch.set_num_threads(1); started=time.perf_counter()
    origin=CACHE/f'episode_{index:03d}';dest=ROOT/'sparse_events'/'episodes'/f'episode_{index:03d}'
    dest.mkdir(parents=True,exist_ok=True); chunks=[];audit=[];access=[]
    with h5py.File(origin/'trajectory_cache.h5','r') as h:
        ts=np.asarray(h['timestamps_s'][:],dtype=np.float64)
        width,height=map(int,h.attrs['camera_resolution_wh'])
        focal=height/(2*np.tan(np.deg2rad(float(h.attrs['vertical_fov_deg']))/2))
        cam=SimpleNamespace(res=(width,height),intrinsics=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1.]]))
        scene=SimpleNamespace(cur_t=float(ts[0])); current=[0]
        ids=h['segmentation_pose_ids'][:]
        def frame(s,c):
            k=current[0];access.append({'kind':'frame','index':k,'t':float(ts[k]),'datasets':['radiance','depth','segmentation']})
            return dict(radiance=h['radiance'][k],depth=h['depth'][k],seg=h['segmentation'][k])
        def motion(s,c):
            k=current[0];poses=h['object_poses'][k]
            access.append({'kind':'geometry','index':k,'t':float(ts[k]),'datasets':['camera_to_world_cv','object_poses']})
            return dict(camera_to_world=h['camera_to_world_cv'][k],object_to_world={int(v):poses[j] for j,v in enumerate(ids)})
        plugin=GenesisPhysicsEventPlugin(noise_config=response(),device='cpu',frame_provider=frame,
            motion_state_provider=motion,warp_composite='coverage_time_log_linear',warp_backend='numpy',
            disocclusion_policy='hold',strict_fidelity=False)
        plugin.attach(scene,cam)
        count=len(ts) if max_frames is None else min(int(max_frames),len(ts))
        for k in range(count):
            current[0]=k;scene.cur_t=float(ts[k]);events=plugin.capture()
            if len(events):chunks.append(events)
            audit.append(dict(index=k,t=float(ts[k]),events=len(events),uncertainty=plugin.uncertainty))
            if k%10==0 or k==count-1:
                (dest/'progress.json').write_text(json.dumps(dict(episode=index,frame=k,total=count,
                    events=sum(map(len,chunks)),wall_s=time.perf_counter()-started)))
        assert [x['index'] for x in access if x['kind']=='frame']==list(range(count))
        assert plugin.intermediate_state_provider is None
    events=np.concatenate(chunks) if chunks else np.empty((0,4),dtype=float)
    if len(events):events=events[np.argsort(events[:,0],kind='stable')]
    np.savez_compressed(dest/'direct_hdr_events_1khz_2khz.npz',events_2000hz=events)
    metadata=json.loads((origin/'episode_result.json').read_text())
    fields=['episode_id','seed','launch_speed_m_s','start_y_m','first_contact_s','lift_start_s','sim_duration_s']
    row={key:(index if key=='episode_id' else metadata.get(key)) for key in fields}
    with (dest/'episode.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerow(row)
    shutil.copyfile(origin/'oracle_trace.jsonl',dest/'state_control_trace.jsonl')
    def serial(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        return str(v)
    (dest/'reconstruction_audit.json').write_text(json.dumps(dict(
        endpoint_only=True,renderer_calls=0,dense_radiance_reads=0,intermediate_state_provider=None,
        public_path='GenesisPhysicsEventPlugin.capture -> PhysicsInterpolator.interpolate -> projected geometric warp -> DVS',
        mode='coverage_time_log_linear',strict_fidelity=False,min_coverage_fraction=.995,
        approximate_unknown_regions=True,source_path=str(origin),
        source_cache_sha256=sha(origin/'trajectory_cache.h5'),
        timestamp_sha256=hashlib.sha256(ts.tobytes()).hexdigest(),
        access=access,intervals=audit),default=serial))
    result=dict(episode=index,frames=count,events=len(events),wall_s=time.perf_counter()-started,
        output_sha256=sha(dest/'direct_hdr_events_1khz_2khz.npz'))
    (dest/'complete.json').write_text(json.dumps(result));print(json.dumps(result),flush=True);return result

if __name__=='__main__':generate(int(sys.argv[1]),int(sys.argv[2]) if len(sys.argv)>2 else None)
