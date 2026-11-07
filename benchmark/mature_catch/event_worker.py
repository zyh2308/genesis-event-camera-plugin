"""Frozen sparse EventGenesis in the existing CUDA environment.

Binary pipe accepts rendered endpoints and poses, and returns only events.
No renderer or object-state estimator runs in this process.
"""
import hashlib
import os
import json
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

def main():
    wire=sys.stdout.buffer;sys.stdout=sys.stderr
    import numpy as np
    import torch
    from event_wire import encode,decode
    from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    base=Path('/home/zhaoyuhan/Eventbased_WAM')
    profile_path=base/'frozen_baselines/robot-sparse-cuda-r2-20261102/event_sparse_reconstruction_20261102.json'
    profile=json.loads(profile_path.read_text());core=base/'runs/robot_sparse_cuda_acceptance_20261102c/core/genesis_event_plugin'
    for name,digest in profile['core_canonical_LF_sha256'].items():
        assert hashlib.sha256((core/name).read_text().replace('\r\n','\n').encode()).hexdigest()==digest,name
    assert torch.cuda.is_available();torch.set_num_threads(2)
    parity_path=Path(os.environ.get("CATCHIT_TEMPORAL_PARITY_FILE",str(base/"runs/robot_scenes/mature_catch/catchit_20261107a/adapter_validation_v1/memory_parity16.json")))
    assert json.loads(parity_path.read_text())["status"]=="PASS"
    from bounded_torch_executor import install
    install(temporal_chunk=16)
    plugin=None;current=None;access=0;started=time.monotonic()
    def reply(data):pickle.dump(encode(data),wire,protocol=5);wire.flush()
    reply(dict(ready=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),core_files_verified=17,external_execution_adapter="temporal chunks=16; all samples retained; parity validated"))
    while True:
        try:request=decode(pickle.load(sys.stdin.buffer))
        except EOFError:break
        if request['kind']=='close':break
        current=request['packet'];t=float(current['time_s'])
        if request['kind']=='reset':
            scene=SimpleNamespace(cur_t=t);camera=SimpleNamespace(res=tuple(current['resolution_wh']),intrinsics=current['intrinsics'])
            def frame(s,c):return dict(radiance=current['radiance'],depth=current['depth'],seg=current['segmentation'])
            def motion(s,c):return dict(camera_to_world=current['camera_to_world_cv'],object_to_world=current['object_poses'])
            kwargs=dict(profile['plugin_kwargs']);kwargs.update(device='cuda:0',warp_backend='torch')
            plugin=GenesisPhysicsEventPlugin(noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),frame_provider=frame,motion_state_provider=motion,**kwargs)
            plugin.attach(scene,camera);access=0
        else:
            assert plugin is not None
            scene.cur_t=t
        before=time.monotonic();events=plugin.capture();access+=1
        events=events.detach().cpu().numpy() if hasattr(events,'detach') else np.asarray(events)
        assert not len(events) or float(events[:,0].max())<=t+1e-8
        reply(dict(events=events,packet_end_s=t,endpoint_reads=access,dense_HDR_reads=0,
                   renderer_calls_in_reconstruction=0,wall_s=time.monotonic()-before))

if __name__=='__main__':main()
