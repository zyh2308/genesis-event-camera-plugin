"""Mechanism control, not downstream repair acceptance."""
from types import SimpleNamespace
from pathlib import Path
import json,numpy as np
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from check_sparse import response

expected=np.array([.2,.4,.6,.8])/.9*.01
results={}
for mode in ['next_primary','weighted','same_surface_log_linear','coverage_time_log_linear']:
    scene=SimpleNamespace(cur_t=0.)
    cam=SimpleNamespace(res=(10,8),intrinsics=np.array([[20,0,5],[0,20,4],[0,0,1.]]))
    p=GenesisPhysicsEventPlugin(noise_config=response(),device='cpu',
        frame_provider=lambda s,c:dict(radiance=np.full((8,10),np.exp(.9*s.cur_t/.01),np.float32),
            depth=np.ones((8,10),np.float32),seg=np.zeros((8,10),int)),
        motion_state_provider=lambda s,c:dict(camera_to_world=np.eye(4),object_to_world={}),
        warp_composite=mode,warp_backend='numpy',strict_fidelity=False)
    p.attach(scene,cam);p.capture();scene.cur_t=.01;ev=p.capture()
    per_pixel=ev[(ev[:,1]==5)&(ev[:,2]==4),0]
    results[mode]=dict(event_count=len(ev),example_times_ms=(per_pixel*1000).tolist(),
        expected_times_ms=(expected*1000).tolist(),
        max_timestamp_error_ms=float(np.max(np.abs(per_pixel-expected)))*1000 if len(per_pixel)==len(expected) else None)
out=dict(control='Known 10ms log-brightness ramp, two actual endpoints only',
    downstream_acceptance=False,results=results)
Path(__file__).with_name('ramp_mechanism_control.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
