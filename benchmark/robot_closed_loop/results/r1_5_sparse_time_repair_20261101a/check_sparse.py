"""Analytic controls for the original endpoint-only public capture path."""
import json
from types import SimpleNamespace
import numpy as np
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator, RenderRequiredError
from genesis_event_plugin.physics_warp import depth_aware_bidirectional_warp_projected_diagnostics as warp
from genesis_event_plugin.noise_model import DvsNoiseConfig

def response():
    return DvsNoiseConfig(pos_thres=.2,neg_thres=.2,sigma_thres=0,cutoff_hz=0,
        leak_rate_hz=0,leak_jitter_fraction=0,shot_noise_rate_hz=0,
        noise_rate_cov_decades=0,refractory_period_s=0,log_input=False,
        radiance_input=True,radiance_scale=1,radiance_white_level=1,
        response_mode='physical_log',radiometric_floor=1e-6,
        radiometric_calibration_status='uncalibrated',readout_model='ideal_readout',seed=42)

def main():
    h,w=8,10; yy,xx=np.mgrid[:h,:w]; uv=np.stack([xx,yy]).astype(float)
    d=np.ones((h,w)); ok=np.ones((h,w),bool); seg=np.zeros((h,w),int)
    a=np.ones((h,w),np.float32); b=np.full_like(a,np.exp(.9))
    def run(alpha,**kw):
        return warp(a,d,uv,d,ok,b,d,uv,d,ok,composite='coverage_time_log_linear',
                    alpha=alpha,surface_id0=seg,surface_id1=seg,**kw)
    for alpha in [0,.1,.25,.5,.75,1]:
        image,valid,diag=run(alpha)
        assert np.allclose(image,np.exp(.9*alpha),rtol=3e-7)
        assert valid.all()
    assert np.array_equal(run(0)[0],a) and np.array_equal(run(1)[0],b)
    # Motion changes the reconstructed spatial position, not just brightness.
    dot=np.zeros((h,w),np.float32)+.1;dot[3,2]=1
    dot1=np.zeros_like(dot)+.1;dot1[3,6]=1
    uv0=uv.copy();uv0[0]+=2;uv1=uv.copy();uv1[0]-=2
    out,_,_=warp(dot,d,uv0,d,ok,dot1,d,uv1,d,ok,
        composite='coverage_time_log_linear',alpha=.5,surface_id0=seg,surface_id1=seg)
    assert np.unravel_index(np.nanargmax(out),out.shape)==(3,4)
    # Partial support does not become a trusted observation.
    partial=uv.copy();partial[0]+=.25
    _,valid,diag=warp(a,d,partial,d,ok,b,d,partial,d,ok,
        composite='coverage_time_log_linear',alpha=.5,surface_id0=seg,surface_id1=seg)
    assert not valid[:,0].any() and (diag['coverage0'][:,0]<.995).all()
    scene=SimpleNamespace(cur_t=0.); cam=SimpleNamespace(res=(w,h),intrinsics=np.array([[20,0,w/2],[0,20,h/2],[0,0,1.]]))
    calls=[]
    def frame(s,c):
        calls.append(float(s.cur_t))
        return dict(radiance=a if s.cur_t==0 else b,depth=d,seg=seg)
    p=GenesisPhysicsEventPlugin(noise_config=response(),device='cpu',
        frame_provider=frame,motion_state_provider=lambda s,c:dict(camera_to_world=np.eye(4),object_to_world={}),
        warp_composite='coverage_time_log_linear',warp_backend='numpy',strict_fidelity=False)
    p.attach(scene,cam); assert len(p.capture())==0
    scene.cur_t=.01; events=p.capture()
    assert calls==[0.,.01] and p.intermediate_state_provider is None
    assert len(events)==h*w*4,len(events)
    expected=np.array([.2,.4,.6,.8])/.9*.01
    measured=np.unique(np.round(events[:,0],10))
    assert np.allclose(measured,expected,atol=2e-8),(measured,expected)
    # Unchanged endpoints yield no static events through capture itself.
    p.start_episode();scene.cur_t=.01;p.capture();scene.cur_t=.02
    assert len(p.capture())==0
    try:
        PhysicsInterpolator(cam.intrinsics,composite='coverage_time_log_linear',warp_backend='torch')
    except ValueError: pass
    else: raise AssertionError('unvalidated backend accepted')
    print(json.dumps(dict(status='PASS',controls=['log ramp','exact endpoints','geometric translation',
        'unknown support retained','public capture endpoint-only audit','analytic threshold timestamps',
        'static zero events','unsupported backend rejected'],event_timing_max_error_s=float(np.max(np.abs(measured-expected))))))

if __name__=='__main__':main()
