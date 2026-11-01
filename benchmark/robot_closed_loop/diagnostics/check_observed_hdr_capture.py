from pathlib import Path
import sys,json
sys.path.insert(0,sys.argv[1])
import numpy as np
from types import SimpleNamespace
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_interpolator import RenderRequiredError

def config():
 return DvsNoiseConfig(pos_thres=.2,neg_thres=.2,sigma_thres=0,cutoff_hz=0,leak_rate_hz=0,
  leak_jitter_fraction=0,shot_noise_rate_hz=0,noise_rate_cov_decades=0,refractory_period_s=0,
  log_input=False,radiance_input=True,response_mode='physical_log',radiometric_floor=1e-6,
  radiometric_calibration_status='uncalibrated',readout_model='ideal_readout',seed=42)
def make():
 scene=SimpleNamespace(cur_t=0.)
 cam=SimpleNamespace(res=(4,3),intrinsics=np.array([[4.,0,2],[0,4.,1.5],[0,0,1.]]))
 plugin=GenesisPhysicsEventPlugin(noise_config=config(),strict_fidelity=True,max_photometric_interval_s=.0005,device='cpu',
  motion_state_provider=lambda s,c: {'camera_to_world':np.eye(4),'object_to_world':{}})
 plugin.attach(scene,cam);plugin.start_episode()
 return plugin,scene
def cap(p,s,t,image):
 s.cur_t=t
 return p.capture_observed(image,t,provenance='unit controlled observed radiance',available_at_s=t)
def canonical(a):
 a=np.asarray(a).reshape(-1,4)
 return a[np.lexsort((a[:,3],a[:,2],a[:,1],a[:,0]))] if len(a) else a

checks={}
p,s=make();direct=DvsEmulator(res=(3,4),config=config(),device='cpu')
images=[np.full((3,4),np.exp(v),np.float32) for v in np.r_[np.linspace(0,1,21),np.linspace(.95,-.3,26)]]
got=[];expected=[]
for i,img in enumerate(images):
 t=i*.0005
 actual=cap(p,s,t,img)
 if i==0:direct.initialize(img,t)
 else:
  target=direct.generate_events(img,t)
  got.extend(actual);expected.extend(target)
g,e=canonical(got),canonical(expected)
assert np.array_equal(g,e),(g.shape,e.shape)
checks['continuous_up_down_ramp_exact_event_equivalence']={'events':len(g),'max_time_error_s':0.}

p,s=make();assert len(cap(p,s,0.,np.ones((3,4))))==0
before=p.total_steps
s.cur_t=.001
try:cap(p,s,.001,np.full((3,4),2.))
except RenderRequiredError as exc:assert exc.requests[0].t==.0005
else:raise AssertionError('missing actual sample was accepted')
assert p.total_steps==before and p._last_observed_time==0.
cap(p,s,.0005,np.full((3,4),1.1))
cap(p,s,.001,np.full((3,4),1.2))
checks['missing_frame_rejected_without_advancing_sensor_then_recoverable']=True

p,s=make();cap(p,s,0.,np.ones((3,4)))
for kind in ['future','available_before_observed','nonfinite','negative','wrong_shape','repeat_time','missing_provenance','integer_display']:
 s.cur_t=.0005;image=np.ones((3,4));t=.0005;available=t;provenance='controlled'
 if kind=='future':available=.001
 if kind=='available_before_observed':available=0.
 if kind=='nonfinite':image[0,0]=np.nan
 if kind=='negative':image[0,0]=-1
 if kind=='wrong_shape':image=np.ones((4,4))
 if kind=='repeat_time':t=0.;available=0.
 if kind=='missing_provenance':provenance=''
 if kind=='integer_display':image=np.ones((3,4),dtype=np.uint8)
 try:p.capture_observed(image,t,provenance=provenance,available_at_s=available)
 except ValueError:pass
 else:raise AssertionError(kind+' was accepted')
 assert p._last_observed_time==0. and p.total_steps==1
checks['invalid_input_and_causality_fail_closed_before_sensor_update']=True
try:p.capture()
except RuntimeError as exc:assert 'capture modes' in str(exc)
else:raise AssertionError('capture modes were mixed')
p.start_episode();assert len(cap(p,s,0.,np.full((3,4),5.)))==0
assert len(cap(p,s,.0005,np.full((3,4),5.)))==0
checks['episode_reset_static_identity_and_mode_separation']=True
p,s=make();p.input_space='srgb'
try:cap(p,s,0.,np.ones((3,4),np.float32))
except ValueError:pass
else:raise AssertionError('sRGB declaration accepted as actual linear HDR')
checks['display_dtype_and_nonlinear_declaration_rejected']=True
scene=SimpleNamespace(cur_t=0.)
cam=SimpleNamespace(res=(4,3),intrinsics=np.array([[4.,0,2],[0,4.,1.5],[0,0,1.]]))
legacy=GenesisPhysicsEventPlugin(noise_config=config(),device='cpu',
 motion_state_provider=lambda s,c:{'camera_to_world':np.eye(4),'object_to_world':{}},
 frame_provider=lambda s,c:{'rgb':np.ones((3,4,3),np.float32),'radiance':np.ones((3,4),np.float32),
  'depth':np.ones((3,4),np.float32),'seg':np.zeros((3,4),np.int32)})
legacy.attach(scene,cam);legacy.start_episode()
for t in [0.,.001,.002]:
 scene.cur_t=t;assert len(legacy.capture())==0
checks['existing_default_capture_static_regression']=True
print(json.dumps({'status':'PASS','checks':checks},indent=2))
