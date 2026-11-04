"""Native HDR adapter -> integrated GPU plugin -> original recorder, actual Genesis."""
import gc,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np,h5py,torch
ROOT=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM')
os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
sys.path[:0]=[str(ROOT/'release_candidate_static'),str(P/'runtime/sim2real/genesis_hdr_v2')]
import genesis as gs
from genesis.ext.pyrender.platforms.osmesa import OSMesaPlatform
OSMesaPlatform.supports_framebuffers=lambda self:True
from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.genesis_adapter import make_frame_provider,camera_to_world_cv
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
torch.set_num_threads(2)
profile=json.loads((ROOT/'release_candidate_static/stage9_sensor_profile.json').read_text())
out=ROOT/'genesis_smoke_accepted';out.mkdir(exist_ok=False)
gs.init(backend=gs.cpu,logging_level='warning')
scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.001,gravity=(0,0,0)),vis_options=gs.options.VisOptions(ambient_light=(.4,.4,.4)),show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(pos=(0,0,.15),size=(.3,.3,.3)),surface=gs.surfaces.Default(color=(.8,.2,.1,1.)))
cam=scene.add_camera(res=(320,240),pos=(.8,.8,.6),lookat=(0,0,.15),fov=50,GUI=False)
scene.build()
context=cam._rasterizer._context
context.ambient_light=np.asarray((10.,10.,10.),dtype=np.float32)
context._scene.ambient_light=context.ambient_light
context.jit.set_light(context._scene,context._scene.light_nodes,context.ambient_light)
provider=make_frame_provider(cam,native_radiance=True)
current={};motion=lambda *args:dict(camera_to_world=camera_to_world_cv(cam),object_to_world={})
plugins=[]
for name,cls in [('reference',GenesisPhysicsEventPlugin),('optimized',OptimizedGenesisPhysicsEventPlugin)]:
    plugin=cls(noise_config=DvsNoiseConfig(**profile['noise_config_kwargs']),sampler_config=AdaptiveSamplingConfig(**profile['sampler_kwargs']),
        frame_provider=lambda *args:current['payload'],motion_state_provider=motion,output_dir=str(out/name),**profile['plugin_kwargs'])
    plugin.attach(scene,cam).start_episode();plugins.append(plugin)
assert plugins[1].optimization_active,plugins[1].optimization_status
events=[[],[]];render=[];capture=[[],[]];radiance=[]
for j in range(3):
    if j:
        for _ in range(10):scene.step()
    cam.set_pose(pos=(.8-.005*j,.8,.6),lookat=(0,0,.15))
    begin=time.perf_counter();current['payload']=provider(scene,cam);render.append(time.perf_counter()-begin)
    image=np.asarray(current['payload']['radiance']);assert image.dtype==np.float32 and np.isfinite(image).all()
    assert (image>1).any(),'native HDR clipped to display range'
    radiance.append(dict(shape=list(image.shape),max=float(image.max()),sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    for k,plugin in enumerate(plugins):
        begin=time.perf_counter();events[k].append(plugin.capture());torch.cuda.synchronize();capture[k].append(time.perf_counter()-begin)
for p in plugins:p.end_episode();p.close()
a,b=[np.concatenate(x) for x in events]
assert a.shape==b.shape and len(a)>0 and np.array_equal(a[:,1:],b[:,1:])
error=float(np.abs(a[:,0]-b[:,0]).max(initial=0));assert error<=3e-9,error
files=[next((out/name).glob('*.h5')) for name in ['reference','optimized']]
payloads=[]
for path in files:
    with h5py.File(path,'r') as h:
        values={}
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset) and 'events' in name:values[name]=np.asarray(obj)
        h.visititems(visit);payloads.append(values)
assert payloads[0].keys()==payloads[1].keys() and len(payloads[0])>0
assert all(np.array_equal(payloads[0][k],payloads[1][k]) for k in payloads[0])
report=dict(passed=True,actual_genesis=True,native_hdr=True,genesis_module=str(Path(gs.__file__).resolve()),
    simulation_times_s=[0,.01,.02],render_wall_s=render,capture_wall_s=capture,radiance=radiance,
    event_count=len(a),raw_bitwise_equal=bool(np.array_equal(a,b)),max_timestamp_error_s=error,recorder_bitwise_equal=True,
    scope='integration smoke; not the five-scene performance benchmark',optimization_active=True)
(ROOT/'genesis_smoke_accepted.json').write_text(json.dumps(report,indent=2)+'\n')
gs.destroy();print('GENESIS_INTEGRATION_PASS',json.dumps(report),flush=True)
