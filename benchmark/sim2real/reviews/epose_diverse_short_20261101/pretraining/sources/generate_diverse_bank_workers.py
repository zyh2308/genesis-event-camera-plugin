"""Matched-CAD textured camera-motion bank; immutable event core, CPU Mesa."""
import os,sys,json,time,hashlib,math
from pathlib import Path
import numpy as np,h5py,cv2,trimesh
from scipy.spatial.transform import Rotation
R=Path(os.environ['EPOSE_REPAIR_RUN']);P=Path('/home/zhaoyuhan/Eventbased_WAM');OLD=P/'runs/sim2real/20261101-epose-wood-short-research-v1'
sys.path[:0]=[str(OLD/'sources'),str(OLD/'sources/benchmark/sim2real/scripts')]
from genesis.ext.pyrender.platforms.osmesa import OSMesaPlatform
OSMesaPlatform.supports_framebuffers=lambda self:True
from generate_stage4_flow_bank import geometry_intrinsics,GeometryCameraProxy,renderer_segmentation_index_map,segmentation_render_to_link
from generate_stage5_flow_coverage_bank import radiance_to_v2e_luma_u8
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.genesis_adapter import camera_to_world_cv,link_to_world_cv
import genesis as gs
cfg=json.loads((R/'bank_protocol.json').read_text());bank=Path(os.environ.get('EPOSE_BANK_ROOT',str(R/'bank')));bank.mkdir(exist_ok=True);(bank/'preview').mkdir(exist_ok=True);worker=int(os.environ.get('BANK_WORKER_INDEX','-1'));workers=int(os.environ.get('BANK_WORKERS','1'));tag='' if worker<0 else f'_worker{worker}'
cad=R/'assets/036_wood_block/google_16k/nontextured.ply';meshfile=cad.with_name('textured.obj');mesh=trimesh.load(cad,process=False);verts=np.asarray(mesh.vertices);lo,hi=mesh.bounds
assert hashlib.sha256(cad.read_bytes()).hexdigest()=='8c653731426cf73b5b5d4ffbfa3bdd4ebb14cb82d413da82559bce1b5ee3ab37'
textured=trimesh.load(meshfile,process=False,force='mesh');assert np.max(np.abs(textured.bounds-mesh.bounds))<1e-7
corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]);keys=np.vstack([(lo+hi)/2,corners]);np.save(R/'cad_keypoints.npy',keys);np.save(R/'cad_vertices.npy',verts)
gs.init(backend=gs.cpu,logging_level='warning')
scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.001,gravity=(0,0,0)),vis_options=gs.options.VisOptions(ambient_light=(.6,.6,.6),segmentation_level='link'),show_viewer=False)
ground=scene.add_entity(gs.morphs.Plane(),surface=gs.surfaces.Default(color=(.25,.29,.32,1)))
entity=scene.add_entity(gs.morphs.Mesh(file=str(meshfile),pos=(0,0,.25),fixed=True,collision=False,decimate=False,file_meshes_are_zup=True),surface=gs.surfaces.Default())
cam=scene.add_camera(res=(346,260),pos=(0,-.65,.40),lookat=(0,0,.25),fov=float(np.degrees(2*np.arctan(260/(2*293.19)))),near=.03,far=4,GUI=False)
scene.build();rng=np.random.default_rng(cfg['physical_seed']);manifest={'protocol':cfg,'cad_sha256':hashlib.sha256(cad.read_bytes()).hexdigest(),'mesh_bounds_error_m':float(np.max(np.abs(textured.bounds-mesh.bounds))),'renderer':'CPU llvmpipe OSMesa; process-local FBO compatibility; no shadows/reflections','rollouts':[]}
with h5py.File(bank/f'shared_trajectory{tag}.h5','w') as h:
 ep=h.create_group('episodes')
 for i in range(cfg['train_rollouts']+cfg['validation_rollouts']):
  t0=time.time();split='train' if i<cfg['train_rollouts'] else 'validation';j=i if split=='train' else i-cfg['train_rollouts'];name=f'{split}_{j:02d}'
  target=np.array([0.,0.,.25]);angles=rng.uniform([-.35,-.35,-math.pi],[.35,.35,math.pi]);entity.set_pos(target);entity.set_quat(np.roll(Rotation.from_euler('xyz',angles).as_quat(),1))
  az=rng.uniform(-.85,.85);distance=rng.uniform(.48,.88);height=rng.uniform(.08,.30);offset=rng.uniform([-.12,-.08],[.12,.08]);velocity=rng.uniform(.10,.50)*rng.choice([-1,1]);vz=rng.uniform(-.08,.08)
  if worker>=0 and i%workers!=worker:continue
  # CAD origin is not its geometric center. Orbit the transformed CAD centroid.
  center=target+Rotation.from_euler('xyz',angles).as_matrix()@((lo+hi)/2)
  for attempt in range(6):
   good=True
   for t in [0,cfg['duration_ms']*.001]:
    a=az+velocity*t/distance;cam.set_pose(pos=center+np.array([distance*np.sin(a),-distance*np.cos(a),height+vz*t]),lookat=center+np.array([offset[0],0,offset[1]]));_,KK=geometry_intrinsics(cam);CC=camera_to_world_cv(cam);TT=link_to_world_cv(entity.links[0]);xx=(np.linalg.inv(CC)@TT@np.c_[keys,np.ones(9)].T).T[:,:3];uu=(KK@xx.T).T;uu=uu[:,:2]/uu[:,2:]
    good=good and (uu>[3,3]).all() and (uu<[343,257]).all()
   if good:break
   distance*=1.12;offset*=.8
  assert good,(name,'physical visibility preflight failed')
  cam.set_pose(pos=center+np.array([distance*np.sin(az),-distance*np.cos(az),height]),lookat=center+np.array([offset[0],0,offset[1]]))
  origin=float(scene.cur_t);_,K=geometry_intrinsics(cam);proxy=GeometryCameraProxy(cam,K);cache={}
  def frame_provider(sc,ca):
   rgb,_,_,_=ca.render(rgb=True,depth=False,segmentation=False,radiance=True,force_render=True)
   _,dep,seg,_=ca.render(rgb=False,depth=True,segmentation=True,force_render=True)
   f={'rgb':np.asarray(rgb,dtype=np.float32),'radiance':np.asarray(rgb,dtype=np.float32),'depth':np.asarray(dep,dtype=np.float32),'seg':segmentation_render_to_link(np.asarray(seg,dtype=np.int32),renderer_segmentation_index_map(ca))};cache['f']=f;return f
  def state_provider(sc,ca):return {'camera_to_world':camera_to_world_cv(ca),'object_to_world':{int(l.idx):link_to_world_cv(l) for l in sc.rigid_solver.links}}
  noise=DvsNoiseConfig(pos_thres=.2,neg_thres=.2,sigma_thres=.03,cutoff_hz=300.,leak_rate_hz=.01,leak_jitter_fraction=.1,noise_rate_cov_decades=.1,shot_noise_rate_hz=.001,refractory_period_s=.0005,seed=42,radiance_input=True,response_mode='physical_log',radiometric_calibration_status='uncalibrated')
  plugin=GenesisPhysicsEventPlugin(input_space='linear',response_mode='physical_log',radiometric_calibration_status='uncalibrated',noise_config=noise,sampler_config=AdaptiveSamplingConfig(),frame_provider=frame_provider,motion_state_provider=state_provider,device='cpu',warp_backend='numpy');plugin.attach(scene,proxy).start_episode()
  events=[];times=[];displays=[];poses=[];cameras=[];errs=[];radiance_max=[]
  for k in range(cfg['duration_ms']+1):
   t=k*.001;a=az+velocity*t/distance;eye=center+np.array([distance*np.sin(a),-distance*np.cos(a),height+vz*t]);look=center+np.array([offset[0],0,offset[1]])
   cam.set_pose(pos=eye,lookat=look)
   if k:scene.step()
   ev=np.asarray(plugin.capture(),dtype=float).reshape(-1,4);f=cache['f'];T=link_to_world_cv(entity.links[0]);C=camera_to_world_cv(cam)
   if k:ev[:,0]=np.round(ev[:,0]-origin,12);events.append(ev)
   times.append(round(float(scene.cur_t)-origin,12));displays.append(radiance_to_v2e_luma_u8(f['radiance']));poses.append(T);cameras.append(C);radiance_max.append(float(f['radiance'].max()))
   if k%20==0:print(json.dumps({'phase':'physical_frame','worker':worker,'rollout':name,'frame_ms':k,'wall_s':time.time()-t0}),flush=True)
   if k%10==0:
    xyz=(np.linalg.inv(C)@T@np.c_[verts,np.ones(len(verts))].T).T[:,:3];uv=(K@xyz.T).T;uv=uv[:,:2]/uv[:,2:];yy,xx=np.where(f['seg']==int(entity.links[0].idx));assert len(xx)>200,(name,k,'object missing')
    err=float(np.max(np.abs(np.r_[uv.min(0),uv.max(0)]-np.array([xx.min(),yy.min(),xx.max(),yy.max()]))));assert err<2.5,(name,k,err);errs.append(err)
    xyz=(np.linalg.inv(C)@T@np.c_[keys,np.ones(9)].T).T[:,:3];uv=(K@xyz.T).T;uv=uv[:,:2]/uv[:,2:];assert (uv>[2,2]).all() and (uv<[344,258]).all(),(name,k,'corners off screen',uv.tolist())
    if k==cfg['warmup_ms']:cv2.imwrite(str(bank/'preview'/f'{name}.png'),cv2.cvtColor(np.uint8(np.clip(f['radiance'],0,1)*255),cv2.COLOR_RGB2BGR))
  g=ep.create_group(name);g.attrs['split']=split
  for key,value in [('frame_timestamps_s',np.array(times)),('camera_intrinsics_K_px',K),('camera_to_world_cv',np.array(cameras)),('target_pose',np.array(poses)),('v2e_display_luma_u8',np.array(displays)),('eventgenesis_events_txyp',np.concatenate(events))]:g.create_dataset(key,data=value,compression='gzip')
  item={'rollout_id':name,'split':split,'events':sum(len(e) for e in events),'max_geometry_error_px':max(errs),'radiance_max':max(radiance_max),'camera_azimuth':az,'camera_distance':distance,'camera_height':height,'look_offset':offset.tolist(),'camera_speed_m_s':velocity,'object_euler_rad':angles.tolist(),'wall_s':time.time()-t0};manifest['rollouts'].append(item);(bank/f'manifest{tag}.json').write_text(json.dumps(manifest,indent=2));h.flush();plugin.close();print(json.dumps(item),flush=True)
  if os.environ.get('EPOSE_BANK_PROBE')=='1':break
(bank/'config.json').write_text(json.dumps({'simulation':{'target_box':{'size_m':(hi-lo).tolist()}}},indent=2));gs.destroy();print('BANK_COMPLETE',flush=True)
