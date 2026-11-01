"""Small Genesis bank of the exact E-POSE wood-block mesh; frozen event core."""
import os,sys,json,math,time,hashlib
from pathlib import Path
import numpy as np,h5py,cv2,trimesh
from scipy.spatial.transform import Rotation
from PIL import Image
R=Path(os.environ['EPOSE_SHORT_RUN']);S=R/'sources';sys.path[:0]=[str(S),str(S/'benchmark/sim2real/scripts')]
from generate_stage4_flow_bank import (geometry_intrinsics,GeometryCameraProxy,renderer_segmentation_index_map,segmentation_render_to_link)
from generate_stage5_flow_coverage_bank import radiance_to_v2e_luma_u8
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.genesis_adapter import camera_to_world_cv,link_to_world_cv
import genesis as gs
P=Path('/home/zhaoyuhan/Eventbased_WAM');cad=P/'external/sim2real_assets/epose_wood_block_20261024/nontextured.ply'
cad_mesh=trimesh.load(cad,process=False)
vertices=np.asarray(cad_mesh.vertices);lo,hi=vertices.min(0),vertices.max(0)
render_mesh=R/'cad_woodblock.obj';cad_mesh.export(render_mesh,file_type='obj',digits=10)
assert np.allclose(np.asarray(trimesh.load(render_mesh,process=False).bounds),np.array([lo,hi]),atol=1e-8)
corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
keypoints=np.vstack([(lo+hi)/2,corners]);np.save(R/'cad_keypoints.npy',keypoints);np.save(R/'cad_vertices.npy',vertices)
bank=R/'bank';bank.mkdir(exist_ok=True);preview=bank/'preview';preview.mkdir(exist_ok=True)
manifest={'scope':'exact CAD-matched small synthetic-only pilot, not full E-POSE benchmark','cad':str(cad),'cad_sha256':hashlib.sha256(cad.read_bytes()).hexdigest(),'render_mesh_obj':str(render_mesh),'render_mesh_sha256':hashlib.sha256(render_mesh.read_bytes()).hexdigest(),'mesh_conversion':'lossless coordinate-preserving PLY-to-OBJ; Genesis does not accept PLY; AABB agreement verified to1e-8m','cad_aabb':[lo.tolist(),hi.tolist()],'keypoints':keypoints.tolist(),'core':'e79e656ac00f7847b98db9e12bf43b093f3afff1','rollouts':[],'genesis_source':gs.__file__,'genesis_version':gs.__version__}
assert manifest['cad_sha256']=='8c653731426cf73b5b5d4ffbfa3bdd4ebb14cb82d413da82559bce1b5ee3ab37'
gs.init(backend=gs.cpu,logging_level='warning')
scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.001,gravity=(0,0,0)),vis_options=gs.options.VisOptions(ambient_light=(.45,.45,.45),segmentation_level='link'),show_viewer=False)
ground=scene.add_entity(gs.morphs.Plane(),surface=gs.surfaces.Default(color=(.22,.25,.28,1)))
entity=scene.add_entity(gs.morphs.Mesh(file=str(render_mesh),pos=(0,0,.22),scale=1.,fixed=True,collision=False,decimate=False,file_meshes_are_zup=True),surface=gs.surfaces.Default(color=(.63,.42,.24,1)))
cam=scene.add_camera(res=(346,260),pos=(0,-.75,.23),lookat=(0,0,.22),fov=float(np.degrees(2*np.arctan(260/(2*293.19)))),near=.03,far=4,GUI=False)
scene.build()
manifest['scene_reuse']='one compiled static environment; independently reset object/camera poses and fresh sensor instance per rollout; shared plane/material; episode clocks normalized to zero'
with h5py.File(bank/'shared_trajectory.h5','w') as h:
 eps=h.create_group('episodes')
 for i in range(6):
  name=('train_' if i<4 else 'validation_')+f'{i if i<4 else i-4:02d}';out=bank/name;out.mkdir(exist_ok=True)
  base=np.array([[-.07,0,.21],[.06,.03,.23],[0,-.06,.22],[.04,.04,.25],[-.035,.025,.235],[.02,-.025,.215]][i])
  angles=np.array([[.15,.12,-.2],[-.18,.1,.2],[.12,-.15,.05],[-.1,-.12,-.1],[.05,.16,.12],[-.14,.02,-.15]][i])
  q=Rotation.from_euler('xyz',angles).as_quat();q=np.roll(q,1)
  entity.set_pos(base);entity.set_quat(q);cam.set_pose(pos=(.04*math.sin(i),-.75-.04*i,.23),lookat=(0,0,.22))
  origin=float(scene.cur_t);_,K=geometry_intrinsics(cam);proxy=GeometryCameraProxy(cam,K);cache={}
  def frame_provider(sc,ca):
   rgb,_,_,_=ca.render(rgb=True,depth=False,segmentation=False,radiance=True,force_render=True)
   _,dep,seg,_=ca.render(rgb=False,depth=True,segmentation=True,force_render=True)
   payload={'rgb':np.asarray(rgb,dtype=np.float32),'radiance':np.asarray(rgb,dtype=np.float32),'depth':np.asarray(dep,dtype=np.float32),'seg':segmentation_render_to_link(np.asarray(seg,dtype=np.int32),renderer_segmentation_index_map(ca))};cache['frame']=payload;return payload
  def state_provider(sc,ca):return {'camera_to_world':camera_to_world_cv(ca),'object_to_world':{int(l.idx):link_to_world_cv(l) for l in sc.rigid_solver.links}}
  sensor=DvsNoiseConfig(pos_thres=.2,neg_thres=.2,sigma_thres=.03,cutoff_hz=300.,leak_rate_hz=.01,leak_jitter_fraction=.1,noise_rate_cov_decades=.1,shot_noise_rate_hz=.001,refractory_period_s=.0005,seed=42,radiance_input=True,response_mode='physical_log',radiometric_calibration_status='uncalibrated')
  plugin=GenesisPhysicsEventPlugin(input_space='linear',response_mode='physical_log',radiometric_calibration_status='uncalibrated',noise_config=sensor,sampler_config=AdaptiveSamplingConfig(),frame_provider=frame_provider,motion_state_provider=state_provider,device='cpu',warp_backend='numpy')
  plugin.attach(scene,proxy).start_episode();times=[];rad=[];display=[];poses=[];cp=[];events=[];bbox_errors=[];dense_errors=[]
  for k in range(5):
   if k:
    for j in range(10):
     alpha=(k-1+(j+1)/10)/4
     pos=base+np.array([(.025 if i%2==0 else -.025)*alpha,.006*math.sin(alpha*math.pi),.012*(1 if i%3 else -1)*alpha])
     a=angles+np.array([.09*alpha,-.10*alpha,(.14 if i%2 else -.14)*alpha]);q=np.roll(Rotation.from_euler('xyz',a).as_quat(),1)
     entity.set_pos(pos);entity.set_quat(q);scene.step()
   ev=np.asarray(plugin.capture(),dtype=float).reshape(-1,4);frame=cache['frame'];T=link_to_world_cv(entity.links[0]);C=camera_to_world_cv(cam)
   times.append(round(float(scene.cur_t)-origin,12));rad.append(frame['radiance']);display.append(radiance_to_v2e_luma_u8(frame['radiance']));poses.append(T);cp.append(C)
   if k:
    ev[:,0]=np.round(ev[:,0]-origin,12);events.append(ev)
   xyz=(np.linalg.inv(C)@T@np.c_[corners,np.ones(8)].T).T[:,:3];uv=(K@xyz.T).T;uv=uv[:,:2]/uv[:,2:]
   yy,xx=np.where(frame['seg']==int(entity.links[0].idx));assert len(xx)>200,'object is invisible'
   projected=np.r_[uv.min(0),uv.max(0)];segbox=np.array([xx.min(),yy.min(),xx.max(),yy.max()]);err=np.max(np.abs(projected-segbox));bbox_errors.append(float(err));assert err<9,(name,k,projected,segbox,err)
   vxyz=(np.linalg.inv(C)@T@np.c_[vertices,np.ones(len(vertices))].T).T[:,:3];vuv=(K@vxyz.T).T;vuv=vuv[:,:2]/vuv[:,2:];densebox=np.r_[vuv.min(0),vuv.max(0)];derr=float(np.max(np.abs(densebox-segbox)));dense_errors.append(derr);assert derr<2.5,(name,k,'dense mesh projection',densebox,segbox,derr)
   if k==0:
    im=np.uint8(np.clip(frame['radiance'],0,1)*255);im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    for v in uv.astype(int):cv2.circle(im,tuple(v),2,(0,255,0),-1)
    cv2.imwrite(str(preview/f'{name}.png'),im)
  g=eps.create_group(name);g.attrs['split']='train' if i<4 else 'validation';g.create_dataset('frame_timestamps_s',data=np.array(times));g.create_dataset('camera_intrinsics_K_px',data=K);g.create_dataset('camera_to_world_cv',data=np.array(cp));g.create_dataset('target_pose',data=np.array(poses));g.create_dataset('v2e_display_luma_u8',data=np.array(display),compression='gzip');g.create_dataset('eventgenesis_events_txyp',data=np.concatenate(events) if events else np.zeros((0,4)),compression='gzip')
  np.save(out/'radiance_frames.npy',np.array(rad,dtype=np.float32));np.save(out/'times.npy',np.array(times));(out/'v2e_input').mkdir()
  for k,im in enumerate(display):Image.fromarray(im).save(out/'v2e_input'/f'{k:08d}.png')
  item={'rollout_id':name,'split':g.attrs['split'],'eventgenesis_events':int(sum(len(v) for v in events)),'bbox_projection_error_max_px':max(bbox_errors),'dense_mesh_bbox_projection_error_max_px':max(dense_errors),'scene_clock_origin_s':origin,'frame_times_s':times};manifest['rollouts'].append(item);(bank/'manifest.json').write_text(json.dumps(manifest,indent=2));h.flush();print(json.dumps(item),flush=True)
  plugin.close()
(bank/'config.json').write_text(json.dumps({'simulation':{'target_box':{'size_m':(hi-lo).tolist()}},'geometry':'exact original CAD coordinates, keypoint centroid plus ordered min/max corners; no centered generic cube substitution'},indent=2))
gs.destroy()
print('BANK_COMPLETE',flush=True)
