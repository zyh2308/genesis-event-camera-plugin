"""Render-only native HDR/LDR lamp probe on archived Catch It! physical states.

No scene.step, no policy inference, no event generator, no frozen-core changes.
Run using the existing g1 Genesis HDR environment/PYTHONPATH overlay.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--helpers',type=Path,required=True);p.add_argument('--intensities',type=float,nargs='+',default=[0,25,100,400,1600]);p.add_argument('--release-offsets',type=float,nargs='+',default=[.05,.2,.4]);a=p.parse_args()
 assert not a.output.exists(),a.output;a.output.mkdir(parents=True);started=time.monotonic()
 import numpy as np,cv2,h5py,imageio.v2 as imageio,genesis as gs
 cv2.setNumThreads(2)
 def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
 protocol=json.loads((a.run/'protocol.json').read_text());result=json.loads((a.run/'episode_000/result.json').read_text());trace=json.loads((a.run/'episode_000/trace.json').read_text());sensor=a.run/'migration_audit/sensor_episode_000';release=float(result['initial']['release_time_s'])
 assert protocol['lighting']=='normal','Use a normal-light archived source to preserve initializer setting'
 with h5py.File(sensor/'sparse_hdr_endpoints.h5','r') as h:
  times=h['time_s'][:];K=h['intrinsics'][:];height,width=h['radiance'].shape[1:3];T0=h['camera_to_world_cv'][0]
  states=[]
  for offset in a.release_offsets:
   target=release+offset;row=min(trace,key=lambda x:abs(x['t']-target));j=int(np.argmin(abs(times-row['t'])));assert abs(float(times[j])-row['t'])<1e-6,(row['t'],times[j]);states.append((offset,row,h['camera_to_world_cv'][j]))
 sys.path.insert(0,str(a.helpers));from rgbd_sphere_initialization import initialize_sphere_rgbd
 rgb=imageio.imread(sensor/'initial_rgb.png');depth0=np.load(sensor/'initial_depth_camera_z.npy');initial=initialize_sphere_rgbd(rgb,depth0,K,T0,.04)
 # This is the exact archived t=0 initialization; no nonexistent initial qpos is invented.
 (a.output/'INITIALIZATION_CHECK.json').write_text(json.dumps({'source':'exact archived actual normal-light RGB-D, checked once; no rerendered initial qpos assumed','valid':initial.get('valid',False),'detection':initial,'rgb_sha256':hashlib.sha256((sensor/'initial_rgb.png').read_bytes()).hexdigest(),'depth_sha256':hashlib.sha256((sensor/'initial_depth_camera_z.npy').read_bytes()).hexdigest()},indent=2));assert initial['valid'],initial
 gs.init(backend=gs.cpu,precision='64',logging_level='warning');lamp_pos=np.array([0.,-.5,2.5]);scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.002),vis_options=gs.options.VisOptions(segmentation_level='link',ambient_light=(.1,.1,.1),lights=[dict(type='directional',dir=(-1,-1,-1),color=(1.,1.,1.),intensity=5.),dict(type='point',pos=tuple(lamp_pos),color=(1.,1.,1.),intensity=0.)]),show_viewer=False)
 robot=scene.add_entity(gs.morphs.MJCF(file=str(a.run/'migration_audit/genesis_task.xml'),default_armature=0.,convexify=True))
 if protocol['background_wall']:scene.add_entity(gs.morphs.Box(pos=(0.,4.,1.6),size=(4.,.02,3.2),fixed=True,collision=False),surface=gs.surfaces.Default(color=(.95,.95,.95,1.)))
 fov=math.degrees(2*math.atan(height/(2*float(K[1,1]))));camera=scene.add_camera(res=(width,height),pos=(0.,0.,1.4),lookat=(0.,1.,1.),fov=fov,GUI=False);scene.build();assert np.allclose(array(camera.intrinsics),K,atol=1e-6),array(camera.intrinsics)
 native_scene=scene.visualizer.context._scene;lamps=[n for n in native_scene.light_nodes if type(n.light).__name__=='PointLight' and np.linalg.norm(native_scene.get_pose(n)[:3,3]-lamp_pos)<1e-5];assert len(lamps)==1,'Native lamp API/identity differs: '+str([(type(n.light).__name__,native_scene.get_pose(n).tolist()) for n in native_scene.light_nodes]);lamp=lamps[0]
 hand=robot.get_link('link6');ball=robot.get_link('object');target_id=None;records=[];maximum_pose_error=0.;kernel=np.ones((3,3),np.uint8)
 def values(x):
  return {'n':int(x.size),'median':float(np.median(x)) if x.size else None,'p95':float(np.percentile(x,95)) if x.size else None}
 def linlog(x):
  x=np.asarray(x,dtype=float);return np.where(x<=20,x*math.log(20)/20,np.log(np.maximum(x,1e-12)))
 for offset,row,Tcv in states:
  robot.set_qpos(np.asarray(row['qpos']),zero_velocity=True);positions=array(robot.get_links_pos(relative=False));hand_error=float(np.linalg.norm(positions[hand.idx_local]-row['hand_position']));ball_error=float(np.linalg.norm(positions[ball.idx_local]-row['object_position']));maximum_pose_error=max(maximum_pose_error,hand_error,ball_error);assert max(hand_error,ball_error)<1e-5
  Tgl=np.asarray(Tcv).copy();Tgl[:3,1:3]*=-1;camera.set_pose(transform=Tgl);assert np.max(abs(array(camera.transform)-Tgl))<1e-9
  state_qpos=array(robot.get_qpos()).copy();state_time=float(scene.cur_t) if hasattr(scene,'cur_t') else None
  for intensity in a.intensities:
   assert math.isfinite(intensity) and intensity>=0;lamp.light.intensity=float(intensity);assert abs(lamp.light.intensity-intensity)<1e-12
   hdr,depth,seg,_=camera.render(rgb=True,depth=True,segmentation=True,radiance=True,force_render=True);hdr=array(hdr);depth=array(depth);seg=array(seg);native=array(camera.render(rgb=True,depth=False,segmentation=False,radiance=False,force_render=True)[0]);assert hdr.dtype.kind=='f' and native.dtype==np.uint8
   # Camera segmentation encoding varies across Genesis versions. Infer the actual
   # decoded label at the known sphere centre and independently verify its surface
   # depth; GT is used solely for this offline photometric measurement.
   pc=(np.linalg.inv(Tcv)@np.r_[row['object_position'],1.])[:3];q=K@pc;uv=q[:2]/q[2];cx,cy=np.floor(uv).astype(int);assert pc[2]>0 and 0<=cx<width and 0<=cy<height
   ray=np.linalg.inv(K)@np.array([cx+.5,cy+.5,1.]);aa=float(ray@ray);bb=float(ray@pc);cc=float(pc@pc)-.04**2;disc=bb*bb-aa*cc;assert disc>0;expected_surface_z=(bb-math.sqrt(disc))/aa*ray[2];surface_error=abs(float(depth[cy,cx])-expected_surface_z);assert surface_error<.01,('GT centre is occluded or depth convention differs',surface_error,uv.tolist())
   decoded_label=int(seg[cy,cx])
   if target_id is None:target_id=decoded_label
   assert decoded_label==target_id,('Target decoded label changed',target_id,decoded_label)
   gray=cv2.cvtColor(native,cv2.COLOR_RGB2GRAY);Y=hdr@np.array([.2126,.7152,.0722]);mask=seg==target_id;assert np.any(mask),'Target not rendered at selected pose';ring=cv2.dilate(mask.astype(np.uint8),kernel).astype(bool)&~mask;logY=np.log(np.maximum(Y,1e-6));logG=linlog(gray);edge_hdr=[];edge_ldr=[];edge_equal=[];edge_sat=[]
   for dy,dx in [(0,1),(1,0)]:
    ma=mask[:height-dy,:width-dx];mb=mask[dy:,dx:];take=ma^mb
    if take.any():
     ya=logY[:height-dy,:width-dx][take];yb=logY[dy:,dx:][take];ga=gray[:height-dy,:width-dx][take];gb=gray[dy:,dx:][take];la=logG[:height-dy,:width-dx][take];lb=logG[dy:,dx:][take];edge_hdr.extend(abs(ya-yb));edge_ldr.extend(abs(la-lb));edge_equal.extend(ga==gb);edge_sat.extend((ga>=254)&(gb>=254))
   eh=np.asarray(edge_hdr);el=np.asarray(edge_ldr);ee=np.asarray(edge_equal,dtype=bool);es=np.asarray(edge_sat,dtype=bool);count=len(eh);strong=eh>=.2;stem='offset_%04dms_I_%g'%(round(offset*1000),intensity)
   np.savez_compressed(a.output/(stem+'.npz'),radiance=hdr,native_rgb=native,native_gray8=gray,segmentation=seg,depth_camera_z=depth,intrinsics=K,camera_to_world_cv=Tcv,time_s=row['t'],intensity=intensity,target_segmentation_id=target_id)
   imageio.imwrite(a.output/(stem+'_native.png'),native)
   rec={'requested_release_offset_s':offset,'actual_state_time_s':row['t'],'actual_release_offset_s':row['t']-release,'point_intensity_renderer_units':intensity,'target_segmentation_id':target_id,'target_projected_center_uv':uv.tolist(),'target_surface_depth_error_m':surface_error,'target_label_source':'actual decoded segmentation at GT projected centre, verified by known-sphere surface depth','visible_target_pixels':int(mask.sum()),'target_HDR_luminance':values(Y[mask]),'ring_HDR_luminance':values(Y[ring]),'target_native_gray':values(gray[mask]),'ring_native_gray':values(gray[ring]),'target_native_saturated_fraction':float(np.mean(gray[mask]>=254)),'target_native_black_fraction':float(np.mean(gray[mask]<=1)),'native_boundary_pairs':count,'HDR_boundary_abs_log_contrast':values(eh),'V2E_native_boundary_abs_linlog_contrast':values(el),'strong_HDR_boundary_pairs':int(strong.sum()),'strong_HDR_equal_LDR_pairs':int(np.count_nonzero(strong&ee)),'strong_HDR_equal_LDR_fraction':float(np.mean(ee[strong])) if strong.any() else None,'both_native_saturated_boundary_fraction':float(np.mean(es)) if count else None,'numeric_threshold_both_is_not_scene_linear_response_calibration':True,'hand_replay_error_m':hand_error,'ball_replay_error_m':ball_error,'saved_arrays':stem+'.npz'};records.append(rec);print(json.dumps(rec),flush=True)
   assert np.array_equal(array(robot.get_qpos()),state_qpos),'Rendering changed stored qpos'
   if state_time is not None:assert float(scene.cur_t)==state_time,'Rendering advanced scene time'
 payload={'status':'PASS','scope':'actual native Genesis scene renders only, no physics steps, no controller or event-generator execution','source_run':str(a.run),'initialization_valid':initial['valid'],'camera_intrinsics':K.tolist(),'sensor_transform_source':'actual stored camera_to_world_cv at same archived qpos time, converted back to OpenGL by column sign flip','maximum_replay_position_error_m':maximum_pose_error,'lighting':{'ambient':[.1]*3,'directional_direction':[-1,-1,-1],'directional_intensity':5,'point_position':lamp_pos.tolist(),'point_intensities':a.intensities},'GT_segmentation_role':'offline photometric diagnosis only, never controller input','selection_rule':'measure target-specific information loss before any new closed-loop outcomes; do not use global image saturation alone','wall_s':time.monotonic()-started,'records':records};(a.output/'PROBE_RESULTS.json').write_text(json.dumps(payload,indent=2));print(json.dumps({'status':'PASS','frames':len(records),'wall_s':payload['wall_s']}),flush=True)

if __name__=='__main__':main()
