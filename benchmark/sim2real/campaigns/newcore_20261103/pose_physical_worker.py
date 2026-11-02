"""Matched textured-CAD, whole-frame domain-randomized synthetic pose cache."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--campaign',type=Path,required=True)
p.add_argument('--slot',type=int,required=True)
args=p.parse_args();R=args.campaign.resolve();P=Path('/home/zhaoyuhan/Eventbased_WAM')
core=Path(json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())['core_path'])
sys.path[:0]=[str(P/'runtime/sim2real/genesis_hdr_v2'),str(core),str(R/'sources/benchmark/sim2real/scripts')]
import numpy as np
import h5py
import trimesh
from scipy.spatial.transform import Rotation
from genesis.ext.pyrender.platforms.osmesa import OSMesaPlatform
OSMesaPlatform.supports_framebuffers=lambda self:True
import genesis as gs
from genesis_event_plugin.genesis_adapter import camera_to_world_cv,link_to_world_cv
from generate_stage4_flow_bank import geometry_intrinsics,renderer_segmentation_index_map,segmentation_render_to_link,radiance_to_v2e_luma_u8

cfg=json.loads((R/'sources/pose_bank_protocol.json').read_text())
asset=P/'runs/sim2real/20261101-epose-diverse-bank-v8/assets/036_wood_block/google_16k'
cad=asset/'nontextured.ply';texture=asset/'textured.obj'
assert hashlib.sha256(cad.read_bytes()).hexdigest()==cfg['cad_sha256']
mesh=trimesh.load(cad,process=False);lo,hi=np.asarray(mesh.bounds)
keys=np.asarray([[(lo+hi)[j]/2 for j in range(3)]]+[[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
gs.init(backend=gs.cpu,logging_level='warning')
scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.001,gravity=(0,0,0)),
    vis_options=gs.options.VisOptions(ambient_light=(.6,.6,.6),segmentation_level='link'),show_viewer=False)
ground=scene.add_entity(gs.morphs.Plane(),surface=gs.surfaces.Default(color=(.3,.34,.38,1)))
target=scene.add_entity(gs.morphs.Mesh(file=str(texture),pos=(0,0,.4),fixed=True,collision=False,decimate=False,file_meshes_are_zup=True),surface=gs.surfaces.Default())
for j in range(4):
    scene.add_entity(gs.morphs.Box(pos=(.3*math.cos(j*np.pi/2),.3*math.sin(j*np.pi/2),.09),size=(.08,.08,.18),fixed=True,collision=False),surface=gs.surfaces.Default(color=((j+1)/6,.3,(4-j)/6,1)))
camera=scene.add_camera(res=(346,260),pos=(0,-.65,.6),lookat=(0,0,.4),fov=float(np.degrees(2*np.arctan(260/(2*293.19)))),near=.03,far=4,GUI=False)
scene.build();ids=np.array([int(l.idx) for l in scene.rigid_solver.links],dtype=np.int32)
links={int(l.idx):l for l in scene.rigid_solver.links}
out=R/'physical';out.mkdir(exist_ok=True)
total=cfg['train_rollouts']+cfg['validation_rollouts']
for i in range(args.slot,total,cfg['workers']):
    split='train' if i<cfg['train_rollouts'] else 'validation'
    j=i if split=='train' else i-cfg['train_rollouts']
    rid=split+'_'+str(j).zfill(3)
    rng=np.random.default_rng(cfg['physical_seed']+i*1009)
    angles=rng.uniform([-1.2,-1.2,-np.pi],[1.2,1.2,np.pi])
    rot0=Rotation.from_euler('xyz',angles);omega=rng.normal(size=3);omega=omega/np.linalg.norm(omega)*rng.uniform(.2,.8)
    az=rng.uniform(-np.pi,np.pi);elevation=rng.uniform(.1,.8)
    f=np.array([np.sin(az)*np.cos(elevation),np.cos(az)*np.cos(elevation),-np.sin(elevation)])
    right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(f,right)
    distance=float(rng.uniform(.38,1.15))
    target.set_pos((0,0,.4));target.set_quat(np.roll(rot0.as_quat(),1))
    T0=link_to_world_cv(target.links[0]);center=(T0@np.r_[(lo+hi)/2,1])[:3]
    _,K=geometry_intrinsics(camera)
    desired=rng.uniform([.1,.1],[.9,.9])*np.array([346,260])
    dx=(desired[0]-K[0,2])/K[0,0]*distance;dy=(desired[1]-K[1,2])/K[1,1]*distance
    eye0=center-distance*f-dx*right-dy*down
    vr=-np.sign(.5-desired[0]/346)*rng.uniform(.2,.8)
    vd=-np.sign(.5-desired[1]/260)*rng.uniform(.05,.3)
    if vr==0:vr=.3
    if vd==0:vd=.1
    velocity=right*vr+down*vd
    # Bound camera travel so the centroid stays on the sensor; box corners may
    # be truncated. This uses synthetic geometry only, never real GT statistics.
    duration=cfg['duration_s']
    toward_x=desired[0] if vr>0 else 346-desired[0]
    toward_y=desired[1] if vd>0 else 260-desired[1]
    factor=min(1.,max(.05,(toward_x-8)*distance/(K[0,0]*abs(vr)*duration)),max(.05,(toward_y-8)*distance/(K[1,1]*abs(vd)*duration)))
    velocity*=factor
    camera.set_pose(pos=tuple(eye0),lookat=tuple(eye0+f))
    start=float(scene.cur_t);begin=time.monotonic();nframes=int(round(duration*100))+1
    destination=out/(rid+'.h5');assert not destination.exists()
    with h5py.File(destination.with_suffix('.h5.partial'),'w') as h:
        g=h.create_group('episodes').create_group(rid)
        g.attrs.update(split=split,stratum='matched_wood_block_fullframe',scene_seed=cfg['physical_seed']+i*1009,
            source='native Genesis textured CAD rendering',allow_truncated_box=True,cad_sha256=cfg['cad_sha256'])
        g.create_dataset('camera_intrinsics_K_px',data=K);g.create_dataset('object_segmentation_link_ids',data=ids)
        cameras=[];objects=[];times=[];visibility=[];centers=[];corner_uv=[]
        for k in range(nframes):
            if k:
                for sub in range(1,11):
                    t=((k-1)*10+sub)*.001
                    q=(rot0*Rotation.from_rotvec(omega*t)).as_quat();target.set_quat(np.roll(q,1))
                    eye=eye0+velocity*t;camera.set_pose(pos=tuple(eye),lookat=tuple(eye+f));scene.step()
            rad,_,_,_=camera.render(rgb=True,depth=False,segmentation=False,radiance=True,force_render=True)
            _,depth,raw,_=camera.render(rgb=False,depth=True,segmentation=True,force_render=True)
            rad=np.asarray(rad,dtype=np.float32);depth=np.asarray(depth,dtype=np.float32)
            seg=segmentation_render_to_link(np.asarray(raw,dtype=np.int32),renderer_segmentation_index_map(camera))
            assert rad.shape==(260,346,3) and np.isfinite(rad).all() and np.isfinite(depth).all()
            for name,array in {'radiance_linear_rgb':rad,'depth_m':depth,'segmentation_link_idx':seg,'v2e_display_luma_u8':radiance_to_v2e_luma_u8(rad)}.items():
                if name not in g:g.create_dataset(name,shape=(nframes,)+array.shape,dtype=array.dtype,chunks=(1,)+array.shape,compression='gzip',compression_opts=1)
                g[name][k]=array
            C=camera_to_world_cv(camera);T=link_to_world_cv(target.links[0])
            xyz=(np.linalg.inv(C)@T@np.c_[keys,np.ones(9)].T).T[:,:3]
            uv=(K@xyz.T).T;uv=uv[:,:2]/uv[:,2:]
            assert np.isfinite(uv).all() and (xyz[:,2]>.03).all()
            cameras.append(C);objects.append(np.stack([link_to_world_cv(links[int(sid)]) for sid in ids]));times.append(float(scene.cur_t)-start)
            centers.append(uv[0]);corner_uv.append(uv[1:]);visibility.append(int((seg==int(target.links[0].idx)).sum()))
            if k%5==0:print(json.dumps({'phase':'matched_pose_native_HDR','rollout':rid,'frame':k,'total_frames':nframes,'wall_s':time.monotonic()-begin}),flush=True)
        g.create_dataset('frame_timestamps_s',data=np.asarray(times));g.create_dataset('camera_to_world_cv',data=np.stack(cameras));g.create_dataset('object_to_world',data=np.stack(objects))
        g.create_dataset('target_pose',data=np.stack(objects)[:,list(ids).index(int(target.links[0].idx))])
        g.create_dataset('cad_keypoints',data=keys);g.create_dataset('projected_centroid_uv',data=np.stack(centers));g.create_dataset('projected_corners_uv',data=np.stack(corner_uv));g.create_dataset('target_visible_pixels',data=np.array(visibility))
        g.attrs['target_link_id']=int(target.links[0].idx)
        g.attrs['physical_parameters']=json.dumps({'distance_m':distance,'desired_uv':desired.tolist(),'angles':angles.tolist(),'camera_velocity':velocity.tolist(),'angular_velocity':omega.tolist()})
    destination.with_suffix('.h5.partial').rename(destination)
    print(json.dumps({'rollout_complete':rid,'bytes':destination.stat().st_size,'wall_s':time.monotonic()-begin,'minimum_visible_pixels':min(visibility)}),flush=True)
gs.destroy();print('POSE_PHYSICAL_WORKER_COMPLETE',args.slot,flush=True)
