"""Render one preregistered rich layout and four 2-second physical trajectories."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--campaign',type=Path,required=True)
p.add_argument('--slot',type=int,required=True)
p.add_argument('--probe-frames',type=int,default=0)
args=p.parse_args()
R=args.campaign.resolve();P=Path('/home/zhaoyuhan/Eventbased_WAM')
CORE=Path(json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())['core_path'])
sys.path[:0]=[str(P/'runtime/sim2real/genesis_hdr_v2'),str(CORE),str(R/'sources'),str(R/'sources/benchmark/sim2real/scripts')]
import numpy as np
import h5py
from genesis.ext.pyrender.platforms.osmesa import OSMesaPlatform
OSMesaPlatform.supports_framebuffers=lambda self:True
import stage9_replicad_scene_loader as scene_impl
from genesis_event_plugin.genesis_adapter import camera_to_world_cv,link_to_world_cv
from generate_stage4_flow_bank import radiance_to_v2e_luma_u8

cfg=json.loads((R/'sources/rich_bank_protocol.json').read_text())
record=cfg['layouts'][args.slot]
asset=P/'external/sim2real_assets/replicacad_interactive_v1_5/extracted_v1_5'
scene_file=asset/'configs/scenes'/record['scene_config']
scene_json=json.loads(scene_file.read_text())
counts=collections.Counter(x['template_name'].rsplit('/',1)[-1] for x in scene_json['object_instances'])
moving=None
for term in ['tvstand','sofa','coffeetable','table','chair']:
    options=sorted(k for k,n in counts.items() if n==1 and term in k.lower())
    if options:
        moving=options[0];break
assert moving is not None,'No unique scripted rigid object in '+str(scene_file)
setup=R/'physical_scene_setup'/('layout'+str(args.slot))
output=R/'physical'
output.mkdir(exist_ok=True)
camera_choice={}

def choose_camera(scene,camera,entity):
    start=camera_to_world_cv(camera)
    target=link_to_world_cv(entity.links[0])[:3,3]+np.array([0.,0.,.35])
    positions=[(start[:3,3],start[:3,3]+start[:3,2])]
    positions += [(target+np.asarray(v),target) for v in [[0,-2,1],[.7,-1.8,.9],[-1.8,0,.9],[0,2,1],[1.6,1.2,1.]]]
    sid=int(entity.links[0].idx)
    for k,(eye,look) in enumerate(positions):
        camera.set_pose(pos=tuple(eye),lookat=tuple(look))
        _,depth,raw,_=camera.render(rgb=False,depth=True,segmentation=True,force_render=True)
        mapping=scene_impl.renderer_segmentation_link_map(camera)
        seg=scene_impl.remap_segmentation_to_links(np.asarray(raw),mapping)
        count=int((seg==sid).sum())
        if count>=100 and float((np.asarray(depth)>0).mean())>.5:
            camera_choice.update(candidate_index=k,visible_object_pixels=count,selection='first deterministic geometry-valid camera; no event counts/model scores')
            return
    raise RuntimeError('No geometry-valid camera candidate')

def find_urdf(template):
    matches=sorted(p for p in (asset/'urdf').rglob('*.urdf') if p.stem.lower()==template.lower())
    return str(matches[0].relative_to(asset)) if len(matches)==1 else None

def capture(scene,camera,output_dir,translation_h,object_delta_h,moving_entity,moving_link_id,links_by_id,moving_base_g):
    base=camera_to_world_cv(camera)
    eye0=base[:3,3].copy();forward=base[:3,2].copy();right=base[:3,0].copy();down=base[:3,1].copy()
    K=np.asarray(camera.intrinsics,dtype=np.float64).copy();K[0,2]-=.5;K[1,2]-=.5
    ids=np.array(sorted(links_by_id),dtype=np.int32)
    results=[]
    for stratum in range(4):
        rid=f's{stratum}_{args.slot:02d}'
        destination=output/(rid+'.h5')
        assert not destination.exists(),str(destination)
        rng=np.random.default_rng(cfg['physical_seed']+stratum*1009+args.slot)
        scale=cfg['motion_scales_m'][args.slot%3]
        frequency=cfg['frequencies_hz'][(args.slot//2+stratum)%3]
        phase=rng.uniform(-np.pi,np.pi,3)
        camera_amp=np.array([scale,.4*scale,.2*scale])
        object_amp=np.array([.06,.04,.02])*(1+.5*(args.slot%3))
        if stratum==0:object_amp*=0
        if stratum==1:camera_amp*=0
        nframes=args.probe_frames or 201
        moving_entity.set_pos(tuple(moving_base_g))
        camera.set_pose(pos=tuple(eye0),lookat=tuple(eye0+forward))
        origin=float(scene.cur_t)
        begin=time.monotonic()
        partial=destination.with_suffix('.h5.partial')
        with h5py.File(partial,'w') as h:
            g=h.create_group('episodes').create_group(rid)
            g.attrs.update(split=record['split'],stratum=cfg['strata'][stratum],scene_id=record['scene_config'],scene_seed=cfg['physical_seed']+args.slot,
                layout_slot=args.slot,physical_trajectory_kind='Genesis stepped1ms, scripted smooth rigid trajectories; native100Hz HDR endpoints',
                pixel_center_contract='renderer principal point minus0.5 for integer pixel ray grid',radiance_source='native float HDR from isolated unchanged overlay')
            h.attrs.update(schema='stage9_rich_physical_cache_v1',endpoint_hz=100,physics_dt_s=.001,real_data_used=False)
            g.create_dataset('camera_intrinsics_K_px',data=K)
            g.create_dataset('object_segmentation_link_ids',data=ids)
            times=[];cams=[];poses=[]
            for k in range(nframes):
                if k:
                    for sub in range(1,11):
                        t=((k-1)*10+sub)*.001
                        wave=np.sin(2*np.pi*frequency*t+phase)-np.sin(phase)
                        delta=camera_amp*wave
                        eye=eye0+right*delta[0]+down*delta[1]+forward*delta[2]
                        yaw=.04*np.sin(2*np.pi*.6*t) if stratum>=2 else 0.
                        direction=np.cos(yaw)*forward+np.sin(yaw)*right
                        camera.set_pose(pos=tuple(eye),lookat=tuple(eye+direction))
                        moving_entity.set_pos(tuple(np.asarray(moving_base_g)+object_amp*wave))
                        scene.step()
                rad,_,_,_=camera.render(rgb=True,depth=False,segmentation=False,radiance=True,force_render=True)
                _,dep,raw,_=camera.render(rgb=False,depth=True,segmentation=True,force_render=True)
                rad=np.asarray(rad,dtype=np.float32);dep=np.asarray(dep,dtype=np.float32)
                seg=scene_impl.remap_segmentation_to_links(np.asarray(raw),scene_impl.renderer_segmentation_link_map(camera)).astype(np.int32)
                assert rad.shape==(260,346,3) and dep.shape==seg.shape==(260,346)
                assert np.isfinite(rad).all() and np.isfinite(dep).all()
                frame_fields={'radiance_linear_rgb':rad,'depth_m':dep,'segmentation_link_idx':seg,'v2e_display_luma_u8':radiance_to_v2e_luma_u8(rad)}
                for name,array in frame_fields.items():
                    if name not in g:g.create_dataset(name,shape=(nframes,)+array.shape,dtype=array.dtype,chunks=(1,)+array.shape,compression='gzip',compression_opts=1)
                    g[name][k]=array
                times.append(float(scene.cur_t)-origin)
                cams.append(camera_to_world_cv(camera))
                poses.append(np.stack([link_to_world_cv(links_by_id[int(sid)]) for sid in ids]))
                if k%20==0:
                    print(json.dumps({'phase':'native_physical_render','rollout':rid,'frame':k,'total_frames':nframes,'wall_s':time.monotonic()-begin}),flush=True)
                    (R/'physical_scene_setup'/('layout'+str(args.slot)+'_progress.json')).write_text(json.dumps({'rollout':rid,'frame':k,'total_frames':nframes,'wall_s':time.monotonic()-begin}))
            times=np.asarray(times)
            assert np.allclose(times,np.arange(nframes)*.01,atol=1e-7,rtol=0)
            g.create_dataset('frame_timestamps_s',data=times)
            g.create_dataset('camera_to_world_cv',data=np.stack(cams))
            g.create_dataset('object_to_world',data=np.stack(poses))
            if nframes==201:
                ff=[];vv=[]
                for i in range(0,198,3):
                    a={int(sid):poses[i][j] for j,sid in enumerate(ids)}
                    b={int(sid):poses[i+3][j] for j,sid in enumerate(ids)}
                    flow,valid,_=scene_impl.visibility_checked_flow(np.asarray(g['depth_m'][i]),np.asarray(g['segmentation_link_idx'][i]),np.asarray(g['depth_m'][i+3]),K,cams[i],cams[i+3],a,b)
                    ff.append(flow);vv.append(valid)
                g.create_dataset('flow_forward_30ms_px',data=np.stack(ff),compression='gzip',compression_opts=1)
                g.create_dataset('flow_valid_30ms',data=np.stack(vv),compression='gzip',compression_opts=1)
            g.attrs['camera_selection']=json.dumps(camera_choice)
            g.attrs['motion_parameters']=json.dumps({'scale_m':scale,'frequency_hz':frequency,'phase':phase.tolist(),'camera_amplitude':camera_amp.tolist(),'object_amplitude':object_amp.tolist()})
        partial.rename(destination)
        row={'rollout_id':rid,'split':record['split'],'layout':record['scene_config'],'path':str(destination),'frames':nframes,'wall_s':time.monotonic()-begin,'bytes':destination.stat().st_size}
        results.append(row);print(json.dumps(row),flush=True)
        if args.probe_frames:break
    (R/'physical'/('layout'+str(args.slot)+'_complete.json')).write_text(json.dumps({'rollouts':results,'scene_config':record['scene_config'],'camera_selection':camera_choice},indent=2)+'\n')
    return {'schema':'stage9_long_native_physical_rollouts','rollouts':results,'events_generated':False}

scene_impl._s9_choose_camera=choose_camera
scene_impl._s9_find_urdf=find_urdf
scene_impl.capture_dynamic_motion_probe=capture
sys.argv=[str(Path(scene_impl.__file__)),'--asset-root',str(asset),'--scene-config',record['scene_config'],'--output-dir',str(setup),'--moving-object-template',moving]
scene_impl.main()
metadata=setup/'result.json'
if metadata.exists():metadata.rename(setup/'legacy_scene_setup_metadata.json')
print('PHYSICAL_LAYOUT_COMPLETE',args.slot,flush=True)
