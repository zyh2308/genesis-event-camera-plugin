"""Actual configured-cadence Genesis HDR endpoints -> frozen CUDA worker -> event tracker."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from event_sphere_tracker import EventSphereTracker
from event_wire import encode,decode

def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

def link_transforms(entity):
    positions=array(entity.get_links_pos(relative=False));quats=array(entity.get_links_quat(relative=False))
    rotations=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
    matrices=np.tile(np.eye(4),(len(positions),1,1));matrices[:,:3,:3]=rotations;matrices[:,:3,3]=positions
    return {int(link.idx)+1:matrices[i] for i,link in enumerate(entity.links)}

class SparseEventSensor:
    def __init__(self,env,output,radius):
        self.env=env;self.camera=env.sensor_camera;self.root=Path(output);self.root.mkdir(exist_ok=True)
        self.endpoint_hz=float(getattr(env,'event_endpoint_hz',100.))
        self.K=array(self.camera.intrinsics).astype(float)
        self.tracker=EventSphereTracker(self.K,self.camera.res,radius)
        self.tracker.gravity=np.asarray(env.Dcmm.model.opt.gravity,dtype=float).copy()
        self.chunks=[];self.packet_audit=[];self.frame_index=0;self.worker=None;self.h=None
        self.initialization_only=os.environ.get('CATCHIT_INITIALIZATION_ONLY')=='1'
        self.initialization_probe=os.environ.get('CATCHIT_INITIALIZATION_PROBE')=='1'
        self.initialization=None;self.initialization_failed=False
        self.capture_suspension=None;self.skipped_post_grasp_queries=0
        self.generator_method=os.environ.get('CATCHIT_EVENT_GENERATOR','ours')
        assert self.generator_method in ('ours','v2e')
        if os.environ.get('CATCHIT_RGB_INITIALIZE')=='1':
            import imageio.v2 as imageio
            from rgb_sphere_initialization import detect_sphere
            self.env.update_sensor_pose()
            use_depth=os.environ.get('CATCHIT_RGBD_INITIALIZE')=='1'
            initial_outputs=self.camera.render(rgb=True,depth=use_depth,segmentation=False,radiance=False,force_render=True)
            rgb=array(initial_outputs[0])
            imageio.imwrite(self.root/'initial_rgb.png',rgb)
            T=array(self.camera.transform).astype(float);T[:3,1:3]*=-1
            if use_depth:
                from rgbd_sphere_initialization import initialize_sphere_rgbd
                initial_depth=array(initial_outputs[1])
                np.save(self.root/'initial_depth_camera_z.npy',initial_depth)
                detection=initialize_sphere_rgbd(rgb,initial_depth,self.K,T,radius)
            else:detection=detect_sphere(rgb,self.K,T,radius)
            self.initialization=dict(detection,time_s=0.,available_s=0.,RGB_frames_read=1,
                                     RGB_available_at_reset=True,control_depth_frames_read=int(use_depth),depth_frame_time_s=0. if use_depth else None,lighting=os.environ['CATCHIT_LIGHTING'],public_world_gravity_m_s2=self.tracker.gravity.tolist())
            (self.root/'initialization.json').write_text(json.dumps(self.initialization,indent=2))
            self.initialization_failed=not detection['valid']
            if detection['valid']:self.tracker.initialize_rgb(detection,time_s=0.,available_s=0.)
            if self.initialization_only or self.initialization_probe or self.initialization_failed:return
        base=Path('/home/zhaoyuhan/Eventbased_WAM');self.core=base/'runs/robot_sparse_cuda_acceptance_20261102c/core'
        gpu_index=os.environ['EGL_DEVICE_ID']
        uuid=subprocess.check_output(['nvidia-smi','-i',gpu_index,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        worker_env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=str(self.core),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        self.log=(self.root/'event_worker.log').open('w')
        worker_name='event_worker.py' if self.generator_method=='ours' else 'v2e_event_worker.py'
        self.worker=subprocess.Popen(['/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python','-u',str(Path(__file__).with_name(worker_name))],
            env=worker_env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=1024*1024)
        try:
            ready=decode(pickle.load(self.worker.stdout));assert ready['ready']
            (self.root/'worker_start.json').write_text(json.dumps(dict(pid=self.worker.pid,gpu_uuid=uuid,**ready),indent=2))
            self.h=h5py.File(self.root/'sparse_hdr_endpoints.h5','w');self.h.attrs['endpoint_rate_hz']=self.endpoint_hz
            self.h.attrs['source']='actual sparse Genesis float32 scene-linear HDR; no dense reference in generation'
            self.h.attrs['sensor_response']='frozen EventGenesis ideal profile' if self.generator_method=='ours' else 'native V2E SSM39 plus EventEmulator, controlled ideal noise settings; native uint8 LDR grayscale input'
            self.h.attrs['event_generation_method']=self.generator_method
            self.capture(0.,reset=True)
        except BaseException:
            self.close();raise

    def capture(self,t,reset=False):
        self.env.update_sensor_pose()
        if self.initialization_only or self.initialization_probe or self.initialization_failed:return
        if os.environ.get('CATCHIT_STOP_EVENTS_AFTER_GRASP')=='1' and self.tracker.attached_offset is not None:
            if self.capture_suspension is None:self.capture_suspension=dict(time_s=float(t),reason='estimated grasp; controller already uses hand proprioception',GT_contact_used=False)
            self.skipped_post_grasp_queries+=1
            return
        hdr_started=time.monotonic()
        hdr,depth,seg,_=self.camera.render(rgb=True,depth=True,segmentation=True,radiance=True,force_render=True)
        hdr_capture_s=time.monotonic()-hdr_started;gray8=None;ldr_capture_s=0.
        if self.generator_method=='v2e':
            import cv2
            ldr_started=time.monotonic()
            ldr=array(self.camera.render(rgb=True,depth=False,segmentation=False,radiance=False,force_render=True)[0])
            assert ldr.dtype==np.uint8
            gray8=cv2.cvtColor(ldr,cv2.COLOR_RGB2GRAY)
            ldr_capture_s=time.monotonic()-ldr_started
        T=array(self.camera.transform).astype(float);T[:3,1:3]*=-1
        poses=link_transforms(self.env.robot)
        if getattr(self.env,'background_wall',None) is not None:poses.update(link_transforms(self.env.background_wall))
        packet=dict(time_s=float(t),resolution_wh=self.camera.res,intrinsics=self.K,radiance=array(hdr).astype(np.float32),
                    depth=array(depth).astype(np.float32),segmentation=array(seg).astype(np.int32),camera_to_world_cv=T,object_poses=poses)
        worker_packet=packet if self.generator_method=='ours' else dict(time_s=float(t),gray8=gray8)
        started=time.monotonic();pickle.dump(encode(dict(kind='reset' if reset else 'frame',packet=worker_packet)),self.worker.stdin,protocol=5);self.worker.stdin.flush()
        reply=decode(pickle.load(self.worker.stdout));events=reply['events']
        available=float(t)+.010
        if len(events):self.chunks.append(events.copy())
        self.tracker.push(events,available_s=available,packet_time_s=t,camera_to_world=T)
        self.packet_audit.append(dict(time_s=float(t),available_s=available,events=len(events),worker_wall_s=reply['wall_s'],
            transfer_and_worker_wall_s=time.monotonic()-started,endpoint_reads=reply['endpoint_reads'],dense_HDR_reads=0,renderer_calls_in_reconstruction=0,
            method=self.generator_method,endpoint_rate_hz=self.endpoint_hz,subframes=reply.get('subframes'),native_ldr_capture_s=ldr_capture_s,hdr_capture_s=hdr_capture_s,
            hdr_used_only_for_audit=self.generator_method=='v2e',worker_input_keys=sorted(worker_packet)))
        data=dict(time_s=np.asarray(t),radiance=packet['radiance'],depth=packet['depth'],segmentation=packet['segmentation'],
                  camera_to_world_cv=T,object_poses=np.stack([poses[k] for k in sorted(poses)]))
        if gray8 is not None:data['native_ldr_gray8']=gray8
        if 'segmentation_pose_ids' not in self.h:self.h.create_dataset('segmentation_pose_ids',data=sorted(poses))
        if 'intrinsics' not in self.h:self.h.create_dataset('intrinsics',data=self.K)
        for name,value in data.items():
            if name not in self.h:
                self.h.create_dataset(name,shape=(0,)+value.shape,maxshape=(None,)+value.shape,chunks=(1,)+value.shape,dtype=value.dtype,**(dict(compression='lzf') if value.ndim else {}))
            ds=self.h[name];ds.resize(self.frame_index+1,axis=0);ds[self.frame_index]=value
        self.frame_index+=1
        if self.frame_index%10==0:
            self.h.flush();(self.root/'progress.json').write_text(json.dumps(dict(**self.packet_audit[-1],
                detections_so_far=len(self.tracker.samples),last_detection=self.tracker.samples[-1] if self.tracker.samples else None)))

    def observe(self,t,hand_pose,hand_joints):return self.tracker.observe(t,hand_pose,hand_joints)

    def close(self):
        if self.worker is not None:
            if self.worker.poll() is None:
                try:pickle.dump(dict(kind='close'),self.worker.stdin,protocol=5);self.worker.stdin.flush();self.worker.stdin.close()
                except (BrokenPipeError,OSError):pass
                try:self.worker.wait(timeout=15)
                except subprocess.TimeoutExpired:self.worker.terminate();self.worker.wait(timeout=10)
            self.worker=None
        if self.h is not None:self.h.close();self.h=None
        if hasattr(self,'log'):self.log.close()
        (self.root/'capture_schedule.json').write_text(json.dumps(dict(active_endpoint_rate_hz=self.endpoint_hz,post_grasp_suspension=self.capture_suspension,skipped_post_grasp_queries=self.skipped_post_grasp_queries,initialization_only=self.initialization_only),indent=2))
        events=np.concatenate(self.chunks) if self.chunks else np.empty((0,4))
        np.savez_compressed(self.root/'raw_events.npz',events=events)
        (self.root/'packet_audit.json').write_text(json.dumps(self.packet_audit))
        (self.root/'event_samples.json').write_text(json.dumps(self.tracker.samples))
        (self.root/'estimator_states.json').write_text(json.dumps(self.tracker.diagnostics))
        (self.root/'grasp_hypothesis_events.json').write_text(json.dumps(getattr(self.tracker,'hold_events',[])))
        (self.root/'front_end_quality.json').write_text(json.dumps(dict(partial_contour_rejections=getattr(self.tracker,'partial_rejections',0))))
        (self.root/'event_observation_windows.json').write_text(json.dumps(dict(
            observation_window_ms=getattr(self.tracker,'observation_window_ms',0.),
            information_availability='Every child retains its original parent packet availability; controller polls unchanged',
            camera_pose_source='Approximate SE3 interpolation between available robot camera endpoint poses; no target truth',
            windows=getattr(self.tracker,'observation_window_audit',[])),indent=2))
