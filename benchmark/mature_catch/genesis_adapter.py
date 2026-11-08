"""Catch It! native controller on Genesis physics; MuJoCo is kinematics only.

No mj_step runs in this adapter. Contacts come from the Genesis solver.
This is a backend migration probe, initially using the author's GT observations.
"""
import ast
import inspect
import json
import os
from pathlib import Path
import textwrap
import types
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv as NativeEnv

def array(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

class GenesisEnv(NativeEnv):
    physics_backend='Genesis 1.2.2; original MuJoCo model used only for kinematics/IK'

    def __init__(self,*args,**kwargs):
        self.gs_ready=False;self.sensor=None;self.sensor_probe=[];self.episode_number=-1;self._gs_step_count=0
        self.front_glare_lamp=None;self.lighting_schedule=None;self.lighting_traces={};self._last_lighting_trace_key=None
        super().__init__(*args,**kwargs)
        from cadence_config import endpoint_stride
        self.event_endpoint_hz=float(os.environ.get('CATCHIT_EVENT_ENDPOINT_HZ','100'))
        self.event_endpoint_stride=endpoint_stride(self.event_endpoint_hz,float(self.Dcmm.model.opt.timestep))
        import genesis as gs
        self.gs=gs
        from contact_calibration import profile_config
        self.contact_profile_name=os.environ.get('CATCHIT_CONTACT_PROFILE','baseline')
        self.contact_profile=profile_config(self.contact_profile_name)
        self.adapter_output=Path(os.environ['CATCHIT_ADAPTER_OUTPUT']);self.adapter_output.mkdir(parents=True,exist_ok=True)
        source=Path.cwd();tree=ET.ElementTree(ET.fromstring(self.Dcmm.model_xml_string));root=tree.getroot()
        root.find('compiler').set('meshdir',str(source/'assets/meshes'))
        for tex in root.findall('.//texture'):
            if tex.get('file'):tex.set('file',str(source/tex.get('file')))
        xmlpath=self.adapter_output/'genesis_task.xml';tree.write(xmlpath)
        gs.init(backend=gs.cpu,precision='64',logging_level='warning')
        lighting=os.environ.get('CATCHIT_LIGHTING','normal')
        vis=dict(segmentation_level='link')
        if lighting=='backlit':vis.update(ambient_light=(.01,.01,.01),lights=[dict(type='point',pos=(0.,3.8,1.8),color=(1.,1.,1.),intensity=100.)])
        elif lighting=='front_glare':
            from light_schedule import front_glare_schedule
            self.lighting_schedule=front_glare_schedule(os.environ.get('CATCHIT_FRONT_GLARE_PEAK','400'))
            vis.update(ambient_light=(.1,.1,.1),lights=[dict(type='directional',dir=(-1.,-1.,-1.),color=(1.,1.,1.),intensity=5.),
                dict(type='point',pos=tuple(self.lighting_schedule['point_position']),color=(1.,1.,1.),intensity=0.)])
        self.scene=gs.Scene(sim_options=gs.options.SimOptions(dt=float(self.Dcmm.model.opt.timestep),substeps=self.contact_profile['substeps']),vis_options=gs.options.VisOptions(**vis),
            rigid_options=gs.options.RigidOptions(enable_self_collision=True,enable_neutral_collision=True,enable_mujoco_compatibility=True),show_viewer=False)
        # MuJoCo collides each mesh as its convex hull. Avoid Genesis splitting
        # concave target meshes into multiple collision pieces during migration.
        self.robot=self.scene.add_entity(gs.morphs.MJCF(file=str(xmlpath),default_armature=0.0,convexify=True))
        self.background_wall=None
        if os.environ.get('CATCHIT_BACKGROUND_WALL')=='1':
            self.background_wall=self.scene.add_entity(gs.morphs.Box(pos=(0.,4.,1.6),size=(4.,.02,3.2),fixed=True,collision=False),surface=gs.surfaces.Default(color=(.95,.95,.95,1.)))
        self.demo_camera=self.scene.add_camera(res=(640,480),pos=(3.1,3.2,2.0),lookat=(0,.8,.9),fov=48,GUI=False)
        camid=int(self.Dcmm.model.camera('top').id);self.sensor_camera_id=camid
        self.sensor_camera=self.scene.add_camera(res=(640,480),pos=(0,0,1.4),lookat=(0,1,1.),fov=float(self.Dcmm.model.cam_fovy[camid]),GUI=False)
        self.scene.build();self.gs_ready=True;self._gs_time=0.;self._gs_released=False
        if self.lighting_schedule is not None:
            native=self.scene.visualizer.context._scene;position=np.asarray(self.lighting_schedule['point_position'])
            lamps=[node for node in native.light_nodes if type(node.light).__name__=='PointLight' and np.linalg.norm(native.get_pose(node)[:3,3]-position)<1e-5]
            if len(lamps)!=1:raise RuntimeError('Expected one actual native front_glare point lamp at configured position')
            self.front_glare_lamp=lamps[0]
        assert self.robot.n_qs==self.Dcmm.model.nq and self.robot.n_dofs==self.Dcmm.model.nv
        joints={j.name:j for j in self.robot.joints}
        mapping=[]
        for i in range(self.Dcmm.model.njnt):
            mj=self.Dcmm.model.joint(i);j=joints[mj.name]
            qi=np.asarray(j.q_idx_local).reshape(-1);di=np.asarray(j.dof_idx_local).reshape(-1)
            assert int(qi[0])==int(self.Dcmm.model.jnt_qposadr[i]),mj.name
            assert int(di[0])==int(self.Dcmm.model.jnt_dofadr[i]),mj.name
            mapping.append(dict(name=mj.name,q=int(qi[0]),dof=int(di[0])))
        self.geom_map={};self.geom_notes=[]
        for geom in self.robot.geoms:
            name=getattr(geom,'metadata',geom._metadata).get('name','')
            idx=mujoco.mj_name2id(self.Dcmm.model,mujoco.mjtObj.mjOBJ_GEOM,name) if name else -1
            if idx<0:
                body=mujoco.mj_name2id(self.Dcmm.model,mujoco.mjtObj.mjOBJ_BODY,geom.link.name)
                candidates=np.where((self.Dcmm.model.geom_bodyid==body)&((self.Dcmm.model.geom_contype!=0)|(self.Dcmm.model.geom_conaffinity!=0)))[0]
                if len(candidates):
                    differences=np.linalg.norm(self.Dcmm.model.geom_pos[candidates]-np.asarray(geom._init_pos),axis=1)
                    idx=int(candidates[np.argmin(differences)])
                    assert float(differences.min())<1e-5,(geom.idx,geom.link.name,differences)
            if idx<0:raise RuntimeError('Unmapped collision geom '+str(geom.idx))
            self.geom_map[int(geom.idx)]=int(idx)
            self.geom_notes.append(dict(gs_geom=int(geom.idx),original_geom=int(idx),name=name,link=geom.link.name))
        # Select hand collision bodies from the native semantic range. Match
        # actual body and local position; imported mesh metadata names repeat
        # across fingers and wheels and cannot identify individual geoms.
        hand_geom_ids=[];hand_selection=[]
        for geom in self.robot.geoms:
            body=mujoco.mj_name2id(self.Dcmm.model,mujoco.mjtObj.mjOBJ_BODY,geom.link.name)
            candidates=np.where((self.Dcmm.model.geom_bodyid==body)&((self.Dcmm.model.geom_contype!=0)|(self.Dcmm.model.geom_conaffinity!=0)))[0]
            distances=np.linalg.norm(self.Dcmm.model.geom_pos[candidates]-np.asarray(geom._init_pos),axis=1)
            nearest=candidates[np.isclose(distances,distances.min(),atol=1e-7,rtol=0)]
            categories=[self.hand_start_id<=int(i)<self.object_id for i in nearest]
            if min(distances)>1e-5 or len(set(categories))!=1:
                raise RuntimeError('Ambiguous contact calibration collider: '+str((geom.idx,geom.link.name,nearest.tolist())))
            if categories[0]:
                hand_geom_ids.append(int(geom.idx))
                hand_selection.append(dict(gs_geom=int(geom.idx),link=geom.link.name,native_geom_candidates=nearest.tolist()))
        from contact_calibration import ContactCalibration
        self.contact_calibration=ContactCalibration(self.robot,self.contact_profile_name,hand_geom_ids)
        self.contact_calibration_audit=self.contact_calibration.apply()
        self.contact_calibration_audit['selection']=hand_selection
        # All unnamed geoms live before the hand except the hand mounting geom.
        # The original model's named collision geoms map exactly; retain audit.
        self.actuator_dofs=[]
        for i in range(self.Dcmm.model.nu):
            jnt=int(self.Dcmm.model.actuator_trnid[i,0]);start=int(self.Dcmm.model.jnt_dofadr[jnt])
            length=6 if self.Dcmm.model.jnt_type[jnt]==mujoco.mjtJoint.mjJNT_FREE else 1
            self.actuator_dofs.append((start,length,self.Dcmm.model.actuator_gear[i,:length].copy()))
        tree=ast.parse(textwrap.dedent(inspect.getsource(NativeEnv._step_mujoco_simulation)))
        class RedirectStep(ast.NodeTransformer):
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='mujoco':
                    if node.func.attr=='mj_step':return ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_genesis_step',ctx=ast.Load()),args=[],keywords=[])
                    if node.func.attr=='mj_rnePostConstraint':return ast.Constant(value=None)
                return node
        tree=ast.fix_missing_locations(RedirectStep().visit(tree));namespace=dict(NativeEnv._step_mujoco_simulation.__globals__)
        exec(compile(tree,'original_control_genesis_step','exec'),namespace)
        self._step_mujoco_simulation=types.MethodType(namespace['_step_mujoco_simulation'],self)
        masses=[]
        for link in self.robot.links:
            body=mujoco.mj_name2id(self.Dcmm.model,mujoco.mjtObj.mjOBJ_BODY,link.name)
            if body>=0 and link.inertial_mass is not None:
                masses.append(dict(link=link.name,genesis=float(link.inertial_mass),original=float(self.Dcmm.model.body_mass[body])))
        mass_delta=max([abs(x['genesis']-x['original']) for x in masses],default=0.)
        assert mass_delta<1e-5,('Imported mass mismatch',mass_delta)
        self.adapter_metadata=dict(joint_mapping=mapping,geometry_mapping=self.geom_notes,masses=masses,maximum_mass_delta_kg=mass_delta,physics_backend=self.physics_backend,
            controller='original per-step PID and IK, only physics stepping redirected',contacts='Genesis solver mapped to original geometry categories',
            original_mj_step_calls=0,maximum_kinematic_position_delta_m=0.,lighting=lighting,background_wall_collision=False,
            lighting_schedule=self.lighting_schedule,lighting_trace_by_episode=self.lighting_traces,
            contact_calibration=self.contact_calibration_audit,contact_profile=self.contact_profile,
            contact_hand_selection=hand_selection,outer_physics_dt_s=float(self.Dcmm.model.opt.timestep),
            effective_substep_dt_s=float(self.Dcmm.model.opt.timestep)/self.contact_profile['substeps'])
        def reject_native_step(*args,**kwargs):
            self.adapter_metadata['original_mj_step_calls']+=1
            raise RuntimeError('MuJoCo physics stepping forbidden in Genesis backend')
        for name in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,name,reject_native_step)
        self._write_adapter_metadata()

    def _write_adapter_metadata(self):
        (self.adapter_output/'adapter_audit.json').write_text(json.dumps(self.adapter_metadata,indent=2))

    def _update_public_lighting(self):
        if self.front_glare_lamp is None:return
        from light_schedule import front_glare_intensity
        t=float(self._gs_time);peak=self.lighting_schedule['peak_point_intensity_renderer_units']
        scheduled=front_glare_intensity(t,peak);self.front_glare_lamp.light.intensity=scheduled
        actual=float(self.front_glare_lamp.light.intensity)
        if abs(actual-scheduled)>1e-9:raise RuntimeError('Native point lamp intensity readback mismatch')
        key=(self.episode_number,t)
        if self.episode_number>=0 and key!=self._last_lighting_trace_key:
            rows=self.lighting_traces.setdefault(str(self.episode_number),[])
            rows.append(dict(time_s=t,scheduled_point_intensity=scheduled,actual_point_intensity=actual,time_source='public reset-relative _gs_time'))
            self._last_lighting_trace_key=key
            if hasattr(self,'adapter_metadata') and (len(rows)==1 or len(rows)%10==0):self._write_adapter_metadata()

    def _sync_kinematics(self):
        self.Dcmm.data.qpos[:]=array(self.robot.get_qpos())
        self.Dcmm.data.qvel[:]=array(self.robot.get_dofs_velocity())
        self.Dcmm.data.time=self._gs_time
        mujoco.mj_forward(self.Dcmm.model,self.Dcmm.data)

    def reset(self):
        if self.sensor is not None:self.sensor.close();self.sensor=None
        obs,info=super().reset()
        if self.gs_ready:
            self.episode_number+=1;self.scene.reset();self._gs_time=0.;self._gs_released=False;self._gs_step_count=0
            self.contact_calibration_audit=self.contact_calibration.apply()
            self.adapter_metadata['contact_calibration']=self.contact_calibration_audit
            self._write_adapter_metadata()
            self._last_lighting_trace_key=None
            self.robot.set_qpos(self.Dcmm.data.qpos.copy(),zero_velocity=True)
            self.robot.set_dofs_velocity(self.Dcmm.data.qvel.copy())
            self.scene.sim.rigid_solver.set_gravity(self.Dcmm.model.opt.gravity.copy())
            self._sync_kinematics()
            # Audit positions once per reset; no extra observation/noise draws.
            positions=array(self.robot.get_links_pos(relative=False))
            discrepancies=[]
            for link in self.robot.links:
                body=mujoco.mj_name2id(self.Dcmm.model,mujoco.mjtObj.mjOBJ_BODY,link.name)
                if body>=0:discrepancies.append(float(np.linalg.norm(positions[link.idx_local]-self.Dcmm.data.xpos[body])))
            error=max(discrepancies,default=0.);assert error<1e-6,('Kinematic mismatch',error)
            self.adapter_metadata['maximum_kinematic_position_delta_m']=max(self.adapter_metadata['maximum_kinematic_position_delta_m'],error)
            self._write_adapter_metadata()
            self.update_sensor_pose()
            self.sensor_output=self.adapter_output/f'sensor_episode_{self.episode_number:03d}';self.sensor_output.mkdir(exist_ok=True)
            if os.environ.get('CATCHIT_EVENT_CONTROL')=='1':
                from sparse_event_sensor import SparseEventSensor
                self.sensor=SparseEventSensor(self,self.sensor_output,float(os.environ['CATCHIT_SPHERE_RADIUS']))
            elif os.environ.get('CATCHIT_SENSOR_PROBE')=='1':self.capture_sensor_probe(0.)
        return obs,info

    def _genesis_step(self):
        held=self.Dcmm.data.time-self.start_time<self.object_static_time
        if held or not self._gs_released:
            q=array(self.robot.get_qpos()).copy();v=array(self.robot.get_dofs_velocity()).copy()
            q[37:44]=self.Dcmm.data.qpos[37:44];v[36:42]=self.Dcmm.data.qvel[36:42]
            self.robot.set_qpos(q,zero_velocity=False);self.robot.set_dofs_velocity(v)
            if not held:self._gs_released=True
        force=np.zeros(self.Dcmm.model.nv)
        for i,(start,length,gear) in enumerate(self.actuator_dofs):
            ctrl=float(self.Dcmm.data.ctrl[i])
            if self.Dcmm.model.actuator_ctrllimited[i]:ctrl=float(np.clip(ctrl,*self.Dcmm.model.actuator_ctrlrange[i]))
            force[start:start+length]+=gear*ctrl
        self.robot.control_dofs_force(force)
        self.scene.step();self._gs_time+=float(self.Dcmm.model.opt.timestep);self._sync_kinematics();self._gs_step_count+=1
        if self.sensor is not None and self._gs_step_count%self.event_endpoint_stride==0:self.sensor.capture(self._gs_time)
        elif os.environ.get('CATCHIT_SENSOR_PROBE')=='1' and self._gs_step_count%int(os.environ.get('CATCHIT_PROBE_STRIDE','20'))==0:self.capture_sensor_probe(self._gs_time)

    def update_sensor_pose(self):
        self._update_public_lighting()
        i=self.sensor_camera_id;T=np.eye(4);T[:3,:3]=self.Dcmm.data.cam_xmat[i].reshape(3,3);T[:3,3]=self.Dcmm.data.cam_xpos[i]
        pitch=float(os.environ.get('CATCHIT_SENSOR_PITCH_DEG','0'))
        assert -15.<=pitch<=15.
        if pitch:
            from scipy.spatial.transform import Rotation
            T[:3,:3]=T[:3,:3]@Rotation.from_euler('x',pitch,degrees=True).as_matrix()
        self.sensor_camera.set_pose(transform=T)

    def capture_sensor_probe(self,t):
        self.update_sensor_pose()
        hdr,_,seg,_=self.sensor_camera.render(rgb=True,depth=False,segmentation=True,radiance=True,force_render=True)
        hdr=array(hdr);seg=array(seg);K=array(self.sensor_camera.intrinsics);T=array(self.sensor_camera.transform).astype(float);T[:3,1:3]*=-1
        point=self.Dcmm.data.body('object').xpos;pc=(np.linalg.inv(T)@np.r_[point,1.])[:3];q=K@pc;uv=q[:2]/q[2]
        ball=seg==(int(self.robot.get_link('object').idx)+1);Y=hdr@np.array([.2126,.7152,.0722])
        value=float(np.median(Y[ball])) if ball.any() else None;bg=float(np.percentile(Y[~ball],99.))
        row=dict(episode=self.episode_number,time_s=float(t),GT_uv=uv.tolist(),GT_camera_z=float(pc[2]),visible_pixels=int(ball.sum()),
                 target_median_radiance=value,background_p99_radiance=bg,bright_to_target_ratio=bg/max(value,1e-8) if value is not None else None,
                 GT_is_scoring_only=True)
        self.sensor_probe.append(row)
        (self.adapter_output/'sensor_visibility.json').write_text(json.dumps(self.sensor_probe))
        if self._gs_step_count in (0,260,400,600,800,1200):
            import imageio.v2 as imageio
            display=np.clip(hdr*.2,0,1)**(1/2.2);imageio.imwrite(self.sensor_output/f'sensor_t{t:.3f}.jpg',(display*255).astype(np.uint8))
            np.savez_compressed(self.sensor_output/f'hdr_t{t:.3f}.npz',radiance=hdr,camera_to_world_cv=T,intrinsics=K)

    def event_state(self):
        if self.sensor is None:return None
        from sparse_event_sensor import link_transforms
        hand=link_transforms(self.robot)[int(self.robot.get_link('link6').idx)+1]
        return self.sensor.observe(self._gs_time,hand,self._get_hand_obs())

    def _get_contacts(self):
        if not self.gs_ready:return super()._get_contacts()
        contacts=self.robot.get_contacts()
        a=array(contacts['geom_a']).reshape(-1);b=array(contacts['geom_b']).reshape(-1)
        pairs=np.asarray([(self.geom_map[int(x)],self.geom_map[int(y)]) for x,y in zip(a,b)],dtype=int).reshape(-1,2)
        def other_for(mask):
            if not len(pairs):return np.asarray([],dtype=int)
            return np.concatenate([pairs[mask(pairs[:,0]),1],pairs[mask(pairs[:,1]),0]])
        return dict(hand_contacts=other_for(lambda x:(x<self.object_id)&(x>=self.hand_start_id)),
                    object_contacts=other_for(lambda x:x==self.object_id),base_contacts=other_for(lambda x:x==self.base_id))

    def render(self):
        # The original observation ignores these display arrays; Genesis camera is separate.
        return np.zeros((0,self.img_size[0],self.img_size[1]))

    def render_demo(self):
        self._update_public_lighting()
        return array(self.demo_camera.render(rgb=True)[0]).copy()

    def close(self):
        if self.sensor is not None:self.sensor.close();self.sensor=None
        if self.gs_ready:self._write_adapter_metadata()
        super().close()
