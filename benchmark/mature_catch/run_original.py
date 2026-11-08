"""Bounded Catch It! reproduction. Reuse original inference methods and scoring."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',default='1701,1702,1703,1704,1705,1706')
    parser.add_argument('--object-eval',action='store_true')
    parser.add_argument('--asset-seed',type=int,help='Model-construction seed, separate from episode seeds')
    parser.add_argument('--gate-until',type=float,default=0.,help='Diagnostic only: hold until this time, then original GT policy')
    parser.add_argument('--gate-after-release',type=float,default=None,help='Diagnostic delay from each actual release; GT oracle only')
    parser.add_argument('--speed-scale',type=float,default=1.,help='Scale horizontal launch velocity only')
    parser.add_argument('--horizontal-speed',type=float,help='Use a specified horizontal speed inside the native task range')
    parser.add_argument('--launch-config',type=Path,help='Task-only fixed launch position/velocity JSON; never exposed to perception')
    parser.add_argument('--backend',choices=['mujoco','genesis'],default='mujoco')
    parser.add_argument('--sphere-radius',type=float)
    parser.add_argument('--lighting',choices=['normal','backlit'],default='normal')
    parser.add_argument('--background-wall',action='store_true')
    parser.add_argument('--sensor-probe',action='store_true')
    parser.add_argument('--event-control',action='store_true')
    parser.add_argument('--rgb-initialize',action='store_true')
    parser.add_argument('--rgbd-initialize',action='store_true')
    parser.add_argument('--initialization-probe',action='store_true')
    parser.add_argument('--initialization-only',action='store_true')
    parser.add_argument('--stop-events-after-grasp',action='store_true')
    args=parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    launch_override=json.loads(args.launch_config.resolve().read_text()) if args.launch_config else None
    if args.rgbd_initialize:args.rgb_initialize=True
    args.output.mkdir(parents=True,exist_ok=True)
    os.chdir(args.source);sys.path.insert(0,str(args.source));sys.path.insert(0,str(args.source/'gym_dcmm'))
    import numpy as np
    import torch
    import mujoco
    import imageio.v2 as imageio
    if args.sphere_radius is not None:
        import configs.env.DcmmCfg as cfg
        assert .035<=args.sphere_radius<=.045
        cfg.object_shape=['sphere']*5;cfg.object_size['sphere']=np.array([[args.sphere_radius,args.sphere_radius]])
        args.object_eval=False
    if args.event_control:assert args.backend=='genesis' and args.sphere_radius is not None
    if args.rgb_initialize:assert args.event_control
    if args.initialization_probe or args.initialization_only:assert args.rgb_initialize
    os.environ.update(CATCHIT_LIGHTING=args.lighting,CATCHIT_BACKGROUND_WALL='1' if args.background_wall else '0',
                      CATCHIT_SENSOR_PROBE='1' if args.sensor_probe else '0',CATCHIT_EVENT_CONTROL='1' if args.event_control else '0',
                      CATCHIT_RGB_INITIALIZE='1' if args.rgb_initialize else '0',
                      CATCHIT_RGBD_INITIALIZE='1' if args.rgbd_initialize else '0',CATCHIT_INITIALIZATION_PROBE='1' if args.initialization_probe else '0',
                      CATCHIT_INITIALIZATION_ONLY='1' if args.initialization_only else '0',
                      CATCHIT_STOP_EVENTS_AFTER_GRASP='1' if args.stop_events_after_grasp else '0')
    if args.sphere_radius is not None:os.environ['CATCHIT_SPHERE_RADIUS']=str(args.sphere_radius)
    if args.backend=='genesis':
        os.environ['CATCHIT_ADAPTER_OUTPUT']=str(args.output/'migration_audit')
        from genesis_adapter import GenesisEnv as DcmmVecEnv
    else:
        from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    from gym_dcmm.algs.ppo_dcmm.models_catch import ActorCritic
    from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
    torch.set_num_threads(2)
    seeds=[int(s) for s in args.seeds.split(',')]
    asset_seed=seeds[0] if args.asset_seed is None else args.asset_seed
    np.random.seed(asset_seed);random.seed(asset_seed);torch.manual_seed(asset_seed)
    env=DcmmVecEnv(task='Catching',camera_name=['top'],render_per_step=False,render_mode='rgb_array',
                   object_name='object',img_size=(112,112),imshow_cam=False,viewer=False,
                   object_eval=args.object_eval,env_time=2.5,steps_per_policy=20)
    original_random_pose=env.random_object_pose
    def sample_pose():
        original_random_pose()
        if args.horizontal_speed is not None:
            env.object_vel6d[:2]*=args.horizontal_speed/np.linalg.norm(env.object_vel6d[:2])
        env.object_vel6d[:2]*=args.speed_scale
        if launch_override is not None:
            position=np.asarray(launch_override['position'],dtype=float);velocity=np.asarray(launch_override['velocity'],dtype=float)
            assert position.shape==velocity.shape==(3,) and np.isfinite(position).all() and np.isfinite(velocity).all()
            env.object_pos3d=position.copy();env.object_vel6d[:3]=velocity
    env.random_object_pose=sample_pose
    weights=args.source/'assets/models/catch_two_stage.pth'
    checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
    agent=types.SimpleNamespace(env=types.SimpleNamespace(call=lambda name:[getattr(env,name)]),device='cpu',
        model=ActorCritic(dict(actor_units=[256,128],actions_num=env.act_c_dim,input_shape=(env.obs_c_dim,),separate_value_mlp=True)),
        running_mean_std_track=RunningMeanStd((env.obs_t_dim,)),running_mean_std_hand=RunningMeanStd((12,)),
        action_track_denorm=[1.5,.025,.01],action_catch_denorm=[1.5,.025,.15])
    # Genesis precision=64 changes torch's default dtype; author's actor is float32.
    agent.model.float();agent.model.load_state_dict(checkpoint['model']);agent.model.eval()
    for key in ('running_mean_std_track','running_mean_std_hand'):
        getattr(agent,key).load_state_dict(checkpoint[key]);getattr(agent,key).eval()
    # Compile these methods verbatim from the pinned author's evaluator.
    trainer=args.source/'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_two_stage.py'
    cls=next(x for x in ast.parse(trainer.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='PPO_Catch_TwoStage')
    methods=['obs2tensor','model_act','action2dict'];namespace={'np':np,'torch':torch}
    for node in cls.body:
        if isinstance(node,ast.FunctionDef) and node.name in methods:
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(trainer),'exec'),namespace)
            setattr(agent,node.name,types.MethodType(namespace[node.name],agent))
    renderer=mujoco.Renderer(env.Dcmm.model,height=480,width=640) if args.backend=='mujoco' else None
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,1,.8];camera.distance=4.3;camera.azimuth=180;camera.elevation=-18
    def frame():
        if args.backend=='genesis':return env.render_demo()
        renderer.update_scene(env.Dcmm.data,camera=camera)
        return renderer.render().copy()
    result=[];start=time.monotonic()
    manifest=dict(launch_override=launch_override,sensor_probe_stride=int(os.environ.get('CATCHIT_PROBE_STRIDE','20')),source=str(args.source),upstream_commit='b6af28c67586fc029f3ab43faf5d787e7d3a1609',
        backend=getattr(env,'physics_backend','MuJoCo original reproduction; NOT Genesis'),observation=(('one actual RGB-D frame at reset, then events + original proprioception' if args.rgbd_initialize else 'one actual RGB frame at reset, then events + original proprioception') if args.rgb_initialize else 'event-only known-radius front end + original proprioception') if args.event_control else 'original GT + original noise',
        inference_methods='verbatim AST from original PPO evaluator',checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        timestep=float(env.Dcmm.model.opt.timestep),steps_per_policy=20,env_time=2.5,seeds=seeds,
        success='original evaluator: truncated at env_time > 2.5s; terminated means failure',
        asset_initialization_seed=asset_seed,object_eval=args.object_eval,versions=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,mujoco=mujoco.__version__))
    manifest['gate_until_s']=args.gate_until
    manifest['gate_after_release_s']=args.gate_after_release
    manifest['horizontal_speed_scale']=args.speed_scale
    manifest['horizontal_speed_m_s']=args.horizontal_speed
    manifest.update(sphere_radius_m=args.sphere_radius,lighting=args.lighting,background_wall=args.background_wall,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),event_control=args.event_control,
                    RGB_frame_budget=1 if args.rgb_initialize else 0,control_depth_frame_budget=1 if args.rgbd_initialize else 0,initialization_only=args.initialization_only,initialization_probe=args.initialization_probe,
                    initialization_available_at_reset=args.rgb_initialize,presentation_camera_images_not_used_by_policy=True,
                    stop_event_capture_after_estimated_grasp=args.stop_events_after_grasp,
                    revocable_grasp=os.environ.get('CATCHIT_REVOCABLE_GRASP')=='1',
                    reject_partial_contours=os.environ.get('CATCHIT_REJECT_PARTIAL_CONTOURS')=='1',
                    event_flight_fit=os.environ.get('CATCHIT_EVENT_FLIGHT_FIT')=='1',
                    sensor_pitch_offset_degrees=float(os.environ.get('CATCHIT_SENSOR_PITCH_DEG','0')))
    (args.output/'protocol.json').write_text(json.dumps(manifest,indent=2))
    try:
        for episode,seed in enumerate(seeds):
            np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
            folder=args.output/f'episode_{episode:03d}';folder.mkdir(exist_ok=False)
            obs,info=env.reset();initial=dict(position=env.object_pos3d.tolist(),velocity=env.object_vel6d.tolist(),gravity=env.Dcmm.model.opt.gravity.tolist(),release_time_s=float(env.object_static_time))
            if args.event_control:
                runtime={}
                for name in ['genesis_adapter','sparse_event_sensor','event_sphere_tracker','event_sphere_tracker_kalman','event_sphere_tracker_contour_q80_occlusion','rgb_sphere_initialization']+(['rgbd_sphere_initialization'] if args.rgbd_initialize else [])+['event_sphere_tracker_rgbd_base','event_bearing_flight']:
                    module=sys.modules[name];path=Path(module.__file__).resolve()
                    assert path.parent==Path(__file__).resolve().parent,(name,str(path))
                    runtime[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                (args.output/'loaded_module_receipt.json').write_text(json.dumps(runtime,indent=2))
            if args.initialization_probe:
                detection=env.sensor.initialization
                probe=dict(seed=seed,scope='single RGB initialization probe, no physics rollout',initialization=detection,
                           scoring_only_GT_position=env.Dcmm.data.body('object').xpos.tolist())
                if detection['valid']:probe['scoring_only_position_error_mm']=float(np.linalg.norm(np.asarray(detection['position'])-env.Dcmm.data.body('object').xpos)*1000)
                (folder/'initialization_probe.json').write_text(json.dumps(probe,indent=2));result.append(probe)
                (args.output/'probe_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(probe),flush=True);continue
            if args.rgb_initialize and env.sensor.initialization_failed:
                failed=dict(episode=episode,seed=seed,success=False,steps=0,sim_time=0.,failure_reason='single RGB initialization failed',event_control=True)
                (folder/'result.json').write_text(json.dumps(failed,indent=2));result.append(failed)
                (args.output/'episodes.json').write_text(json.dumps(result,indent=2));print(json.dumps(failed),flush=True);continue
            rows=[];contacts=0;steps=0;reward_sum=0.;first_touch=None;previous_proprio=None;valid_event_polls=0
            with imageio.get_writer(folder/('genesis.mp4' if args.backend=='genesis' else 'original_mujoco.mp4'),fps=env.fps,codec='libx264',quality=8,macro_block_size=None) as video:
                img=frame();video.append_data(img);imageio.imwrite(folder/'initial.png',img)
                while True:
                    event_state=env.event_state() if args.event_control else None
                    proprio=dict(base=env.Dcmm.data.body('arm_base').xpos.copy(),
                                 yaw=float(2*np.arctan2(env.Dcmm.data.body('base_link').xquat[3],env.Dcmm.data.body('base_link').xquat[0])))
                    if args.event_control and event_state is not None:
                        p=np.asarray(event_state['position']);dt=1/env.fps
                        def relative(point,robot):
                            d=point-robot['base'];c=np.cos(robot['yaw']);s=np.sin(robot['yaw'])
                            return np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1],d[2]])
                        rel=relative(p,proprio);acceleration=np.asarray(event_state.get('acceleration',[0.,0.,-9.81]));prev_p=p-np.asarray(event_state['velocity'])*dt+.5*acceleration*dt*dt
                        if env.sensor.tracker.attached_offset is not None and previous_proprio is not None:
                            # Once visually acquired, proprioception propagates an assumed grasp.
                            # No physical contact flag or object truth enters this estimate.
                            from sparse_event_sensor import link_transforms
                            handT=link_transforms(env.robot)[int(env.robot.get_link('link6').idx)+1]
                            prev_p=(previous_proprio.get('handT',handT)@np.r_[env.sensor.tracker.attached_offset,1.])[:3]
                        vel=(rel-relative(prev_p,previous_proprio or proprio))*env.fps
                        obs['object']=dict(pos3d=rel,v_lin_3d=vel)
                        assert event_state.get('maximum_event_s') is None or event_state['maximum_event_s']<=float(env.Dcmm.data.time)+1e-8
                        assert event_state.get('RGB_available_s',0.)<=float(env.Dcmm.data.time)+1e-8
                        valid_event_polls+=1
                    if args.event_control and event_state is None:
                        # The original GT object observation is never passed to the actor.
                        action=torch.zeros((1,env.act_c_dim),dtype=torch.float32)
                        action_dict={key:value[0] for key,value in agent.action2dict(action).items()}
                    else:
                        batched={k:({s:np.asarray(v)[None] for s,v in value.items()} if isinstance(value,dict) else np.asarray(value)[None]) for k,value in obs.items()}
                        with torch.no_grad():
                            action=agent.model_act({'obs':agent.obs2tensor(batched)},inference=True)['actions']
                            action=torch.clamp(action,-1,1)
                            action=torch.nn.functional.pad(action,(0,env.act_c_dim-action.shape[1]),value=0)
                            action_dict={key:value[0] for key,value in agent.action2dict(action).items()}
                    if args.event_control:
                        from sparse_event_sensor import link_transforms
                        proprio['handT']=link_transforms(env.robot)[int(env.robot.get_link('link6').idx)+1]
                    previous_proprio=proprio
                    gate=max(args.gate_until,env.object_static_time+args.gate_after_release if args.gate_after_release is not None else 0.)
                    held=float(env.Dcmm.data.time-env.start_time)<gate or (args.event_control and event_state is None)
                    if held:
                        action.zero_()
                        action_dict={key:np.zeros_like(value) for key,value in action_dict.items()}
                    before=float(env.Dcmm.data.time)
                    scoring_position_before=env.Dcmm.data.body('object').xpos.copy()
                    scoring_uv_before=None
                    predicted_uv=None;scoring_pixel_error=None
                    policy_object_input=({k:np.asarray(v).tolist() for k,v in obs['object'].items()} if (not args.event_control or event_state is not None) else None)
                    if args.backend=='genesis':
                        T=np.asarray(env.sensor_camera.transform,dtype=float).copy();T[:3,1:3]*=-1
                        pc=(np.linalg.inv(T)@np.r_[scoring_position_before,1.])[:3];q=np.asarray(env.sensor_camera.intrinsics)@pc
                        scoring_uv_before=(q[:2]/q[2]).tolist()
                        if event_state is not None:
                            epc=(np.linalg.inv(T)@np.r_[event_state['position'],1.])[:3];eq=np.asarray(env.sensor_camera.intrinsics)@epc
                            predicted_uv=(eq[:2]/eq[2]).tolist();scoring_pixel_error=float(np.linalg.norm(np.asarray(predicted_uv)-scoring_uv_before))
                    obs,reward,terminated,truncated,info=env.step(action_dict)
                    steps+=1;reward_sum+=float(reward);contacts+=int(env.step_touch)
                    if env.step_touch and first_touch is None:first_touch=float(info['env_time'])
                    rows.append(dict(step=steps,t_before=before,t=float(env.Dcmm.data.time),terminated=bool(terminated),truncated=bool(truncated),
                        scoring_only_GT_position_at_control=scoring_position_before.tolist(),scoring_only_GT_uv_at_control=scoring_uv_before,
                        policy_object_input=policy_object_input,predicted_event_uv=predicted_uv,scoring_only_pixel_error=scoring_pixel_error,
                        stage=env.stage,held_before_observation=held,hand_contact=bool(env.step_touch),ee_distance=float(info['ee_distance']),
                        event_state=({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in event_state.items()} if event_state else None),
                        reward=float(reward),action=action[0].tolist(),qpos=env.Dcmm.data.qpos.tolist(),qvel=env.Dcmm.data.qvel.tolist(),
                        object_position=env.Dcmm.data.body('object').xpos.tolist(),hand_position=env.Dcmm.data.body('link6').xpos.tolist()))
                    video.append_data(frame())
                    if steps%5==0:(folder/'trace.partial.json').write_text(json.dumps(rows))
                    if terminated or truncated:break
                    if steps>100:raise RuntimeError('Unexpected original task length')
                imageio.imwrite(folder/'final.png',frame())
            row=dict(episode=episode,seed=seed,success=bool(truncated),terminated=bool(terminated),steps=steps,
                sim_time=float(info['env_time']),hand_contact_steps=contacts,first_touch=first_touch,reward=reward_sum,initial=initial,
                final_ee_distance=float(info['ee_distance']),valid_event_polls=valid_event_polls,event_control=args.event_control,
                max_base_displacement_m=float(max(np.linalg.norm(np.asarray(x['qpos'][:2])-np.asarray(rows[0]['qpos'][:2])) for x in rows)))
            (folder/'trace.json').write_text(json.dumps(rows));(folder/'result.json').write_text(json.dumps(row,indent=2));result.append(row)
            (args.output/'episodes.json').write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
        if args.initialization_probe:
            (args.output/'probe_summary.json').write_text(json.dumps(dict(probes=len(result),valid=sum(x['initialization']['valid'] for x in result),scope='no physics rollout'),indent=2))
        else:
            (args.output/'summary.json').write_text(json.dumps(dict(episodes=len(result),successes=sum(x['success'] for x in result),wall_s=time.monotonic()-start,backend=args.backend,event_control=args.event_control),indent=2))
    finally:
        if renderer is not None:renderer.close()
        env.close()

if __name__=='__main__':main()
