"""Cost-only shadow replay around sealed run_original.py; no source-file edits.

Physics receives a fixed archived action sequence. The actual selected frontend
and original actor still execute, but their actions cannot change this cost
trajectory. A typed exception ends the fixed horizon without creating a fake
task success. This program requires the normal server Genesis/CUDA environments.
"""
from __future__ import annotations
import argparse
import ast
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import runpy
import socket
import sys
import time
from types import SimpleNamespace
sys.dont_write_bytecode = True
from cost_ledger import WallLedger, sha256_file
from query_witness import QueryWitness,raw_event_summary


class CostHorizonComplete(Exception):
    pass


PHYSICS_SCHEMA='fixed_archived_actions_public_cost_physics_v1'
RECORDED='PHYSICS_REFERENCE_RECORDED'
REGISTERED_PHYSICS_SHA256='41458da631174aad39846d96e0ab7b753aa45c79bfcc0046658ba913d02b32a9'


def validate_public_physics_reference(payload, controls, expected):
    """Validate one predeclared real recording, not a per-method fitted path."""
    if payload.get('schema')!=PHYSICS_SCHEMA or payload.get('status')!=RECORDED:
        raise ValueError('Not a completed public physical reference recording')
    if payload.get('utility_result') is not False or payload.get('cost_measurement') is not False:
        raise ValueError('A reference recording is neither utility nor formal cost')
    for name,value in expected.items():
        if payload.get(name)!=value:
            raise ValueError('Public physical reference differs: '+name)
    rows=payload.get('controls',[])
    if len(rows)!=len(controls):
        raise ValueError('Public physical reference does not cover every fixed control')
    for index,(row,original) in enumerate(zip(rows,controls)):
        if row.get('step')!=index or row.get('action')!=original['action']:
            raise ValueError('Archived action sequence was changed in the public reference')
        if any(abs(row[key]-original[key])>1e-8 for key in ('t_before','t')):
            raise ValueError('Public physical reference changed the fixed control timestamps')
        for key in ('qpos','qvel'):
            values=row.get(key,[])
            if len(values)!=len(original[key]) or not all(math.isfinite(x) for x in values):
                raise ValueError('Invalid actual recorded physical state: '+key)
    return rows


class NullDataset:
    def __init__(self, shape=None, data=None, **kwargs):
        self.shape = tuple(shape) if shape is not None else getattr(data, 'shape', ())
    def resize(self, size, axis=0):
        shape = list(self.shape)
        shape[axis] = size
        self.shape = tuple(shape)
    def __setitem__(self, key, value):
        pass


class NullArchive:
    """Explicitly disabled audit HDF writer; no fake HDR/HDF file is created."""
    def __init__(self):
        self.attrs, self.datasets = {}, {}
    def __contains__(self, key):
        return key in self.datasets
    def __getitem__(self, key):
        return self.datasets[key]
    def create_dataset(self, name, **kwargs):
        value = NullDataset(**kwargs)
        self.datasets[name] = value
        return value
    def flush(self):
        pass
    def close(self):
        pass


class NullMovie:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def append_data(self, frame):
        pass


def load_json(path):
    return json.loads(Path(path).read_text())


def prewarm_physics_snapshot(env, actual_action, archived_action, np):
    """Read-only forensic state; never draw RNG values or alter an IK target.

    Used only before the measured .32s boundary. In particular the official
    controller's private data_arm is not captured by a public qpos audit.
    """
    def array_or_none(value):
        return np.asarray(value).copy().tolist() if value is not None else None
    dcmm=env.Dcmm
    out=dict(
        numpy_random_state_sha256=hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=5)).hexdigest(),
        actual_actor_action_dtypes={k:str(np.asarray(v).dtype) for k,v in actual_action.items()},
        replay_action_dtypes={k:str(np.asarray(v).dtype) for k,v in archived_action.items()},
        actual_actor_vs_archived_decoded_max_abs={k:float(np.max(np.abs(np.asarray(actual_action[k])-v)))
            for k,v in archived_action.items()},
        private_arm_kinematic_qpos=array_or_none(dcmm.data_arm.qpos),
        target_arm_qpos=array_or_none(getattr(dcmm,'target_arm_qpos',None)),
        target_hand_qpos=array_or_none(getattr(dcmm,'target_hand_qpos',None)),
        target_base_vel=array_or_none(getattr(dcmm,'target_base_vel',None)),
        noise_and_PID={k:array_or_none(getattr(env,k,None)) for k in
            ('k_act','k_arm','k_drive','k_steer','k_hand','k_obs_base','k_obs_arm','k_obs_hand','k_obs_object')},
        action_buffers={k:dict(length=len(v),maxlen=(int(v.maxlen) if getattr(v,'maxlen',None) is not None else None))
            for k,v in env.action_buffer.items()})
    return out


def main():
    launcher_begin = time.perf_counter_ns()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-meta', type=Path, required=True, help='Sealed FAIR_RUNTIME.json')
    p.add_argument('--reference-run', type=Path, required=True, help='Archived successful combined-physics actor run')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--record-physics-reference',action='store_true',help='Record a new public physical path from the same fixed archived actions; no utility or cost score')
    mode.add_argument('--physics-reference',type=Path,help='Strictly match this previously recorded public PHYSICS_REFERENCE.json instead of the legacy utility qpos')
    p.add_argument('--frontend-json', type=Path, required=True, help='Locked frontend env or {env:...}; no tuning here')
    p.add_argument('--method', choices=['ours', 'v2e'], required=True)
    p.add_argument('--rate', type=float, choices=[12.5, 25., 50., 100.], required=True)
    p.add_argument('--repeat-id', default='r00')
    p.add_argument('--v2e-parity-receipt', type=Path, help='Native V2E receipt matching this actual source cadence')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup-s', type=float, default=.32)
    p.add_argument('--horizon-s', type=float, default=1.12)
    p.add_argument('--acquisition', choices=['legacy', 'required'], default='required')
    p.add_argument('--native-parity-receipt','--cost-equivalence-receipt',dest='native_parity_receipt',type=Path)
    p.add_argument('--equivalence-kind',choices=['raw_exact','task_functional'],default='raw_exact')
    p.add_argument('--profile-worker', action='store_true', help='Separate diagnostic pass, not primary throughput')
    p.add_argument('--synchronize-worker-stages', action='store_true')
    p.add_argument('--retain-audit-io', action='store_true', help='Separate audit-overhead pass; primary defaults off')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        p.error('Use a fresh cost output directory')
    if not 0 <= args.warmup_s < args.horizon_s or any(abs(t/.08-round(t/.08)) > 1e-8 for t in (args.warmup_s, args.horizon_s)):
        p.error('Warmup/horizon must be fixed, increasing multiples of80ms for all four cadences')
    if args.synchronize_worker_stages and not args.profile_worker:
        p.error('Per-stage synchronization belongs only to a separate --profile-worker pass')
    if args.record_physics_reference:
        p.error('v4 reuses the registered v3 public physical reference; no rerecording is permitted')
    if args.physics_reference and sha256_file(args.physics_reference)!=REGISTERED_PHYSICS_SHA256:
        p.error('v4 must retain the registered v3 physical reference SHA; do not replace a failing reference')
    if args.equivalence_kind=='task_functional' and not args.physics_reference:
        p.error('task_functional mode requires the registered public physical reference')
    if args.record_physics_reference and (args.method!='ours' or args.rate!=25 or
            args.acquisition!='legacy' or not args.retain_audit_io or args.profile_worker):
        p.error('Predeclared reference recipe is Ours25 legacy with retained audit, original unprofiled worker')
    if not os.environ.get('EGL_DEVICE_ID'):
        p.error('Run inside an already allocated server GPU job with EGL_DEVICE_ID; this script does not reserve a GPU')
    meta = load_json(args.source_meta)
    source = Path(meta['source'])
    for name, digest in meta['runtime_files'].items():
        if sha256_file(source/name) != digest:
            p.error('Sealed source mismatch: '+name)
    reference = load_json(args.reference_run/'protocol.json')
    frozen_receipt_path=args.reference_run.parents[2]/'EVALUATION_PROTOCOL.json'
    frozen=load_json(frozen_receipt_path)
    if len(frozen['core_files'])!=17:
        p.error('Expected the frozen17-file event-core manifest')
    for name,digest in frozen['core_files'].items():
        if sha256_file(Path(frozen['core_root'])/'genesis_event_plugin'/name)!=digest:
            p.error('Frozen event core differs: '+name)
    if sha256_file(frozen['profile_file'])!=frozen['profile_sha256']:
        p.error('Frozen event sensor profile differs')
    trace_path = args.reference_run/'episode_000/trace.json'
    trace = load_json(trace_path)
    ref_result = load_json(args.reference_run/'episode_000/result.json')
    physics_receipt_path = args.reference_run/'migration_audit/adapter_audit.json'
    physics_receipt = load_json(physics_receipt_path)
    if physics_receipt['contact_profile']['name'] != 'combined' or reference.get('sensor_pitch_offset_degrees') != 7:
        p.error('Cost reference must use the fixed combined physics and7degree camera')
    if not ref_result['success'] or ref_result['terminated'] or ref_result['sim_time'] < args.horizon_s:
        p.error('Reference must actually reach the same fixed cost horizon successfully')
    if ref_result.get('first_touch') is not None and ref_result['first_touch'] <= args.horizon_s:
        p.error('This initial cost protocol fixes a precontact horizon; choose a later-contact reference')
    selected = [row for row in trace if row['t_before'] < args.horizon_s-1e-8]
    if not selected or abs(selected[-1]['t']-args.horizon_s) > 1e-8:
        p.error('Reference controls do not exactly cover the planned horizon')
    catch_source = Path(reference['source'])
    checkpoint = catch_source/'assets/models/catch_two_stage.pth'
    if sha256_file(checkpoint) != reference['checkpoint_sha256']:
        p.error('Original actor checkpoint differs from the archived reference')
    physics_expected=dict(original_action_trace_sha256=sha256_file(trace_path),
        source_meta_sha256=sha256_file(args.source_meta),source_code_commit=meta['code_commit'],
        original_policy_sha256=sha256_file(checkpoint),frozen_core_files=frozen['core_files'],
        original_controller_files={name:sha256_file(catch_source/name) for name in (
            'gym_dcmm/envs/DcmmVecEnv.py','gym_dcmm/agents/MujocoDcmm.py',
            'gym_dcmm/utils/ik_pkg/ik_arm.py','gym_dcmm/algs/ppo_dcmm/models_catch.py',
            'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_two_stage.py')},
        frozen_sensor_profile_sha256=frozen['profile_sha256'],
        warmup_simulated_s=args.warmup_s,horizon_simulated_s=args.horizon_s,
        reference_seed=reference['seeds'][0],asset_initialization_seed=reference['asset_initialization_seed'],
        launch_override=reference['launch_override'],recording_host=socket.gethostname(),
        fixed_scene=dict(contact_profile='combined',camera_pitch_degrees=7,resolution_wh=[640,480],
            sphere_radius_m=reference['sphere_radius_m'],lighting=reference['lighting'],background_wall=reference.get('background_wall',False)))
    if args.physics_reference:
        try:
            physics_controls=validate_public_physics_reference(load_json(args.physics_reference),selected,physics_expected)
        except (KeyError,ValueError,TypeError,OSError) as exc:
            p.error(str(exc))
        physics_reference_sha256=sha256_file(args.physics_reference)
        physics_reference_kind='public_recorded_fixed_archived_actions'
    else:
        physics_controls=selected
        physics_reference_sha256=None if args.record_physics_reference else sha256_file(trace_path)
        physics_reference_kind='record_only_new_public_reference' if args.record_physics_reference else 'legacy_utility_rollout'
    sys.path[:0] = [str(source), str(catch_source), str(catch_source/'gym_dcmm')]
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        CATCHIT_EVENT_ENDPOINT_HZ=str(args.rate), CATCHIT_EVENT_GENERATOR=args.method,
        CATCHIT_CONTACT_PROFILE='combined', CATCHIT_SENSOR_PITCH_DEG='7',
        CATCHIT_REVOCABLE_GRASP='0', CATCHIT_REJECT_PARTIAL_CONTOURS='0')
    frontend_module = importlib.import_module('frontend_config')
    frontend_payload = load_json(args.frontend_json)
    frontend = frontend_module.validate_frontend(frontend_payload.get('env', frontend_payload))
    os.environ.update(frontend)
    frontend_id=frontend_payload.get('frontend_id') or ('w'+frontend['CATCHIT_EVENT_OBSERVATION_WINDOW_MS']+
        '_fit'+frontend['CATCHIT_EVENT_FLIGHT_FIT']+'_gap'+frontend['CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS'])
    if args.v2e_parity_receipt:
        os.environ['CATCHIT_V2E_PARITY_RECEIPT']=str(args.v2e_parity_receipt.resolve())
    if args.method=='v2e':
        receipt_path=os.environ.get('CATCHIT_V2E_PARITY_RECEIPT')
        if not receipt_path:
            p.error('V2E requires --v2e-parity-receipt for this cadence')
        native_receipt=load_json(receipt_path)
        expected_up=max(2,int(math.ceil((1/args.rate)/.001-1e-9)))
        if native_receipt.get('status')!='PASS' or native_receipt.get('minimum_upsampling')!=expected_up:
            p.error('V2E native parity receipt does not cover this cadence/upsampling')
    args.output.mkdir(parents=True)
    runner_output = args.output/'runner_output'
    launch_path = args.output/'fixed_launch.json'
    launch_path.write_text(json.dumps(reference['launch_override'], indent=2))
    import numpy as np
    import torch
    import h5py
    import imageio.v2 as imageio
    adapter = importlib.import_module('genesis_adapter')
    sensor_module = importlib.import_module('sparse_event_sensor')
    tracker_module = importlib.import_module('event_sphere_tracker')
    models = importlib.import_module('gym_dcmm.algs.ppo_dcmm.models_catch')
    sensor_cls, env_cls = sensor_module.SparseEventSensor, adapter.GenesisEnv
    ledger = WallLedger()
    query_witness=QueryWitness(np,torch)
    active_env=[None]
    process_begin = launcher_begin
    checks = []
    recorded_controls=[]
    def owned(path):
        try:
            return Path(path).resolve().is_relative_to(runner_output)
        except (TypeError,ValueError,OSError):
            return False

    def measured_method(cls, name, stage, clock=None):
        original = getattr(cls, name)
        @functools.wraps(original)
        def measured(self, *a, **kw):
            t = clock(self, a, kw) if clock else None
            with ledger.span(stage, t):
                return original(self, *a, **kw)
        setattr(cls, name, measured)
        return original

    # Decode the archived normalized action with the author's actual action2dict,
    # rather than approximating IK/PID commands or changing policy weights.
    trainer = catch_source/'gym_dcmm/algs/ppo_dcmm/ppo_dcmm_catch_two_stage.py'
    cls = next(n for n in ast.parse(trainer.read_text()).body if isinstance(n, ast.ClassDef) and n.name=='PPO_Catch_TwoStage')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='action2dict')
    scope = {'np':np, 'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(trainer), 'exec'), scope)
    denorm = SimpleNamespace(env=SimpleNamespace(call=lambda name:['Catching']),
        action_track_denorm=[1.5,.025,.01], action_catch_denorm=[1.5,.025,.15])
    original_step = env_cls.step
    step_index = [0]
    @functools.wraps(original_step)
    def fixed_step(self, ignored_actor_action):
        i = step_index[0]
        if i >= len(selected):
            raise RuntimeError('Fixed action horizon exceeded')
        row = selected[i]
        physics_row=physics_controls[i]
        before = float(self.Dcmm.data.time)
        if abs(before-row['t_before']) > 1e-8:
            raise RuntimeError('Reference control timestamp differs before physics')
        caller=sys._getframe(1)
        if Path(caller.f_code.co_filename).resolve()!= (source/'run_original.py').resolve() or any(
                key not in caller.f_locals for key in ('held','event_state','action')):
            raise RuntimeError('Cannot witness actual sealed runner held/None/action flags')
        query_witness.complete_query(before,caller.f_locals['held'],caller.f_locals['event_state'] is None,
            ignored_actor_action,caller.f_locals['action'])
        action = torch.tensor([row['action']], dtype=torch.float32)
        actions = {k:v[0] for k,v in scope['action2dict'](denorm, action).items()}
        debug=prewarm_physics_snapshot(self,ignored_actor_action,actions,np) if before<args.warmup_s-1e-8 else None
        with ledger.span('environment_step_inclusive', before):
            result = original_step(self, actions)
        q = np.asarray(self.Dcmm.data.qpos)
        v = np.asarray(self.Dcmm.data.qvel)
        qerr = float(np.max(np.abs(q-np.asarray(physics_row['qpos']))))
        verr = float(np.max(np.abs(v-np.asarray(physics_row['qvel']))))
        now = float(self.Dcmm.data.time)
        check=dict(step=i,t_before=before,t=now,
            qpos_max_abs_error=None if args.record_physics_reference else qerr,
            qvel_max_abs_error=None if args.record_physics_reference else verr)
        if args.record_physics_reference:
            check.update(legacy_archive_qpos_max_abs_delta=qerr,legacy_archive_qvel_max_abs_delta=verr)
            recorded_controls.append(dict(step=i,t_before=before,t=now,action=row['action'],qpos=q.copy().tolist(),qvel=v.copy().tolist()))
        if debug is not None:
            check['prewarm_diagnostics']=debug
        checks.append(check)
        if abs(now-physics_row['t'])>1e-8 or (not args.record_physics_reference and (qerr>1e-6 or verr>1e-5)):
            check['divergence_details']=dict(actual_qpos=q.tolist(),expected_qpos=physics_row['qpos'],
                actual_qvel=v.tolist(),expected_qvel=physics_row['qvel'],
                qpos_largest_error_index=int(np.argmax(np.abs(q-np.asarray(physics_row['qpos'])))),
                qvel_largest_error_index=int(np.argmax(np.abs(v-np.asarray(physics_row['qvel'])))),
                decoded_archived_action={k:np.asarray(value).tolist() for k,value in actions.items()},
                private_arm_kinematic_qpos_after=np.asarray(self.Dcmm.data_arm.qpos).tolist(),
                target_arm_qpos_after=np.asarray(self.Dcmm.target_arm_qpos).tolist(),
                numpy_random_state_sha256_after=hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=5)).hexdigest())
            raise RuntimeError('Actual physics diverged from the fixed archived trajectory; cost result invalid')
        step_index[0] += 1
        if now >= args.horizon_s-1e-8:
            ledger.stop_hot()
            raise CostHorizonComplete()
        if result[2] or result[3]:
            raise RuntimeError('Cost physics terminated before the common horizon')
        return result
    env_cls.step = fixed_step
    measured_method(env_cls, '_genesis_step', 'common_physics_driver_inclusive', lambda s,a,k:float(s._gs_time))
    measured_method(env_cls, '_get_ctrl', 'common_pid_control')
    original_actor=models.ActorCritic.act_inference
    @functools.wraps(original_actor)
    def witnessed_actor(self,*a,**kw):
        if active_env[0] is None:
            raise RuntimeError('Actual actor query lacks its active environment')
        call=query_witness.model_input(a,kw,float(active_env[0].Dcmm.data.time))
        with ledger.span('original_actor_inference'):
            result=original_actor(self,*a,**kw)
        query_witness.model_output(call,result)
        return result
    models.ActorCritic.act_inference=witnessed_actor
    original_observe = tracker_module.EventSphereTracker.observe
    def measured_observe(self, now_s, *a, **kw):
        query_witness.begin_query(now_s)
        if now_s >= args.warmup_s-1e-8:
            ledger.start_hot()
        with ledger.span('frontend_observe', float(now_s)):
            result=original_observe(self, now_s, *a, **kw)
        query_witness.frontend_result(result)
        return result
    tracker_module.EventSphereTracker.observe = measured_observe

    original_sensor_init = sensor_cls.__init__
    def sensor_init(self, env, *a, **kw):
        active_env[0]=env
        camera = env.sensor_camera
        render = camera.render
        def measured_render(*ra, **rk):
            if rk.get('radiance', False):
                stage = 'ours_HDR_depth_seg_acquisition' if args.method=='ours' else 'v2e_audit_only_HDR_depth_seg'
            elif rk.get('depth', False):
                stage = 'shared_initial_RGBD_acquisition'
            else:
                stage = 'v2e_native_LDR_acquisition'
            with ledger.span(stage):
                return render(*ra, **rk)
        camera.render = measured_render
        return original_sensor_init(self, env, *a, **kw)
    sensor_cls.__init__ = sensor_init
    original_link_transforms = sensor_module.link_transforms
    def measured_poses(*a, **kw):
        in_capture = ledger.inside('sensor_capture_inclusive')
        stage = ('ours_geometry_pose_acquisition' if args.method=='ours' else 'v2e_audit_only_geometry_poses') if in_capture else 'shared_robot_proprioception'
        with ledger.span(stage):
            return original_link_transforms(*a, **kw)
    sensor_module.link_transforms = measured_poses
    array_original=sensor_module.array
    def measured_array(*a,**kw):
        stage='capture_array_readback' if ledger.inside('sensor_capture_inclusive') else 'robot_array_readback'
        with ledger.span(stage):
            return array_original(*a,**kw)
    sensor_module.array=measured_array
    for name,stage in (('encode','worker_RPC_array_wire_encode'),('decode','worker_RPC_array_wire_decode')):
        original=getattr(sensor_module,name)
        def measured_wire(*a,_fn=original,_stage=stage,**kw):
            with ledger.span(_stage):
                return _fn(*a,**kw)
        setattr(sensor_module,name,measured_wire)
    original_capture = sensor_cls.capture
    def required_v2e_capture(self, t, reset=False):
        """Same native LDR/SSM input; omit only unused audit HDR/geometry readouts."""
        self.env.update_sensor_pose()
        if self.initialization_only or self.initialization_probe or self.initialization_failed:
            return
        import cv2
        started = time.perf_counter_ns()
        ldr = sensor_module.array(self.camera.render(rgb=True,depth=False,segmentation=False,radiance=False,force_render=True)[0])
        if ldr.dtype != np.uint8:
            raise RuntimeError('Native renderer LDR must stay uint8')
        gray8 = cv2.cvtColor(ldr,cv2.COLOR_RGB2GRAY)
        ldr_s = (time.perf_counter_ns()-started)/1e9
        T = sensor_module.array(self.camera.transform).astype(float)
        T[:3,1:3] *= -1
        packet = dict(time_s=float(t), gray8=gray8)
        rpc_start = time.perf_counter_ns()
        pickle.dump(sensor_module.encode(dict(kind='reset' if reset else 'frame',packet=packet)),self.worker.stdin,protocol=5)
        self.worker.stdin.flush()
        reply = sensor_module.decode(pickle.load(self.worker.stdout))
        events = reply['events']
        if len(events):
            self.chunks.append(events.copy())
        self.tracker.push(events,available_s=float(t)+.010,packet_time_s=t,camera_to_world=T)
        self.packet_audit.append(dict(time_s=float(t),available_s=float(t)+.010,events=len(events),
            worker_wall_s=reply['wall_s'],transfer_and_worker_wall_s=(time.perf_counter_ns()-rpc_start)/1e9,
            native_ldr_capture_s=ldr_s,hdr_capture_s=0.,hdr_acquired=False,hdr_used_only_for_audit=False,
            method='v2e',endpoint_rate_hz=self.endpoint_hz,subframes=reply.get('subframes'),
            endpoint_reads=reply['endpoint_reads'],dense_HDR_reads=0,renderer_calls_in_reconstruction=0,
            worker_input_keys=['gray8','time_s'],cost_only_native_acquisition=True))
        # If explicitly retaining audit data, store only real LDR and camera pose;
        # never synthesize empty/fake HDR, depth or segmentation datasets.
        self.h.attrs['source'] = 'Cost-only native LDR capture; no audit HDR/depth/segmentation acquired'
        for name,value in dict(time_s=np.asarray(t),native_ldr_gray8=gray8,camera_to_world_cv=T).items():
            if name not in self.h:
                self.h.create_dataset(name,shape=(0,)+value.shape,maxshape=(None,)+value.shape,chunks=(1,)+value.shape,dtype=value.dtype)
            ds=self.h[name]
            ds.resize(self.frame_index+1,axis=0)
            ds[self.frame_index]=value
        self.frame_index += 1
    def capture(self, t, reset=False):
        with ledger.span('sensor_capture_inclusive', float(t)):
            if args.method=='v2e' and args.acquisition=='required':
                return required_v2e_capture(self,t,reset)
            return original_capture(self,t,reset)
    sensor_cls.capture = capture
    # Time the transport boundary. Worker compute is inside this wait and must
    # not be added to it again when composing a total wall budget.
    dump, load = pickle.dump, pickle.load
    def measured_dump(value, file, *a, **kw):
        if ledger.inside('sensor_capture_inclusive'):
            with ledger.span('worker_RPC_send_serialization'):
                return dump(value,file,*a,**kw)
        return dump(value,file,*a,**kw)
    def measured_load(file, *a, **kw):
        if ledger.inside('sensor_capture_inclusive'):
            with ledger.span('worker_RPC_receive_wait'):
                return load(file,*a,**kw)
        return load(file,*a,**kw)
    pickle.dump, pickle.load = measured_dump, measured_load

    if not args.retain_audit_io:
        h5open = h5py.File
        def maybe_null_h5(path,*a,**kw):
            if owned(path) and Path(path).name=='sparse_hdr_endpoints.h5':
                return NullArchive()
            return h5open(path,*a,**kw)
        h5py.File = maybe_null_h5
        writer, imwrite = imageio.get_writer, imageio.imwrite
        imageio.get_writer = lambda path,*a,**kw: NullMovie() if owned(path) else writer(path,*a,**kw)
        def write_image(path,*a,**kw):
            if owned(path) and Path(path).name in ('initial.png','final.png'):
                return None
            return imwrite(path,*a,**kw)
        imageio.imwrite = write_image
        env_cls.render_demo = lambda self: np.zeros((480,640,3),dtype=np.uint8)
    else:
        measured_method(env_cls,'render_demo','audit_presentation_render')

    if args.profile_worker:
        import subprocess as sp
        popen = sp.Popen
        def profiled_worker(command,*a,**kw):
            if isinstance(command,list) and command and Path(command[-1]).name in ('event_worker.py','v2e_event_worker.py'):
                name = 'ours' if Path(command[-1]).name=='event_worker.py' else 'v2e'
                command = [command[0],'-u',str(Path(__file__).with_name('cost_worker_wrapper.py')),
                    '--worker-script',command[-1],'--source-dir',str(source),'--method',name,
                    '--sidecar',str(args.output/'worker_detailed_profile.json')]
                if args.synchronize_worker_stages:
                    command.append('--synchronize-stages')
            return popen(command,*a,**kw)
        sp.Popen = profiled_worker

    argv = [str(source/'run_original.py'),'--source',str(catch_source),'--output',str(runner_output),
        '--backend','genesis','--event-control','--rgbd-initialize','--sphere-radius',str(reference['sphere_radius_m']),
        '--seeds',str(reference['seeds'][0]),'--asset-seed',str(reference['asset_initialization_seed']),
        '--lighting',reference['lighting'],'--launch-config',str(launch_path)]
    if reference.get('background_wall'):
        argv.append('--background-wall')
    if reference['lighting']=='front_glare':
        p.error('This first fixed cost pilot uses the registered normal-light trace; add timed lighting replay separately')
    sys.argv = argv
    status, error = 'ERROR', None
    try:
        runpy.run_path(str(source/'run_original.py'),run_name='__main__')
        error = 'Original runner ended before the fixed cost sentinel; do not publish this cost'
    except CostHorizonComplete:
        status = RECORDED if args.record_physics_reference else 'MEASURED_FIXED_TRAJECTORY'
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
    finally:
        # All writes below are isolated measurement outputs; no original result,
        # utility score or frozen source file is rewritten.
        for name,digest in meta['runtime_files'].items():
            if sha256_file(source/name) != digest:
                status,error='ERROR','Source bytes changed during measurement: '+name
        if args.physics_reference and sha256_file(args.physics_reference)!=physics_reference_sha256:
            status,error='ERROR','Public physical reference bytes changed during measurement'
        elapsed = (time.perf_counter_ns()-process_begin)/1e9
        summary = ledger.summary()
        sensor_root=runner_output/'migration_audit/sensor_episode_000'
        packet_path=sensor_root/'packet_audit.json'
        packets=load_json(packet_path) if packet_path.is_file() else []
        expected_packets=int(round(args.horizon_s*args.rate))+1
        complete_capture=(len(packets)==expected_packets and np.allclose(
            [x['time_s'] for x in packets],np.arange(expected_packets)/args.rate,rtol=0,atol=1e-8))
        if status in ('MEASURED_FIXED_TRAJECTORY',RECORDED) and not complete_capture:
            status,error='ERROR','Unconditional endpoint capture did not cover the fixed source-frame budget'
        if status==RECORDED:
            public=dict(schema=PHYSICS_SCHEMA,status=RECORDED,utility_result=False,cost_measurement=False,
                **physics_expected,controls=recorded_controls,
                recording_recipe=dict(method=args.method,rate_hz=args.rate,frontend_env=frontend,
                    acquisition_mode=args.acquisition,retain_audit_io=args.retain_audit_io,
                    action_source='Unchanged original archived normalized actions; computed current actor actions discarded'),
                recording_cpu_affinity=(sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None),
                recording_instrumentation_sha256=sha256_file(__file__),
                contract='This recording defines one common real physical trajectory. Every later method/rate/repeat must independently match qpos<=1e-6,qvel<=1e-5,time<=1e-8; no method-specific rerecording.')
            validate_public_physics_reference(public,selected,physics_expected)
            public_path=args.output/'PHYSICS_REFERENCE.json'
            public_path.write_text(json.dumps(public,indent=2),encoding='utf-8')
            physics_reference_sha256=sha256_file(public_path)
        worker_start_path=sensor_root/'worker_start.json'
        worker_start=load_json(worker_start_path) if worker_start_path.is_file() else None
        witness_payload=None
        witness_error=None
        try:
            witness_payload=query_witness.checked_payload([row['t_before'] for row in selected])
        except (ValueError,RuntimeError) as exc:
            witness_error=str(exc)
            if status=='MEASURED_FIXED_TRAJECTORY':
                status,error='ERROR','Actual policy-query witness failed: '+str(exc)
        witness_path=args.output/'ACTOR_QUERY_WITNESS.json'
        witness_path.write_text(json.dumps(dict(status='COMPLETE' if witness_payload else 'INCOMPLETE',
            error=witness_error,source_meta_sha256=sha256_file(args.source_meta),
            original_policy_sha256=sha256_file(checkpoint),physics_reference_sha256=physics_reference_sha256,
            payload=witness_payload,incomplete_queries=query_witness.queries if witness_payload is None else None),indent=2),encoding='utf-8')
        raw_path=sensor_root/'raw_events.npz'
        raw_summary=None
        if raw_path.is_file():
            with np.load(raw_path) as raw:
                raw_summary=raw_event_summary(raw['events'],np)
        parity = load_json(args.native_parity_receipt) if args.native_parity_receipt else None
        expected_status='PASS_TASK_FUNCTIONAL_EQUIVALENCE' if args.equivalence_kind=='task_functional' else 'PASS_RAW_AND_TASK_EQUIVALENCE'
        functional_ok=bool(parity and witness_payload and parity.get('functional_query_exact') is True and
            parity.get('actor_query_payload_sha256')==witness_payload['query_payload_sha256'] and
            parity.get('actual_query_count')==len(selected))
        raw_matches_reference=bool(parity and raw_summary and
            parity.get('raw_reference_content_sha256')==raw_summary['content_sha256'])
        raw_ok=bool(parity and parity.get('raw_exact') is True and raw_matches_reference)
        parity_ok = bool(parity and parity.get('status')==expected_status and functional_ok and
            parity.get('equivalence_kind')==args.equivalence_kind and
            parity.get('equivalence_protocol')=='actual_query_bytes_v1' and
            parity.get('equivalence_checker_sha256')==sha256_file(Path(__file__).with_name('compare_cost_event_parity.py')) and
            (args.equivalence_kind=='task_functional' or raw_ok) and
            parity.get('reference_trace_sha256')==sha256_file(trace_path) and
            parity.get('physics_reference_sha256')==physics_reference_sha256 and
            parity.get('physics_reference_kind')==physics_reference_kind and
            parity.get('rate_hz')==args.rate and parity.get('frontend_env')==frontend and
            parity.get('method')==args.method and parity.get('horizon_s')==args.horizon_s and
            parity.get('warmup_s')==args.warmup_s and parity.get('source_meta_sha256')==sha256_file(args.source_meta) and
            parity.get('instrumentation_sha256')=={f.name:sha256_file(f) for f in (
                Path(__file__),Path(__file__).with_name('cost_ledger.py'),Path(__file__).with_name('query_witness.py'))} and
            parity.get('same_host')==socket.gethostname() and worker_start is not None and
            parity.get('same_gpu_uuid')==worker_start['gpu_uuid'])
        try:
            import resource
            own_usage=resource.getrusage(resource.RUSAGE_SELF)
            child_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
            resource_usage=dict(parent_peak_RSS_KiB=int(own_usage.ru_maxrss),
                terminated_children_peak_RSS_KiB=int(child_usage.ru_maxrss),
                parent_user_s=own_usage.ru_utime,parent_system_s=own_usage.ru_stime,
                terminated_children_user_s=child_usage.ru_utime,terminated_children_system_s=child_usage.ru_stime,
                scope='Linux rusage lifetime counters; child maxRSS is a maximum, not summed resident memory')
        except ImportError:
            resource_usage=None
        payload = dict(status=status,error=error,cost_instrumentation_version='v4_actual_query_functional_equivalence',cost_only_shadow=True,utility_result=False,
            cost_measurement=not args.record_physics_reference,
            physics_reference_kind=physics_reference_kind,physics_reference_sha256=physics_reference_sha256,
            physics_reference_path=str(args.physics_reference.resolve()) if args.physics_reference else
                (str(args.output/'PHYSICS_REFERENCE.json') if status==RECORDED else str(trace_path)),
            physical_actions='fixed archived actions; selected frontend/actor actions are computed then discarded',
            reference_run=str(args.reference_run.resolve()),reference_trace_sha256=sha256_file(trace_path),
            source_meta_sha256=sha256_file(args.source_meta),source=str(source),source_code_commit=meta['code_commit'],
            original_policy_sha256=sha256_file(checkpoint),method=args.method,source_rate_hz=args.rate,
            frontend_id=frontend_id,repeat_id=args.repeat_id,reference_seed=reference['seeds'][0],
            frozen_core_files=frozen['core_files'],frozen_core_verified_files=17,
            frozen_sensor_profile_sha256=frozen['profile_sha256'],
            host=socket.gethostname(),worker_start=worker_start,
            parent_python=sys.version,parent_numpy=np.__version__,parent_torch=torch.__version__,
            cpu_affinity=(sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None),
            resource_usage=resource_usage,
            frontend_env=frontend,acquisition_mode=args.acquisition,acquisition_parity_verified=parity_ok,
            equivalence_kind=args.equivalence_kind,functional_query_exact=functional_ok,
            equivalence_protocol='actual_query_bytes_v1',
            raw_event_exact=raw_matches_reference if parity else None,raw_events=raw_summary,
            actor_query_payload_sha256=witness_payload['query_payload_sha256'] if witness_payload else None,
            actor_query_count=witness_payload['query_count'] if witness_payload else len(query_witness.queries),
            actor_query_witness_sha256=sha256_file(witness_path),
            native_parity_receipt_sha256=sha256_file(args.native_parity_receipt) if args.native_parity_receipt else None,
            eligible_as_native_primary=(status=='MEASURED_FIXED_TRAJECTORY' and not args.profile_worker and
                not args.retain_audit_io and args.acquisition=='required' and parity_ok and
                physics_reference_sha256==REGISTERED_PHYSICS_SHA256),
            warmup_simulated_s=args.warmup_s,horizon_simulated_s=args.horizon_s,
            measured_simulated_duration_s=args.horizon_s-args.warmup_s,
            process_wall_including_setup_and_close_s=elapsed,
            process_wall_origin='launcher main entry; interpreter startup excluded; includes instrumentation/source preflight',
            cold_setup_until_hot_s=((ledger.hot_start_ns-process_begin)/1e9 if ledger.hot_start_ns is not None else None),
            hot_rtf=(args.horizon_s-args.warmup_s)/summary['hot_wall_s'] if summary['hot_wall_s'] and not args.record_physics_reference else None,
            audit_hdf_and_demo_output_suppressed=not args.retain_audit_io,
            capture_not_stopped_after_estimated_grasp=True,profiled_worker=args.profile_worker,
            actual_matched_control_steps=0 if args.record_physics_reference else len(checks),
            actual_recorded_control_steps=len(recorded_controls),physics_qpos_qvel_time_checks=checks,
            expected_original_packets=expected_packets,actual_original_packets=len(packets),
            fixed_capture_budget_pass=complete_capture,
            gpu_cumulative_time_s=None,gpu_time_status='unknown; no kernel/activity profiler collected',
            timing=None if args.record_physics_reference else summary,runner_argv=argv,
            instrumentation_sha256={f.name:sha256_file(f) for f in (
                Path(__file__),Path(__file__).with_name('cost_ledger.py'),Path(__file__).with_name('query_witness.py'))})
        (args.output/'COST_RESULT.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        ledger.write(args.output/'COST_SPANS.json')
        print(json.dumps(dict(status=status,method=args.method,rate=args.rate,
            hot_wall_s=None if args.record_physics_reference else summary['hot_wall_s'],
            eligible_as_native_primary=payload['eligible_as_native_primary'],error=error),indent=2))
    if status not in ('MEASURED_FIXED_TRAJECTORY',RECORDED):
        raise SystemExit(1)


if __name__=='__main__':
    main()
