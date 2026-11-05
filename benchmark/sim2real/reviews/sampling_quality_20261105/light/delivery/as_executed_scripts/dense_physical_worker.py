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
sys.dont_write_bytecode=True
SOURCE=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/sources')
sys.path[:0]=[str(P/'runtime/sim2real/genesis_hdr_v2'),str(CORE),str(SOURCE),str(SOURCE/'benchmark/sim2real/scripts')]
import numpy as np
import h5py
from genesis.ext.pyrender.platforms.osmesa import OSMesaPlatform
OSMesaPlatform.supports_framebuffers=lambda self:True
import stage9_replicad_scene_loader as scene_impl
from genesis_event_plugin.genesis_adapter import camera_to_world_cv,link_to_world_cv
from generate_stage4_flow_bank import radiance_to_v2e_luma_u8

cfg=json.loads((SOURCE/'rich_bank_protocol.json').read_text())
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

GEOMETRY_QUALITY_SPEC = {'schema': 'sampling_quality_stage9_dense_render_spec_v1', 'slot': 0, 'stratum': 3, 'start_tick_1ms': 1000, 'duration_ticks_1ms': 100, 'baseline_bank_file': '/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/rich_bank/physical/s3_00.h5', 'source_dir': '/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1/sources', 'source_sha256': {'rich_physical_worker.py': '0eafc57043eb2a4b4e963fbb6ccf4e58d4286df12d3f1d20856637f839097d17', 'stage9_replicad_scene_loader.py': '8aa173ff3bc0d91c7e432fb5edbd1414338f805f5d2f71865c6b94068adef5dd', 'rich_bank_protocol.json': 'e3fd835346e422deedeeb77d2d110fc27bf0d6e568708ec7c690f7473bbb13ff'}, 'source_capture_replaced_only_in_private_worker': True, 'unchanged_trajectory_equations': True, 'genesis_physics_dt_s': 0.001, 'actual_subframe_render_dt_s': 0.001, 'noise_or_event_generator_called': False}

def capture(scene, camera, output_dir, translation_h, object_delta_h,
            moving_entity, moving_link_id, links_by_id, moving_base_g):
    """Stage9's exact scripted1ms trajectory, with additional real renders.

    This function is transplanted only into a new isolated worker by the builder.
    It does not call any event interpolator or change Genesis's1ms physics dt.
    """
    import hashlib
    import json
    import random
    import time
    from pathlib import Path
    import h5py
    import numpy as np
    import torch

    spec = GEOMETRY_QUALITY_SPEC
    assert args.slot == spec['slot']
    stratum = spec['stratum']
    rid = f's{stratum}_{args.slot:02d}'
    destination = output/(rid+'_dense_1ms.h5')
    assert not destination.exists()
    partial = destination.with_suffix('.h5.partial')
    assert not partial.exists()

    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024**2), b''):
                digest.update(block)
        return digest.hexdigest()

    def rawsha(value):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    def rng_snapshot():
        return dict(python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                    numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
                    torch_cpu=rawsha(torch.get_rng_state().numpy()))

    base = camera_to_world_cv(camera)
    eye0 = base[:3, 3].copy()
    forward, right, down = base[:3, 2].copy(), base[:3, 0].copy(), base[:3, 1].copy()
    K = np.asarray(camera.intrinsics, dtype=np.float64).copy()
    K[0, 2] -= .5
    K[1, 2] -= .5
    ids = np.array(sorted(links_by_id), dtype=np.int32)
    # These are the original Stage9 worker's trajectory equations and RNG.
    rng = np.random.default_rng(cfg['physical_seed']+stratum*1009+args.slot)
    scale = cfg['motion_scales_m'][args.slot%3]
    frequency = cfg['frequencies_hz'][(args.slot//2+stratum)%3]
    phase = rng.uniform(-np.pi, np.pi, 3)
    camera_amp = np.array([scale, .4*scale, .2*scale])
    object_amp = np.array([.06, .04, .02])*(1+.5*(args.slot%3))
    if stratum == 0:
        object_amp *= 0
    if stratum == 1:
        camera_amp *= 0
    moving_entity.set_pos(tuple(moving_base_g))
    camera.set_pose(pos=tuple(eye0), lookat=tuple(eye0+forward))
    origin = float(scene.cur_t)
    first_tick = spec['start_tick_1ms']
    final_tick = first_tick+spec['duration_ticks_1ms']
    nframes = spec['duration_ticks_1ms']+1
    old_path = Path(spec['baseline_bank_file'])
    original_hash = sha(old_path)
    times, prescribed_times, cams, poses, repeat_checks = [], [], [], [], []
    begun = time.perf_counter()

    def render_actual():
        rad, _, _, _ = camera.render(rgb=True, depth=False, segmentation=False,
                                     radiance=True, force_render=True)
        _, dep, raw, _ = camera.render(rgb=False, depth=True, segmentation=True, force_render=True)
        rad, dep = np.asarray(rad, dtype=np.float32), np.asarray(dep, dtype=np.float32)
        seg = scene_impl.remap_segmentation_to_links(
            np.asarray(raw), scene_impl.renderer_segmentation_link_map(camera)).astype(np.int32)
        assert rad.shape == (260, 346, 3) and dep.shape == seg.shape == (260, 346)
        assert np.isfinite(rad).all() and (rad >= 0).all() and np.isfinite(dep).all()
        return dict(radiance_linear_rgb=rad, depth_m=dep, segmentation_link_idx=seg)

    rng_before = rng_snapshot()
    with h5py.File(partial, 'x') as h:
        g = h.create_group('episodes').create_group(rid)
        h.attrs.update(schema='sampling_quality_actual_genesis_dense_v1', real_data_used=False,
                       physics_dt_s=.001, render_dt_s=.001, event_generator_called=False,
                       reference_role='withheld real Genesis renders; evaluation only')
        g.attrs.update(scene_id=record['scene_config'], stratum=cfg['strata'][stratum],
                       layout_slot=args.slot, physical_trajectory_kind='same Stage9 scripted poses, same1ms Genesis stepping',
                       source_worker_sha256=spec['source_sha256']['rich_physical_worker.py'],
                       source_loader_sha256=spec['source_sha256']['stage9_replicad_scene_loader.py'],
                       source_protocol_sha256=spec['source_sha256']['rich_bank_protocol.json'],
                       motion_parameters=json.dumps(dict(scale_m=scale, frequency_hz=frequency,
                           phase=phase.tolist(), camera_amplitude=camera_amp.tolist(), object_amplitude=object_amp.tolist())),
                       camera_selection=json.dumps(camera_choice))
        g.create_dataset('camera_intrinsics_K_px', data=K)
        g.create_dataset('object_segmentation_link_ids', data=ids)
        for tick in range(final_tick+1):
            if tick:
                t = tick*.001
                wave = np.sin(2*np.pi*frequency*t+phase)-np.sin(phase)
                delta = camera_amp*wave
                eye = eye0+right*delta[0]+down*delta[1]+forward*delta[2]
                yaw = .04*np.sin(2*np.pi*.6*t) if stratum >= 2 else 0.
                direction = np.cos(yaw)*forward+np.sin(yaw)*right
                camera.set_pose(pos=tuple(eye), lookat=tuple(eye+direction))
                moving_entity.set_pos(tuple(np.asarray(moving_base_g)+object_amp*wave))
                scene.step()
            if tick < first_tick:
                continue
            k = tick-first_tick
            fields = render_actual()
            if k in {0, nframes//2, nframes-1}:
                before_repeat = rng_snapshot()
                repeated = render_actual()
                after_repeat = rng_snapshot()
                repeat_checks.append(dict(index=k, prescribed_t_s=tick*.001,
                    raw_bytes_equal={name: value.tobytes() == repeated[name].tobytes() for name,value in fields.items()},
                    rng_unchanged=before_repeat == after_repeat))
            for name, value in fields.items():
                if name not in g:
                    g.create_dataset(name, shape=(nframes,)+value.shape, dtype=value.dtype,
                                     chunks=(1,)+value.shape, compression='gzip', compression_opts=1)
                g[name][k] = value
            times.append(float(scene.cur_t)-origin)
            prescribed_times.append(tick*.001)
            cams.append(camera_to_world_cv(camera))
            poses.append(np.stack([link_to_world_cv(links_by_id[int(sid)]) for sid in ids]))
            if k%10 == 0:
                print(json.dumps(dict(phase='actual_genesis_dense_render', case=rid,
                    dense_frame=k, total_frames=nframes, wall_s=time.perf_counter()-begun)), flush=True)
        times = np.asarray(times, dtype=np.float64)
        prescribed_times = np.asarray(prescribed_times, dtype=np.float64)
        assert len(times) == nframes and np.allclose(times, prescribed_times, rtol=0, atol=1e-9)
        g.create_dataset('frame_timestamps_s', data=times)
        g.create_dataset('prescribed_trajectory_timestamps_s', data=prescribed_times)
        g.create_dataset('camera_to_world_cv', data=np.stack(cams))
        g.create_dataset('object_to_world', data=np.stack(poses))
    partial.rename(destination)
    rng_after = rng_snapshot()
    checks = []
    with h5py.File(destination, 'r') as dense, h5py.File(old_path, 'r') as old:
        dg, og = dense['episodes'][rid], old['episodes'][rid]
        assert str(og.attrs['scene_id']) == record['scene_config']
        assert np.array_equal(ids, np.asarray(og['object_segmentation_link_ids']))
        assert np.array_equal(K, np.asarray(og['camera_intrinsics_K_px']))
        for di in range(0, nframes, 10):
            oi = (first_tick+di)//10
            row = dict(dense_index=di, original_index=oi,
                       timestamp_delta_s=float(times[di]-og['frame_timestamps_s'][oi]), fields={})
            for name in ('radiance_linear_rgb', 'depth_m', 'segmentation_link_idx',
                         'camera_to_world_cv', 'object_to_world'):
                x, y = np.asarray(dg[name][di]), np.asarray(og[name][oi])
                row['fields'][name] = dict(raw_bytes_equal=x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(),
                    max_abs=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))),
                    changed_values=int(np.count_nonzero(x != y)))
            checks.append(row)
    passed = all(all(item['raw_bytes_equal'] for item in row['fields'].values())
                 and abs(row['timestamp_delta_s']) <= 1e-9 for row in checks)
    repeats_passed = all(all(row['raw_bytes_equal'].values()) and row['rng_unchanged'] for row in repeat_checks)
    assert sha(old_path) == original_hash, 'Original physical bank changed during read-only replay'
    receipt = dict(status='complete', case=rid, output=str(destination), output_sha256=sha(destination),
        specification=spec, source_10ms_bank=str(old_path), source_10ms_bank_sha256=original_hash,
        actual_genesis_render=True, endpoint_interpolation_used_for_reference=False,
        renderer_backend=str(scene_impl.gs.backend), genesis_version=scene_impl.gs.__version__,
        physics_dt_s=.001, render_dt_s=.001, frames=nframes,
        coarse_endpoint_checks=checks, coarse_endpoint_replay_passed=passed,
        repeated_actual_render_checks=repeat_checks, repeated_actual_render_passed=repeats_passed,
        render_global_rng_before=rng_before, render_global_rng_after=rng_after,
        render_global_rng_unchanged=rng_before == rng_after,
        reference_temporal_convergence_established=False,
        event_timestamp_20us_accuracy_established=False,
        limitation='1ms geometry/occlusion reference only; not a microsecond event oracle or proof of convergence',
        wall_s=time.perf_counter()-begun, event_generator_called=False, real_data_used=False)
    receipt_path = R/(rid+'_dense_reference.json')
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print('DENSE_REFERENCE_COMPLETE', json.dumps(dict(case=rid, replay_passed=passed,
        repeated_render_passed=repeats_passed, reference=str(destination), receipt=str(receipt_path))), flush=True)
    assert passed, 'Source10ms replay differs; preserve evidence and investigate before scoring'
    assert repeats_passed, 'Repeated renderer output/RNG differs; reference not accepted'
    return receipt


scene_impl._s9_choose_camera=choose_camera
scene_impl._s9_find_urdf=find_urdf
scene_impl.capture_dynamic_motion_probe=capture
sys.argv=[str(Path(scene_impl.__file__)),'--asset-root',str(asset),'--scene-config',record['scene_config'],'--output-dir',str(setup),'--moving-object-template',moving]
scene_impl.main()
metadata=setup/'result.json'
if metadata.exists():metadata.rename(setup/'legacy_scene_setup_metadata.json')
print('PHYSICAL_LAYOUT_COMPLETE',args.slot,flush=True)
