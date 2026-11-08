"""Generate one locked synthetic holdout and score frozen pose candidates."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

N = 96
PHYSICAL_SEED = 20261106017
METHODS = ('eventgenesis', 'v2e')
SEEDS = (0, 1, 2)
CASES = ('original', 'median', 'center_median', 'learned')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)


def call(path, *args, env=None):
    subprocess.run([sys.executable, '-u', str(path), *map(str, args)], env=env, check=True)


def generate(args, count, physical_seed):
    out, root = args.out, args.campaign
    src = out/'sources'
    src.mkdir(parents=True)
    names = ['benchmark', 'pose_physical_worker.py', 'stage9_sensor_profile.json', 'v2e_source_pin.json',
             'replay_new_events.py', 'replay_v2e_parallel.py']
    for name in names:
        original = root/'sources'/name
        (src/name).symlink_to(original, target_is_directory=original.is_dir())
    pins = {n: sha(src/n) for n in names if (src/n).is_file()}
    config = json.loads((root/'sources/pose_bank_protocol.json').read_text())
    config.update(physical_seed=physical_seed, train_rollouts=0, validation_rollouts=count, workers=1)
    dump(src/'pose_bank_protocol.json', config)
    bank = out/'pose_bank'
    (bank/'cache').mkdir(parents=True)
    (bank/'sources').symlink_to(src, target_is_directory=True)
    project = root.parents[2]
    private_lib = project/'runs/sim2real/20261101-epose-diverse-bank-v8/runtime/osmesa/usr/lib/x86_64-linux-gnu'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYOPENGL_PLATFORM='osmesa', GALLIUM_DRIVER='llvmpipe',
               LIBGL_ALWAYS_SOFTWARE='1', MESA_GL_VERSION_OVERRIDE='4.5', LD_LIBRARY_PATH=str(private_lib),
               LP_NUM_THREADS='1', QD_NUM_THREADS='2', QD_OFFLINE_CACHE='1',
               QD_OFFLINE_CACHE_FILE_PATH=str(bank/'cache/quad'), GS_CACHE_FILE_PATH=str(bank/'cache/genesis'))
    print('HOLDOUT_PHASE', 'physical', count, flush=True)
    call(src/'pose_physical_worker.py', '--campaign', bank, '--slot', 0, env=env)
    files = sorted((bank/'physical').glob('*.h5'))
    assert len(files) == count
    physical = []
    for i, path in enumerate(files):
        with h5py.File(path, 'r') as stream:
            g = stream['episodes'][path.stem]
            assert g.attrs['split'] == 'validation' and g.attrs['scene_seed'] == physical_seed + 1009*i
            assert len(g['frame_timestamps_s']) == 21
            assert np.allclose(np.diff(g['frame_timestamps_s'][:]), 0.01, rtol=0, atol=1e-7)
        physical.append(dict(rollout=path.stem, sha256=sha(path)))
    dump(bank/'bank_manifest.json', dict(rollouts=physical, source_pins=pins,
         physical_config_sha256=sha(src/'pose_bank_protocol.json'), real_data_used=False))
    for method, script in zip(METHODS, ['replay_new_events.py', 'replay_v2e_parallel.py'], strict=True):
        print('HOLDOUT_PHASE', method, count, flush=True)
        call(src/script, '--campaign', out, '--kind', 'pose', '--workers', 1)
        receipt = json.loads((bank/'events'/method/'manifest.json').read_text())
        assert receipt['physical_manifest_sha256'] == sha(bank/'bank_manifest.json')
        assert len(receipt['rollouts']) == count and not receipt['real_data_used']
    return bank, files


def adapter(bank, files, method, encode):
    rows = []
    folder = bank/'events'/method
    event_receipt = json.loads((folder/'manifest.json').read_text())
    items = {r['rollout_id']: r for r in event_receipt['rollouts']}
    for path in files:
        rid = path.stem
        raw_path = folder/(rid+'.npy')
        assert sha(raw_path) == items[rid]['events_sha256']
        raw = np.load(raw_path, allow_pickle=False)
        t, x, y, p = raw[:, 0], raw[:, 1].astype(int), raw[:, 2].astype(int), (raw[:, 3] > 0).astype(np.uint8)
        with h5py.File(path, 'r') as stream:
            g = stream['episodes'][rid]
            ts = g['frame_timestamps_s'][:]
            K, keys = g['camera_intrinsics_K_px'][:], g['cad_keypoints'][:]
            for k in range(4, 20):
                T = np.linalg.inv(g['camera_to_world_cv'][k+1]) @ g['target_pose'][k+1]
                xyz = keys @ T[:3, :3].T + T[:3, 3]
                projected = xyz @ K.T
                uv = projected[:, :2] / projected[:, 2:]
                assert np.isfinite(uv).all() and (xyz[:, 2] > 0.03).all()
                cached = np.vstack([g['projected_centroid_uv'][k+1], g['projected_corners_uv'][k+1]])
                np.testing.assert_allclose(uv, cached, atol=1e-7)
                if not ((uv[0] > 0).all() and (uv[0] < [346, 260]).all()):
                    continue
                image = encode(t, x, y, p, ts[k], ts[k+1], 346, 260, 8.0)
                rows.append(dict(image=image, uv=(uv/[346, 260]).astype(np.float32), T=T, K=K, keys=keys,
                                 id=f'{rid}/{k:03d}', rid=rid))
    return rows


def gate(results):
    rng = np.random.default_rng(50855)
    comparisons = []
    for method in METHODS:
        runs = [r for r in results if r['method'] == method]
        assert tuple(r['seed'] for r in runs) == SEEDS
        ids = list(runs[0]['by_rollout'])
        assert len(ids) == N and all(list(r['by_rollout']) == ids for r in runs)
        for baseline in ('original', 'median'):
            for metric, sign in [('depth_mae', -1), ('success', 1)]:
                delta = np.array([[sign * (r['by_rollout'][key]['learned'][metric] -
                                           r['by_rollout'][key][baseline][metric]) for key in ids] for r in runs])
                seeds = delta.mean(1)
                draws = delta.mean(0)[rng.integers(N, size=(4000, N))].mean(1)
                lo, hi = np.quantile(draws, [0.025, 0.975])
                comparisons.append(dict(method=method, baseline=baseline, metric=metric,
                                        seed_improvement=seeds.tolist(), ci=[float(lo), float(hi)],
                                        passed=bool((seeds > 0).all() and lo > 0)))
    return dict(passed=all(r['passed'] for r in comparisons), comparisons=comparisons)


def launch_real(args):
    recovery = args.campaign/'recovery_20261105_mainline'
    original = recovery/'pose_full_real_results_v5'
    paths = dict(synthetic=args.rotation_run/'completion.json', protocol=recovery/'pose_prepared_full_v5/protocol.json',
                 scale=args.scale_source, rotation=args.rotation_source, scale_completion=args.scale_run/'completion.json',
                 confirmation=args.out/'completion.json', confirmation_source=Path(__file__), pilot=args.pilot_source,
                 native=recovery/'stage9_epose_v5.py', score=args.score_source,
                 original_registration=original/'evaluation_registration.json',
                 frozen=args.campaign/'recovery_20261103/pose_frozen_checkpoints.json')
    hashes = {k: sha(v) for k, v in paths.items()}
    tags = [f'{method}_seed{seed}' for method in METHODS for seed in SEEDS]
    authorization = args.out/'epose_authorization.json'
    dump(authorization, dict(score_launch_enabled=True, hashes=hashes,
         baseline_predictions={tag: sha(original/(tag+'_predictions.jsonl')) for tag in tags},
         user_gate='fixed independent synthetic confirmation passed before E-POSE',
         training_or_selection_on_real_data=False))
    print('HOLDOUT_PHASE', 'epose_after_gate', flush=True)
    call(args.score_source, '--campaign', args.campaign, '--synthetic', args.rotation_run, '--scale-run', args.scale_run,
         '--rotation-source', args.rotation_source, '--scale-source', args.scale_source, '--pilot-source', args.pilot_source,
         '--confirmation', paths['confirmation'], '--confirmation-source', paths['confirmation_source'],
         '--authorization', authorization, '--out', args.out/'epose')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'out', 'scale-run', 'rotation-run', 'scale-source', 'rotation-source',
                 'pilot-source', 'helper', 'score-source'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--real-on-pass', action='store_true')
    parser.add_argument('--smoke-receipt', type=Path)
    parser.add_argument('--smoke-completion', type=Path)
    args = parser.parse_args()
    assert not (args.smoke and args.real_on_pass)
    if not args.smoke:
        assert args.smoke_receipt is not None and args.smoke_completion is not None
        smoke_exit = json.loads(args.smoke_receipt.read_text())
        smoke = json.loads(args.smoke_completion.read_text())
        assert smoke_exit['exit_code'] == 0 and smoke['status'] == 'complete' and smoke['smoke']
        assert smoke['physical_seed'] != PHYSICAL_SEED and len(smoke['results']) == len(METHODS)
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)
    count = 2 if args.smoke else N
    physical_seed = PHYSICAL_SEED + 400000 if args.smoke else PHYSICAL_SEED
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out/'protocol.json', dict(smoke=args.smoke, physical_seed=physical_seed, holdout_rollouts=count,
         source_sha256=sha(__file__), source_campaign=str(args.campaign), scope='synthetic only',
         rotation_completion_sha256=sha(args.rotation_run/'completion.json'),
         scale_completion_sha256=sha(args.scale_run/'completion.json'), fixed_weights=True,
         real_launch_requires_synthetic_gate=args.real_on_pass))
    bank, files = generate(args, count, physical_seed)
    pilot = load('pilot', args.pilot_source)
    scale = load('scale', args.scale_source)
    rotation = load('rotation', args.rotation_source)
    real_score = load('shared_pose_metric', args.score_source)
    decode = pilot.helpers(args.helper, args.campaign)['decode']
    root = args.campaign
    scripts = root/'sources/benchmark/sim2real/scripts'
    sys.path.insert(0, str(scripts))
    from run_pose_stage2_pilot import encode_event_window
    models = root.parents[2]/'external/sources/downstream/SingleShotPose'
    sys.path.insert(0, str(models))
    from darknet import Darknet
    assets = json.loads((root/'recovery_20261105_mainline/pose_prepared_full_v5/protocol.json').read_text())['assets']
    assert sha(assets['vertices']['path']) == assets['vertices']['sha256']
    vertices = np.load(assets['vertices']['path'], allow_pickle=False)
    tree = cKDTree(vertices)
    scale_runs = json.loads((args.scale_run/'completion.json').read_text())['results']
    rotation_runs = json.loads((args.rotation_run/'completion.json').read_text())['results']
    scale.CASES = pilot.CASES = CASES
    results = []
    paired_digest = None
    for method in METHODS:
        samples = adapter(bank, files, method, encode_event_window)
        assert len({r['rid'] for r in samples}) == count
        labels = hashlib.sha256()
        for sample in samples:
            labels.update(sample['id'].encode())
            for k in ('T', 'K', 'keys', 'uv'):
                labels.update(sample[k].tobytes())
        paired_digest = paired_digest or labels.hexdigest()
        assert paired_digest == labels.hexdigest()
        images = scale.images(samples)
        intrinsics = [sample['K'] for sample in samples]
        for seed in ((0,) if args.smoke else SEEDS):
            tag = f'{method}_seed{seed}'
            rs = next(r for r in rotation_runs if (r['method'], r['seed']) == (method, seed))
            ss = next(r for r in scale_runs if (r['method'], r['seed']) == (method, seed))
            rp, sp = args.rotation_run/(tag+'_rotation.pt'), args.scale_run/(tag+'_scale.pt')
            assert sha(rp) == rs['weights_sha256'] and sha(sp) == ss['weights_sha256']
            rw, sw = (torch.load(p, map_location='cpu', weights_only=True) for p in (rp, sp))
            assert rw['scale_sha256'] == ss['weights_sha256'] and sha(sw['base_path']) == sw['base_sha256']
            base, net, head = Darknet(str(models/'cfg/yolo-pose.cfg')).cuda().eval(), scale.CenterScale().cuda().eval(), rotation.RotationHead().cuda().eval()
            base.load_state_dict(torch.load(sw['base_path'], map_location='cpu', weights_only=False)['model'], strict=True)
            net.load_state_dict(sw['model'], strict=True)
            head.load_state_dict(rw['model'], strict=True)
            points = scale.original_keypoints(base, samples, decode)
            center, depth, box = scale.estimate(net, images, intrinsics)
            prediction, rays = rotation.estimate_rotation(head, images, center, box, intrinsics)
            rows = []
            for sample, uv, R, ray, z in zip(samples, points, prediction, rays, depth, strict=True):
                pose = pilot.epnp(sample['keys'], uv, sample['K'])
                baseline = dict(sequence=sample['rid'], pnp_valid=pose is not None)
                if pose is not None:
                    original_R, t = pose
                    T = sample['T']
                    truth = vertices @ T[:3, :3].T + T[:3, 3]
                    baseline.update(estimated_R=original_R, estimated_t_m=t,
                                    adds_m=float(tree.query((truth-t) @ original_R, workers=1)[0].mean()))
                row = real_score.score_row(scale, dict(index=sample['id'], target_to_camera=sample['T']),
                                           baseline, R, ray, z, rw['median_m'], sample['keys'], vertices, tree)
                rows.append(row)
            summary, by_rollout = pilot.summarize(rows, 0.1*assets['diameter_m'])
            run = dict(method=method, seed=seed, summary=summary, by_rollout=by_rollout,
                       rotation_sha256=rs['weights_sha256'], scale_sha256=ss['weights_sha256'])
            dump(args.out/(tag+'.json'), dict(run, rows=rows))
            results.append(run)
            print('HOLDOUT_CASE_COMPLETE', method, seed, len(rows), flush=True)
            del net, head, base
        del samples, images
    decision = None if args.smoke else gate(results)
    dump(args.out/'completion.json', dict(status='complete', smoke=args.smoke, real_data_used=False,
         physical_seed=physical_seed, label_sha256=paired_digest, results=results,
         gate=decision))
    if args.real_on_pass and decision['passed']:
        launch_real(args)


if __name__ == '__main__':
    main()
