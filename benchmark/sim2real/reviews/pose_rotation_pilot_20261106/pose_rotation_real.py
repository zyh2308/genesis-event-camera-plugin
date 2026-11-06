"""Score frozen center, scale and rotation branches on native E-POSE."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def score_row(scale, sample, baseline, rotation, ray, depth, median, keys, vertices, tree):
    T = np.asarray(sample['target_to_camera'])
    gt_z = float(T[2, :3] @ keys[0] + T[2, 3])
    result = dict(sample=sample['index'], rollout=baseline['sequence'], gt_depth_m=gt_z,
                  adds_m=dict.fromkeys(scale.CASES), predicted_depth_m=dict.fromkeys(scale.CASES))
    truth = vertices @ T[:3, :3].T + T[:3, 3]
    alternatives = [('center_median', median, ray, rotation), ('learned', depth, ray, rotation)]
    if baseline['pnp_valid']:
        R, t = np.asarray(baseline['estimated_R']), np.asarray(baseline['estimated_t_m'])
        original = R @ keys[0] + t
        result['adds_m']['original'] = baseline['adds_m']
        result['predicted_depth_m']['original'] = float(original[2])
        alternatives.append(('median', median, original / original[2], R))
    for name, z, direction, R in alternatives:
        if baseline['pnp_valid']:
            result['predicted_depth_m'][name] = float(z)
        new_t = direction * z - R @ keys[0]
        if ((keys @ R.T + new_t)[:, 2] > 0).all():
            result['adds_m'][name] = float(tree.query((truth - new_t) @ R, workers=1)[0].mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'synthetic', 'scale-run', 'rotation-source', 'scale-source', 'pilot-source',
                 'confirmation', 'confirmation-source', 'authorization', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    recovery = root / 'recovery_20261105_mainline'
    source = recovery / 'stage9_epose_v5.py'
    native = module('native_pose', source)
    scale = module('scale_pilot', args.scale_source)
    rotation = module('rotation_pilot', args.rotation_source)
    pilot = module('depth_pilot', args.pilot_source)
    fresh = module('fresh_gate', args.confirmation_source)
    sha, dump = native.sha, native.dump
    auth = json.loads(args.authorization.read_text())
    completion = args.synthetic / 'completion.json'
    prepared = recovery / 'pose_prepared_full_v5/protocol.json'
    existing = recovery / 'pose_full_real_results_v5'
    registration = existing / 'evaluation_registration.json'
    frozen_path = root / 'recovery_20261103/pose_frozen_checkpoints.json'
    for key, path in [('synthetic', completion), ('protocol', prepared), ('scale', args.scale_source),
                      ('rotation', args.rotation_source), ('scale_completion', args.scale_run/'completion.json'),
                      ('confirmation', args.confirmation), ('confirmation_source', args.confirmation_source),
                      ('pilot', args.pilot_source), ('native', source), ('score', __file__),
                      ('original_registration', registration), ('frozen', frozen_path)]:
        assert sha(path) == auth['hashes'][key]
    assert auth['score_launch_enabled']
    synthetic = json.loads(completion.read_text())
    scale_run = json.loads((args.scale_run/'completion.json').read_text())
    assert synthetic['status'] == 'complete' and not synthetic['smoke'] and not synthetic['real_data_used']
    confirmation = json.loads(args.confirmation.read_text())
    assert confirmation['status'] == 'complete' and not confirmation['smoke'] and not confirmation['real_data_used']
    assert confirmation['gate']['passed'] and fresh.gate(confirmation['results']) == confirmation['gate']
    original = json.loads(registration.read_text())
    assert original['protocol_sha256'] == sha(prepared)
    assert original['frozen_checkpoints_sha256'] == sha(frozen_path)
    frozen = json.loads(frozen_path.read_text())
    protocol = json.loads(prepared.read_text())
    assets, sequences = protocol['assets'], protocol['sequences']
    assert protocol['success_operator'] == 'lt' and len(sequences) == 18
    assert sum(seq['sample_count'] for seq in sequences) == 20916
    for key in ['vertices', 'keypoints']:
        assert sha(assets[key]['path']) == assets[key]['sha256']
    for seq in sequences:
        assert sha(seq['sample_manifest']) == seq['sample_manifest_sha256']
        assert sha(seq['raw_h5']) == seq['raw_h5_sha256']
    vertices = np.load(assets['vertices']['path'], allow_pickle=False)
    keys = np.load(assets['keypoints']['path'], allow_pickle=False)
    tree = cKDTree(vertices)
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    scale.CASES = pilot.CASES = ('original', 'median', 'center_median', 'learned')
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / 'registration.json', dict(auth, authorization_sha256=sha(args.authorization)))
    results = []
    with ExitStack() as stack:
        models = {}
        for run in synthetic['results']:
            method, seed = run['method'], run['seed']
            tag = f'{method}_seed{seed}'
            weights = args.synthetic / (tag + '_rotation.pt')
            assert sha(weights) == run['weights_sha256']
            state = torch.load(weights, map_location='cpu', weights_only=True)
            record = next(row for row in frozen['runs'] if (row['method'], row['seed']) == (method, seed))
            assert (state['method'], state['seed']) == (method, seed)
            scale_record = next(row for row in scale_run['results'] if (row['method'], row['seed']) == (method, seed))
            confirmed = next(row for row in confirmation['results'] if (row['method'], row['seed']) == (method, seed))
            assert confirmed['rotation_sha256'] == run['weights_sha256']
            assert confirmed['scale_sha256'] == scale_record['weights_sha256']
            scale_weights = args.scale_run / (tag + '_scale.pt')
            assert sha(scale_weights) == scale_record['weights_sha256'] == state['scale_sha256']
            scale_state = torch.load(scale_weights, map_location='cpu', weights_only=True)
            assert (scale_state['method'], scale_state['seed']) == (method, seed)
            assert scale_state['base_sha256'] == record['selected_checkpoint_sha256']
            model = scale.CenterScale().cuda().eval()
            model.load_state_dict(scale_state['model'], strict=True)
            head = rotation.RotationHead().cuda().eval()
            head.load_state_dict(state['model'], strict=True)
            old_path = existing / (tag + '_predictions.jsonl')
            assert sha(old_path) == auth['baseline_predictions'][tag]
            baseline = stack.enter_context(old_path.open())
            output = stack.enter_context((args.out / (tag + '.jsonl')).open('x'))
            models[tag] = (model, head, state['median_m'], baseline, output)
        for seq in sequences:
            name, group = seq['sequence'], seq['group']
            manifest = [json.loads(line) for line in Path(seq['sample_manifest']).read_text().splitlines()]
            assert len(manifest) == seq['sample_count']
            lut = native.rectification_lut(np.asarray(manifest[0]['K']), np.asarray(protocol['distortion_k1_k2_p1_p2_k3']))
            scored = {tag: [] for tag in models}
            with h5py.File(seq['raw_h5'], 'r') as stream:
                data = stream[seq['raw_dataset']]
                for k in range(0, len(manifest), 32):
                    block = manifest[k:k+32]
                    images = scale.images([{'image': native.frame_from_raw(data, row, lut)[0]} for row in block])
                    intrinsics = [np.asarray(row['K']) for row in block]
                    for tag, (model, head, median, baseline, output) in models.items():
                        centers, depths, boxes = scale.estimate(model, images, intrinsics)
                        rotations, rays = rotation.estimate_rotation(head, images, centers, boxes, intrinsics)
                        for row, R, ray, depth in zip(block, rotations, rays, depths, strict=True):
                            original_row = json.loads(next(baseline))
                            assert f"{original_row['method']}_seed{original_row['seed']}" == tag
                            assert (original_row['sequence'], original_row['group'], original_row['index']) == (name, group, row['index'])
                            result = score_row(scale, row, original_row, R, ray, depth, median, keys, vertices, tree)
                            scored[tag].append(result)
                            output.write(json.dumps(result, allow_nan=False) + '\n')
                    if k % 512 == 480 or k + len(block) == len(manifest):
                        print('EPOSE_SEQUENCE_PROGRESS', name, k + len(block), len(manifest), flush=True)
            for tag, rows in scored.items():
                assert len(rows) == seq['sample_count']
                summary, _ = pilot.summarize(rows, 0.1 * assets['diameter_m'])
                results.append(dict(tag=tag, sequence=name, group=group, samples=len(rows), summary=summary))
            dump(args.out / (name.replace('/', '__') + '_summary.json'), results[-len(models):])
            print('EPOSE_SEQUENCE_COMPLETE', name, group, seq['sample_count'], flush=True)
        assert all(next(baseline, None) is None for _, _, _, baseline, _ in models.values())
    dump(args.out / 'completion.json', dict(status='complete', results=results, real_feedback_used=False))


if __name__ == '__main__':
    main()
