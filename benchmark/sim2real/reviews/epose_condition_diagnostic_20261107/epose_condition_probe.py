"""Decompose the frozen EventGenesis/V2E E-POSE gap without fitting on real data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F

METHODS = ('eventgenesis', 'v2e')
SEEDS = (0, 1, 2)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


@torch.inference_mode()
def infer(scale, rotation, nets, frames, K):
    x = F.interpolate(torch.from_numpy(np.stack(frames)), size=(scale.SIZE, scale.SIZE), mode='area')
    centers, depths, sizes = scale.estimate(nets[0], x, K)
    rays, bases = rotation.ray_frame(centers, K)
    intrinsics = np.asarray(K)
    aspect = np.sqrt(intrinsics[:, 0, 0] * 260 / (intrinsics[:, 1, 1] * 346))
    patch, _ = rotation.crop(x, torch.as_tensor(centers), torch.as_tensor(sizes),
                             torch.as_tensor(aspect, dtype=torch.float32), torch.zeros(len(x)))
    rotations = bases @ nets[1](patch).numpy()
    assert np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))) < 1e-4
    return centers.astype(float), depths.astype(float), sizes.astype(float), rotations, rays


def gt(sample, keys, vertices):
    T, K = np.asarray(sample['target_to_camera']), np.asarray(sample['K'])
    truth = vertices @ T[:3, :3].T + T[:3, 3]
    center = T[:3, :3] @ keys[0] + T[:3, 3]
    assert center[2] > 0 and (truth[:, 2] > 0).all()
    pixels = truth @ K.T
    uv = pixels[:, :2] / pixels[:, 2:] / [346, 260]
    box = np.stack((uv.min(0), uv.max(0)))
    U, _, Vt = np.linalg.svd(T[:3, :3])
    polar = U @ Vt
    assert np.linalg.det(polar) > 0
    return center, box, polar, truth


def add(tree, keys, truth, R, center):
    t = center - R @ keys[0]
    if not ((keys @ R.T + t)[:, 2] > 0).all():
        return None
    return float(tree.query((truth - t) @ R, workers=1)[0].mean())


def metrics(pred, sample, target, keys, tree, expected):
    uv, depth, size, R, ray = pred
    center, box, polar, truth = target
    z = float(center[2])
    true_ray = center / z
    gt_box = float(np.sqrt(np.prod(box[1] - box[0])))
    gt_center = np.asarray(sample['gt_uv_ideal'])[0]
    full = add(tree, keys, truth, R, ray * depth)
    expected_score = expected['adds_m']['learned']
    assert (full is None) == (expected_score is None)
    delta = abs(full - expected_score) if full is not None else 0.0
    expected_z = expected['predicted_depth_m']['learned']
    depth_delta = abs(depth - expected_z) if expected_z is not None else 0.0
    assert delta < 0.0003 and depth_delta < 0.0003, (delta, depth_delta, sample['index'])
    return dict(signed_z_m=float(depth - z), center_px=float(np.linalg.norm(uv*[346, 260] - gt_center)),
                lateral_m=float(np.linalg.norm((ray - true_ray) * z)),
                log_scale=float(np.log(size / gt_box)), log_shape=float(np.log(depth / z * size / gt_box)),
                log_z=float(np.log(depth / z)), full_adds_m=expected_score,
                exact_z=add(tree, keys, truth, R, ray*z), exact_center=add(tree, keys, truth, R, true_ray*depth),
                exact_rotation=add(tree, keys, truth, polar, ray*depth),
                rotation_only=add(tree, keys, truth, R, center), parity_adds_m=delta, parity_depth_m=depth_delta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    A = args.campaign/'recovery_20261105_mainline'
    D = A/'jobs/119_pose_fresh_confirm_paired_20261106/results'
    scale_path = A/'pose_scale_pilot_20261106.py'
    rotation_path = A/'pose_rotation_pilot_20261106.py'
    native_path = A/'stage9_epose_v5.py'
    protocol_path = A/'pose_prepared_full_v5/protocol.json'
    registration = json.loads((D/'epose/registration.json').read_text())
    for name, path in [('scale', scale_path), ('rotation', rotation_path), ('native', native_path), ('protocol', protocol_path)]:
        assert digest(path) == registration['hashes'][name]
    scale, rotation, native = (load(name, path) for name, path in
                               [('scale', scale_path), ('rotation', rotation_path), ('native', native_path)])
    protocol = json.loads(protocol_path.read_text())
    assets = protocol['assets']
    for name in ['vertices', 'keypoints']:
        assert digest(assets[name]['path']) == assets[name]['sha256']
    vertices = np.load(assets['vertices']['path'], allow_pickle=False)
    keys = np.load(assets['keypoints']['path'], allow_pickle=False)
    tree = cKDTree(vertices)
    fresh = json.loads((D/'completion.json').read_text())
    assert fresh['gate']['passed'] and digest(D/'completion.json') == registration['hashes']['confirmation']
    sroot = A/'jobs/115_pose_scale_paired_20261106/results'
    rroot = A/'jobs/117_pose_rotation_paired_20261106/results'
    models, archived = {}, {}
    for row in fresh['results']:
        tag = (row['method'], row['seed'])
        name = f'{tag[0]}_seed{tag[1]}'
        sp, rp = sroot/(name+'_scale.pt'), rroot/(name+'_rotation.pt')
        assert digest(sp) == row['scale_sha256'] and digest(rp) == row['rotation_sha256']
        s, r = (torch.load(path, map_location='cpu', weights_only=True) for path in [sp, rp])
        assert (s['method'], s['seed']) == (r['method'], r['seed']) == tag
        model, head = scale.CenterScale().eval(), rotation.RotationHead().eval()
        model.load_state_dict(s['model'], strict=True)
        head.load_state_dict(r['model'], strict=True)
        models[tag] = (model, head)
        archived[tag] = {}
        with (D/'epose'/(name+'.jsonl')).open() as stream:
            for line in stream:
                row = json.loads(line)
                archived[tag][(row['rollout'], row['sample'])] = row
        assert len(archived[tag]) == 20916
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/'protocol.json', dict(synthetic_only_training=True, real_fit=False, diagnostic_only=True,
          samples_per_sequence=4 if args.smoke else 64, sample_policy='uniform ordinal positions within each frozen sequence',
          threshold_m=.1*assets['diameter_m'], seed_count=3, source_sha256=digest(__file__),
          rotations='full CAD centered ADD-S; oracle uses polar rotation while native pose preserved for scores'))
    sequences = protocol['sequences'][:1] if args.smoke else protocol['sequences']
    for seq in sequences:
        name = seq['sequence']
        manifest, raw = Path(seq['sample_manifest']), Path(seq['raw_h5'])
        assert digest(manifest) == seq['sample_manifest_sha256'] and digest(raw) == seq['raw_h5_sha256']
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        assert len(rows) == seq['sample_count']
        chosen = [rows[i] for i in np.linspace(0, len(rows)-1, 4 if args.smoke else 64, dtype=int)]
        K = [np.asarray(row['K']) for row in chosen]
        lut = native.rectification_lut(K[0], np.asarray(protocol['distortion_k1_k2_p1_p2_k3']))
        output = []
        with h5py.File(raw, 'r') as stream:
            ds = stream[seq['raw_dataset']]
            for off in range(0, len(chosen), 16):
                block = chosen[off:off+16]
                frames = [native.frame_from_raw(ds, row, lut)[0] for row in block]
                predicts = {k: list(zip(*infer(scale, rotation, net, frames, K[off:off+16]), strict=True))
                            for k, net in models.items()}
                for i, row in enumerate(block):
                    target = gt(row, keys, vertices)
                    record = dict(index=row['index'], event_count=row['raw_event_count'], depth_m=float(target[0][2]),
                                  models={}, swaps={})
                    for seed in SEEDS:
                        values = [predicts[m, seed][i] for m in METHODS]
                        for method, pred in zip(METHODS, values, strict=True):
                            tag = f'{method}_seed{seed}'
                            expected = archived[method, seed][(name, row['index'])]
                            record['models'][tag] = metrics(pred, row, target, keys, tree, expected)
                        mixed = []
                        for mask in range(8):
                            if mask in (0, 7):
                                tag = f'{METHODS[mask != 0]}_seed{seed}'
                                mixed.append(record['models'][tag]['full_adds_m'])
                            else:
                                depth = values[(mask >> 0) & 1][1]
                                ray = values[(mask >> 1) & 1][4]
                                R = values[(mask >> 2) & 1][3]
                                mixed.append(add(tree, keys, target[3], R, ray*depth))
                        record['swaps'][str(seed)] = mixed
                    output.append(record)
        write(args.out/(name.replace('/', '__')+'.json'), dict(sequence=name, group=seq['group'], rows=output))
        print('CONDITION_COMPLETE', name, len(output), flush=True)
    write(args.out/'completion.json', dict(status='complete', count=len(sequences), real_fit=False, diagnostic_only=True))


if __name__ == '__main__':
    main()
