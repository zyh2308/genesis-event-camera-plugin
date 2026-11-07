"""Measure frozen pose branches on matched, physically attenuated radiance replays."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

import pose_light_robustness as robust
import prepare_pose_radiance_pairs as light


def fingerprint(samples):
    return [(row['id'], *(row[key].tobytes() for key in ('K', 'T', 'keys', 'uv'))) for row in samples]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'radiance', 'robustness', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--clean-parent',type=Path)
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    R = args.campaign
    A = R/'recovery_20261105_mainline'
    D = A/'jobs/119_pose_fresh_confirm_paired_20261106/results'
    scale = robust.load('scale', A/'pose_scale_pilot_20261106.py')
    rotation = robust.load('rotation', A/'pose_rotation_pilot_20261106.py')
    fresh = robust.load('fresh', A/'pose_fresh_confirm_20261106_v3.py')
    sys.path.insert(0, str(R/'sources/benchmark/sim2real/scripts'))
    from run_pose_stage2_pilot import encode_event_window
    source_result = json.loads((D/'completion.json').read_text())
    assert source_result['gate']['passed'] and not source_result['real_data_used']
    result = json.loads((args.robustness/'completion.json').read_text())
    replay = json.loads((args.radiance/'completion.json').read_text())
    assert result['status'] == replay['status'] == 'complete'
    assert result['gate'] is not None and replay['replayed']
    assert not result['real_data_used'] and not replay['real_data_used']
    arms = tuple(result['results'][0]['arms'])
    assert all(tuple(run['arms']) == arms for run in result['results'])
    selection=None
    if args.clean_parent:
        selection=json.loads((args.robustness/'selection.json').read_text())
        assert not selection['real_data_used']
        for run in result['results']:
            tag=f"{run['method']}_seed{run['seed']}"
            path=args.clean_parent/f'{tag}_clean.pt'
            assert light.sha(path)==selection['parent_sha256'][tag]['clean_sha256']
    assets = json.loads((A/'pose_prepared_full_v5/protocol.json').read_text())['assets']
    assert light.sha(assets['vertices']['path']) == assets['vertices']['sha256']
    vertices = np.load(assets['vertices']['path'], allow_pickle=False)
    tree = cKDTree(vertices)
    banks = {'normal': D/'pose_bank', **{name: args.radiance/name/'pose_bank' for name in light.LEVELS}}
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = dict(real_data_used=False, tracks=light.SELECTED, levels={'normal': 1., **light.LEVELS},
                      scope='small held-out radiance intervention, fixed rotations per view',
                      clean_parent=str(args.clean_parent) if args.clean_parent else None,
                      source_sha256=light.sha(__file__))
    light.dump(args.out/'protocol.json', provenance)
    results, identity = [], None
    for method in robust.METHODS:
        for condition, bank in banks.items():
            physical = [bank/'physical'/(rid+'.h5') for rid in light.SELECTED]
            samples = fresh.adapter(bank, physical, method, encode_event_window)
            ident = fingerprint(samples)
            identity = ident if identity is None else identity
            assert ident == identity and len({s['rid'] for s in samples}) == len(light.SELECTED)
            frames = robust.tensors(samples, device)
            gt_size = robust.labels(scale, samples, vertices, device)[:,2].cpu().numpy()
            intrinsics = [s['K'] for s in samples]
            for seed in range(3):
                tag = f'{method}_seed{seed}'
                frozen = next(r for r in source_result['results'] if (r['method'], r['seed']) == (method, seed))
                sp = A/f'jobs/115_pose_scale_paired_20261106/results/{tag}_scale.pt'
                rp = A/f'jobs/117_pose_rotation_paired_20261106/results/{tag}_rotation.pt'
                assert light.sha(sp) == frozen['scale_sha256'] and light.sha(rp) == frozen['rotation_sha256']
                fixed = scale.CenterScale().to(device).eval()
                fixed.load_state_dict(torch.load(sp, map_location='cpu', weights_only=True)['model'], strict=True)
                head = rotation.RotationHead().to(device).eval()
                head.load_state_dict(torch.load(rp, map_location='cpu', weights_only=True)['model'], strict=True)
                center, depth, size = scale.estimate(fixed, frames, intrinsics)
                rotations = robust.frozen_rotation(rotation, head, frames, center, size, intrinsics)
                scores = {}
                hashes = {'original': frozen['scale_sha256'], 'rotation': frozen['rotation_sha256']}
                for arm in arms:
                    if arm != 'original':
                        path = args.robustness/f'{tag}_{arm}.pt'
                        if arm=='clean' and args.clean_parent:
                            path=args.clean_parent/path.name
                        hashes[arm] = light.sha(path)
                        model = scale.CenterScale().to(device).eval()
                        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
                        center, depth, size = scale.estimate(model, frames, intrinsics)
                    scores[arm] = robust.score(rotation, samples, center, depth, size, gt_size, rotations,
                                               vertices, tree, .1*assets['diameter_m'])
                entry = dict(method=method, seed=seed, condition=condition, samples=len(samples), scores=scores,
                             weights_sha256=hashes)
                results.append(entry)
                light.dump(args.out/f'{tag}_{condition}.json', entry)
                print('RADIANCE_SCORE', tag, condition, len(samples), flush=True)
    light.dump(args.out/'completion.json', dict(status='complete', real_data_used=False, results=results))


if __name__ == '__main__':
    main()
