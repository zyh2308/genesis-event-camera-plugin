"""Estimate pose translation from event-only center and apparent scale."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

METHODS = ('eventgenesis', 'v2e')
SEEDS = (0, 1, 2)
EPOCHS = 24
BATCH = 32
SIZE = 128
CASES = ('original', 'median', 'previous', 'depth_only', 'center_median', 'learned')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CenterScale(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        width = 3
        for channels in (24, 40, 64):
            layers.extend([nn.Conv2d(width, channels, 5, stride=2, padding=2, bias=False),
                           nn.GroupNorm(4, channels), nn.SiLU(),
                           nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                           nn.GroupNorm(4, channels), nn.SiLU()])
            width = channels
        self.features = nn.Sequential(*layers)
        self.center = nn.Conv2d(width, 1, 1)
        self.size = nn.Sequential(nn.Linear(2 * width, 64), nn.SiLU(), nn.Linear(64, 2))
        coords = (torch.arange(SIZE // 8) + 0.5) / (SIZE // 8)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('grid', torch.stack((xx, yy), dim=-1).reshape(-1, 2))

    def forward(self, images):
        features = self.features(images).flatten(2)
        logits = self.center(features.reshape(len(images), -1, SIZE // 8, SIZE // 8)).flatten(1)
        attention = logits.softmax(dim=1)
        uv = attention @ self.grid
        local = (features * attention[:, None]).sum(dim=2)
        context = features.mean(dim=2)
        size, shape = self.size(torch.cat((local, context), dim=1)).unbind(dim=1)
        return logits, uv, size - 2.3, shape - 2.3


def focal(K):
    return float(0.5 * np.log(K[0, 0] * K[1, 1] / (346 * 260)))


def targets(rows, vertices, depth_target):
    values = []
    for row in rows:
        camera = vertices @ row['T'][:3, :3].T + row['T'][:3, 3]
        assert (camera[:, 2] > 0).all()
        pixels = camera @ row['K'].T
        uv = pixels[:, :2] / pixels[:, 2:] / [346, 260]
        box_size = np.ptp(uv, axis=0)
        log_size = float(np.log(box_size).mean())
        log_z = float(np.log(depth_target(row)))
        log_f = focal(row['K'])
        values.append([*row['uv'][0], log_size, log_f, log_z])
    array = np.asarray(values, dtype=np.float32)
    assert np.isfinite(array).all()
    return torch.from_numpy(array).cuda()


def images(rows):
    return torch.cat([F.interpolate(torch.from_numpy(np.stack([row['image'] for row in rows[k:k+32]])).cuda(),
                                    size=(SIZE, SIZE), mode='area') for k in range(0, len(rows), 32)])


def augment(inputs, labels, rng):
    B = len(inputs)
    transform = torch.from_numpy(rng.uniform(-1, 1, (B, 3)).astype(np.float32)).cuda()
    zoom = 1 + 0.15 * transform[:, 0]
    shift = 0.06 * transform[:, 1:]
    theta = torch.zeros((B, 2, 3), device=inputs.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / zoom
    theta[:, :, 2] = -2 * shift / zoom[:, None]
    grid = F.affine_grid(theta, inputs.shape, align_corners=False)
    inputs = F.grid_sample(inputs, grid, align_corners=False)
    result = labels.clone()
    result[:, :2] = (labels[:, :2] - 0.5) * zoom[:, None] + 0.5 + shift
    result[:, 2:4] += zoom.log()[:, None]
    return inputs, result


def fit(train_x, labels, seed, smoke):
    torch.manual_seed(62330 + seed)
    model = CenterScale()
    initial = hashlib.sha256(b''.join(t.numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = np.random.default_rng(19731 + seed)
    history = []
    for _ in range(1 if smoke else EPOCHS):
        losses = []
        indices = rng.permutation(len(train_x))
        if smoke:
            indices = indices[:BATCH]
        for start in range(0, len(indices), BATCH):
            idx = torch.from_numpy(indices[start:start+BATCH]).cuda()
            x, y = augment(train_x[idx], labels[idx], rng)
            logits, uv, log_size, log_shape = model(x)
            dist = (model.grid[None] - y[:, None, :2]).square().sum(-1)
            heatmap = (-0.5 * dist * (SIZE // 8) ** 2).softmax(1)
            center_loss = -(heatmap * logits.log_softmax(1)).sum(1).mean()
            expected_shape = y[:, 4] - y[:, 3] + y[:, 2]
            depth = log_shape + y[:, 3] - log_size
            smooth = lambda a, b: F.smooth_l1_loss(a, b, beta=0.1)
            loss = (center_loss + 4 * smooth(uv, y[:, :2]) + 2 * smooth(log_size, y[:, 2])
                    + smooth(log_shape, expected_shape) + 2 * smooth(depth, y[:, 4]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            assert torch.isfinite(loss) and norm > 0
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model.eval(), initial, history


@torch.inference_mode()
def estimate(model, inputs, intrinsics):
    center, depth, scale = [], [], []
    for k in range(0, len(inputs), BATCH):
        _, uv, log_size, log_shape = model(inputs[k:k+BATCH])
        log_f = torch.tensor([focal(K) for K in intrinsics[k:k+BATCH]], device=inputs.device)
        center.extend(uv.cpu().numpy())
        depth.extend((log_shape + log_f - log_size).exp().cpu().numpy())
        scale.extend(log_size.exp().cpu().numpy())
    assert np.isfinite(center).all() and np.isfinite(depth).all() and (np.asarray(depth) > 0).all()
    return np.asarray(center), np.asarray(depth), np.asarray(scale)


@torch.inference_mode()
def original_keypoints(model, rows, decode):
    output = []
    for k in range(0, len(rows), 16):
        inputs = torch.from_numpy(np.stack([row['image'] for row in rows[k:k+16]])).cuda()
        coords, _ = decode(model(inputs))
        output.extend(coords)
    return np.asarray(output) * [346, 260]


def geometry(pilot, keys, K, corners, center):
    base = pilot.epnp(keys, corners, K)
    if base is None:
        return None
    R, t = base
    offset = R @ keys[0]
    predicted = offset + t
    base_ray = predicted / predicted[2]
    ray = np.linalg.solve(K, [center[0] * 346, center[1] * 260, 1.0])
    return R, offset, base_ray, ray, predicted[2]


def score(pilot, rows, keypoints, center, depth, scale, prior, median, vertices, tree):
    output = []
    for sample, uv, xy, z, box, old in zip(rows, keypoints, center, depth, scale, prior, strict=True):
        assert sample['id'] == old['sample'] and sample['rid'] == old['rollout']
        assert abs(pilot.depth_target(sample) - old['gt_depth_m']) < 1e-10
        keys = sample['keys']
        geo = geometry(pilot, keys, sample['K'], uv, xy)
        additions = dict(depth_only=(z, None), center_median=(median, None), learned=(z, None))
        errors = {k: old['adds_m'][k] for k in ('original', 'median')}
        depths = {k: old['predicted_depth_m'][k] for k in ('original', 'median')}
        errors['previous'], depths['previous'] = old['adds_m']['learned'], old['predicted_depth_m']['learned']
        if geo is not None:
            R, offset, base_ray, ray, base_depth = geo
            additions = dict(depth_only=(z, base_ray), center_median=(median, ray), learned=(z, ray))
            assert abs(base_depth - depths['original']) < 0.01
            truth = vertices @ sample['T'][:3, :3].T + sample['T'][:3, 3]
        for case, (value, ray) in additions.items():
            depths[case] = float(value) if geo is not None else None
            errors[case] = None
            if ray is not None:
                t = ray * value - offset
                if ((keys @ R.T + t)[:, 2] > 0).all():
                    errors[case] = float(tree.query((truth - t) @ R, workers=1)[0].mean())
        output.append(dict(sample=sample['id'], rollout=sample['rid'], gt_depth_m=old['gt_depth_m'],
                           predicted_depth_m=depths, adds_m=errors, predicted_center=xy.tolist(),
                           center_error_px=float(np.linalg.norm((xy - sample['uv'][0]) * [346, 260])),
                           scale=float(box)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'prior', 'pilot-source', 'helper', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)
    pilot = load_module('depth_pilot', args.pilot_source)
    h = pilot.helpers(args.helper, args.campaign)
    sha, dump = h['sha'], h['dump']
    root = args.campaign
    source = root.parents[2] / 'external/sources/downstream/SingleShotPose'
    sys.path.insert(0, str(source))
    from darknet import Darknet
    asset = json.loads((root / 'recovery_20261105_mainline/pose_prepared_full_v5/protocol.json').read_text())['assets']
    assert sha(asset['vertices']['path']) == asset['vertices']['sha256']
    vertices = np.load(asset['vertices']['path'], allow_pickle=False)
    tree = cKDTree(vertices)
    threshold = 0.1 * asset['diameter_m']
    reference = json.loads((args.prior / 'completion.json').read_text())
    assert reference['status'] == 'complete' and not reference['real_data_used']
    frozen = json.loads((root / 'recovery_20261103/pose_frozen_checkpoints.json').read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / 'protocol.json', dict(real_data_used=False, smoke=args.smoke, primary='learned', cases=CASES,
         epochs=EPOCHS, batch=BATCH, threshold_m=threshold, seed=SEEDS, prior_sha256=sha(args.prior/'completion.json'),
         source_sha256=sha(__file__), old_source_sha256=sha(args.pilot_source), helper_sha256=sha(args.helper)))
    results, starts = [], {}
    for method in METHODS:
        train, val = (h['load'](method, split) for split in ('train', 'validation'))
        for split, rows, count in [('train', train, 1529), ('validation', val, 510)]:
            digest = hashlib.sha256()
            for row in rows:
                digest.update(row['id'].encode() + row['rid'].encode())
                for key in ('keys', 'K', 'T', 'uv'):
                    digest.update(row[key].tobytes())
            assert len(rows) == count and digest.hexdigest() == reference['label_hashes'][split]
        if args.smoke:
            train, val = train[:64], val[:16]
        train_x, val_x = images(train), images(val)
        labels = targets(train, vertices, pilot.depth_target)
        del train
        for seed in ((0,) if args.smoke else SEEDS):
            tag = f'{method}_seed{seed}'
            ref = next(r for r in reference['results'] if (r['method'], r['seed']) == (method, seed))
            prior = json.loads((args.prior / (tag + '.json')).read_text())
            assert prior['summary'] == ref['summary'] and prior['base_sha256'] == ref['base_sha256']
            record = next(r for r in frozen['runs'] if (r['method'], r['seed']) == (method, seed))
            path = Path(record['selected_checkpoint'])
            assert sha(path) == ref['base_sha256'] == record['selected_checkpoint_sha256']
            payload = torch.load(path, map_location='cpu', weights_only=False)
            base = Darknet(str(source/'cfg/yolo-pose.cfg')).cuda()
            base.load_state_dict(payload['model'], strict=True)
            del payload
            keypoints = original_keypoints(base.eval(), val, h['decode'])
            del base
            model, initial, history = fit(train_x, labels, seed, args.smoke)
            assert starts.setdefault(seed, initial) == initial
            center, depth, scale = estimate(model, val_x, [row['K'] for row in val])
            rows = score(pilot, val, keypoints, center, depth, scale, prior['rows'][:len(val)],
                         ref['median_m'], vertices, tree)
            pilot.CASES = CASES
            summary, by_rollout = pilot.summarize(rows, threshold)
            if not args.smoke:
                for case in ('original', 'median'):
                    assert summary[case] == ref['summary'][case]
            weights = args.out / (tag + '_scale.pt')
            torch.save(dict(model=model.state_dict(), method=method, seed=seed, base_path=str(path),
                            base_sha256=ref['base_sha256'], median_m=ref['median_m']), weights)
            run = dict(method=method, seed=seed, summary=summary, by_rollout=by_rollout, train_loss=history,
                       weights_sha256=sha(weights), initial_sha256=initial,
                       center_median_error_px=float(np.median([row['center_error_px'] for row in rows])))
            dump(args.out / (tag + '.json'), dict(run, rows=rows))
            print('SCALE_RESULT', json.dumps({k: v for k, v in run.items() if k != 'by_rollout'}), flush=True)
            results.append(run)
            del model
        del train_x, val_x, labels, val
    matched = [dict(run, by_rollout={rid: dict(group, median=group['center_median'])
                                    for rid, group in run['by_rollout'].items()}) for run in results]
    dump(args.out / 'completion.json', dict(status='complete', smoke=args.smoke, real_data_used=False,
         results=results, gate=None if args.smoke else pilot.gate(results),
         matched_center_gate=None if args.smoke else pilot.gate(matched)))


if __name__ == '__main__':
    main()
