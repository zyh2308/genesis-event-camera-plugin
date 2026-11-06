"""Learn event-only allocentric rotation with a fixed center and scale branch."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

SEEDS = (0, 1, 2)
METHODS = ('eventgenesis', 'v2e')
EPOCHS = 20
BATCH = 48
SIZE = 80
CASES = ('original', 'median', 'previous', 'center_median', 'learned', 'oracle_rotation')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class RotationHead(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (3, 24, 40, 64, 96)
        layers = []
        for n, m in zip(widths, widths[1:]):
            layers.extend([nn.Conv2d(n, m, 5, 2, 2, bias=False), nn.GroupNorm(4, m), nn.SiLU(),
                           nn.Conv2d(m, m, 3, 1, 1, bias=False), nn.GroupNorm(4, m), nn.SiLU()])
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(3), nn.Flatten(),
                                     nn.Linear(864, 128), nn.SiLU(), nn.Linear(128, 6))

    def forward(self, images):
        a, b = self.encoder(images).split(3, dim=1)
        u = F.normalize(a, dim=1)
        v = F.normalize(b - u * (b * u).sum(dim=1, keepdim=True), dim=1)
        return torch.stack((u, v, torch.cross(u, v, dim=1)), dim=2)


def ray_frame(center, intrinsics):
    K = np.asarray(intrinsics)
    pix = np.column_stack((center * [346, 260], np.ones(len(center))))
    rays = np.linalg.solve(K, pix[..., None]).squeeze(-1)
    z = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return rays, np.stack((x, np.cross(z, x), z), axis=2)


def crop(inputs, center, box_size, aspect, angle):
    c, s = angle.cos(), angle.sin()
    rotate = torch.stack((c, -s, s, c), dim=1).reshape(-1, 2, 2)
    bounds = (2.4 * box_size).clamp(0.16, 0.9)
    xy_scale = torch.stack((bounds * aspect, bounds / aspect), dim=1)
    theta = torch.cat((xy_scale[:, :, None] * rotate, (2 * center - 1)[:, :, None]), dim=2)
    grid = F.affine_grid(theta, (len(inputs), 3, SIZE, SIZE), align_corners=False)
    return F.grid_sample(inputs, grid, align_corners=False), rotate


def train(inputs, center, box, aspect, truth, weights, seed, smoke):
    torch.manual_seed(seed + 77856)
    head = RotationHead()
    initial = hashlib.sha256(b''.join(t.numpy().tobytes() for t in head.state_dict().values())).hexdigest()
    head.cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=0.0007, weight_decay=0.01)
    rng = np.random.default_rng(seed + 11879)
    loss_history = []
    for _ in range(1 if smoke else EPOCHS):
        losses = []
        order = rng.permutation(len(inputs))[:BATCH if smoke else len(inputs)]
        for start in range(0, len(order), BATCH):
            ids = torch.from_numpy(order[start:start+BATCH]).cuda()
            angle = torch.from_numpy(rng.uniform(-np.pi, np.pi, len(ids)).astype(np.float32)).cuda()
            frames, transform = crop(inputs[ids], center[ids], box[ids], aspect[ids], angle)
            prediction = head(frames)
            target = truth[ids].clone()
            target[:, :2] = transform.transpose(1, 2) @ target[:, :2]
            dot = (target * prediction).sum(dim=1)
            loss = ((1 - dot.square()) * weights).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
            assert torch.isfinite(loss) and norm > 0
            opt.step()
            losses.append(float(loss.detach()))
        loss_history.append(float(np.mean(losses)))
    return head.eval(), initial, loss_history


@torch.inference_mode()
def estimate_rotation(head, inputs, centers, boxes, intrinsics, rotations=None):
    rays, axes = ray_frame(centers, intrinsics)
    K = np.asarray(intrinsics)
    aspect = np.sqrt(K[:, 0, 0] * 260 / (K[:, 1, 1] * 346))
    center, box, aspect = (torch.as_tensor(x, dtype=torch.float32, device='cuda') for x in (centers, boxes, aspect))
    if rotations is not None:
        truth = axes.transpose(0, 2, 1) @ rotations
        return center, box, aspect, torch.as_tensor(truth, dtype=torch.float32, device='cuda')
    output = []
    for i in range(0, len(inputs), BATCH):
        sl = slice(i, i+BATCH)
        frames, _ = crop(inputs[sl], center[sl], box[sl], aspect[sl], torch.zeros(len(center[sl]), device='cuda'))
        output.append(head(frames).cpu().numpy())
    predicted = axes @ np.concatenate(output)
    assert np.max(np.abs(predicted.transpose(0, 2, 1) @ predicted - np.eye(3))) < 1e-4
    assert np.min(np.linalg.det(predicted)) > 0.9999
    return predicted, rays


def score(pilot, rows, prior, predictions, rays, median, vertices, tree):
    output = []
    for sample, old, R, ray in zip(rows, prior, predictions, rays, strict=True):
        assert (sample['id'], sample['rid']) == (old['sample'], old['rollout'])
        assert abs(pilot.depth_target(sample) - old['gt_depth_m']) < 1e-10
        keys, T = sample['keys'], sample['T']
        truth = vertices @ T[:3, :3].T + T[:3, 3]
        errors = {k: old['adds_m'][k] for k in ('original', 'median')}
        depths = {k: old['predicted_depth_m'][k] for k in ('original', 'median')}
        errors['previous'] = old['adds_m']['learned']
        depths['previous'] = depth = old['predicted_depth_m']['learned']
        for case, rotation, z in [('learned', R, depth), ('center_median', R, median),
                                  ('oracle_rotation', T[:3, :3], depth)]:
            depths[case] = z if depth is not None else None
            errors[case] = None
            if depth is not None:
                t = ray * z - rotation @ keys[0]
                if ((keys @ rotation.T + t)[:, 2] > 0).all():
                    errors[case] = float(tree.query((truth - t) @ rotation, workers=1)[0].mean())
        output.append(dict(sample=sample['id'], rollout=sample['rid'], gt_depth_m=old['gt_depth_m'],
                           predicted_depth_m=depths, adds_m=errors))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'prior', 'scale-source', 'pilot-source', 'helper', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    pilot = module('depth_pilot', args.pilot_source)
    scale = module('scale_pilot', args.scale_source)
    h = pilot.helpers(args.helper, args.campaign)
    sha, dump = h['sha'], h['dump']
    assets = json.loads((args.campaign/'recovery_20261105_mainline/pose_prepared_full_v5/protocol.json').read_text())['assets']
    assert sha(assets['vertices']['path']) == assets['vertices']['sha256']
    vertices = np.load(assets['vertices']['path'], allow_pickle=False)
    tree = cKDTree(vertices)
    extent = np.ptp(vertices, axis=0)
    weights = torch.as_tensor(extent / extent.sum(), dtype=torch.float32, device='cuda')
    prior = json.loads((args.prior/'completion.json').read_text())
    assert prior['status'] == 'complete' and not prior['smoke'] and not prior['real_data_used']
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out/'protocol.json', dict(smoke=args.smoke, real_data_used=False, cases=CASES, seeds=SEEDS,
         epochs=EPOCHS, prior_sha256=sha(args.prior/'completion.json'), scale_sha256=sha(args.scale_source),
         rotation_source_sha256=sha(__file__), threshold_m=0.1*assets['diameter_m']))
    results, starts = [], {}
    for method in METHODS:
        tr, va = (h['load'](method, split) for split in ('train', 'validation'))
        assert len(tr) == 1529 and len(va) == 510
        if args.smoke:
            tr, va = tr[:64], va[:16]
        x, y = scale.images(tr), scale.images(va)
        K_train, K_val = ([r['K'] for r in rows] for rows in (tr, va))
        rotation_labels = np.stack([r['T'][:3, :3] for r in tr])
        del tr
        for seed in ((0,) if args.smoke else SEEDS):
            tag = f'{method}_seed{seed}'
            ref = next(r for r in prior['results'] if (r['method'], r['seed']) == (method, seed))
            source = args.prior/(tag+'_scale.pt')
            assert sha(source) == ref['weights_sha256']
            payload = torch.load(source, map_location='cpu', weights_only=True)
            assert (payload['method'], payload['seed']) == (method, seed)
            net = scale.CenterScale().cuda().eval()
            net.load_state_dict(payload['model'], strict=True)
            centers, _, boxes = scale.estimate(net, x, K_train)
            val_center, val_depth, val_box = scale.estimate(net, y, K_val)
            del net
            crop_inputs = estimate_rotation(None, x, centers, boxes, K_train, rotations=rotation_labels)
            head, initial, loss = train(x, *crop_inputs, weights, seed, args.smoke)
            assert starts.setdefault(seed, initial) == initial
            rotations, rays = estimate_rotation(head, y, val_center, val_box, K_val)
            previous = json.loads((args.prior/(tag+'.json')).read_text())
            old_rows = previous['rows'][:len(va)]
            np.testing.assert_allclose(val_center, [r['predicted_center'] for r in old_rows], atol=1e-6)
            np.testing.assert_allclose(val_depth, [r['predicted_depth_m']['learned'] for r in old_rows], atol=1e-5)
            rows = score(pilot, va, old_rows, rotations, rays, payload['median_m'], vertices, tree)
            pilot.CASES = CASES
            summary, by_rollout = pilot.summarize(rows, 0.1*assets['diameter_m'])
            if not args.smoke:
                for case in ('original', 'median'):
                    assert summary[case] == ref['summary'][case]
            dest = args.out/(tag+'_rotation.pt')
            torch.save(dict(model=head.state_dict(), method=method, seed=seed, scale_sha256=ref['weights_sha256'],
                            scale_path=str(source), median_m=payload['median_m']), dest)
            run = dict(method=method, seed=seed, summary=summary, by_rollout=by_rollout, train_loss=loss,
                       weights_sha256=sha(dest), initial_sha256=initial)
            dump(args.out/(tag+'.json'), dict(run, rows=rows))
            print('ROTATION_RESULT', json.dumps({k: v for k, v in run.items() if k != 'by_rollout'}), flush=True)
            results.append(run)
            del head
    matched = [dict(run, by_rollout={rid: dict(group, median=group['center_median'])
                                    for rid, group in run['by_rollout'].items()}) for run in results]
    dump(args.out/'completion.json', dict(status='complete', results=results, smoke=args.smoke, real_data_used=False,
         gate=None if args.smoke else pilot.gate(results), matched_center_gate=None if args.smoke else pilot.gate(matched)))


if __name__ == '__main__':
    main()
