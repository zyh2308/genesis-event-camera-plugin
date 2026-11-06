"""Train and gate a paired event-only depth branch on synthetic windows."""
import argparse
import ast
import hashlib
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
CASES = ('original', 'median', 'learned')
SEEDS = (0, 1, 2)
EPOCHS = 8
BATCH = 64


def helpers(path, campaign):
    names = {'load', 'decode', 'sha', 'dump'}
    source = ast.parse(path.read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    context = dict(globals(), R=campaign, device=torch.device('cuda:0'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), context)
    return context


def extract(model, samples, decode):
    captured = []
    handle = model.models[-2].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    features, coordinates = [], []
    try:
        with torch.no_grad():
            for offset in range(0, len(samples), 16):
                batch = samples[offset:offset + 16]
                images = torch.from_numpy(np.stack([row['image'] for row in batch])).cuda()
                raw = model(images)
                assert len(captured) == 1 and raw.shape[1:] == (20, 13, 13)
                feature = captured.pop()
                assert feature.shape[2:] == raw.shape[2:]
                cells = raw[:, 18].flatten(1).argmax(1)
                local = feature.flatten(2)[torch.arange(len(batch), device=raw.device), :, cells]
                avg = F.adaptive_avg_pool2d(feature, 2).flatten(1)
                maximum = F.adaptive_max_pool2d(feature, 2).flatten(1)
                features.append(torch.cat((local, avg, maximum), dim=1))
                predicted, decoded_cells = decode(raw)
                np.testing.assert_array_equal(decoded_cells, cells.cpu().numpy())
                coordinates.append(predicted)
    finally:
        handle.remove()
    return torch.cat(features), np.concatenate(coordinates)


def make_head(width):
    return nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))


def depth_target(sample):
    return float(sample['T'][2, :3] @ sample['keys'][0] + sample['T'][2, 3])


def fit_head(train, target, validation, seed, smoke):
    mean = train.mean(0)
    scale = train.std(0).clamp_min(0.01)
    train = ((train - mean) / scale).clamp(-8, 8)
    validation = ((validation - mean) / scale).clamp(-8, 8)
    median = float(np.median(target))
    log_target = torch.from_numpy(np.log(target / median).astype(np.float32)).cuda()
    torch.manual_seed(seed + 41731)
    head = make_head(train.shape[1])
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    initial = hashlib.sha256(b''.join(t.numpy().tobytes() for t in head.state_dict().values())).hexdigest()
    head.cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    rng = np.random.default_rng(seed + 51623)
    history = []
    for epoch in range(1 if smoke else EPOCHS):
        indices = rng.permutation(len(train))
        if smoke:
            indices = indices[:BATCH]
        total = 0.0
        for offset in range(0, len(indices), BATCH):
            batch = torch.from_numpy(indices[offset:offset + BATCH]).cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(head(train[batch]).squeeze(1), log_target[batch], beta=0.1)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
            assert norm > 0
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append(total / len(indices))
    with torch.no_grad():
        depth = (head(validation).squeeze(1).exp() * median).cpu().numpy()
    assert np.isfinite(depth).all() and (depth > 0).all()
    state = dict(head={k: v.cpu() for k, v in head.state_dict().items()},
                 mean=mean.cpu(), scale=scale.cpu(), median_m=median, seed=seed,
                 initial_sha256=initial, epochs=len(history), train_loss=history)
    return depth, state


def epnp(keys, coordinates, K):
    try:
        ok, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(keys[1:], dtype=np.float64),
                                    np.ascontiguousarray(coordinates[1:], dtype=np.float64),
                                    K.astype(np.float64), None, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return None
    if not ok:
        return None
    rotation = cv2.Rodrigues(rvec)[0]
    translation = tvec[:, 0]
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
        return None
    if not ((keys @ rotation.T + translation)[:, 2] > 0).all():
        return None
    return rotation, translation


def predictions(keys, coordinates, K, median, learned):
    base = epnp(keys, coordinates, K)
    poses = dict.fromkeys(CASES)
    depths = dict.fromkeys(CASES)
    if base is None:
        return poses, depths
    rotation, translation = base
    center_offset = rotation @ keys[0]
    center = center_offset + translation
    poses['original'] = base
    depths.update(original=float(center[2]), median=median, learned=float(learned))
    for case in ('median', 'learned'):
        new_t = center * (depths[case] / center[2]) - center_offset
        if ((keys @ rotation.T + new_t)[:, 2] > 0).all():
            poses[case] = rotation, new_t
    return poses, depths


def measure(samples, coordinates, depths, median, vertices, tree):
    rows = []
    for sample, coords, depth in zip(samples, coordinates, depths, strict=True):
        truth = vertices @ sample['T'][:3, :3].T + sample['T'][:3, 3]
        poses, predicted_z = predictions(sample['keys'], coords * [346, 260], sample['K'], median, depth)
        errors = dict.fromkeys(CASES)
        for case, pose in poses.items():
            if pose is not None:
                R, t = pose
                errors[case] = float(tree.query((truth - t) @ R, workers=1)[0].mean())
        gt_z = depth_target(sample)
        rows.append(dict(sample=sample['id'], rollout=sample['rid'], gt_depth_m=gt_z,
                         predicted_depth_m=predicted_z, adds_m=errors,
                         epe_px=float(np.linalg.norm((coords - sample['uv']) * [346, 260], axis=1).mean())))
    return rows


def summarize(rows, threshold):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['rollout'], []).append(row)
    by_rollout = {}
    for rid, group in sorted(grouped.items()):
        stats = {}
        for case in CASES:
            errors = [row['adds_m'][case] for row in group]
            absolute_depth = [abs(row['predicted_depth_m'][case] - row['gt_depth_m'])
                              for row in group if row['predicted_depth_m'][case] is not None]
            stats[case] = dict(success=float(np.mean([e is not None and e < threshold for e in errors])),
                               auc=float(np.mean([max(1 - e / 0.1, 0) if e is not None else 0 for e in errors])),
                               depth_mae=float(np.mean(absolute_depth)) if absolute_depth else None,
                               valid=sum(e is not None for e in errors))
        by_rollout[rid] = stats
    summary = {}
    for case in CASES:
        values = [group[case] for group in by_rollout.values()]
        depths = [v['depth_mae'] for v in values if v['depth_mae'] is not None]
        abs_errors = [abs(row['predicted_depth_m'][case] - row['gt_depth_m'])
                      for row in rows if row['predicted_depth_m'][case] is not None]
        summary[case] = dict(success=float(np.mean([v['success'] for v in values])),
                             auc=float(np.mean([v['auc'] for v in values])),
                             depth_mae_m=float(np.mean(depths)), depth_median_abs_m=float(np.median(abs_errors)),
                             valid=sum(v['valid'] for v in values), depth_valid=len(abs_errors))
    return summary, by_rollout


def gate(results):
    rng = np.random.default_rng(50855)
    comparisons = []
    for method in METHODS:
        cases = [item for item in results if item['method'] == method]
        assert tuple(item['seed'] for item in cases) == SEEDS
        rollouts = list(cases[0]['by_rollout'])
        assert len(rollouts) == 32 and all(list(case['by_rollout']) == rollouts for case in cases)
        for baseline in ('original', 'median'):
            for metric, sign in [('depth_mae', -1), ('success', 1)]:
                delta = np.array([[sign * (case['by_rollout'][rid]['learned'][metric] -
                                           case['by_rollout'][rid][baseline][metric])
                                   for rid in rollouts] for case in cases])
                seeds = delta.mean(1)
                draws = delta.mean(0)[rng.integers(len(rollouts), size=(4000, len(rollouts)))].mean(1)
                lo, hi = np.quantile(draws, [0.025, 0.975])
                comparisons.append(dict(method=method, baseline=baseline, metric=metric,
                                        seed_improvement=seeds.tolist(), ci=[float(lo), float(hi)],
                                        passed=bool((seeds > 0).all() and lo > 0)))
    return dict(passed=all(row['passed'] for row in comparisons), comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    h = helpers(args.helper, args.campaign)
    sha, dump = h['sha'], h['dump']
    root = args.campaign
    model_source = root.parents[2] / 'external/sources/downstream/SingleShotPose'
    sys.path.insert(0, str(model_source))
    from darknet import Darknet
    cfg = model_source / 'cfg/yolo-pose.cfg'
    frozen = json.loads((root / 'recovery_20261103/pose_frozen_checkpoints.json').read_text())
    assets = json.loads((root / 'recovery_20261105_mainline/pose_prepared_full_v5/protocol.json').read_text())['assets']
    vertices_path = Path(assets['vertices']['path'])
    assert sha(vertices_path) == assets['vertices']['sha256']
    vertices = np.load(vertices_path, allow_pickle=False)
    tree = cKDTree(vertices)
    threshold = 0.1 * assets['diameter_m']
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / 'protocol.json', dict(real_data_used=False, smoke=args.smoke, methods=METHODS,
         seeds=(0,) if args.smoke else SEEDS, epochs=EPOCHS, batch=BATCH, base='frozen synthetic EPE best',
         selection='fixed final head epoch; no validation-based selection', threshold_m=threshold,
         source_sha256=sha(__file__), helper_sha256=sha(args.helper), cfg_sha256=sha(cfg),
         gate='both metrics vs both controls, each seed positive and paired rollout 95% CI lower > 0'))
    bindings = {}
    results, initial = [], {}
    for method in METHODS:
        train = h['load'](method, 'train')
        val = h['load'](method, 'validation')
        for split, rows, size, count in [('train', train, 1529, 96), ('validation', val, 510, 32)]:
            assert len(rows) == size and len({row['rid'] for row in rows}) == count
            digest = hashlib.sha256()
            for row in rows:
                digest.update(row['id'].encode() + row['rid'].encode())
                for key in ('keys', 'K', 'T', 'uv'):
                    digest.update(row[key].tobytes())
            assert bindings.setdefault(split, digest.hexdigest()) == digest.hexdigest()
        if args.smoke:
            train, val = train[:64], val[:16]
        target = np.array([depth_target(row) for row in train])
        assert np.isfinite(target).all() and (target > 0).all()
        for seed in ((0,) if args.smoke else SEEDS):
            record = next(row for row in frozen['runs'] if row['method'] == method and row['seed'] == seed)
            path = Path(record['selected_checkpoint'])
            assert sha(path) == record['selected_checkpoint_sha256']
            payload = torch.load(path, map_location='cpu', weights_only=False)
            assert (payload['generator'], payload['seed']) == (method, seed)
            update = payload['completed_update']
            model = Darknet(str(cfg)).cuda()
            model.load_state_dict(payload['model'], strict=True)
            del payload
            model.requires_grad_(False).eval()
            train_x, _ = extract(model, train, h['decode'])
            val_x, coordinates = extract(model, val, h['decode'])
            del model
            assert torch.isfinite(train_x).all() and torch.isfinite(val_x).all()
            depth, state = fit_head(train_x, target, val_x, seed, args.smoke)
            assert initial.setdefault(seed, state['initial_sha256']) == state['initial_sha256']
            rows = measure(val, coordinates, depth, state['median_m'], vertices, tree)
            summary, by_rollout = summarize(rows, threshold)
            if not args.smoke:
                validation = json.loads((path.parent / f'synthetic_validation_{update:06d}.json').read_text())
                epe = np.mean([np.mean([row['epe_px'] for row in rows if row['rollout'] == rid])
                               for rid in by_rollout])
                assert abs(epe - validation['mean_keypoint_epe_px']) < 0.002
            tag = f'{method}_seed{seed}'
            state.update(method=method, base_path=str(path), base_sha256=record['selected_checkpoint_sha256'])
            head_path = args.out / (tag + '_head.pt')
            torch.save(state, head_path)
            run = dict(method=method, seed=seed, original_update=update, summary=summary, by_rollout=by_rollout,
                       head_sha256=sha(head_path), base_sha256=state['base_sha256'],
                       initial_sha256=state['initial_sha256'], train_loss=state['train_loss'],
                       median_m=state['median_m'])
            dump(args.out / (tag + '.json'), dict(run, rows=rows))
            print('DEPTH_RESULT', json.dumps({k: v for k, v in run.items() if k != 'by_rollout'}), flush=True)
            results.append(run)
            del train_x, val_x
    dump(args.out / 'completion.json', dict(status='complete', results=results, label_hashes=bindings,
         smoke=args.smoke, real_data_used=False, gate=None if args.smoke else gate(results)))


if __name__ == '__main__':
    main()
