"""Measure visible-target support of the original RAW C1 rotation crop on CPU."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rotation_error(a, b):
    trace = np.trace(np.asarray(a).T @ np.asarray(b))
    return float(np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--factor-sha256', required=True)
    parser.add_argument('--peak-sha256', required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    root = args.root
    p = root / '20261108-pose-factor-diagnostic-v1/export/completion.json'
    q = root / '20261109-pose-advantage-and-failure-v1/peak_diagnosis/completion.json'
    assert sha(p) == args.factor_sha256 and sha(q) == args.peak_sha256
    f, peak = [json.loads(x.read_text()) for x in (p, q)]
    assert f['native_parity_passed'] and f['models_and_sources_unchanged']
    assert peak['prediction_cases'] == 1152 and f['real_data_used'] is peak['real_data_used'] is False
    inputs = {str(p): args.factor_sha256, str(q): args.peak_sha256}

    def check(record):
        path = Path(record['path'])
        assert path.stat().st_size == record['size_bytes'] and sha(path) == record['sha256']
        inputs[str(path)] = record['sha256']
        return path

    records = [r for r in f['artifacts'] if Path(r['path']).name == 'predictions.jsonl']
    assert len(records) == 1
    predictions = [r for line in check(records[0]).open() if (r := json.loads(line))['arm'] == 'C1']
    methods, lights = ('eventgenesis', 'v2e'), ('normal', 'half', 'eighth')
    rids = ('validation_000', 'validation_031', 'validation_063', 'validation_095')
    sample_ids = {f'{rid}/{i:03d}' for rid in rids for i in range(4, 20)}
    assert len(predictions) == 1152
    assert {(r['method'], r['seed'], r['condition'], r['sample_id']) for r in predictions} == set(itertools.product(methods, range(3), lights, sample_ids))
    phys = [r for r in peak['input_records'] if '/physical/validation_' in r['path'] and '/radiance/' not in r['path']]
    assert len(phys) == 4 and len({Path(r['path']).parent for r in phys}) == 1
    bank = check(phys[0]).parent.parent
    for record in phys[1:]:
        check(record)
    support = [r for r in peak['input_records'] if Path(r['path']).name == 'event_support.py']
    assert len(support) == 1
    support = load(check(support[0]), 'frozen_crop_region_support')
    rotation = root / '20261103-stage9-direct-v1/recovery_20261105_mainline/pose_rotation_pilot_20261106.py'
    rotation_sha = 'c3da6bcbc05d4adeacf6ec278eb2c6a3753fdf7952903b29f2c080063b3a7423'
    assert sha(rotation) == rotation_sha
    inputs[str(rotation)] = rotation_sha
    crop = load(rotation, 'original_raw_c1_rotation').crop
    ref = {r['sample_id']: r for r in predictions if (r['method'], r['seed'], r['condition']) == ('eventgenesis', 0, 'normal')}
    regions = support.regions_for_rows(bank, [dict(id=r['sample_id'], rid=r['rid'], K=r['K'], uv=[r['uv_gt']]) for r in ref.values()], strict=True)
    masks = {i: support.region_pyramid(v['endpoint'])['input_fraction'] for i, v in regions.items()}
    points = {}
    for i, v in regions.items():
        y, x = np.nonzero(v['endpoint'].mask)
        assert len(x) > 0
        points[i] = np.stack(((x + .5) / 346., (y + .5) / 260.), axis=-1)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(predictions), 48):
            batch = predictions[start:start + 48]
            for r in batch:
                assert all(r[k] == ref[r['sample_id']][k] for k in ('K', 'uv_gt', 'T_gt'))
            x = torch.from_numpy(np.stack([masks[r['sample_id']] for r in batch]))[:, None].expand(-1, 3, -1, -1)
            center = torch.tensor([r['pred_uv'] for r in batch], dtype=torch.float32)
            size = torch.tensor([r['size'] for r in batch], dtype=torch.float32)
            K = np.asarray([r['K'] for r in batch])
            aspect = torch.tensor(np.sqrt(K[:, 0, 0] * 260 / (K[:, 1, 1] * 346)), dtype=torch.float32)
            sample, _ = crop(x, center, size, aspect, torch.zeros(len(batch)))
            bounds = (2.4 * size).clamp(.16, .9)
            extent = torch.stack((bounds * aspect, bounds / aspect), 1).numpy()
            occupancy = sample[:, 0].mean((1, 2)).numpy()
            for r, half, mass in zip(batch, extent / 2, occupancy):
                ij = points[r['sample_id']]
                retained = np.all(np.abs(ij - r['pred_uv']) <= half, axis=1).mean()
                gt_retained = np.all(np.abs(ij - r['uv_gt']) <= half, axis=1).mean()
                rows.append(dict(**{k: r[k] for k in ('method','seed','condition','rid','sample_id')},
                    target_retained=float(retained), crop_target_mass=float(mass), gt_center_same_size_retained=float(gt_retained),
                    rotation_deg=rotation_error(r['R_pred'], np.asarray(r['T_gt'])[:3, :3]),
                    rotation_gt_crop_deg=rotation_error(r['R_crop_gt_center'], np.asarray(r['T_gt'])[:3, :3])))
    assert all(sha(p) == v for p, v in inputs.items())
    groups = defaultdict(list)
    for row in rows:
        groups[row['method'], row['condition'], row['rid']].append(row)
    assert len(groups) == 24 and all(len(v) == 48 for v in groups.values())
    fields = ('target_retained','crop_target_mass','gt_center_same_size_retained','rotation_deg','rotation_gt_crop_deg')
    result = dict(status='complete', optimizer_updates=0, model_forward_calls=0, cases=len(rows), input_sha256=inputs,
                  GT_used_only_posthoc=True, real_data_used=False,
                  by_rollout=[dict(method=m, condition=c, rid=rid, windows=len(v),
                                   fraction_below_90percent=float(np.mean([a['target_retained'] < .9 for a in v])),
                                   **{key: float(np.mean([a[key] for a in v])) for key in fields})
                              for (m, c, rid), v in sorted(groups.items())])
    args.out.mkdir(parents=True)
    (args.out / 'completion.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (args.out / 'cases.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    print('CROP_AUDIT_COMPLETE', len(rows))


if __name__ == '__main__':
    main()
