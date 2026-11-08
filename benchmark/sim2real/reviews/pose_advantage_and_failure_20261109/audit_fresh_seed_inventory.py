"""Read registered pose-bank protocols/H5 attributes; do not generate data."""
import argparse, hashlib, json
from pathlib import Path
import h5py


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--base-seed', type=int, required=True)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    paths = set()
    for run in a.runs.iterdir():
        if not run.is_dir():
            continue
        for sub in ('sources', 'study/sources', 'smoke/sources', 'results/sources', 'bank/sources'):
            p = run/sub/'pose_bank_protocol.json'
            if p.is_file():
                paths.add(p.absolute())
    for pattern in ('recovery_*/jobs/*/results/sources/pose_bank_protocol.json',
                    'recovery_*/jobs/*/smoke/sources/pose_bank_protocol.json'):
        paths.update(p.absolute() for p in a.campaign.glob(pattern))
    proposed = {a.base_seed + 1009*i for i in range(a.count)}
    assert len(proposed) == a.count
    protocol_rows, actual_rows, all_old = [], [], set()
    for p in sorted(paths):
        c = json.loads(p.read_text())
        if not {'physical_seed', 'train_rollouts', 'validation_rollouts'} <= set(c):
            raise ValueError('Registered protocol lacks seed/count: '+str(p))
        n = c['train_rollouts'] + c['validation_rollouts']
        seeds = {int(c['physical_seed'])+1009*i for i in range(n)}
        all_old.update(seeds)
        protocol_rows.append(dict(path=str(p), sha256=sha(p), base_seed=c['physical_seed'],
                                  train_rollouts=c['train_rollouts'], validation_rollouts=c['validation_rollouts']))
        bank = p.parent.parent/'pose_bank/physical'
        if bank.is_dir():
            for h in sorted(bank.glob('*.h5')):
                with h5py.File(h, 'r') as f:
                    groups = list(f['episodes'].values())
                    assert len(groups) == 1
                    g = groups[0]
                    s = int(g.attrs['scene_seed'])
                    all_old.add(s)
                    actual_rows.append(dict(path=str(h), scene_seed=s, split=str(g.attrs['split'])))
    overlap = sorted(proposed & all_old)
    result = dict(schema='registered_pose_seed_inventory_v1',
        scope='Listed run-level source protocols and campaign recovery job result/smoke protocols; H5 attributes under their pose_bank/physical. Not an exhaustive filesystem scan.',
        protocol_count=len(protocol_rows), observed_H5_files=len(actual_rows),
        unique_registered_or_observed_scene_seeds=len(all_old),
        proposed_base_seed=a.base_seed, proposed_count=a.count,
        proposed_scene_seeds=sorted(proposed), overlap=overlap, passed=not overlap,
        protocols=protocol_rows, actual_H5=actual_rows, optimizer_updates=0, real_data_used=False)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ('protocol_count','observed_H5_files','unique_registered_or_observed_scene_seeds','overlap','passed')}))
    assert result['protocol_count'] >= 2 and result['observed_H5_files'] >= 96
    assert not overlap


if __name__ == '__main__':
    main()
