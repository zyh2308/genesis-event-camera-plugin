"""Generate paired relit TRAIN trajectories without touching the frozen simulator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import prepare_pose_radiance_pairs as shared


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    R=args.campaign
    bank=R/'pose_bank';sources=R/'sources'
    manifest=json.loads((bank/'bank_manifest.json').read_text())
    assert not manifest['real_data_used']
    train={row['rollout_id']:row for row in manifest['rollouts'] if row['split']=='train'}
    assert len(train)==96
    selected=tuple(sorted(train)[i] for i in np.linspace(0,95,8,dtype=int))
    assert all(r.startswith('train_') for r in selected)
    tone_path=sources/'benchmark/sim2real/scripts/generate_stage4_flow_bank.py'
    tone=shared.canonical_tone(tone_path)
    args.out.mkdir(parents=True,exist_ok=False)
    shared.dump(args.out/'protocol.json',dict(real_data_used=False,split='train',selected=selected,
        selection='eight uniformly spaced ordinal positions in the sorted 96 training trajectories',
        levels=shared.LEVELS,source_manifest_sha256=shared.sha(bank/'bank_manifest.json'),
        tone_source_sha256=shared.sha(tone_path),source_sha256=shared.sha(__file__),
        scope='fixed profiles and simulator; common physical radiance attenuation; no holdout trajectory in training'))
    for level,gain in shared.LEVELS.items():
        folder=args.out/level;folder.mkdir()
        (folder/'sources').symlink_to(sources,target_is_directory=True)
        target=folder/'pose_bank';(target/'physical').mkdir(parents=True)
        rows=[]
        for rid in selected:
            path=bank/'physical'/(rid+'.h5')
            assert shared.sha(path)==train[rid]['sha256']
            rows.append(shared.relight(path,target/'physical'/(rid+'.h5'),rid,gain,tone))
        shared.dump(target/'bank_manifest.json',dict(rollouts=rows,real_data_used=False,split='train'))
        print('TRAIN_LIGHT_PHYSICAL_READY',level,len(rows),flush=True)
        for script in ('replay_new_events.py','replay_v2e_parallel.py'):
            subprocess.run([sys.executable,'-u',str(sources/script),'--campaign',str(folder),
                            '--kind','pose','--workers','2'],check=True)
    shared.dump(args.out/'completion.json',dict(status='complete',real_data_used=False,selected=selected,levels=shared.LEVELS))


if __name__=='__main__':main()
