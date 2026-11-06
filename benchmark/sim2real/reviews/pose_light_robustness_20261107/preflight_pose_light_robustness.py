"""Local procedural preflight for training, scoring and shared radiance inputs."""
import argparse
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

import pose_light_robustness as trial
import prepare_pose_radiance_pairs as light


def fixture(device, scale, rot):
    rng = np.random.default_rng(1126)
    frames = np.zeros((64, 3, 128, 128), np.float32)
    labels = []
    for i in range(64):
        z = float(rng.uniform(.45, 1.1))
        x, y = rng.integers(34, 94, 2)
        h, w = round(8/z), round(13/z)
        frames[i, 0, y-h:y+h, [x-w, x+w]] = .5
        frames[i, 1, [y-h, y+h], x-w:x+w] = .5
        frames[i, 2] = frames[i, 0] + frames[i, 1]
        labels.append([x/128, y/128, .5*np.log((2*w/128)*(2*h/128)), 0, np.log(z)])
    inputs = torch.from_numpy(frames).to(device)
    labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
    draw = trial.distort(inputs, .65, 715)
    assert torch.equal(draw, trial.distort(inputs, .65, 715)) and not torch.equal(draw, inputs)
    assert torch.isfinite(draw).all() and draw.min() >= 0 and draw.max() <= 1
    state = scale.CenterScale().state_dict()
    K = np.array([[293., 0., 173.], [0., 293., 130.], [0., 0., 1.]])
    losses = {}
    for arm in trial.ARMS:
        model, loss = trial.fit(scale, state, inputs, labels, 0, arm, True)
        assert len(loss) == 1 and np.isfinite(loss).all()
        assert any(not torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items())
        xy, z, size = scale.estimate(model, inputs[:2], [K, K])
        R = trial.frozen_rotation(rot, rot.RotationHead().to(device).eval(), inputs[:2], xy, size, [K, K])
        assert np.max(np.abs(R.transpose(0, 2, 1) @ R - np.eye(3))) < 1e-4
        assert (z > 0).all() and np.isfinite(z).all()
        losses[arm] = dict(optimizer_updates=1, loss=loss[0])
    vertices = np.array([[a,b,c] for a in [-.04,.04] for b in [-.06,.06] for c in [-.05,.05]])
    keys = np.vstack((np.zeros(3), vertices))
    truth = np.eye(4)
    truth[2,3] = .7
    rows = [dict(T=truth, K=K, keys=keys, uv=np.array([[.5,.5]]), rid='fixture')]
    tree = cKDTree(vertices)
    xy = np.array([[.5,.5]])
    best = trial.score(rot, rows, xy, [.7], [.2], [np.log(.2)], [np.eye(3)], vertices, tree, .02)['summary']
    far = trial.score(rot, rows, xy, [.85], [.3], [np.log(.2)], [np.eye(3)], vertices, tree, .02)['summary']
    assert best['success']==1 and best['depth_mae_m']==best['log_size_mae']==0
    assert far['success']==0 and np.isclose(far['depth_mae_m'],.15) and np.isclose(far['log_size_bias'],np.log(1.5))
    return losses


def check_radiance(tone_source):
    tone = light.canonical_tone(tone_source)
    rng = np.random.default_rng(1126)
    with tempfile.TemporaryDirectory() as tmp:
        source, dest = Path(tmp)/'base.h5', Path(tmp)/'dim.h5'
        rgb = rng.uniform(.01, 2, (3, 12, 18, 3)).astype('float32')
        pose = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
        display = np.stack([tone(x) for x in rgb])
        with h5py.File(source, 'w') as stream:
            g = stream.create_group('episodes/validation_000')
            g.create_dataset('radiance_linear_rgb', data=rgb, chunks=(1,12,18,3))
            g.create_dataset('v2e_display_luma_u8', data=display, chunks=(1,12,18))
            g.create_dataset('target_pose', data=pose)
            g.create_dataset('frame_timestamps_s', data=np.arange(3)*.01)
        before = light.sha(source)
        result = light.relight(source, dest, 'validation_000', .125, tone)
        with h5py.File(dest, 'r') as stream:
            g = stream['episodes/validation_000']
            assert np.array_equal(g['target_pose'][:], pose)
            assert np.array_equal(g['radiance_linear_rgb'][:], rgb*.125)
            assert (g['v2e_display_luma_u8'][:] <= display).all()
        assert light.sha(source) == before and result['canonical_luma_verified']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('scale-source', 'rotation-source', 'tone-source', 'out'):
        p.add_argument('--'+arg, type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(127)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    scale = trial.load('scale', args.scale_source)
    rot = trial.load('rotation', args.rotation_source)
    results = {'cpu':fixture(torch.device('cpu'), scale, rot)}
    if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 2*1024**3:
        results['cuda'] = fixture(torch.device('cuda'), scale, rot)
    check_radiance(args.tone_source)
    record = dict(status='complete', scope='local procedural fixture only', real_data_used=False,
                  optimizer=results, geometry_and_size_checks=True, radiance_and_luma_checks=True,
                  source_sha256={n:light.sha(Path(__file__).with_name(n)) for n in
                                 ['pose_light_robustness.py', 'prepare_pose_radiance_pairs.py']})
    light.dump(args.out, record)
    print(json.dumps(record), flush=True)


if __name__=='__main__':main()
