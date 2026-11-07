"""Create immutable, shared radiance intervention banks for frozen event replay."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

SELECTED = ('validation_000', 'validation_031', 'validation_063', 'validation_095')
LEVELS = {'half': 0.5, 'eighth': 0.125}


def sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()


def dump(path, record):
    with path.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


def canonical_tone(path):
    ast_tree = ast.parse(path.read_text())
    defs = [n for n in ast_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'radiance_to_v2e_luma_u8']
    assert len(defs) == 1
    scope = {'np': np}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), 'exec'), scope)
    return scope['radiance_to_v2e_luma_u8']


def relight(path, out, rollout, gain, tone):
    assert 0 < gain < 1
    data_path = '/episodes/' + rollout
    with h5py.File(path, 'r') as src, h5py.File(out, 'x') as dst:
        original = src[data_path]
        group = dst.create_group(data_path)
        group.attrs.update(original.attrs)
        group.attrs['diagnostic_radiance_gain'] = gain
        original_rgb = original['radiance_linear_rgb']
        original_luma = original['v2e_display_luma_u8']
        rgb = group.create_dataset('radiance_linear_rgb', shape=original_rgb.shape, dtype=original_rgb.dtype,
                                   chunks=original_rgb.chunks, compression='gzip', compression_opts=1)
        luma = group.create_dataset('v2e_display_luma_u8', shape=original_luma.shape, dtype=original_luma.dtype,
                                    chunks=original_luma.chunks, compression='gzip', compression_opts=1)
        base_count = changed_count = 0
        for i, frame in enumerate(original_rgb):
            assert np.isfinite(frame).all() and np.array_equal(tone(frame), original_luma[i])
            changed = np.multiply(frame, gain, dtype=original_rgb.dtype)
            rgb[i] = changed
            encoded = tone(changed)
            luma[i] = encoded
            base_count += int(np.count_nonzero(original_luma[i]))
            changed_count += int(np.count_nonzero(encoded))
        frozen_names = sorted(set(original) - {'radiance_linear_rgb', 'v2e_display_luma_u8'})
        for name in frozen_names:
            assert isinstance(original[name], h5py.Dataset)
            group[name] = h5py.ExternalLink(str(path.resolve()), data_path + '/' + name)
    with h5py.File(out, 'r') as dst:
        group = dst[data_path]
        for name in frozen_names:
            link = group.get(name, getlink=True)
            assert Path(link.filename) == path.resolve() and link.path == data_path + '/' + name
    return dict(rollout=rollout, gain=gain, source_sha256=sha(path), output_sha256=sha(out),
                canonical_luma_verified=True, frozen_geometry_fields=frozen_names,
                base_nonzero_luma=base_count, scaled_nonzero_luma=changed_count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    R = args.campaign
    sources = R/'sources'
    frozen = R/'recovery_20261105_mainline/jobs/119_pose_fresh_confirm_paired_20261106/results/pose_bank'
    original_manifest = json.loads((frozen/'bank_manifest.json').read_text())
    assert not original_manifest['real_data_used'] and len(original_manifest['rollouts']) == 96
    originals = {r['rollout']:r['sha256'] for r in original_manifest['rollouts']}
    tone_path = sources/'benchmark/sim2real/scripts/generate_stage4_flow_bank.py'
    tone = canonical_tone(tone_path)
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out/'protocol.json', dict(real_data_used=False, levels=LEVELS, selected=SELECTED,
         source_manifest_sha256=sha(frozen/'bank_manifest.json'), tone_source_sha256=sha(tone_path),
         scope='global linear radiance attenuation; equal physical trajectories and original sensor profiles'))
    for name, gain in LEVELS.items():
        folder = args.out/name
        folder.mkdir()
        (folder/'sources').symlink_to(sources, target_is_directory=True)
        bank = folder/'pose_bank'
        (bank/'physical').mkdir(parents=True)
        rows = []
        for rid in SELECTED:
            source = frozen/'physical'/(rid+'.h5')
            assert sha(source) == originals[rid]
            rows.append(relight(source, bank/'physical'/(rid+'.h5'), rid, gain, tone))
        dump(bank/'bank_manifest.json', dict(rollouts=rows, source_manifest_sha256=sha(frozen/'bank_manifest.json'),
                                           real_data_used=False))
        print('RADIANCE_PAIR_COMPLETE', name, len(rows), flush=True)
        if args.replay:
            for script in ('replay_new_events.py','replay_v2e_parallel.py'):
                subprocess.run([sys.executable,'-u',str(sources/script),'--campaign',str(folder),
                                '--kind','pose','--workers','1'],check=True)
    dump(args.out/'completion.json',dict(status='complete',replayed=args.replay,real_data_used=False))


if __name__=='__main__':main()
