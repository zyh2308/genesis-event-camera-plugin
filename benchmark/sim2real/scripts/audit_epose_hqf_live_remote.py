#!/usr/bin/env python3
"""Read-only, narrow g1 audit of registered HQF and one E-POSE sample."""
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


def main() -> None:
    base = Path('/home/zhaoyuhan/Eventbased_WAM')
    print('utc=' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    print('loadavg=' + Path('/proc/loadavg').read_text().strip())
    print(subprocess.check_output(['free', '-h'], text=True))
    print('top_cpu_processes=' + subprocess.check_output(['bash', '-lc', "ps -eo user,pid,pcpu,comm --sort=-pcpu | head -12"], text=True))
    print('genesis_module_available=' + str(importlib.util.find_spec('genesis') is not None))
    try:
        req = Request('https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset/resolve/main/replicaCAD.scene_dataset_config.json', method='HEAD')
        with urlopen(req, timeout=12) as response:
            print('replicacad_hf_head=' + json.dumps({'status': response.status, 'content_length': response.headers.get('Content-Length'), 'final_url': response.geturl()}))
    except Exception as exc:
        print('replicacad_hf_head_error=' + repr(exc))
    repo = base / 'repositories/gec-sim2real'
    print('repo_head=' + subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip())
    print('repo_branch=' + subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip())
    for root in (base / 'data/sim2real/stage3/real/HQF', base / 'data/sim2real/stage5/20261021-hqf14-memmap-v3'):
        print('root=' + str(root) + ' exists=' + str(root.is_dir()))
        if root.is_dir():
            seq = []
            for directory in sorted(root.iterdir()):
                metadata = directory / 'metadata.json'
                if directory.is_dir() and metadata.is_file():
                    m = json.loads(metadata.read_text())
                    seq.append({'name': directory.name,
                                'n_events': int(len(__import__('numpy').load(directory / 'events_ts.npy', mmap_mode='r'))),
                                'n_images': int(len(__import__('numpy').load(directory / 'images_ts.npy', mmap_mode='r'))),
                                'event_ts_unit': m.get('event_timestamp_unit'), 'image_ts_unit': m.get('image_timestamp_unit')})
            print('sequences=' + json.dumps(seq))

    pose = base / 'data/real/epose/figshare_28106717_v1'
    mat = pose / 'extracted/block/block/r_01_gl/0001_data.mat'
    print('pose_mat=' + str(mat) + ' exists=' + str(mat.is_file()))
    if mat.is_file():
        import scipy.io as sio
        d = sio.loadmat(mat)
        keep = {}
        for key in ('time', 'cls_index', 'intrinsicMatrix', 'cameraPose', 'pose', 'centerInfo', 'keypointprojections', 'events'):
            if key not in d:
                continue
            arr = np.asarray(d[key])
            keep[key] = {'shape': list(arr.shape), 'dtype': str(arr.dtype), 'finite': bool(np.isfinite(arr).all())}
            if key in ('time', 'cls_index', 'intrinsicMatrix', 'cameraPose', 'pose', 'centerInfo', 'keypointprojections'):
                keep[key]['value'] = arr.tolist()
            if key == 'events':
                keep[key].update({'first_row': arr[:1].tolist(), 'last_row': arr[-1:].tolist(),
                                  'xy_minmax': [arr[:, :2].min(axis=0).tolist(), arr[:, :2].max(axis=0).tolist()],
                                  't_minmax': [float(arr[:, 2].min()), float(arr[:, 2].max())],
                                  'p_values': np.unique(arr[:, 3]).tolist()})
        print('mat_fields=' + json.dumps(keep))
        print('mat_sha256=' + hashlib.sha256(mat.read_bytes()).hexdigest())

    h5path = pose / 'extracted/original_data/original_data/block/r_01_gl/data.h5'
    cad_xyz = base / 'external_sources/E_Pose/xyz_data/wood block.xyz'
    cad_ply = base / 'external_sources/E_Pose/obj_files/wood_block/google_16k/nontextured.ply'
    print('cad_paths=' + json.dumps({p.name: {'path': str(p), 'exists': p.is_file(),
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None} for p in (cad_xyz, cad_ply)}))
    if h5path.is_file() and mat.is_file():
        import scipy.io as sio
        import h5py
        from collections import defaultdict, Counter
        mat_data = sio.loadmat(mat)
        mat_events = np.asarray(mat_data['events'])
        with h5py.File(h5path, 'r') as h5:
            info = {}
            for key in h5:
                ds = h5[key]
                row = {'shape': list(ds.shape), 'dtype': str(ds.dtype)}
                if ds.ndim == 2 and ds.shape[1] > 0:
                    row['row_minmax'] = [[float(np.nanmin(ds[i, :])), float(np.nanmax(ds[i, :]))] for i in range(ds.shape[0])]
                    row['row_first_last'] = [[float(ds[i, 0]), float(ds[i, -1])] for i in range(ds.shape[0])]
                info[key] = row
            print('h5_datasets=' + json.dumps(info))
            raw = np.asarray(h5['events_parallel'])
        event_t = raw[3]
        by_pixel = defaultdict(list)
        for x, y, p, t in raw.T:
            by_pixel[(int(x), int(y), int(p))].append(float(t))
        offsets = Counter()
        all_diffs = []
        for x, y, t, p in mat_events:
            for absolute_t in by_pixel.get((int(x), int(y), int(p)), []):
                difference = absolute_t - float(t)
                all_diffs.append(difference)
                offsets[round(difference / 0.0001)] += 1
        if not offsets:
            raise RuntimeError('MAT raw events have no coordinate/polarity matches in original HDF5')
        best_bin, _ = offsets.most_common(1)[0]
        candidates = [value for value in all_diffs if abs(value - best_bin * 0.0001) <= 0.00025]
        offset = float(np.median(candidates))
        left = float(mat_events[:, 2].min() + offset)
        right = float(mat_events[:, 2].max() + offset)
        lo = int(np.searchsorted(event_t, left, side='left'))
        hi = int(np.searchsorted(event_t, right, side='right'))
        candidate = raw[:, lo:hi]
        candidate_keys = defaultdict(list)
        for x, y, p, t in candidate.T:
            candidate_keys[(int(x), int(y), int(p))].append(float(t))
        matched = 0
        errors = []
        for x, y, t, p in mat_events:
            ts = candidate_keys.get((int(x), int(y), int(p)), [])
            if ts:
                err = min(abs(v - (float(t) + offset)) for v in ts)
                errors.append(err)
                matched += int(err <= 2e-5)
        print('mat_h5_time_alignment=' + json.dumps({
            'offset_s': offset, 'offset_mode_100us_bin': best_bin,
            'mat_event_window_s_relative': [float(mat_events[:, 2].min()), float(mat_events[:, 2].max())],
            'mapped_absolute_window_s': [left, right], 'h5_events_in_window': int(candidate.shape[1]),
            'mat_events': int(len(mat_events)), 'matched_xy_p_within_20us': matched,
            'median_nearest_time_error_s': float(np.median(errors)) if errors else None,
            'max_nearest_time_error_s': float(np.max(errors)) if errors else None
        }))
        print('h5_path=' + str(h5path))
        print('h5_sha256=' + hashlib.sha256(h5path.read_bytes()).hexdigest())
        if cad_xyz.is_file() and cad_ply.is_file():
            import cv2
            xyz = np.loadtxt(cad_xyz, usecols=(0, 1, 2))
            with cad_ply.open() as stream:
                header = 0
                for line in stream:
                    header += 1
                    if line.strip() == 'end_header':
                        break
            ply = np.loadtxt(cad_ply, skiprows=header, max_rows=8194, usecols=(0, 1, 2))
            pose_data = np.asarray(mat_data['pose'], dtype=np.float64)
            low, high = xyz.min(axis=0), xyz.max(axis=0)
            corners = np.asarray([
                [high[0], high[1], low[2]], [high[0], low[1], low[2]],
                [low[0], high[1], low[2]], [low[0], low[1], low[2]],
                [high[0], high[1], high[2]], [high[0], low[1], high[2]],
                [low[0], high[1], high[2]], [low[0], low[1], high[2]],
            ])
            camera = corners @ pose_data[:3, :3].T + pose_data[:3, 3]
            kproj = np.asarray(mat_data['keypointprojections'], dtype=np.float64)
            correct_xy = camera[:, :2] / camera[:, 2:3] * np.asarray([292.52, 293.19]) + np.asarray([172.61, 130.125])
            first_z_xy = camera[:, :2] / camera[:1, 2:3] * np.asarray([292.52, 293.19]) + np.asarray([172.61, 130.125])
            K = np.asarray(mat_data['intrinsicMatrix'], dtype=np.float64)
            distortion = np.asarray([[-0.362, 0.115, 0.0009576, 0.00000455, 0.130]], dtype=np.float64)
            correct_undistorted = cv2.undistortPoints(correct_xy.reshape(-1, 1, 2), K, distortion, P=K).reshape(-1, 2)
            first_z_undistorted = cv2.undistortPoints(first_z_xy.reshape(-1, 1, 2), K, distortion, P=K).reshape(-1, 2)
            print('cad_pose_projection=' + json.dumps({
                'xyz_vertex_shape': list(xyz.shape), 'ply_vertex_shape': list(ply.shape),
                'xyz_ply_max_abs_m': float(np.max(np.abs(xyz - ply))),
                'cad_dimensions_m': np.ptp(xyz, axis=0).tolist(), 'pose_translation_m': pose_data[:3, 3].tolist(),
                'camera_corner_depth_m': camera[:, 2].tolist(),
                'mat_camxyz_max_abs_m': float(np.max(np.abs(kproj[:, 2:5] - camera))),
                'correct_undistorted_pixel_max_abs_vs_mat': float(np.max(np.abs(kproj[:, :2] - correct_undistorted))),
                'legacy_first_corner_depth_pixel_max_abs_vs_mat': float(np.max(np.abs(kproj[:, :2] - first_z_undistorted)))
            }))


if __name__ == '__main__':
    main()
