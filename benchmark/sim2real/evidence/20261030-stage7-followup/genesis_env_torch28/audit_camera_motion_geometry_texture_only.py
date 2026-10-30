#!/usr/bin/env python3
"""Audit geometry/texture support without consuming either generator's events."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from genesis_event_plugin.physics_warp import compute_se3_flow


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def audit(name: str, motion: Path, motion_threshold: float, texture_threshold: float) -> dict:
    result = json.loads((motion / 'motion_result.json').read_text())
    rad = np.load(motion / 'radiance_frames.npy', allow_pickle=False)
    depth = np.load(motion / 'depth_frames.npy', allow_pickle=False)
    seg = np.load(motion / 'renderer_segmentation_raw.npy', allow_pickle=False)
    poses = np.load(motion / 'camera_poses_cv.npy', allow_pickle=False)
    K = np.asarray(result['camera_pose_check']['K_pixel_index'], dtype=np.float64)
    if rad.shape != (4, 240, 320, 3) or depth.shape != (4, 240, 320) or seg.shape != depth.shape or poses.shape != (4, 4, 4):
        raise RuntimeError(f'{name}: unexpected saved pilot-array contract')
    H, W = depth.shape[1:]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    rows = []
    for i in range(3):
        flow, valid0 = compute_se3_flow(depth[i], seg[i], K, poses[i], poses[i + 1])
        valid = (valid0 & (xx + flow[0] >= 0) & (xx + flow[0] < W)
                 & (yy + flow[1] >= 0) & (yy + flow[1] < H) & np.isfinite(flow).all(axis=0))
        mag = np.linalg.norm(flow, axis=0)
        luma = 0.2126 * rad[i, :, :, 0] + 0.7152 * rad[i, :, :, 1] + 0.0722 * rad[i, :, :, 2]
        gy, gx = np.gradient(luma)
        grad = np.hypot(gx, gy)
        motion_mask = valid & (mag > motion_threshold)
        texture = valid & (grad > texture_threshold)
        both = motion_mask & texture
        q = np.percentile(mag[valid], [0, 10, 50, 90, 99, 100]) if valid.any() else np.full(6, np.nan)
        rows.append({
            'interval_s': [i * 0.01, (i + 1) * 0.01],
            'valid_inframe_pixels': int(valid.sum()),
            'gt_motion_gt_threshold_pixels': int(motion_mask.sum()),
            'texture_gt_threshold_pixels': int(texture.sum()),
            'texture_and_motion_pixels': int(both.sum()),
            'texture_and_motion_fraction_of_full_frame': float(both.sum() / (H * W)),
            'flow_magnitude_px_quantiles_0_10_50_90_99_100': [float(x) for x in q],
        })
    return {
        'name': name,
        'motion_dir': str(motion),
        'input_sha256': {p.name: sha256(p) for p in [motion / 'motion_result.json', motion / 'radiance_frames.npy', motion / 'depth_frames.npy', motion / 'renderer_segmentation_raw.npy', motion / 'camera_poses_cv.npy']},
        'camera_trajectory': result['trajectory'],
        'flow_check': result['flow'],
        'intervals': rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--criteria', type=Path, required=True)
    p.add_argument('--pilot', action='append', nargs=2, metavar=('NAME', 'MOTION_DIR'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    criteria_path = a.criteria.resolve()
    criteria = json.loads(criteria_path.read_text())
    motion_threshold = float(criteria['flow']['motion_mask'].split('> ')[1].split(' ')[0])
    texture_threshold = float(criteria['texture']['texture_mask'].split('> ')[1].split(' ')[0])
    pilots = [audit(name, Path(path).resolve(), motion_threshold, texture_threshold) for name, path in a.pilot]
    out = {
        'schema': 'stage7_camera_motion_geometry_texture_only_audit_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'criteria_path': str(criteria_path), 'criteria_sha256': sha256(criteria_path),
        'motion_threshold_px_per_10ms': motion_threshold,
        'texture_gradient_threshold_scene_linear_radiance_per_px': texture_threshold,
        'scope': 'Geometry/texture diagnostics only; no EventGenesis/V2E values consumed and no training/model selection. These pilot values describe candidate views only.',
        'pilots': pilots,
    }
    out_path = a.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({name['name']: name['intervals'] for name in pilots}, indent=2))

if __name__ == '__main__':
    main()
