#!/usr/bin/env python3
"""Fixed-rule observability audit for the one ReplicaCAD apt_1 motion pilot."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def event_support(events, shape, t0, t1, time_scale):
    h, w = shape
    t = events[:, 0].astype(np.float64) * time_scale
    keep = (t > t0) & (t <= t1)
    selected = events[keep]
    support = np.zeros((h, w), dtype=bool)
    if len(selected):
        x = np.rint(selected[:, 1]).astype(np.int64)
        y = np.rint(selected[:, 2]).astype(np.int64)
        if np.any((x < 0) | (x >= w) | (y < 0) | (y >= h)):
            raise RuntimeError('event coordinates out of range')
        support[y, x] = True
    return selected, support


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera-motion-dir', type=Path, required=True)
    p.add_argument('--v2e-h5', type=Path, required=True)
    p.add_argument('--criteria', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    warp_path = Path(__file__).resolve().parents[5] / 'genesis_event_plugin/physics_warp.py'
    spec = importlib.util.spec_from_file_location('stage7_physics_warp', warp_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load physical flow implementation: {warp_path}')
    warp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warp_module)
    compute_se3_flow = warp_module.compute_se3_flow
    motion, v2e_path, criteria_path = a.camera_motion_dir.resolve(), a.v2e_h5.resolve(), a.criteria.resolve()
    criteria = json.loads(criteria_path.read_text())
    radiance = np.load(motion / 'radiance_frames.npy', allow_pickle=False)
    depths = np.load(motion / 'depth_frames.npy', allow_pickle=False)
    segmentations = np.load(motion / 'renderer_segmentation_raw.npy', allow_pickle=False)
    poses = np.load(motion / 'camera_poses_cv.npy', allow_pickle=False)
    eg = np.load(motion / 'eventgenesis_events_seconds_xy_p.npy', allow_pickle=False)
    with h5py.File(v2e_path, 'r') as h5:
        v2e = np.asarray(h5['events'], dtype=np.uint32)
    motion_result = json.loads((motion / 'motion_result.json').read_text())
    K = np.asarray(motion_result['camera_pose_check']['K_pixel_index'], dtype=np.float64)
    if radiance.shape != (4,240,320,3) or depths.shape != (4,240,320) or poses.shape != (4,4,4):
        raise RuntimeError('saved camera-motion arrays do not satisfy the registered four-endpoint contract')
    if eg.ndim != 2 or eg.shape[1] != 4 or v2e.ndim != 2 or v2e.shape[1] != 4:
        raise RuntimeError('event matrices must be [N,4]')
    h, w = depths.shape[1:]
    intervals = []
    started = time.monotonic()
    motion_threshold_px = float(criteria['flow']['motion_mask'].split('> ')[1].split(' ')[0])
    texture_threshold = float(criteria['texture']['texture_mask'].split('> ')[1].split(' ')[0])
    for i in range(3):
        t0, t1 = i * 0.01, (i + 1) * 0.01
        flow, valid0 = compute_se3_flow(depths[i], segmentations[i], K, poses[i], poses[i+1])
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        inside = ((xx + flow[0] >= 0) & (xx + flow[0] < w)
                  & (yy + flow[1] >= 0) & (yy + flow[1] < h))
        valid = valid0 & inside & np.isfinite(flow).all(axis=0)
        mag = np.linalg.norm(flow, axis=0)
        motion_mask = valid & (mag > motion_threshold_px)
        luma = 0.2126*radiance[i,:,:,0] + 0.7152*radiance[i,:,:,1] + 0.0722*radiance[i,:,:,2]
        gy, gx = np.gradient(luma)
        grad = np.hypot(gx, gy)
        texture_mask = valid & (grad > texture_threshold)
        eg_chunk, eg_mask = event_support(eg, (h,w), t0, t1, 1.0)
        v2e_chunk, v2e_mask = event_support(v2e, (h,w), t0, t1, 1e-6)
        union = eg_mask | v2e_mask
        texture_motion = texture_mask & motion_mask
        q = np.percentile(mag[valid], [0,10,50,90,99,100]) if valid.any() else np.full(6, np.nan)
        intervals.append({
            'interval_s': [t0,t1], 'flow_valid_inframe_pixels': int(valid.sum()),
            'gt_motion_gt_1px_pixels': int(motion_mask.sum()),
            'texture_gt_0_005_pixels': int(texture_mask.sum()),
            'texture_and_motion_pixels': int(texture_motion.sum()),
            'flow_magnitude_px_quantiles_0_10_50_90_99_100': [float(x) for x in q],
            'eventgenesis_event_count': int(len(eg_chunk)), 'eventgenesis_active_pixels': int(eg_mask.sum()),
            'v2e_event_count': int(len(v2e_chunk)), 'v2e_active_pixels': int(v2e_mask.sum()),
            'shared_union_active_pixels': int(union.sum()),
            'coverage_fraction_over_valid_frame': float((union & valid).sum()/max(1,valid.sum())),
            'coverage_fraction_over_gt_motion_gt_1px': float((union & motion_mask).sum()/max(1,motion_mask.sum())),
            'coverage_fraction_over_texture_and_motion': float((union & texture_motion).sum()/max(1,texture_motion.sum())),
        })
    out = {
        'schema':'stage7_camera_motion_observability_audit_v1',
        'status':'completed_descriptive_single_path_audit',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'criteria':{'path':str(criteria_path),'sha256':sha256(criteria_path),'values':criteria},
        'inputs':{
            'camera_motion_dir':str(motion),
            'radiance_frames_sha256':sha256(motion/'radiance_frames.npy'),
            'depth_frames_sha256':sha256(motion/'depth_frames.npy'),
            'segmentation_sha256':sha256(motion/'renderer_segmentation_raw.npy'),
            'camera_poses_sha256':sha256(motion/'camera_poses_cv.npy'),
            'eventgenesis_events_sha256':sha256(motion/'eventgenesis_events_seconds_xy_p.npy'),
            'v2e_h5':str(v2e_path),'v2e_h5_sha256':sha256(v2e_path),
            'flow_implementation_sha256':sha256(Path(__file__).resolve().parents[5]/'genesis_event_plugin/physics_warp.py'),
            'K_pixel_index':K.tolist(),
        },
        'scope':'one camera-only 30 ms interval divided into three aligned 10 ms windows; not a training-bank acceptance result',
        'intervals':intervals,
        'elapsed_s':time.monotonic()-started,
        'limitations':['single camera pose/path and static layout only','renderer segmentation remains raw and unmapped','radiometric calibration unverified','coverage metrics are descriptive and do not select a scene or generator'],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':out['status'],'intervals':[{k:v for k,v in row.items() if k not in ('flow_magnitude_px_quantiles_0_10_50_90_99_100',)} for row in intervals]},indent=2))

if __name__ == '__main__': main()
