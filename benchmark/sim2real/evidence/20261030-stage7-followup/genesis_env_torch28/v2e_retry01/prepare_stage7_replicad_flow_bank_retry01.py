#!/usr/bin/env python3
"""Bind paired ReplicaCAD events to shared, visibility-checked physical flow labels."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def label_digest(flow: np.ndarray, valid: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(flow, dtype=np.float32).view(np.uint8))
    h.update(np.ascontiguousarray(valid, dtype=np.uint8).view(np.uint8))
    return h.hexdigest()


def visibility_checked_flow(depth0, depth1, seg0, K, pose0, pose1, compute_se3_flow):
    flow, geom_valid = compute_se3_flow(depth0, seg0, K, pose0, pose1)
    h, w = depth0.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    tx, ty = xx + flow[0], yy + flow[1]
    inside = geom_valid & (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1)
    x_near = np.clip(np.rint(tx).astype(np.int64), 0, w - 1)
    y_near = np.clip(np.rint(ty).astype(np.int64), 0, h - 1)

    z0 = np.asarray(depth0, dtype=np.float64)
    points0 = np.stack(((xx - K[0, 2]) * z0 / K[0, 0],
                        (yy - K[1, 2]) * z0 / K[1, 1], z0), axis=-1)
    relative = np.linalg.inv(np.asarray(pose1, dtype=np.float64)) @ np.asarray(pose0, dtype=np.float64)
    points1 = points0 @ relative[:3, :3].T + relative[:3, 3]
    projected_z = points1[..., 2]
    sampled_z = np.asarray(depth1, dtype=np.float32)[y_near, x_near]
    tolerance = np.maximum(0.02, 0.02 * projected_z)
    visible = inside & np.isfinite(projected_z) & (projected_z > 1e-6)
    visible &= np.isfinite(sampled_z) & (sampled_z > 1e-6)
    visible &= np.abs(sampled_z - projected_z) <= tolerance
    visible &= np.isfinite(flow).all(axis=0)
    result = np.asarray(flow, dtype=np.float32).copy()
    result[:, ~visible] = 0.0
    return result, visible.astype(np.uint8), {
        "geometry_valid_pixels": int(geom_valid.sum()),
        "in_frame_pixels": int(inside.sum()),
        "visible_pixels": int(visible.sum()),
        "occluded_or_invalid_pixels": int((~visible).sum()),
        "depth_visibility_tolerance_m": "max(0.02 m, 0.02 * projected target z)",
    }


def load_generator_events(arm: str, motion_dir: Path, v2e_h5: Path):
    if arm == "eventgenesis":
        raw = np.load(motion_dir / "eventgenesis_events_seconds_xy_p.npy", allow_pickle=False)
        events = np.asarray(raw, dtype=np.float64)
    else:
        with h5py.File(v2e_h5, "r") as h5:
            raw = np.asarray(h5["events"], dtype=np.uint32)
        events = np.stack((raw[:, 0].astype(np.float64) * 1e-6,
                           raw[:, 1].astype(np.float64), raw[:, 2].astype(np.float64),
                           np.where(raw[:, 3] > 0, 1.0, -1.0)), axis=1)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"{arm} raw events must be [N,4], got {events.shape}")
    if not np.isfinite(events).all():
        raise ValueError(f"{arm} events contain nonfinite values")
    if len(events) and np.any(np.diff(events[:, 0]) < 0):
        raise ValueError(f"{arm} events are not timestamp-sorted")
    h, w = np.load(motion_dir / "depth_frames.npy", mmap_mode="r").shape[1:]
    keep = (events[:, 0] > 0.0) & (events[:, 0] <= 0.03)
    events = events[keep].copy()
    if len(events):
        xy = events[:, 1:3]
        if np.any((xy[:, 0] < 0) | (xy[:, 0] >= w) | (xy[:, 1] < 0) | (xy[:, 1] >= h)):
            raise ValueError(f"{arm} event coordinate outside {w}x{h}")
        if not np.isin(events[:, 3], (-1, 1)).all():
            raise ValueError(f"{arm} polarity must be mapped to {-1,+1}")
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--old-run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite bank: {args.output_root}")
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo / "benchmark/sim2real/scripts"))
    from prepare_steflow_stage0 import gaussian_count_representation
    warp_path = repo / "genesis_event_plugin/physics_warp.py"
    spec = importlib.util.spec_from_file_location("stage7_physics_warp", warp_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned physical flow implementation: {warp_path}")
    warp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warp_module)
    compute_se3_flow = warp_module.compute_se3_flow

    registration_path = args.registration.resolve()
    registration = json.loads(registration_path.read_text())
    run_root, old_root = args.run_root.resolve(), args.old_run_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True)
    entries, coverage_rows = [], []
    scene_specs = {
        "v3_sc0_staging_00": ("train", "20261030-stage7-replicad-v3_sc0_staging_00-fast-run14",
                              "20261030-stage7-replicad-v3-sc0-v2e-cpu-run17"),
        "v3_sc1_staging_00": ("train", "20261030-stage7-replicad-v3_sc1_staging_00-fast-run15",
                              "20261030-stage7-replicad-v3-sc1-v2e-cpu-run20"),
        "v3_sc2_staging_00": ("validation", "20261030-stage7-replicad-v3_sc2_staging_00-fast-run16",
                              "20261030-stage7-replicad-v3-sc2-v2e-gpu-run21"),
    }
    for scene, (split, old_genesis_name, old_v2e_name) in scene_specs.items():
        path_specs = [
            ("x_060mm", old_root / old_genesis_name / "render_output/camera_motion_30ms",
             old_root / old_v2e_name / "artifacts/v2e_events.h5"),
            ("y_040mm", run_root / scene / "camera_motion_30ms",
             run_root / "v2e_retry01" / scene / "v2e_gpu/events/v2e_events.h5"),
        ]
        for path_id, motion_dir, v2e_h5 in path_specs:
            rollout = f"{scene.removesuffix('_staging_00')}_{path_id}"
            required = [motion_dir / name for name in (
                "camera_poses_cv.npy", "radiance_frames.npy", "depth_frames.npy",
                "renderer_segmentation_raw.npy", "gt_flow_px_t0_to_t30ms.npy",
                "eventgenesis_events_seconds_xy_p.npy", "motion_result.json",
            )] + [v2e_h5]
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"{rollout} missing required inputs: {missing}")
            poses = np.load(motion_dir / "camera_poses_cv.npy", allow_pickle=False)
            radiance = np.load(motion_dir / "radiance_frames.npy", allow_pickle=False)
            depth = np.load(motion_dir / "depth_frames.npy", allow_pickle=False)
            seg = np.load(motion_dir / "renderer_segmentation_raw.npy", allow_pickle=False)
            result = json.loads((motion_dir / "motion_result.json").read_text())
            K = np.asarray(result["camera_pose_check"]["K_pixel_index"], dtype=np.float64)
            independent_flow_p99 = float(result["flow"]["p99_analytic_endpoint_error_px"])
            if not np.isfinite(independent_flow_p99) or independent_flow_p99 > 0.08:
                raise RuntimeError(f"{rollout}: independent camera pinhole check failed: p99={independent_flow_p99}")
            if radiance.shape != (4, 240, 320, 3) or depth.shape != (4, 240, 320):
                raise ValueError(f"{rollout}: unexpected radiance/depth shape {radiance.shape}/{depth.shape}")
            if seg.shape != depth.shape or poses.shape != (4, 4, 4):
                raise ValueError(f"{rollout}: mismatched segmentation/pose shapes")
            flow_raw, geom_valid = compute_se3_flow(depth[0], seg[0], K, poses[0], poses[-1])
            saved_flow = np.load(motion_dir / "gt_flow_px_t0_to_t30ms.npy", allow_pickle=False)
            if saved_flow.shape != (2, 240, 320) or not np.allclose(flow_raw, saved_flow, atol=2e-5, rtol=0):
                raise RuntimeError(f"{rollout}: saved 30 ms flow differs from independent recomputation")
            flow, valid, vis_record = visibility_checked_flow(
                depth[0], depth[-1], seg[0], K, poses[0], poses[-1], compute_se3_flow
            )
            label_sha = label_digest(flow, valid)
            labels_dir = out_root / "shared_labels"
            labels_dir.mkdir(exist_ok=True)
            shared_label_path = labels_dir / f"{rollout}.npz"
            np.savez_compressed(shared_label_path, flow_gt=flow, flow_valid=valid,
                                sample_id=np.asarray(f"{rollout}/dt30"), label_sha256=np.asarray(label_sha))
            gen_events = {}
            gen_support = {}
            for arm in ("eventgenesis", "v2e"):
                events = load_generator_events(arm, motion_dir, v2e_h5 if arm == "v2e" else v2e_h5)
                representation = gaussian_count_representation(events, 240, 320)
                gen_events[arm] = events
                gen_support[arm] = np.any(representation != 0, axis=(0, 3))
                arm_dir = out_root / arm
                arm_dir.mkdir(exist_ok=True)
                artifact = arm_dir / f"{rollout}.npz"
                np.savez_compressed(
                    artifact, events=representation, flow_gt=flow, flow_valid=valid,
                    sample_ids=np.asarray(f"{rollout}/dt30"), label_sha256=np.asarray(label_sha),
                    event_count=np.asarray(len(events), dtype=np.int64),
                    source_resolution_hw=np.asarray([240, 320], dtype=np.int32),
                    flow_unit=np.asarray("pixels per 30ms endpoint interval"),
                )
                entries.append({
                    "generator": arm, "rollout_id": rollout, "scene_family": scene,
                    "path_id": path_id, "split": split, "sample_id": f"{rollout}/dt30",
                    "path": str(artifact), "sha256": sha256(artifact),
                    "size_bytes": artifact.stat().st_size, "samples": 1,
                    "event_count_in_(0,30ms]": int(len(events)),
                    "active_pixels_from_registered_five_bin_tensor": int(gen_support[arm].sum()),
                    "shared_label_path": str(shared_label_path),
                    "shared_label_sha256": label_sha,
                    "source_motion_dir": str(motion_dir),
                    "source_motion_sha256": sha256(motion_dir / "radiance_frames.npy"),
                    "source_v2e_h5": str(v2e_h5),
                    "source_v2e_h5_sha256": sha256(v2e_h5),
                    "event_only_inference": True,
                })
            common_active = gen_support["eventgenesis"] | gen_support["v2e"]
            valid_mask = valid.astype(bool)
            magnitude = np.hypot(flow[0], flow[1])
            motion_mask = valid_mask & (magnitude >= 0.5)
            luma = 0.2126 * radiance[0, ..., 0] + 0.7152 * radiance[0, ..., 1] + 0.0722 * radiance[0, ..., 2]
            gy, gx = np.gradient(luma)
            textured_motion = valid_mask & (np.hypot(gx, gy) > 0.005) & motion_mask
            coverage_rows.append({
                "rollout_id": rollout, "scene_family": scene, "path_id": path_id,
                "split": split, "flow_unit": "pixels per 30ms endpoint interval",
                "visibility": vis_record, "flow_valid_pixels": int(valid_mask.sum()),
                "stored_pilot_independent_flow_p99_px": independent_flow_p99,
                "gt_motion_ge_0_5px_pixels": int(motion_mask.sum()),
                "texture_and_motion_pixels": int(textured_motion.sum()),
                "flow_magnitude_px_quantiles_valid": np.percentile(magnitude[valid_mask], [0, 10, 50, 90, 99, 100]).tolist() if valid_mask.any() else [],
                "generator_event_count_and_active_pixels": {
                    arm: {"events": int(len(gen_events[arm])), "active_pixels": int(gen_support[arm].sum())}
                    for arm in gen_events
                },
                "common_union_active_pixels": int(common_active.sum()),
                "common_union_fraction_valid": float((common_active & valid_mask).sum() / max(1, valid_mask.sum())),
                "common_union_fraction_motion_ge_0_5px": float((common_active & motion_mask).sum() / max(1, motion_mask.sum())),
                "common_union_fraction_texture_and_motion": float((common_active & textured_motion).sum() / max(1, textured_motion.sum())),
                "shared_label_sha256": label_sha,
            })
    manifest = {
        "schema": "stage7_replicad_multiaxis_flow_training_bank_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "frozen_simulator_core": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
        "flow_unit": "pixels per 30ms endpoint interval",
        "event_representation": "pinned STE equal-event-count five-bin Gaussian-count representation, per polarity, from raw events in (0,30ms]",
        "event_window": "left-open, right-closed (0,0.03] seconds; V2E microseconds converted to seconds",
        "label_contract": "recomputed from source renderer z-depth, pixel-index intrinsics and endpoint poses; validity intersects in-frame endpoints and target-depth visibility max(0.02 m, 2% projected z)",
        "split_unit": "complete layout family: sc0/sc1 train; sc2 validation",
        "real_data_used": False,
        "entries": entries,
        "coverage_audit": coverage_rows,
        "limitations": registration["known_limits"] + [
            "small pilot has four train and two validation physical rollouts, one 30 ms sample each",
            "static-scene camera motion only; no dynamic-object ground truth",
        ],
    }
    manifest_path = out_root / "training_bank_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": manifest["status"], "manifest": str(manifest_path),
        "rollouts": len(coverage_rows), "paired_entries": len(entries),
        "train_rollouts": sum(row["split"] == "train" for row in coverage_rows),
        "validation_rollouts": sum(row["split"] == "validation" for row in coverage_rows),
        "coverage": coverage_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
