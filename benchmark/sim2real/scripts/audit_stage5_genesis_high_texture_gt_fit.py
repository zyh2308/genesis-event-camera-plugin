#!/usr/bin/env python3
"""CPU-only label, warp, and event-support audit for the preregistered Stage-5 fit.

The training window and independent validation rollout come from the checked-in
config. This audit never loads a model and does not select a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.5, 0.9, 0.99]).astype(float).tolist() if values.size else []


def photo_residual(previous: np.ndarray, following: np.ndarray, flow: np.ndarray,
                   mask: np.ndarray) -> float | None:
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x, map_y = xx + flow[0], yy + flow[1]
    inside = ((map_x >= 0) & (map_x <= width - 1) &
              (map_y >= 0) & (map_y <= height - 1))
    use = mask & inside
    if not use.any():
        return None
    warped = cv2.remap(following.astype(np.float32), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return float(np.abs(warped[use] - previous[use]).mean())


def one_window(group: h5py.Group, window_index: int, event_ts: np.ndarray,
               events_xy: np.ndarray, times: np.ndarray, flow: np.ndarray,
               valid: np.ndarray, gray: np.ndarray, display: np.ndarray,
               target_link_ids: list[int], run_lk: bool) -> dict:
    endpoint = window_index * 3
    left, right = float(times[endpoint]), float(times[endpoint + 3])
    lo = int(np.searchsorted(event_ts, left, side="right"))
    hi = int(np.searchsorted(event_ts, right, side="right"))
    chunk = events_xy[lo:hi]
    if len(chunk) and not (np.all(chunk[:, 0] > left) and np.all(chunk[:, 0] <= right)):
        raise RuntimeError(f"event-window boundary mismatch at {left, right}")
    active = np.zeros(valid.shape[1:], dtype=bool)
    if len(chunk):
        if not np.allclose(chunk[:, 1:3], np.rint(chunk[:, 1:3]), atol=1e-6, rtol=0):
            raise RuntimeError("EventGenesis x/y coordinates are not integer sensor pixels")
        xy = chunk[:, 1:3].astype(np.int64)
        if np.any(xy[:, 0] < 0) or np.any(xy[:, 0] >= active.shape[1]) or np.any(xy[:, 1] < 0) or np.any(xy[:, 1] >= active.shape[0]):
            raise RuntimeError("EventGenesis coordinates fall outside the registered sensor")
        active[xy[:, 1], xy[:, 0]] = True
    gt = flow[window_index]
    gt_mag = np.linalg.norm(gt, axis=0)
    valid_i = valid[window_index]
    yy, xx = np.mgrid[:valid_i.shape[0], :valid_i.shape[1]].astype(np.float32)
    gx = cv2.Sobel(gray[endpoint], cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray[endpoint], cv2.CV_32F, 0, 1, ksize=3) / 8.0
    textured = np.hypot(gx, gy) > (1.0 / 255.0)
    moving = gt_mag >= 0.5
    moving_texture = valid_i & moving & textured
    active_moving_texture = moving_texture & active

    common_inbounds = valid_i.copy()
    for candidate in (np.zeros_like(gt), gt, -gt):
        map_x, map_y = xx + candidate[0], yy + candidate[1]
        common_inbounds &= ((map_x >= 0) & (map_x <= valid_i.shape[1] - 1) &
                            (map_y >= 0) & (map_y <= valid_i.shape[0] - 1))
    warp_mask = common_inbounds & moving_texture
    warp_regions = {
        "moving_texture_valid": warp_mask,
        "event_active_moving_texture_valid": warp_mask & active,
    }
    candidates = {"zero": np.zeros_like(gt), "gt_forward": gt, "reverse": -gt}
    warp = {}
    for region_name, region in warp_regions.items():
        warp[region_name] = {
            "pixels": int(region.sum()),
            "mae_proxy_luma": {
                name: photo_residual(gray[endpoint], gray[endpoint + 3], candidate, region)
                for name, candidate in candidates.items()
            },
        }

    # A separate image-motion check: LK tracks are measured from previous to
    # next endpoint and compared with forward source-pixel GT at source pixels.
    target_mask = np.isin(group["segmentation_link_idx"][endpoint],
                          np.asarray(target_link_ids, dtype=np.int32)) if run_lk else None
    points = (cv2.goodFeaturesToTrack(display[endpoint], maxCorners=500, qualityLevel=0.01,
                                     minDistance=3, mask=target_mask.astype(np.uint8) * 255,
                                     blockSize=5) if run_lk else None)
    lk = {"source_target_features": 0, "tracked_features": 0,
          "median_lk_displacement_xy_px": None,
          "median_epe_vs_gt_forward_px": None,
          "median_epe_vs_reversed_gt_px": None,
          "run": bool(run_lk)}
    if points is not None and len(points):
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            display[endpoint], display[endpoint + 3], points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if next_points is not None and status is not None:
            p0 = points.reshape(-1, 2)
            p1 = next_points.reshape(-1, 2)
            keep = status.reshape(-1).astype(bool)
            keep &= (p0[:, 0] >= 0) & (p0[:, 0] < valid_i.shape[1])
            keep &= (p0[:, 1] >= 0) & (p0[:, 1] < valid_i.shape[0])
            src_x = np.clip(np.rint(p0[:, 0]).astype(int), 0, valid_i.shape[1] - 1)
            src_y = np.clip(np.rint(p0[:, 1]).astype(int), 0, valid_i.shape[0] - 1)
            keep &= valid_i[src_y, src_x]
            keep &= (p1[:, 0] >= 0) & (p1[:, 0] < valid_i.shape[1])
            keep &= (p1[:, 1] >= 0) & (p1[:, 1] < valid_i.shape[0])
            if keep.any():
                displacement = p1[keep] - p0[keep]
                sampled = gt[:, src_y[keep], src_x[keep]].T
                lk.update({
                    "source_target_features": int(len(points)),
                    "tracked_features": int(keep.sum()),
                    "median_lk_displacement_xy_px": np.median(displacement, axis=0).astype(float).tolist(),
                    "median_epe_vs_gt_forward_px": float(np.median(np.linalg.norm(displacement - sampled, axis=1))),
                    "median_epe_vs_reversed_gt_px": float(np.median(np.linalg.norm(displacement + sampled, axis=1))),
                })
    elif run_lk:
        lk["source_target_features"] = 0

    valid_magnitude = gt_mag[valid_i]
    return {
        "window_index": int(window_index),
        "sample_id": f"{group.name.rsplit('/', 1)[-1]}/dt1_{endpoint * 10:03d}",
        "endpoint_indices": [int(endpoint), int(endpoint + 3)],
        "window_seconds": [left, right],
        "event_count_independently_recounted": int(len(chunk)),
        "valid_pixels": int(valid_i.sum()),
        "event_active_valid_pixels": int((valid_i & active).sum()),
        "event_active_valid_coverage": float((valid_i & active).sum() / max(int(valid_i.sum()), 1)),
        "gt_motion_ge_0_5px_valid_pixels": int((valid_i & moving).sum()),
        "texture_gradient_gt_1_over_255_valid_pixels": int((valid_i & textured).sum()),
        "moving_texture_valid_pixels": int(moving_texture.sum()),
        "event_active_moving_texture_pixels": int(active_moving_texture.sum()),
        "gt_magnitude_px_per_30ms_valid_p50_p90_p99": percentile(valid_magnitude),
        "warp": warp,
        "lk_direction_crosscheck": lk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    bank = args.bank_root.resolve()
    bank_path = bank / "shared_trajectory.h5"
    manifest_path = bank / "bank_manifest.json"
    split_path = bank / "registered_split.json"
    manifest, split = json.loads(manifest_path.read_text()), json.loads(split_path.read_text())
    if manifest.get("schema") != config["shared_bank"]["schema"]:
        raise RuntimeError("shared-bank schema differs from frozen audit config")
    if manifest.get("bank_sha256") != config["shared_bank"]["sha256"]:
        raise RuntimeError("shared-bank SHA in frozen config does not match manifest")
    actual_bank_sha = sha256(bank_path)
    if actual_bank_sha != manifest["bank_sha256"]:
        raise RuntimeError("shared HDF5 does not match bank manifest")
    rows = {row["id"]: row for row in split}
    train_cfg = config["training_sample"]
    val_cfg = config["independent_validation"]
    for item, expected_split in ((train_cfg, "train"), (val_cfg, "validation")):
        row = rows.get(item["rollout_id"])
        if row is None or row.get("split") != expected_split or row.get("stratum") != item["stratum"]:
            raise RuntimeError(f"registered rollout/split/stratum mismatch: {item}")

    results = {"config": str(args.config.resolve()), "config_sha256": sha256(args.config.resolve()),
               "bank_path": str(bank_path), "bank_sha256": actual_bank_sha,
               "observed_utc": datetime.now(timezone.utc).isoformat(),
               "audit_scope": "CPU-only, EventGenesis support only; no model loaded; no training or checkpoint selection",
               "generator_pairing_status": "not_pairable_on_full_bank_until_v2e_events_are_manifest-bound to the same bank",
               "rollouts": {}}
    with h5py.File(bank_path, "r") as shared:
        for rollout_id, window_indices in (
            (train_cfg["rollout_id"], [int(train_cfg["window_index"])]),
            (val_cfg["rollout_id"], list(range(int(val_cfg["window_count"])))),
        ):
            group = shared[f"episodes/{rollout_id}"]
            times = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
            flow = np.asarray(group["flow_forward_30ms_px"], dtype=np.float32)
            valid = np.asarray(group["flow_valid_30ms"], dtype=np.uint8).astype(bool)
            signal = np.asarray(group["linearized_display_rgb_proxy"], dtype=np.float32)
            display = np.asarray(group["v2e_display_luma_u8"], dtype=np.uint8)
            events = np.asarray(group["eventgenesis_events_txyp"], dtype=np.float64)
            if flow.shape != (66, 2, 260, 346) or valid.shape != (66, 260, 346):
                raise RuntimeError(f"unexpected flow/validity shape for {rollout_id}")
            if times.shape != (201,) or not np.allclose(np.diff(times), .01, atol=1e-7, rtol=0):
                raise RuntimeError(f"unexpected endpoint cadence for {rollout_id}")
            if not np.isfinite(flow).all() or not np.isfinite(signal).all():
                raise RuntimeError(f"non-finite label or image data for {rollout_id}")
            if events.ndim != 2 or events.shape[1] != 4 or not np.isfinite(events).all():
                raise RuntimeError(f"invalid EventGenesis events for {rollout_id}")
            if len(events) and np.any(np.diff(events[:, 0]) < 0):
                raise RuntimeError(f"EventGenesis timestamps are not sorted for {rollout_id}")
            if signal.shape != (201, 260, 346, 3) or display.shape != (201, 260, 346):
                raise RuntimeError(f"unexpected endpoint image shape for {rollout_id}")
            gray = signal @ np.asarray([.2126, .7152, .0722], dtype=np.float32)
            raw = events.copy()
            if len(raw) and (np.any(raw[:, 0] < times[0] - 1e-9) or
                             np.any(raw[:, 0] > times[-1] + 1e-5)):
                raise RuntimeError(f"EventGenesis timestamps outside rollout for {rollout_id}")
            target_ids_attr = group.attrs["target_object_segmentation_link_ids"]
            if isinstance(target_ids_attr, bytes):
                target_ids_attr = target_ids_attr.decode("utf-8")
            target_link_ids = json.loads(str(target_ids_attr))
            if rollout_id == train_cfg["rollout_id"]:
                index = int(train_cfg["window_index"])
                if not np.allclose([times[index * 3], times[index * 3 + 3]],
                                   train_cfg["window_seconds"], atol=1e-9, rtol=0):
                    raise RuntimeError("the preregistered training window endpoints do not match this bank")
            per_window = [one_window(group, index, raw[:, 0], raw, times, flow, valid,
                                     gray, display, [int(x) for x in target_link_ids],
                                     run_lk=index == int(train_cfg["window_index"]))
                          for index in window_indices]
            active_coverages = [row["event_active_valid_coverage"] for row in per_window]
            motion_texture = [row["moving_texture_valid_pixels"] for row in per_window]
            active_motion_texture = [row["event_active_moving_texture_pixels"] for row in per_window]
            warp_rows = [row["warp"]["event_active_moving_texture_valid"] for row in per_window]
            results["rollouts"][rollout_id] = {
                "split": rows[rollout_id]["split"],
                "all_windows_audited": len(window_indices) == 66,
                "windows": per_window,
                "summary": {
                    "window_count": len(per_window),
                    "mean_event_active_valid_coverage": float(np.mean(active_coverages)),
                    "median_event_active_valid_coverage": float(np.median(active_coverages)),
                    "windows_with_nonempty_moving_texture": int(sum(x > 0 for x in motion_texture)),
                    "moving_texture_valid_pixels_mean": float(np.mean(motion_texture)),
                    "event_active_moving_texture_pixels_mean": float(np.mean(active_motion_texture)),
                    "windows_with_nonempty_event_active_moving_texture": int(sum(x > 0 for x in active_motion_texture)),
                    "active_moving_texture_warp_pixels_total": int(sum(x["pixels"] for x in warp_rows)),
                    "event_count_total": int(sum(x["event_count_independently_recounted"] for x in per_window)),
                },
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "config_sha256": results["config_sha256"], "bank_sha256": actual_bank_sha,
                      "rollout_summaries": {key: value["summary"] for key, value in results["rollouts"].items()},
                      "train_window": results["rollouts"][train_cfg["rollout_id"]]["windows"][0]}, indent=2))


if __name__ == "__main__":
    main()
