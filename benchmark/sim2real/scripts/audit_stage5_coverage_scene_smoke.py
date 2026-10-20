#!/usr/bin/env python3
"""Audit one paired Stage-5 scene smoke without treating it as a full bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np

from prepare_steflow_stage0 import gaussian_count_representation
from stage4_v2e_artifacts import locate_shard_artifacts


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def active_mask(events: np.ndarray, left_s: float, timestamp_scale: float) -> np.ndarray:
    if not len(events):
        return np.zeros((260, 346), dtype=bool)
    local = events.copy()
    local[:, 0] = (local[:, 0] * timestamp_scale - left_s)
    local[:, 3] = np.where(local[:, 3] > 0, 1.0, -1.0)
    return np.any(gaussian_count_representation(local, 260, 346) != 0, axis=(0, 3))


def photo_warp(gray_previous: np.ndarray, gray_next: np.ndarray, flow: np.ndarray, mask: np.ndarray) -> float | None:
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    map_x = xx + flow[0]
    map_y = yy + flow[1]
    inside = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    use = mask & inside
    if not use.any():
        return None
    warped = cv2.remap(gray_next.astype(np.float32), map_x.astype(np.float32), map_y.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return float(np.abs(warped[use] - gray_previous[use]).mean())


def analytic_warp_fixture() -> dict:
    h, w = 64, 96
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    def texture(x, y):
        return (0.5 + 0.19 * np.sin(0.17 * x) + 0.16 * np.cos(0.13 * y)
                + 0.1 * np.sin(0.08 * (x + 1.7 * y))).astype(np.float32)

    dx, dy = 2.25, -1.5
    previous = texture(xx, yy)
    following = texture(xx - dx, yy - dy)
    valid = (xx + dx >= 1) & (xx + dx < w - 1) & (yy + dy >= 1) & (yy + dy < h - 1)
    true_flow = np.stack((np.full_like(xx, dx), np.full_like(yy, dy)))
    zero = np.zeros_like(true_flow)
    reverse = -true_flow
    residuals = {
        "zero": photo_warp(previous, following, zero, valid),
        "gt": photo_warp(previous, following, true_flow, valid),
        "reverse": photo_warp(previous, following, reverse, valid),
    }
    passed = residuals["gt"] < 0.5 * residuals["zero"] and residuals["gt"] < 0.5 * residuals["reverse"]
    if not passed:
        raise RuntimeError(f"analytic texture translation does not validate warp direction: {residuals}")
    return {"status": "pass", "translation_xy_px": [dx, dy], "valid_pixels": int(valid.sum()),
            "mae": residuals, "direction_contract": "next(x+u,y+v) is compared with previous(x,y)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--v2e-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank, v2e, output = args.bank_root.resolve(), args.v2e_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    bank_path, bank_manifest_path = bank / "shared_trajectory.h5", bank / "bank_manifest.json"
    split_path, v2e_manifest_path = bank / "registered_split.json", v2e / "v2e_complete_manifest.json"
    bank_manifest = json.loads(bank_manifest_path.read_text())
    split = json.loads(split_path.read_text())
    v2e_manifest = json.loads(v2e_manifest_path.read_text())
    if bank_manifest.get("schema") != "stage5_flow_scene_coverage_single_scene_smoke_v2" or len(split) != 1:
        raise RuntimeError("this audit requires exactly one registered scene-smoke rollout")
    if not bank_manifest.get("complete_balanced_pilot") is False:
        raise RuntimeError("one-scene smoke must not claim full balanced-pilot completion")
    if v2e_manifest.get("scope") != "one_registered_rollout_smoke" or v2e_manifest.get("bank_manifest_sha256") != sha256(bank_manifest_path):
        raise RuntimeError("V2E manifest is not bound to this one-scene physical bank")
    if sha256(bank_path) != bank_manifest.get("bank_sha256"):
        raise RuntimeError("physical bank HDF5 hash mismatch")
    rollout = split[0]["id"]
    if rollout != "s5p_00_train" or bank_manifest.get("generated_episode_ids") != [rollout]:
        raise RuntimeError(f"unexpected or unregistered smoke rollout {rollout}")
    v2e_raw_path, v2e_status_path = locate_shard_artifacts(v2e, rollout)
    v2e_status = json.loads(v2e_status_path.read_text())
    v2e_row = next(row for row in v2e_manifest["episodes"] if row["rollout_id"] == rollout)
    if v2e_status.get("exit_code") != 0 or sha256(v2e_raw_path) != v2e_row["raw_events_sha256"]:
        raise RuntimeError("V2E output status/hash does not match the manifest")

    with h5py.File(bank_path, "r") as shared:
        if set(shared["episodes"].keys()) != {rollout}:
            raise RuntimeError("single-rollout HDF5 groups differ from registered split")
        group = shared[f"episodes/{rollout}"]
        times = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
        signal = np.asarray(group["linearized_display_rgb_proxy"], dtype=np.float32)
        display = np.asarray(group["v2e_display_luma_u8"], dtype=np.uint8)
        flow = np.asarray(group["flow_forward_30ms_px"], dtype=np.float32)
        valid = np.asarray(group["flow_valid_30ms"], dtype=np.uint8).astype(bool)
        eg = np.asarray(group["eventgenesis_events_txyp"], dtype=np.float64)
        K = np.asarray(group["camera_intrinsics_K_px"], dtype=np.float64)
        segmentation = np.asarray(group["segmentation_link_idx"], dtype=np.int32)
    if times.shape != (201,) or not np.allclose(np.diff(times), .01, atol=1e-7, rtol=0):
        raise RuntimeError("shared scene endpoints are not 201 uniform 100 Hz timestamps")
    if signal.shape != (201, 260, 346, 3) or not np.isfinite(signal).all():
        raise RuntimeError("linearized display RGB proxy endpoints are missing, non-finite, or wrong-sized")
    if np.any(signal < 0) or np.any(signal > 1):
        raise RuntimeError("linearized display RGB proxy is outside [0,1]")
    if display.shape != (201, 260, 346) or segmentation.shape != display.shape:
        raise RuntimeError("V2E display / renderer segmentation endpoint contract failed")
    if flow.shape != (66, 2, 260, 346) or valid.shape != (66, 260, 346) or not np.isfinite(flow).all():
        raise RuntimeError("shared 30 ms flow labels/validity do not cover all endpoints")
    if K.shape != (3, 3) or not np.isfinite(K).all() or eg.ndim != 2 or eg.shape[1] != 4:
        raise RuntimeError("camera or EventGenesis artifact schema invalid")
    if len(eg) and (not np.isfinite(eg).all() or np.any(np.diff(eg[:, 0]) < 0)):
        raise RuntimeError("EventGenesis events are non-finite or unsorted")
    with h5py.File(v2e_raw_path, "r") as raw:
        v2 = np.asarray(raw["events"], dtype=np.float64)
    if v2.ndim != 2 or v2.shape[1] != 4 or (len(v2) and (not np.isfinite(v2).all() or np.any(np.diff(v2[:, 0]) < 0))):
        raise RuntimeError("V2E raw events are non-finite, unsorted or wrong-shaped")
    for events, time_scale, name in ((eg, 1.0, "EventGenesis"), (v2, 1e-6, "V2E")):
        if len(events) and (np.min(events[:, 1]) < 0 or np.max(events[:, 1]) >= 346
                            or np.min(events[:, 2]) < 0 or np.max(events[:, 2]) >= 260
                            or events[:, 0].min() * time_scale < -1e-9
                            or events[:, 0].max() * time_scale > times[-1] + 1e-5):
            raise RuntimeError(f"{name} event time or coordinates exceed the shared scene")

    eg_us = eg.copy(); eg_us[:, 0] *= 1e6
    gray = signal @ np.asarray([.2126, .7152, .0722], dtype=np.float32)
    rows = []
    for index in range(66):
        endpoint = index * 3
        left_us, right_us = times[endpoint] * 1e6, times[endpoint + 3] * 1e6
        chunks = {}
        for arm, events in (("eventgenesis", eg_us), ("v2e", v2)):
            timestamps = events[:, 0] if len(events) else np.empty((0,), dtype=np.float64)
            lo = int(np.searchsorted(timestamps, left_us, side="right"))
            hi = int(np.searchsorted(timestamps, right_us, side="right"))
            chunk = events[lo:hi]
            if len(chunk) and (np.any(chunk[:, 0] <= left_us) or np.any(chunk[:, 0] > right_us)):
                raise RuntimeError(f"{arm}/{index}: event interval violates (left,right] endpoints")
            chunks[arm] = chunk
        active = {
            "eventgenesis": active_mask(chunks["eventgenesis"], times[endpoint], 1.0),
            "v2e": active_mask(chunks["v2e"], times[endpoint], 1e-6),
        }
        union = active["eventgenesis"] | active["v2e"]
        valid_i = valid[index]
        magnitude = np.linalg.norm(flow[index], axis=0)
        moving = magnitude >= .5
        gx = cv2.Sobel(gray[endpoint], cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(gray[endpoint], cv2.CV_32F, 0, 1, ksize=3) / 8.0
        textured = np.hypot(gx, gy) > (1.0 / 255.0)
        moving_texture = valid_i & moving & textured
        mistimed_index = min(index + 8, 65)
        candidate_flows = (np.zeros_like(flow[index]), flow[index], -flow[index], flow[mistimed_index])
        yy, xx = np.mgrid[:260, :346].astype(np.float32)
        warp_common = valid_i.copy()
        for candidate_flow in candidate_flows:
            map_x = xx + candidate_flow[0]
            map_y = yy + candidate_flow[1]
            warp_common &= (map_x >= 0) & (map_x <= 345) & (map_y >= 0) & (map_y <= 259)
        photo_values = {}
        zero = np.zeros_like(flow[index])
        photo_values["zero"] = photo_warp(gray[endpoint], gray[endpoint + 3], zero, warp_common)
        photo_values["gt"] = photo_warp(gray[endpoint], gray[endpoint + 3], flow[index], warp_common)
        photo_values["reverse"] = photo_warp(gray[endpoint], gray[endpoint + 3], -flow[index], warp_common)
        photo_values["mistimed_plus_8_windows"] = photo_warp(gray[endpoint], gray[endpoint + 3], flow[mistimed_index], warp_common)
        # The moving-texture subset is deliberately shared by all warp variants.
        moving_texture_common = moving_texture & warp_common
        photo_values["moving_texture_subset_pixels"] = int(moving_texture_common.sum())
        photo_values["moving_texture_mae_zero"] = photo_warp(gray[endpoint], gray[endpoint + 3], zero, moving_texture_common)
        photo_values["moving_texture_mae_gt"] = photo_warp(gray[endpoint], gray[endpoint + 3], flow[index], moving_texture_common)
        photo_values["moving_texture_mae_reverse"] = photo_warp(gray[endpoint], gray[endpoint + 3], -flow[index], moving_texture_common)
        rows.append({
            "sample_id": f"{rollout}/dt1_{endpoint * 10:03d}",
            "window_index": index,
            "endpoint_indices": [endpoint, endpoint + 3],
            "window_seconds": [float(times[endpoint]), float(times[endpoint + 3])],
            "events_independently_counted": {arm: int(len(chunk)) for arm, chunk in chunks.items()},
            "valid_pixels": int(valid_i.sum()),
            "active_valid_pixels": {arm: int((valid_i & mask).sum()) for arm, mask in active.items()},
            "common_union_active_valid_pixels": int((valid_i & union).sum()),
            "common_union_active_valid_coverage": float((valid_i & union).sum() / max(int(valid_i.sum()), 1)),
            "gt_motion_percentiles_px_per_30ms": np.quantile(magnitude[valid_i], [.5, .9, .99]).tolist() if valid_i.any() else [],
            "gt_motion_ge_0_5px_valid_pixels": int((valid_i & (magnitude >= .5)).sum()),
            "common_union_active_and_gt_ge_0_5px": int((valid_i & union & (magnitude >= .5)).sum()),
            "texture_pixels_gradient_gt_1_over_255": int((valid_i & textured).sum()),
            "valid_moving_texture_pixels": int(moving_texture.sum()),
            "photo_warp_mae_linearized_display_proxy_luma": photo_values,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "stage5_flow_scene_smoke_audit_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "real_data_used": False,
        "real_scoring_enabled": False,
        "bank": {"path": str(bank_path), "size_bytes": bank_path.stat().st_size, "sha256": sha256(bank_path),
                 "manifest_sha256": sha256(bank_manifest_path), "config_sha256": bank_manifest["config_sha256"]},
        "v2e_manifest_sha256": sha256(v2e_manifest_path),
        "event_artifacts": {
            "eventgenesis_count": int(len(eg)), "eventgenesis_time_unit": "seconds relative to rollout",
            "v2e_count": int(len(v2)), "v2e_time_unit": "microseconds from pinned V2E HDF5",
            "eventgenesis_hdf5_sha256": sha256(bank_path), "v2e_raw_hdf5_sha256": sha256(v2e_raw_path),
        },
        "linearized_display_signal": {"representation": "gamma-2.2-decoded Genesis uint8 display RGB proxy; not native HDR radiance",
                     "shape": list(signal.shape), "range": [float(signal.min()), float(signal.max())],
                     "mean": float(signal.mean()),
                     "saturated_channel_fraction_mean": float(group.attrs["render_signal_saturated_channel_fraction_mean"]),
                     "saturated_channel_fraction_max": float(group.attrs["render_signal_saturated_channel_fraction_max"]),
                     "renderer_backend": str(group.attrs["genesis_renderer_backend"]), "K_px": K.tolist(),
                     "v2e_display_mapping": "fixed Rec.709 linear Y from shared proxy, exposure 1, Reinhard, sRGB, uint8"},
        "ground_truth": {"flow_shape": list(flow.shape), "valid_shape": list(valid.shape),
                         "unit": "forward source-pixel displacement per 30 ms endpoint interval",
                         "valid_fraction_mean": float(valid.mean())},
        "analytic_warp_fixture": analytic_warp_fixture(),
        "window_rule": "independently recounted half-open event windows (t_i,t_i+3] on shared 100 Hz endpoints",
        "support_rule": "common normal-input active union intersect shared valid GT; no event-count filtering",
        "photo_warp_rule": "GT forward endpoint correspondence evaluates next(x+flow)->previous(x); same mask for zero, GT, reverse and mistimed flow",
        "windows": rows,
        "summary": {
            "windows": len(rows),
            "mean_common_union_active_valid_coverage": float(np.mean([row["common_union_active_valid_coverage"] for row in rows])),
            "mean_valid_moving_texture_pixels": float(np.mean([row["valid_moving_texture_pixels"] for row in rows])),
            "windows_with_nonempty_moving_texture": int(sum(row["valid_moving_texture_pixels"] > 0 for row in rows)),
            "mean_event_count": {arm: float(np.mean([row["events_independently_counted"][arm] for row in rows])) for arm in ("eventgenesis", "v2e")},
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "summary": result["summary"],
                      "analytic_warp": result["analytic_warp_fixture"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
