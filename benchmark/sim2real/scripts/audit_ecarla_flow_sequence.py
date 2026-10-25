#!/usr/bin/env python3
"""Audit one preregistered eCARLA event/flow sequence and export small windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import hdf5plugin  # noqa: F401; registers the official eWiz HDF5 filter.
import h5py
import inverse_optical_flow
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode_window(x: np.ndarray, y: np.ndarray, t_abs: np.ndarray, polarity: np.ndarray,
                  start_us: int, end_us: int, height: int, width: int, bins: int = 5) -> tuple[np.ndarray, int]:
    # Match eWiz WriterFlow.map_flow_to_events: first event at/after each boundary.
    left = int(np.searchsorted(t_abs, start_us, side="left"))
    right = int(np.searchsorted(t_abs, end_us, side="left"))
    ts = t_abs[left:right]
    xs = x[left:right].astype(np.int32, copy=False)
    ys = y[left:right].astype(np.int32, copy=False)
    ps = polarity[left:right].astype(bool, copy=False)
    if ts.size and (xs.min() < 0 or xs.max() >= width or ys.min() < 0 or ys.max() >= height):
        raise RuntimeError("event coordinate outside the registered sensor size")
    edges = np.linspace(start_us, end_us, bins + 1, dtype=np.float64)
    which_bin = np.searchsorted(edges, ts, side="right") - 1
    if np.any((which_bin < 0) | (which_bin >= bins)):
        raise RuntimeError("event timestamp escaped the left-closed/right-open temporal-bin contract")
    counts = np.zeros((2, height, width, bins), dtype=np.uint16)
    for b in range(bins):
        keep = which_bin == b
        if not np.any(keep):
            continue
        channel = np.where(ps[keep], 0, 1)
        linear = channel * (height * width) + ys[keep] * width + xs[keep]
        bincount = np.bincount(linear, minlength=2 * height * width)
        if int(bincount.max(initial=0)) > np.iinfo(np.uint16).max:
            raise RuntimeError("event count overflows uint16")
        counts[..., b] = bincount.reshape(2, height, width).astype(np.uint16)
    return counts, int(right - left)


def endpoint_mask(flow: np.ndarray) -> np.ndarray:
    height, width = flow.shape[1:]
    yy, xx = np.mgrid[0:height, 0:width]
    u, v = flow
    finite = np.isfinite(flow).all(axis=0)
    return finite & (xx + u >= 0) & (xx + u <= width - 1) & (yy + v >= 0) & (yy + v <= height - 1)


def photometric_check(gray0: np.ndarray, gray1: np.ndarray, flow: np.ndarray, valid_gt: np.ndarray,
                      event_active: np.ndarray, texture_threshold: float = 5.0,
                      motion_threshold: float = 0.5) -> dict:
    height, width = gray0.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    flow = np.nan_to_num(flow.astype(np.float32, copy=False))
    u, v = flow
    plus_ok = (xx + u >= 1) & (xx + u < width - 2) & (yy + v >= 1) & (yy + v < height - 2)
    minus_ok = (xx - u >= 1) & (xx - u < width - 2) & (yy - v >= 1) & (yy - v < height - 2)
    common = plus_ok & minus_ok & valid_gt
    g0 = gray0.astype(np.float32)
    g1 = gray1.astype(np.float32)
    zero = cv2.remap(g1, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    forward = cv2.remap(g1, xx + u, yy + v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    reverse = cv2.remap(g1, xx - u, yy - v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    grad_x = cv2.Sobel(g0, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(g0, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    textured = np.hypot(grad_x, grad_y) >= texture_threshold
    moving = np.hypot(u, v) >= motion_threshold
    masks = {"in_bounds_common": common, "textured_motion": common & textured & moving,
             "event_textured_motion": common & textured & moving & event_active}
    result = {"common_in_bounds_pixels": int(common.sum()), "supports": {}}
    for name, mask in masks.items():
        result["supports"][name] = int(mask.sum())
        if mask.any():
            result["supports"][name] = {
                "pixels": int(mask.sum()), "coverage_of_image": float(mask.mean()),
                "zero_flow_mae": float(np.abs(g0 - zero)[mask].mean()),
                "forward_gt_mae": float(np.abs(g0 - forward)[mask].mean()),
                "reverse_gt_mae": float(np.abs(g0 - reverse)[mask].mean()),
            }
        else:
            result["supports"][name] = {"pixels": 0, "coverage_of_image": 0.0,
                                         "zero_flow_mae": None, "forward_gt_mae": None,
                                         "reverse_gt_mae": None}
    return result


def independent_track_check(gray0: np.ndarray, gray1: np.ndarray, raw_flow: np.ndarray,
                            forward_flow: np.ndarray, inverse_mask: np.ndarray,
                            options: dict) -> dict:
    """Use grayscale feature tracks as an independent direction/unit check."""
    p0 = cv2.goodFeaturesToTrack(
        gray0, maxCorners=int(options["max_corners"]), qualityLevel=float(options["quality_level"]),
        minDistance=float(options["min_distance_px"]), blockSize=int(options["block_size"]),
    )
    if p0 is None:
        return {"detected_corners": 0, "tracked_corners": 0, "raw": None, "converted_forward": None}
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(options["lk_iterations"]), float(options["lk_epsilon"]))
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        gray0, gray1, p0, None, winSize=tuple(int(x) for x in options["lk_window_wh"]),
        maxLevel=int(options["lk_max_level"]), criteria=criteria,
    )
    p0 = p0.reshape(-1, 2)
    p1 = p1.reshape(-1, 2)
    height, width = gray0.shape
    inbounds = ((p1[:, 0] >= 0) & (p1[:, 0] < width - 1) &
                (p1[:, 1] >= 0) & (p1[:, 1] < height - 1))
    keep_track = status.reshape(-1).astype(bool) & inbounds
    points = p0[keep_track]
    observed = p1[keep_track] - points
    if len(points) == 0:
        return {"detected_corners": int(len(p0)), "tracked_corners": 0,
                "raw": None, "converted_forward": None}
    gx = points[:, 0].reshape(-1, 1).astype(np.float32)
    gy = points[:, 1].reshape(-1, 1).astype(np.float32)
    observed_mag = np.linalg.norm(observed, axis=1)

    def score(candidate: np.ndarray, require_valid: bool) -> dict | None:
        sampled = np.column_stack([
            cv2.remap(candidate[0], gx, gy, cv2.INTER_LINEAR).reshape(-1),
            cv2.remap(candidate[1], gx, gy, cv2.INTER_LINEAR).reshape(-1),
        ])
        candidate_mag = np.linalg.norm(sampled, axis=1)
        eligible = np.isfinite(sampled).all(axis=1) & (observed_mag >= 1.0) & (candidate_mag >= 0.5)
        if require_valid:
            valid_sample = cv2.remap(inverse_mask.astype(np.float32), gx, gy, cv2.INTER_NEAREST).reshape(-1)
            eligible &= valid_sample == 0
        if not np.any(eligible):
            return None
        cosine = (sampled[eligible] * observed[eligible]).sum(axis=1) / (
            candidate_mag[eligible] * observed_mag[eligible] + 1e-6
        )
        epe = np.linalg.norm(sampled[eligible] - observed[eligible], axis=1)
        return {
            "eligible_tracks": int(eligible.sum()),
            "median_epe_px": float(np.median(epe)),
            "median_direction_cosine": float(np.median(cosine)),
            "median_gt_to_track_magnitude_ratio": float(np.median(candidate_mag[eligible] / (observed_mag[eligible] + 1e-6))),
            "fraction_direction_cosine_positive": float(np.mean(cosine > 0)),
        }

    return {
        "detected_corners": int(len(p0)), "tracked_corners": int(keep_track.sum()),
        "raw": score(raw_flow, False),
        "converted_forward": score(forward_flow, True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data_root, config_path, out = args.data_root.resolve(), args.config.resolve(), args.output_root.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {out}")
    config = json.loads(config_path.read_text())
    dims = config["source"]["sensor_resolution_wh"]
    width, height = int(dims[0]), int(dims[1])
    event_path, gray_path, flow_path = (data_root / n for n in ("events.hdf5", "gray.hdf5", "flow.hdf5"))
    for path in (event_path, gray_path, flow_path, data_root / "props.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    with h5py.File(event_path, "r") as ev, h5py.File(gray_path, "r") as gray, h5py.File(flow_path, "r") as flow_file:
        x = np.asarray(ev["events/x"])
        y = np.asarray(ev["events/y"])
        event_t = np.asarray(ev["events/time"])
        polarity = np.asarray(ev["events/polarity"])
        event_offset = int(ev["time_offset"][0])
        event_abs_t = event_t.astype(np.int64) + event_offset
        flow_t = np.asarray(flow_file["time"], dtype=np.int64) + int(flow_file["time_offset"][0])
        gray_t = np.asarray(gray["time"], dtype=np.int64) + int(gray["time_offset"][0])
        flow_to_events = np.asarray(flow_file["flow_to_events"], dtype=np.int64)
        flow_dtype = str(flow_file["flows"].dtype)
        height_actual, width_actual = map(int, gray["gray_images"].shape[1:])
        if (width_actual, height_actual) != (width, height):
            raise RuntimeError(f"sensor size mismatch: config {(width, height)}, files {(width_actual, height_actual)}")
        if len(flow_t) != len(gray_t) or len(flow_t) != len(flow_file["flows"]):
            raise RuntimeError("flow/grayscale sample count mismatch")
        if not np.all(np.diff(event_abs_t) >= 0) or not np.all(np.diff(flow_t) > 0):
            raise RuntimeError("timestamps are not monotonically ordered")
        gray_flow_delta = np.abs(gray_t - flow_t)
        if gray_flow_delta.max(initial=0) != 0:
            raise RuntimeError("grayscale endpoints and flow labels do not share exact timestamps")

        train_ids = [int(i) for i in config["fixed_windows"]["train_flow_indices"]]
        val_ids = [int(i) for i in config["fixed_windows"]["validation_flow_indices"]]
        warp_ids = [int(i) for i in config["fixed_windows"]["photometric_warp_indices"]]
        all_ids = sorted(set(train_ids + val_ids))
        if any(i < 0 or i + 1 >= len(flow_t) for i in all_ids + warp_ids):
            raise RuntimeError("fixed interval index outside the dataset")

        exported: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
        targets: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
        valid_masks: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
        ids_out: dict[str, list[str]] = {"train": [], "validation": []}
        per_window = []
        mapping_comparisons = []
        conversion_checks = []
        for split, indices in (("train", train_ids), ("validation", val_ids)):
            for i in indices:
                start_us, end_us = int(flow_t[i]), int(flow_t[i + 1])
                ev_counts, event_count = encode_window(x, y, event_abs_t, polarity, start_us, end_us,
                                                       height, width, bins=5)
                raw_flow = np.asarray(flow_file["flows"][i], dtype=np.float32)
                if raw_flow.shape != (2, height, width) or not np.isfinite(raw_flow).all():
                    raise RuntimeError(f"invalid raw GT flow at fixed interval {i}")
                flow_gt, inverse_mask = inverse_optical_flow.avg_method(raw_flow)
                flow_gt = np.asarray(flow_gt, dtype=np.float32)
                inverse_mask = np.asarray(inverse_mask)
                if flow_gt.shape != (2, height, width) or inverse_mask.shape != (height, width):
                    raise RuntimeError(f"inverse-flow adapter returned unexpected shapes at interval {i}: "
                                       f"flow={flow_gt.shape}, mask={inverse_mask.shape}")
                finite_flow = np.isfinite(flow_gt).all(axis=0)
                valid = (inverse_mask == 0) & finite_flow & endpoint_mask(np.nan_to_num(flow_gt))
                flow_gt = np.nan_to_num(flow_gt, nan=0.0, posinf=0.0, neginf=0.0)
                active = ev_counts.sum(axis=(0, 3)) > 0
                magnitude = np.hypot(flow_gt[0], flow_gt[1])
                flow_to_event_count = int(flow_to_events[i + 1] - flow_to_events[i])
                lo = int(np.searchsorted(event_abs_t, start_us, side="left"))
                hi = int(np.searchsorted(event_abs_t, end_us, side="left"))
                mapping_comparisons.append({
                    "flow_index": i, "window_start_us": start_us, "window_end_us": end_us,
                    "independent_raw_event_count": event_count, "flow_to_events_delta": flow_to_event_count,
                    "event_count_delta_matches": event_count == flow_to_event_count,
                    "start_event_index_left": lo, "end_event_index_left": hi,
                })
                conversion_checks.append({
                    "flow_index": i, "raw_direction": "opposite source-to-destination grayscale tracks",
                    "conversion": "inverse_optical_flow.avg_method(raw_flow)",
                    "inverse_mask_invalid_pixels": int((inverse_mask != 0).sum()),
                    "valid_forward_pixels": int(valid.sum()),
                })
                exported[split].append(ev_counts)
                targets[split].append(flow_gt)
                valid_masks[split].append(valid.astype(np.uint8))
                ids_out[split].append(f"{config['source']['scene']}/flow_{i:04d}")
                per_window.append({
                    "split": split, "flow_index": i, "timestamp_start_us": start_us,
                    "timestamp_end_us": end_us, "label_timestamp_us": start_us,
                    "duration_us": end_us - start_us, "events": event_count,
                    "event_active_pixels": int(active.sum()), "event_active_coverage": float(active.mean()),
                    "valid_gt_pixels": int(valid.sum()),
                    "gt_motion_gt_0_5px_valid_pixels": int((valid & (magnitude >= 0.5)).sum()),
                    "event_active_and_motion_pixels": int((active & valid & (magnitude >= 0.5)).sum()),
                    "gt_magnitude_px_per_interval": {
                        "mean_valid": float(magnitude[valid].mean()),
                        "p50_valid": float(np.quantile(magnitude[valid], 0.50)),
                        "p90_valid": float(np.quantile(magnitude[valid], 0.90)),
                        "p99_valid": float(np.quantile(magnitude[valid], 0.99)),
                        "max_valid": float(magnitude[valid].max()),
                    },
                    "event_bins_count": ev_counts.sum(axis=(1, 2)).astype(np.int64).tolist(),
                })

        warp_rows = []
        for i in warp_ids:
            gray0 = np.asarray(gray["gray_images"][i], dtype=np.uint8)
            gray1 = np.asarray(gray["gray_images"][i + 1], dtype=np.uint8)
            raw_flow = np.asarray(flow_file["flows"][i], dtype=np.float32)
            flow_gt, inverse_mask = inverse_optical_flow.avg_method(raw_flow)
            flow_gt = np.asarray(flow_gt, dtype=np.float32)
            inverse_mask = np.asarray(inverse_mask)
            finite_flow = np.isfinite(flow_gt).all(axis=0)
            valid_gt = (inverse_mask == 0) & finite_flow & endpoint_mask(np.nan_to_num(flow_gt))
            flow_gt = np.nan_to_num(flow_gt, nan=0.0, posinf=0.0, neginf=0.0)
            start_us, end_us = int(flow_t[i]), int(flow_t[i + 1])
            ev_counts, _ = encode_window(x, y, event_abs_t, polarity, start_us, end_us, height, width, bins=5)
            warp_rows.append({
                "flow_index": i, "start_us": start_us, "end_us": end_us,
                "independent_tracks": independent_track_check(
                    gray0, gray1, raw_flow, flow_gt, inverse_mask, config["independent_track_check"]),
                "photometric": photometric_check(
                    gray0, gray1, flow_gt, valid_gt, ev_counts.sum(axis=(0, 3)) > 0,
                    texture_threshold=float(config["validity"]["texture_region"].split(">=")[1].split()[0]),
                    motion_threshold=float(config["validity"]["motion_region_threshold_px"])),
            })

        out.mkdir(parents=True)
        artifact_path = out / "ecarla_fixed_windows.npz"
        np.savez_compressed(
            artifact_path,
            train_events=np.stack(exported["train"]), train_flow=np.stack(targets["train"]),
            train_valid=np.stack(valid_masks["train"]), train_ids=np.asarray(ids_out["train"]),
            validation_events=np.stack(exported["validation"]), validation_flow=np.stack(targets["validation"]),
            validation_valid=np.stack(valid_masks["validation"]), validation_ids=np.asarray(ids_out["validation"]),
        )

    result = {
        "schema": "ecarla_flow_sequence_audit_v2",
        "status": "passed_corrected_data_contract_audit; model_training_not_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "data_root": str(data_root),
        "data_members": {p.name: {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
                          for p in (data_root / "props.json", event_path, gray_path, flow_path)},
        "external_source": {"official_eWiz_commit": "7e6f9132bcedb5ebaf49743f27a0679075017ad0",
                            "flow_sync_contract": "ReaderFlow uses FlowSync at the interval start and returns displacement",
                            "inverse_optical_flow_version": "0.0.2",
                            "raw_flow_direction_contract": "raw artifact disagrees with independently tracked forward grayscale motion; this audit applies the upstream inverse_flow adapter explicitly"},
        "time_contract": {
            "event_time_monotonic": True, "flow_time_monotonic": True,
            "flow_gray_timestamp_max_abs_delta_us": int(gray_flow_delta.max(initial=0)),
            "flow_interval_us_min_median_max": [int(np.diff(flow_t).min()), int(np.median(np.diff(flow_t))), int(np.diff(flow_t).max())],
            "event_offset_us": event_offset,
            "event_flow_abs_start_delta_us": int((flow_t[0] - event_abs_t[0])),
            "event_window_convention": "[start,end), using left insertion indices as eWiz WriterFlow",
            "fixed_window_event_count_mapping_match_count": sum(r["event_count_delta_matches"] for r in mapping_comparisons),
            "fixed_window_count": len(mapping_comparisons),
            "mapping_comparisons": mapping_comparisons,
        },
        "flow_gt": {
            "native_shape": [2, height, width], "stored_dtype": flow_dtype,
            "raw_direction": "opposite to forward grayscale feature tracks on fixed audit windows",
            "diagnostic_direction": "converted to forward displacement with upstream inverse_optical_flow.avg_method",
            "valid_mask_in_dataset": False,
            "diagnostic_mask": "upstream inverse disocclusion mask == 0 and finite forward endpoint inside image",
        },
        "fixed_windows": per_window,
        "flow_conversion_checks": conversion_checks,
        "photometric_warp_checks": warp_rows,
        "exported_diagnostic_artifact": {"path": str(artifact_path), "size_bytes": artifact_path.stat().st_size,
                                           "sha256": sha256(artifact_path), "training_windows": len(train_ids),
                                           "validation_windows": len(val_ids), "model_training_performed": False},
        "real_data_used": False, "real_scoring_enabled": False,
    }
    (out / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "fixed_windows"}, indent=2))


if __name__ == "__main__":
    main()
