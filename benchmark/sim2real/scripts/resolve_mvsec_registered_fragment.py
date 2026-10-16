#!/usr/bin/env python3
"""Resolve the preregistered first-second MVSEC APS windows without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-h5", type=Path, required=True)
    parser.add_argument("--gt-h5", type=Path, required=True)
    parser.add_argument("--calibration-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    with h5py.File(args.data_h5, "r") as f:
        base = "davis/left"
        ev = f[f"{base}/events"]
        if ev.ndim != 2 or ev.shape[1] != 4:
            raise ValueError(f"MVSEC raw events must be [N,4] [x,y,t,p], got {ev.shape}")
        # Timestamp is the third stored event field. Read only this column so
        # endpoint resolution does not materialize the several-GB event array.
        event_shape = tuple(ev.shape)
        event_times = np.asarray(ev[:, 2], dtype=np.float64)
        aps_times = np.asarray(f[f"{base}/image_raw_ts"], dtype=np.float64).reshape(-1)
        image_shape = tuple(f[f"{base}/image_raw"].shape)
        image_event_indices = np.rint(np.asarray(
            f[f"{base}/image_raw_event_inds"], dtype=np.float64
        ).reshape(-1)).astype(np.int64)
    with h5py.File(args.gt_h5, "r") as f:
        gt_base = "davis/left"
        gt_times = np.asarray(f[f"{gt_base}/flow_dist_ts"], dtype=np.float64).reshape(-1)
        flow_shape = tuple(f[f"{gt_base}/flow_dist"].shape)

    for name, ts in (("event", event_times), ("APS", aps_times), ("GT", gt_times)):
        if ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all() or np.any(np.diff(ts) < 0):
            raise ValueError(f"{name} timestamps are invalid or unsorted")
    if np.any(np.diff(aps_times) <= 0) or np.any(np.diff(gt_times) <= 0):
        raise ValueError("APS and flow-GT timestamps must be strictly increasing")
    if image_shape != (len(aps_times), 260, 346):
        raise ValueError(f"unexpected MVSEC APS image shape {image_shape}")
    if flow_shape[1:] != (2, 260, 346):
        raise ValueError(f"unexpected MVSEC flow GT shape {flow_shape}")
    if len(gt_times) not in (flow_shape[0], flow_shape[0] + 1):
        raise ValueError(
            "flow timestamp count must equal the flow-map count or include its final endpoint: "
            f"{flow_shape[0]} maps vs {len(gt_times)} timestamps"
        )
    if len(image_event_indices) != len(aps_times):
        raise ValueError("APS-to-event index array length mismatch")

    common_start = max(float(event_times[0]), float(aps_times[0]), float(gt_times[0]))
    common_end = min(float(event_times[-1]), float(aps_times[-1]), float(gt_times[-1]))
    aps_anchor = int(np.searchsorted(aps_times, common_start, side="left"))
    if aps_anchor >= len(aps_times) or aps_times[aps_anchor] > common_end:
        raise ValueError("no APS endpoint falls in the common event/APS/GT timestamp range")
    anchor_time = float(aps_times[aps_anchor])
    fragment_end = min(anchor_time + 1.0, common_end)
    windows: dict[str, list[dict]] = {}
    for dt in (1, 4):
        indices = []
        for i in range(aps_anchor, len(aps_times) - dt):
            start, end = float(aps_times[i]), float(aps_times[i + dt])
            if start < anchor_time:
                continue
            if end > fragment_end or end > common_end:
                break
            lo = int(np.searchsorted(event_times, start, side="right"))
            hi = int(np.searchsorted(event_times, end, side="right"))
            if hi < lo:
                raise RuntimeError("timestamp event interval indices are reversed")
            event_boundary = int(image_event_indices[i + dt])
            if abs(hi - event_boundary) > 1:
                raise ValueError(
                    f"event timestamp count and official APS event index differ by >1 at endpoint {i+dt}: "
                    f"{hi} vs {event_boundary}"
                )
            indices.append({
                "aps_start_index": i,
                "aps_end_index": i + dt,
                "start_time_s": start,
                "end_time_s": end,
                "event_left_exclusive_index": lo,
                "event_stop_exclusive_index": hi,
                "event_count": hi - lo,
                "official_endpoint_event_index": event_boundary,
                "timestamp_endpoint_event_index": hi,
                "endpoint_event_index_abs_difference": abs(hi - event_boundary),
            })
        if not indices:
            raise ValueError(f"no dt={dt} window fits the preregistered fragment")
        windows[f"dt{dt}"] = indices

    with zipfile.ZipFile(args.calibration_zip) as archive:
        calibration_members = [
            {"name": info.filename, "size_bytes": info.file_size}
            for info in archive.infolist() if not info.is_dir()
        ]
        calibration_name = "camchain-imucam-indoor_flying.yaml"
        if calibration_name not in archive.namelist():
            raise KeyError(f"official calibration archive is missing {calibration_name}")
        import yaml
        calibration = yaml.safe_load(archive.read(calibration_name))
    cam0 = calibration["cam0"]
    intrinsics = [float(value) for value in cam0["intrinsics"]]
    projection = [[float(value) for value in row] for row in cam0["projection_matrix"]]
    T_cam_imu = [[float(value) for value in row] for row in cam0["T_cam_imu"]]
    if len(intrinsics) != 4 or len(projection) != 3 or any(len(row) != 4 for row in projection):
        raise ValueError("cam0 calibration has an unexpected intrinsics/projection shape")
    result = {
        "status": "pass",
        "scope": "preregistered sample resolution only; no candidate checkpoint/model weights loaded",
        "registered_rule": "first APS endpoint in common event/APS/GT coverage; include every complete dt window ending within the next 1.0 s",
        "sequence": "indoor_flying1",
        "time_unit": "seconds",
        "common_timestamp_range_s": [common_start, common_end],
        "first_common_aps_index": aps_anchor,
        "first_common_aps_timestamp_s": anchor_time,
        "registered_fragment_end_s": fragment_end,
        "sensor_hw": [260, 346],
        "left_camera_calibration": {
            "resolution_wh": [int(v) for v in cam0["resolution"]],
            "intrinsics_fx_fy_cx_cy": intrinsics,
            "K_from_intrinsics": [
                [intrinsics[0], 0.0, intrinsics[2]],
                [0.0, intrinsics[1], intrinsics[3]],
                [0.0, 0.0, 1.0],
            ],
            "projection_matrix_3x4": projection,
            "distortion_model": str(cam0["distortion_model"]),
            "distortion_coefficients": [float(v) for v in cam0["distortion_coeffs"]],
            "camera_model": str(cam0["camera_model"]),
            "T_cam_imu": T_cam_imu,
            "rostopic": str(cam0["rostopic"]),
            "note": "dataset-space choice between intrinsics and projection matrix must follow official MVSEC flow conventions; both are recorded here",
        },
        "aps_image_shape": list(image_shape),
        "flow_gt_shape": list(flow_shape),
        "event_shape": list(event_shape),
        "event_timestamp_count": len(event_times),
        "aps_timestamp_count": len(aps_times),
        "gt_timestamp_count": len(gt_times),
        "timestamp_convention": "event windows are (APS[i], APS[i+dt]]; independent event index recount by binary search",
        "windows": windows,
        "inputs": {
            str(p.resolve()): {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in (args.data_h5, args.gt_h5, args.calibration_zip)
        },
        "calibration_zip_members": calibration_members,
        "resolved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "first_common_aps_index": aps_anchor,
        "first_common_aps_timestamp_s": anchor_time, "fragment_end_s": fragment_end,
        "dt1_indices": [row["aps_start_index"] for row in windows["dt1"]],
        "dt4_indices": [row["aps_start_index"] for row in windows["dt4"]],
        "calibration_members": calibration_members,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
