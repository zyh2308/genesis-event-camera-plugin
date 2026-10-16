#!/usr/bin/env python3
"""Audit real MVSEC event/APS/GT alignment for the preregistered fixed fragment."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import yaml

from run_mvsec_candidate_inference import (
    build_window_representation,
    candidate_valid_mask,
    load_mvsec_events_and_endpoints,
    official_flow_association_from_h5,
    verify_event_index_boundary,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-h5", type=Path, required=True)
    parser.add_argument("--gt-h5", type=Path, required=True)
    parser.add_argument("--calibration-zip", type=Path, required=True)
    parser.add_argument("--resolution-json", type=Path, required=True)
    parser.add_argument("--slice-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    resolved = json.loads(args.resolution_json.read_text())
    slice_manifest = json.loads(args.slice_manifest.read_text())
    data = load_mvsec_events_and_endpoints(args.data_h5)
    resolved_sha = hashlib.sha256(args.resolution_json.read_bytes()).hexdigest()
    expected_event_count = int(slice_manifest["derived_slice"]["data_h5"]["event_rows_original"][1]
                               - slice_manifest["derived_slice"]["data_h5"]["event_rows_original"][0])
    if slice_manifest.get("preregistration_sha256") != resolved_sha:
        raise ValueError("fixed data slice manifest does not reference the registered timestamp manifest")
    if len(data.events_xyts_p) != expected_event_count:
        raise ValueError(f"compact data event count {len(data.events_xyts_p)} != fixed-slice count {expected_event_count}")
    source_manifest_sha = resolved_sha
    if data.registration_json_sha256 != source_manifest_sha:
        raise ValueError("derived data registration manifest differs from the registered timestamp manifest")

    with h5py.File(args.gt_h5, "r") as gt_file:
        flow = gt_file["davis/left/flow_dist"]
        gt_ts = np.asarray(gt_file["davis/left/flow_dist_ts"], dtype=np.float64).reshape(-1)
        if flow.shape[1:] != (2, 260, 346) or len(gt_ts) != flow.shape[0] + 1:
            raise ValueError(f"derived flow/timestamp contract differs from expected [N,2,260,346]/N+1: {flow.shape}/{len(gt_ts)}")
        rows = {}
        for dt in (1, 4):
            dt_rows = []
            for registered in resolved["windows"][f"dt{dt}"]:
                index = int(registered["aps_start_index"])
                native, padded, offset_yx, interval_meta = build_window_representation(
                    data.events_xyts_p, data.event_timestamps_s, data.aps_timestamps_s,
                    index, dt, data.height, data.width, 352,
                )
                if native.shape != (2, 260, 346, 5 * dt) or padded.shape != (2, 352, 352, 5 * dt):
                    raise ValueError("real MVSEC event representation shape mismatch")
                if tuple(offset_yx) != (46, 3):
                    raise ValueError(f"unexpected registered symmetric padding offset {offset_yx}")
                event_count = int(sum(item["event_count"] for item in interval_meta))
                if event_count != int(registered["event_count"]):
                    raise ValueError(f"registered event count differs for APS {index}, dt={dt}")
                start = float(data.aps_timestamps_s[index])
                end = float(data.aps_timestamps_s[index + dt])
                if interval_meta[0]["start_time_s"] != start or interval_meta[-1]["end_time_s"] != end:
                    raise ValueError(f"dt={dt} event window endpoints do not match APS endpoints at {index}")
                endpoint_checks = [verify_event_index_boundary(data, index),
                                   verify_event_index_boundary(data, index + dt)]
                gt, gt_alignment = official_flow_association_from_h5(flow, gt_ts, start, end)
                if gt.shape != (2, 260, 346):
                    raise ValueError(f"GT endpoint flow shape mismatch: {gt.shape}")
                valid, mask_counts = candidate_valid_mask(gt, native)
                if valid.shape != (260, 346) or int(valid.sum()) != mask_counts["candidate_valid_pixels"]:
                    raise RuntimeError("real MVSEC evaluation support shape/count inconsistency")
                first_idx = int(interval_meta[0]["event_start_index"])
                stop_idx = int(interval_meta[-1]["event_stop_index_exclusive"])
                selected_ts = data.event_timestamps_s[first_idx:stop_idx]
                if len(selected_ts) != event_count or (len(selected_ts) and
                        (np.any(selected_ts <= start) or np.any(selected_ts > end))):
                    raise ValueError(f"raw event timestamps violate preregistered (start,end] for APS {index}, dt={dt}")
                dt_rows.append({
                    "aps_start_index": index,
                    "dt": dt,
                    "start_time_s": start,
                    "end_time_s": end,
                    "label_time_s": end,
                    "event_window_convention": "(start_time_s, end_time_s]",
                    "event_count": event_count,
                    "event_first_time_s": float(selected_ts[0]) if event_count else None,
                    "event_last_time_s": float(selected_ts[-1]) if event_count else None,
                    "event_indices": [first_idx, stop_idx],
                    "aps_interval_indices": [index, index + dt],
                    "event_interval_metadata": interval_meta,
                    "event_index_boundary_checks": endpoint_checks,
                    "gt_shape": list(gt.shape),
                    "gt_finite_pixels": int(np.isfinite(gt).all(axis=0).sum()),
                    "candidate_valid_pixels": mask_counts["candidate_valid_pixels"],
                    "gt_alignment": gt_alignment,
                })
            rows[f"dt{dt}"] = dt_rows

    with zipfile.ZipFile(args.calibration_zip) as archive:
        yaml_name = "camchain-imucam-indoor_flying.yaml"
        calibration = yaml.safe_load(archive.read(yaml_name))
    cam0 = calibration["cam0"]
    intr = [float(value) for value in cam0["intrinsics"]]
    proj = [[float(value) for value in row] for row in cam0["projection_matrix"]]
    source_attrs = {}
    for path, key in ((args.data_h5, "data"), (args.gt_h5, "gt")):
        with h5py.File(path, "r") as f:
            source_attrs[key] = {
                "full_source_path": str(f.attrs.get("source_full_file", "")),
                "full_source_sha256": str(f.attrs.get("source_full_file_sha256", "")),
            }
    result = {
        "status": "pass",
        "scope": "actual fixed indoor_flying1 data slice parsed and checked without loading candidate model weights",
        "sequence": "indoor_flying1",
        "source_timestamp_manifest": str(args.resolution_json.resolve()),
        "source_timestamp_manifest_sha256": source_manifest_sha,
        "slice_manifest": str(args.slice_manifest.resolve()),
        "slice_manifest_sha256": sha256(args.slice_manifest),
        "data_h5": {"path": str(args.data_h5.resolve()), "size_bytes": args.data_h5.stat().st_size, "sha256": sha256(args.data_h5), **source_attrs["data"]},
        "gt_h5": {"path": str(args.gt_h5.resolve()), "size_bytes": args.gt_h5.stat().st_size, "sha256": sha256(args.gt_h5), **source_attrs["gt"]},
        "calibration_zip": {"path": str(args.calibration_zip.resolve()), "size_bytes": args.calibration_zip.stat().st_size, "sha256": sha256(args.calibration_zip)},
        "event_count": len(data.events_xyts_p),
        "aps_endpoint_count_in_slice": len(data.aps_timestamps_s),
        "sensor_hw": [data.height, data.width],
        "event_schema": "[x,y,t_seconds,p]; polarity 0→OFF(-1), +1→ON(+1) in pinned STE representation",
        "left_camera_K_from_intrinsics": [[intr[0], 0.0, intr[2]], [0.0, intr[1], intr[3]], [0.0, 0.0, 1.0]],
        "left_camera_projection_matrix_3x4": proj,
        "left_camera_distortion_model": str(cam0["distortion_model"]),
        "coordinate_resolution_note": "intrinsics and projection_matrix differ; both retained; this is an I/O/association smoke, no calibration-dependent model input or real ranking",
        "padding_hw": [352, 352],
        "padding_offset_yx": [46, 3],
        "windows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "dt1_windows": len(rows["dt1"]), "dt4_windows": len(rows["dt4"]),
                      "event_count": result["event_count"], "sensor_hw": result["sensor_hw"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
