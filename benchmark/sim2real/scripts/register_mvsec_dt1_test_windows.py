#!/usr/bin/env python3
"""Register the frozen MVSEC dt1 windows and label provenance without loading a model.

This script prepares a real-test manifest only. It reads public events, APS
timestamps, calibration and flow GT to bind exact input/label windows, but it
does not load candidate weights, run inference, or compute candidate scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
import yaml

from run_mvsec_candidate_inference import (
    NATIVE_HW,
    build_window_representation,
    load_mvsec_events_and_endpoints,
    official_flow_association_from_h5,
    round_model_size,
    verify_event_index_boundary,
)
from stage4_mvsec_dt1_metrics import all_valid_gt_mask, common_valid_mask


SEQUENCES = ("indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1")
SEQUENCE_FOLDERS = {
    "indoor_flying1": "indoor_flying",
    "indoor_flying2": "indoor_flying",
    "indoor_flying3": "indoor_flying",
    "outdoor_day1": "outdoor_day",
}
LEGACY_INDOOR_ROI = (2, 258, 45, 301)
LEGACY_OUTDOOR_ROI = (2, 192, 45, 301)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_indices(sequence: str, aps_timestamps_s: np.ndarray) -> tuple[list[int], list[int]]:
    """Return the pinned STE-FlowNet start indices and indices dropped at boundaries."""
    endpoint_count = len(aps_timestamps_s)
    if sequence.startswith("indoor_flying"):
        # Pinned Test_loading rule: i > 20 and i+20 < number of APS endpoints.
        candidates = list(range(21, max(21, endpoint_count - 20)))
    elif sequence == "outdoor_day1":
        # Pinned outdoor validation intervals, represented as half-open index ranges.
        candidates = [i for lo, hi in ((9200, 9600), (10500, 10900)) for i in range(lo, hi)]
    else:
        raise ValueError(f"unsupported sequence: {sequence}")
    eligible = [i for i in candidates if 0 <= i and i + 1 < endpoint_count]
    excluded = sorted(set(candidates) - set(eligible))
    return eligible, excluded


def read_calibration(path: Path, camera_key: str = "cam0") -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        yaml_names = [name for name in archive.namelist() if name.endswith(".yaml")]
        if len(yaml_names) != 1:
            raise ValueError(f"expected exactly one calibration YAML in {path}, got {yaml_names}")
        document = yaml.safe_load(archive.read(yaml_names[0]))
    camera = document.get(camera_key)
    if not isinstance(camera, dict):
        raise KeyError(f"calibration {path} has no {camera_key} camera block")
    intrinsics = np.asarray(camera.get("intrinsics"), dtype=np.float64)
    resolution = tuple(int(value) for value in camera.get("resolution", ()))
    if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all():
        raise ValueError(f"invalid {camera_key} intrinsics in {path}: {intrinsics}")
    if resolution != (NATIVE_HW[1], NATIVE_HW[0]):
        raise ValueError(f"calibration resolution must be [W,H]={(NATIVE_HW[1], NATIVE_HW[0])}, got {resolution}")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    k = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return {
        "archive_path": str(path),
        "archive_size_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "yaml_member": yaml_names[0],
        "camera_key": camera_key,
        "camera_model": camera.get("camera_model"),
        "distortion_model": camera.get("distortion_model"),
        "distortion_coeffs": [float(value) for value in camera.get("distortion_coeffs", [])],
        "resolution_wh": list(resolution),
        "intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
        "K_px": k,
    }


def inspect_one_sequence(
    sequence: str,
    root: Path,
    reported_root: str,
    calibration: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    folder = root / SEQUENCE_FOLDERS[sequence]
    data_path = folder / f"{sequence}_data.hdf5"
    gt_path = folder / f"{sequence}_gt.hdf5"
    for path in (data_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    data = load_mvsec_events_and_endpoints(data_path)
    gt_base = "davis/left"
    with h5py.File(gt_path, "r") as gt_file:
        if f"{gt_base}/flow_dist" not in gt_file or f"{gt_base}/flow_dist_ts" not in gt_file:
            raise KeyError(f"{gt_path} lacks official flow_dist/flow_dist_ts fields")
        flow_dataset = gt_file[f"{gt_base}/flow_dist"]
        gt_ts = np.asarray(gt_file[f"{gt_base}/flow_dist_ts"], dtype=np.float64).reshape(-1)
        gt_file_sha256 = sha256_file(gt_path)
        if flow_dataset.ndim != 4 or tuple(flow_dataset.shape[1:]) != (2, *NATIVE_HW):
            raise ValueError(f"{sequence} flow_dist shape is invalid: {flow_dataset.shape}")
        gt_flow_shape = list(flow_dataset.shape)
        gt_flow_dtype = str(flow_dataset.dtype)
        if len(gt_ts) not in (flow_dataset.shape[0], flow_dataset.shape[0] + 1):
            raise ValueError(f"{sequence} GT timestamp/map count mismatch")
        if not np.isfinite(gt_ts).all() or np.any(np.diff(gt_ts) <= 0):
            raise ValueError(f"{sequence} flow timestamps must be finite and strictly increasing seconds")

        candidate_indices, boundary_excluded = resolve_indices(sequence, data.aps_timestamps_s)
        rows: list[dict[str, Any]] = []
        excluded_gt_coverage: list[int] = []
        durations: list[float] = []
        empty_event_windows = 0
        empty_common_full = 0
        empty_common_legacy = 0
        total_events = 0
        active_pixels_total = 0
        common_full_total = 0
        common_legacy_total = 0

        for index in candidate_indices:
            start_s = float(data.aps_timestamps_s[index])
            end_s = float(data.aps_timestamps_s[index + 1])
            if start_s < gt_ts[0] or end_s > gt_ts[-1]:
                excluded_gt_coverage.append(index)
                continue
            volume, _, _, interval_meta = build_window_representation(
                data.events_xyts_p,
                data.event_timestamps_s,
                data.aps_timestamps_s,
                index,
                1,
                data.height,
                data.width,
                round_model_size(data.height, data.width),
            )
            gt, association = official_flow_association_from_h5(
                flow_dataset, gt_ts, start_s, end_s
            )
            full_valid, full_counts = common_valid_mask(gt, volume, volume)
            roi = LEGACY_OUTDOOR_ROI if sequence == "outdoor_day1" else LEGACY_INDOOR_ROI
            legacy_valid, legacy_counts = common_valid_mask(
                gt, volume, volume, evaluation_roi_native_yxyx=roi
            )
            full_gt = all_valid_gt_mask(gt, NATIVE_HW[0])
            legacy_gt = all_valid_gt_mask(gt, NATIVE_HW[0], roi)
            event_meta = interval_meta[0]
            if event_meta["start_time_s"] != start_s or event_meta["end_time_s"] != end_s:
                raise RuntimeError(f"{sequence} APS {index}: event interval differs from APS endpoints")
            start_boundary = verify_event_index_boundary(data, index)
            end_boundary = verify_event_index_boundary(data, index + 1)
            event_count = int(event_meta["event_count"])
            if event_count == 0:
                empty_event_windows += 1
            if not full_valid.any():
                empty_common_full += 1
            if not legacy_valid.any():
                empty_common_legacy += 1
            duration = end_s - start_s
            durations.append(duration)
            total_events += event_count
            active_pixels_total += int(full_counts["common_event_active_pixels"])
            common_full_total += int(full_valid.sum())
            common_legacy_total += int(legacy_valid.sum())

            native_volume_sha = array_sha256(volume)
            gt_sha = array_sha256(gt.astype(np.float32, copy=False))
            full_mask_sha = array_sha256(full_valid.astype(np.uint8))
            legacy_mask_sha = array_sha256(legacy_valid.astype(np.uint8))
            row = {
                "sample_id": f"MVSEC/{sequence}/dt1/aps_{index:06d}",
                "sequence": sequence,
                "rollout_id": f"MVSEC/{sequence}",
                "dt": 1,
                "aps_start_index": int(index),
                "start_time_s": start_s,
                "label_time_s": end_s,
                "end_time_s": end_s,
                "event_window_convention": "(start_time_s, end_time_s]",
                "event_count": event_count,
                "event_index_bounds": {
                    "start_exclusive_index": int(event_meta["event_start_index"]),
                    "end_inclusive_stop_index_exclusive": int(event_meta["event_stop_index_exclusive"]),
                },
                "official_aps_event_index_checks": {
                    "start_endpoint": start_boundary,
                    "end_endpoint": end_boundary,
                },
                "gt_association": association,
                "gt_support": {
                    "finite_nonzero_pixels_full_native": int(full_gt.sum()),
                    "finite_nonzero_pixels_legacy_roi": int(legacy_gt.sum()),
                    "common_event_active_valid_pixels_full_native": int(full_valid.sum()),
                    "common_event_active_valid_pixels_legacy_roi": int(legacy_valid.sum()),
                    "common_mask_sha256_full_native": full_mask_sha,
                    "common_mask_sha256_legacy_roi": legacy_mask_sha,
                },
                "event_active_pixels_native": int(full_counts["common_event_active_pixels"]),
                "event_input_sha256": native_volume_sha,
                "gt_flow_sha256": gt_sha,
                "event_volume_shape": list(volume.shape),
                "event_volume_dtype": str(volume.dtype),
                "data_file_sha256": data.data_file_sha256,
                "gt_file_sha256": gt_file_sha256,
                "data_path_on_g1": str(Path(reported_root) / SEQUENCE_FOLDERS[sequence] / data_path.name),
                "gt_path_on_g1": str(Path(reported_root) / SEQUENCE_FOLDERS[sequence] / gt_path.name),
            }
            rows.append(row)

    per_window_durations = np.asarray(durations, dtype=np.float64)
    sequence_spec = protocol["sequence_specs"][sequence]
    registration = {
        "sequence": sequence,
        "data_path_on_g1": str(Path(reported_root) / SEQUENCE_FOLDERS[sequence] / data_path.name),
        "data_size_bytes": data_path.stat().st_size,
        "data_sha256": data.data_file_sha256,
        "data_hdf5_schema": {
            "events_shape": list(data.events_xyts_p.shape),
            "events_dtype_after_load": str(data.events_xyts_p.dtype),
            "aps_count": int(len(data.aps_timestamps_s)),
            "aps_first_time_s": float(data.aps_timestamps_s[0]),
            "aps_last_time_s": float(data.aps_timestamps_s[-1]),
            "event_first_time_s": float(data.event_timestamps_s[0]),
            "event_last_time_s": float(data.event_timestamps_s[-1]),
            "sensor_hw": [data.height, data.width],
        },
        "gt_path_on_g1": str(Path(reported_root) / SEQUENCE_FOLDERS[sequence] / gt_path.name),
        "gt_size_bytes": gt_path.stat().st_size,
        "gt_sha256": sha256_file(gt_path),
        "flow_gt_shape": gt_flow_shape,
        "flow_gt_dtype": gt_flow_dtype,
        "flow_gt_timestamp_count": int(len(gt_ts)),
        "flow_gt_timestamp_range_s": [float(gt_ts[0]), float(gt_ts[-1])],
        "calibration": calibration,
        "candidate_aps_indices_before_gt_intersection": len(candidate_indices),
        "eligible_registered_windows": len(rows),
        "excluded_by_sequence_boundary_rule": boundary_excluded,
        "excluded_by_gt_timestamp_coverage": excluded_gt_coverage,
        "empty_event_windows_kept": empty_event_windows,
        "empty_common_support_windows_kept": {
            "full_native": empty_common_full,
            "legacy_roi": empty_common_legacy,
        },
        "summary": {
            "total_raw_events_in_registered_windows": int(total_events),
            "mean_duration_s": float(per_window_durations.mean()) if len(per_window_durations) else None,
            "duration_percentiles_s": np.percentile(per_window_durations, [0, 50, 95, 100]).tolist() if len(per_window_durations) else [],
            "mean_active_pixels_native": float(active_pixels_total / len(rows)) if rows else None,
            "mean_common_valid_pixels_full_native": float(common_full_total / len(rows)) if rows else None,
            "mean_common_valid_pixels_legacy_roi": float(common_legacy_total / len(rows)) if rows else None,
        },
        "protocol_rule": sequence_spec["candidate_aps_index_rule"],
    }
    return registration, rows, boundary_excluded + excluded_gt_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory containing the complete verified HDF5 files and calibration ZIPs")
    parser.add_argument("--reported-root-on-g1", required=True,
                        help="the path that will be used by the future g1 inference run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", choices=SEQUENCES, default=list(SEQUENCES))
    args = parser.parse_args()

    base_manifest = json.loads(args.base_manifest.read_text())
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("real_scores_authorized") is not False:
        raise PermissionError("registration expects a candidate protocol with real scoring closed")
    if not set(args.sequences).issubset(set(protocol["sequences"])):
        raise ValueError("requested sequence is not declared in the candidate protocol")
    root = args.data_root.resolve()
    reported_root = args.reported_root_on_g1
    calibration_paths = {
        "indoor_flying": root / "indoor_flying" / "indoor_flying_calib.zip",
        "outdoor_day": root / "outdoor_day" / "outdoor_day_calib.zip",
    }
    required_calibrations = {
        "outdoor_day" if sequence == "outdoor_day1" else "indoor_flying"
        for sequence in args.sequences
    }
    calibration_details = {
        name: read_calibration(calibration_paths[name]) for name in sorted(required_calibrations)
    }

    registrations = {}
    all_rows: list[dict[str, Any]] = []
    for sequence in args.sequences:
        registration, rows, _ = inspect_one_sequence(
            sequence,
            root,
            reported_root,
            calibration_details["outdoor_day" if sequence == "outdoor_day1" else "indoor_flying"],
            protocol,
        )
        registrations[sequence] = registration
        all_rows.extend(rows)

    manifest = dict(base_manifest)
    manifest.update({
        "schema": "mvsec_dt1_test_manifest_candidate_v3",
        "schema_version": 3,
        "manifest_id": "mvsec-steflownet-dt1-fixed-test-20261021-v3-candidate",
        "protocol": str(args.protocol),
        "data_root_on_g1": reported_root,
        "download_manifest_on_g1": str(Path(reported_root) / "public_download_manifest.json"),
        "registered_aps_start_indices": {sequence: [] for sequence in SEQUENCES},
        "windows": all_rows,
        "file_registration": registrations,
        "calibrations": calibration_details,
        "registration": {
            "created_utc": utc_now(),
            "status": "registered_candidate_no_model_inference_no_real_scores",
            "sequences_processed": list(args.sequences),
            "model_weights_loaded": False,
            "real_predictions_computed": False,
            "real_scores_computed": False,
            "real_scoring_enabled": False,
            "window_count": len(all_rows),
            "event_input_contract": "pinned STE-style equal-event-count Gaussian ON/OFF representation; exactly 5 bins per dt1 APS interval",
            "event_window_contract": "raw event timestamps in (APS_i, APS_i+1]; label_time_s=end_time_s=APS_i+1",
            "gt_contract": "pinned estimate_corresponding_gt_flow semantics from exact flow_dist timestamps; native [u,v] endpoint displacement in px_per_interval",
            "mask_candidates": {
                "primary_full_native": "finite nonzero GT AND active pixel in the registered real event tensor on native 260x346 raster",
                "legacy_secondary": "same support restricted to the exact pinned STE center-crop ROI recorded per sequence",
                "empty_support": "keep window row and APS index; later metric is undefined and must be excluded from means, never set to zero",
            },
            "sequences": registrations,
            "source_script": "benchmark/sim2real/scripts/register_mvsec_dt1_test_windows.py",
            "source_code_bindings": {
                "registrar_sha256": sha256_file(Path(__file__).resolve()),
                "inference_adapter_sha256": sha256_file(Path(__file__).with_name("run_mvsec_candidate_inference.py")),
                "metric_evaluator_sha256": sha256_file(Path(__file__).with_name("stage4_mvsec_dt1_metrics.py")),
                "real_window_audit_sha256": sha256_file(Path(__file__).with_name("audit_mvsec_real_window_contract.py")),
                "protocol_sha256": sha256_file(args.protocol.resolve()),
                "base_manifest_sha256": sha256_file(args.base_manifest.resolve()),
            },
            "invocation": {
                "argv": sys.argv,
                "pid": os.getpid(),
                "host_python": platform.python_version(),
            },
        },
        "real_scores_authorized": False,
        "score_gate": {**manifest.get("score_gate", {}), "real_scores_authorized": False},
        "status": "candidate_pending_research_protocol_freeze_and_fixed_sample_inference",
    })
    registered_files: dict[str, Any] = {}
    for sequence, value in registrations.items():
        registered_files[sequence] = {
            "data_path_on_g1": value["data_path_on_g1"],
            "data_size_bytes": value["data_size_bytes"],
            "data_sha256": value["data_sha256"],
            "gt_path_on_g1": value["gt_path_on_g1"],
            "gt_size_bytes": value["gt_size_bytes"],
            "gt_sha256": value["gt_sha256"],
            "status": "hash_verified_registered_unscored",
        }
    calibration_entries = {
        name: {
            "path_on_g1": item["archive_path"],
            "size_bytes": item["archive_size_bytes"],
            "sha256": item["archive_sha256"],
            "status": "hash_verified_registered_unscored",
        }
        for name, item in calibration_details.items()
    }
    manifest["files"] = registered_files
    manifest["calibration_files"] = calibration_entries
    manifest["calibration_zip"] = calibration_entries.get("indoor_flying")
    for row in all_rows:
        manifest["registered_aps_start_indices"][row["sequence"]].append(row["aps_start_index"])
    manifest["source_download_manifest"] = {
        "path_used_for_input_inventory": str(root / "public_download_manifest.json"),
        "sha256": sha256_file(root / "public_download_manifest.json") if (root / "public_download_manifest.json").is_file() else None,
        "note": "the registered HDF5 paths and hashes above were independently read on the execution host",
    }
    manifest["registration"]["runtime"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "opencv": cv2.__version__,
        "torch_imported_for_pinned_adapter": torch.__version__,
        "torch_cuda_used": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps({
        "status": manifest["status"],
        "output": str(args.output),
        "window_count": len(all_rows),
        "real_scores_computed": False,
        "real_scoring_enabled": False,
        "sequences": {key: {
            "eligible_windows": value["eligible_registered_windows"],
            "empty_event_windows_kept": value["empty_event_windows_kept"],
            "empty_common_support_windows_kept": value["empty_common_support_windows_kept"],
        } for key, value in registrations.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
