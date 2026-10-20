#!/usr/bin/env python3
"""Audit one pre-registered MVSEC window without loading a model or scoring.

This reads raw public event timestamps, one APS interval, the associated official
flow maps, and the deterministic STE event representation. It emits provenance
and contract checks only; it never evaluates a candidate prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from run_mvsec_candidate_inference import (
    FLOW_UNIT,
    NATIVE_HW,
    gaussian_count_representation,
    normalize_polarity,
    official_flow_association_from_h5,
    verify_event_index_boundary,
    MVSECData,
)
from stage4_mvsec_dt1_metrics import common_valid_mask

SEQUENCE_FOLDER = {
    "indoor_flying1": "indoor_flying",
    "indoor_flying2": "indoor_flying",
    "indoor_flying3": "indoor_flying",
    "outdoor_day1": "outdoor_day",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", choices=tuple(SEQUENCE_FOLDER), required=True)
    parser.add_argument("--aps-start-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    folder = args.data_root / SEQUENCE_FOLDER[args.sequence]
    data_path = folder / f"{args.sequence}_data.hdf5"
    gt_path = folder / f"{args.sequence}_gt.hdf5"
    index = args.aps_start_index
    with h5py.File(data_path, "r") as data_file:
        base = "davis/left"
        events_ds = data_file[f"{base}/events"]
        aps_ts = np.asarray(data_file[f"{base}/image_raw_ts"], dtype=np.float64).reshape(-1)
        official_indices = np.rint(np.asarray(
            data_file[f"{base}/image_raw_event_inds"], dtype=np.float64
        ).reshape(-1)).astype(np.int64)
        image_shape = tuple(data_file[f"{base}/image_raw"].shape)
        if events_ds.ndim != 2 or events_ds.shape[1] != 4:
            raise ValueError(f"MVSEC events must be [N,4], got {events_ds.shape}")
        if not 0 <= index or index + 1 >= len(aps_ts):
            raise IndexError(f"APS start index {index} is out of range for {len(aps_ts)} endpoints")
        if image_shape != (len(aps_ts), *NATIVE_HW):
            raise ValueError(f"APS raster shape does not match native sensor: {image_shape}")
        event_ts = np.asarray(events_ds[:, 2], dtype=np.float64)
        if not np.isfinite(event_ts).all() or np.any(np.diff(event_ts) < 0):
            raise ValueError("raw event timestamps must be finite and sorted")
        start_s, end_s = float(aps_ts[index]), float(aps_ts[index + 1])
        left = int(np.searchsorted(event_ts, start_s, side="right"))
        right = int(np.searchsorted(event_ts, end_s, side="right"))
        raw = np.asarray(events_ds[left:right], dtype=np.float64)

    raw[:, 3] = normalize_polarity(raw[:, 3])
    if len(raw) and (np.any(raw[:, 2] <= start_s) or np.any(raw[:, 2] > end_s)):
        raise RuntimeError("independent event slice violates the declared (start,end] interval")
    representation = gaussian_count_representation(raw, *NATIVE_HW)

    with h5py.File(gt_path, "r") as gt_file:
        base = "davis/left"
        flow_ds = gt_file[f"{base}/flow_dist"]
        gt_ts = np.asarray(gt_file[f"{base}/flow_dist_ts"], dtype=np.float64).reshape(-1)
        if flow_ds.ndim != 4 or tuple(flow_ds.shape[1:]) != (2, *NATIVE_HW):
            raise ValueError(f"flow_dist must be [N,2,{NATIVE_HW[0]},{NATIVE_HW[1]}], got {flow_ds.shape}")
        ground_truth, association = official_flow_association_from_h5(
            flow_ds, gt_ts, start_s, end_s
        )

    event_boundaries = np.rint(np.asarray([left, right], dtype=np.float64)).astype(np.int64)
    audit_data = MVSECData(
        events_xyts_p=np.empty((0, 4), dtype=np.float64),
        event_timestamps_s=event_ts,
        aps_timestamps_s=aps_ts,
        event_index_boundaries=official_indices,
        height=NATIVE_HW[0],
        width=NATIVE_HW[1],
        data_file_sha256=sha256_file(data_path),
        source_full_file=None,
        source_full_file_sha256=None,
        registration_json_sha256=None,
    )
    endpoint_checks = [
        verify_event_index_boundary(audit_data, index),
        verify_event_index_boundary(audit_data, index + 1),
    ]
    if endpoint_checks[0]["timestamp_right_search_index"] != int(event_boundaries[0]):
        raise RuntimeError("independent start event boundary changed during registration")
    if endpoint_checks[1]["timestamp_right_search_index"] != int(event_boundaries[1]):
        raise RuntimeError("independent end event boundary changed during registration")
    common, support = common_valid_mask(ground_truth, representation, representation)

    result: dict[str, Any] = {
        "schema": "mvsec_real_window_contract_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass_unscored_real_file_window_audit",
        "sequence": args.sequence,
        "aps_start_index": index,
        "data_path": str(data_path.resolve()),
        "data_size_bytes": data_path.stat().st_size,
        "data_sha256": audit_data.data_file_sha256,
        "gt_path": str(gt_path.resolve()),
        "gt_size_bytes": gt_path.stat().st_size,
        "gt_sha256": sha256_file(gt_path),
        "aps_count": int(len(aps_ts)),
        "gt_timestamp_count": int(len(gt_ts)),
        "start_time_s": start_s,
        "end_time_s": end_s,
        "duration_ms": (end_s - start_s) * 1000.0,
        "event_window_convention": "(start_time_s, end_time_s]",
        "event_window_index_bounds": [left, right],
        "event_count_independent_searchsorted": right - left,
        "events_strictly_after_start_at_or_before_end": True,
        "event_endpoint_checks": endpoint_checks,
        "event_representation_shape": list(representation.shape),
        "event_representation_dtype": str(representation.dtype),
        "event_representation_sum": int(representation.sum()),
        "flow_gt_shape": list(ground_truth.shape),
        "flow_gt_unit": FLOW_UNIT,
        "gt_association": association,
        "finite_gt_pixels": int(np.isfinite(ground_truth).all(axis=0).sum()),
        "common_event_active_finite_nonzero_gt_pixels": int(common.sum()),
        "support_counts": support,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__},
        "candidate_weights_loaded": False,
        "predictions_computed": False,
        "real_scores_computed": False,
        "real_scoring_enabled": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
