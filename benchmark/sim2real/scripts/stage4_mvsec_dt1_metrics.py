#!/usr/bin/env python3
"""Stage 4 MVSEC dt=1 metric contract and fail-closed paired evaluator.

The self-test uses only hand-constructed arrays. Real artifact evaluation is
blocked unless a separately reviewed protocol and authorization record are
provided. No real data is opened by --self-test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FLOW_UNIT = "px_per_interval"
SENSOR_HW = (260, 346)
EVENT_SHAPE = (2, *SENSOR_HW, 5)  # ON/OFF, H, W, five STE slices for dt1.
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "protocol" / "mvsec_dt1_protocol_candidate_20261019.json"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "mvsec_dt1_test_manifest_candidate_20261019.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _binary_mask(mask: np.ndarray, shape: tuple[int, int], label: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {value.shape}")
    if not (np.issubdtype(value.dtype, np.bool_) or np.issubdtype(value.dtype, np.number)):
        raise ValueError(f"{label} must be boolean or numeric 0/1")
    if not np.isfinite(value).all() or not np.isin(value, (0, 1)).all():
        raise ValueError(f"{label} must contain only finite boolean/0/1 values")
    return value.astype(bool, copy=False)


def event_support(event_volume: np.ndarray) -> np.ndarray:
    """Return native-raster active pixels from one dt1 STE input tensor."""
    events = np.asarray(event_volume)
    if events.shape != EVENT_SHAPE:
        raise ValueError(f"dt1 event representation must be {EVENT_SHAPE}, got {events.shape}")
    if not np.issubdtype(events.dtype, np.number) or not np.isfinite(events).all():
        raise ValueError("event representation must be finite numeric")
    if np.any(events < 0):
        raise ValueError("STE count representation cannot contain negative values")
    return np.any(events > 0, axis=(0, 3))


def common_valid_mask(
    ground_truth: np.ndarray,
    eventgenesis_events: np.ndarray,
    v2e_events: np.ndarray,
    *,
    evaluation_rows: int = SENSOR_HW[0],
) -> tuple[np.ndarray, dict[str, int]]:
    """Build one pair-shared support; reject mismatched real event inputs.

    Both candidate models receive the exact same public real event stream and
    deterministic STE encoding. Equality is checked before making the mask so
    one arm cannot get a more favorable evaluation region.
    """
    gt = np.asarray(ground_truth)
    if gt.ndim != 3 or gt.shape[0] != 2 or gt.shape[1:] != SENSOR_HW:
        raise ValueError(f"MVSEC dt1 GT must have shape [2,{SENSOR_HW[0]},{SENSOR_HW[1]}], got {gt.shape}")
    if not 0 < int(evaluation_rows) <= SENSOR_HW[0]:
        raise ValueError(f"evaluation_rows must be in [1,{SENSOR_HW[0]}]")
    if not np.issubdtype(gt.dtype, np.number):
        raise ValueError("GT flow must be numeric")
    eg = np.asarray(eventgenesis_events)
    v2e = np.asarray(v2e_events)
    eg_support = event_support(eg)
    v2e_support = event_support(v2e)
    if eg.shape != v2e.shape or not np.array_equal(eg, v2e):
        raise ValueError("paired real event representations differ; refusing arm-specific support")
    if not np.array_equal(eg_support, v2e_support):
        raise RuntimeError("identical event representations produced different active support")
    finite = np.isfinite(gt).all(axis=0)
    finite_gt = np.where(np.isfinite(gt), gt, 0.0).astype(np.float64, copy=False)
    magnitude = np.hypot(finite_gt[0], finite_gt[1])
    nonzero = magnitude > 0
    active = eg_support  # equal to both-arm common support by the assertion above.
    row_domain = np.arange(SENSOR_HW[0])[:, None] < int(evaluation_rows)
    valid = finite & nonzero & active & row_domain
    return valid, {
        "sensor_pixels": int(np.prod(SENSOR_HW)),
        "finite_gt_pixels": int(finite.sum()),
        "finite_gt_pixels_within_evaluation_crop": int((finite & row_domain).sum()),
        "nonzero_finite_gt_pixels": int((finite & nonzero).sum()),
        "common_event_active_pixels": int(active.sum()),
        "common_event_active_pixels_within_evaluation_crop": int((active & row_domain).sum()),
        "evaluation_rows": int(evaluation_rows),
        "common_valid_pixels": int(valid.sum()),
        "coverage_of_nonzero_finite_gt_within_evaluation_crop": float(
            valid.sum() / max(1, int((finite & nonzero & row_domain).sum()))
        ),
    }


def compute_window_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    prediction_unit: str = FLOW_UNIT,
    ground_truth_unit: str = FLOW_UNIT,
) -> dict[str, Any]:
    """Strict vector AEE and candidate/legacy threshold fractions.

    Candidate outlier definition: EPE > 3 px AND EPE > 0.05 * |GT|.
    The standalone >=3 px fraction is reported only for comparison with the
    legacy STE helper, never as the candidate outlier metric.
    """
    pred = np.asarray(prediction)
    gt = np.asarray(ground_truth)
    if pred.shape != (2, *SENSOR_HW) or gt.shape != pred.shape:
        raise ValueError(f"prediction and GT must both be [2,{SENSOR_HW[0]},{SENSOR_HW[1]}], got {pred.shape} and {gt.shape}")
    if not np.issubdtype(pred.dtype, np.number) or not np.isfinite(pred).all():
        raise ValueError("prediction must be finite numeric [u,v] flow")
    if not np.issubdtype(gt.dtype, np.number):
        raise ValueError("GT must be numeric [u,v] flow")
    if prediction_unit != FLOW_UNIT or ground_truth_unit != FLOW_UNIT:
        raise ValueError(f"both flow units must be {FLOW_UNIT!r}")
    mask = _binary_mask(valid_mask, SENSOR_HW, "valid mask")
    count = int(mask.sum())
    if count == 0:
        return {
            "status": "undefined_empty_common_support",
            "valid_pixels": 0,
            "aee_px_per_interval": None,
            "outlier_3px_and_5pct_percent": None,
            "legacy_epe_ge_3px_percent": None,
            "reference_epe_lt_3px_percent": None,
        }
    if not np.isfinite(gt[:, mask]).all():
        raise ValueError("GT contains non-finite values inside common valid mask")
    du = pred[0, mask].astype(np.float64) - gt[0, mask].astype(np.float64)
    dv = pred[1, mask].astype(np.float64) - gt[1, mask].astype(np.float64)
    epe = np.hypot(du, dv)
    gt_mag = np.hypot(gt[0, mask].astype(np.float64), gt[1, mask].astype(np.float64))
    if not np.isfinite(epe).all() or not np.isfinite(gt_mag).all() or np.any(gt_mag <= 0):
        raise ValueError("scored EPE and GT magnitudes must be finite; GT magnitude must be nonzero")
    outlier = (epe > 3.0) & (epe > 0.05 * gt_mag)
    return {
        "status": "scored",
        "valid_pixels": count,
        "aee_px_per_interval": float(epe.mean()),
        "outlier_3px_and_5pct_percent": float(100.0 * outlier.mean()),
        "legacy_epe_ge_3px_percent": float(100.0 * (epe >= 3.0).mean()),
        "reference_epe_lt_3px_percent": float(100.0 * (epe < 3.0).mean()),
    }


def all_valid_gt_mask(ground_truth: np.ndarray, evaluation_rows: int) -> np.ndarray:
    """Optional diagnostic support: finite, nonzero GT within the official crop."""
    gt = np.asarray(ground_truth)
    if gt.shape != (2, *SENSOR_HW) or not np.issubdtype(gt.dtype, np.number):
        raise ValueError(f"GT must be numeric [2,{SENSOR_HW[0]},{SENSOR_HW[1]}], got {gt.shape}")
    if not 0 < int(evaluation_rows) <= SENSOR_HW[0]:
        raise ValueError(f"evaluation_rows must be in [1,{SENSOR_HW[0]}]")
    finite = np.isfinite(gt).all(axis=0)
    safe = np.where(np.isfinite(gt), gt, 0.0)
    nonzero = np.hypot(safe[0], safe[1]) > 0
    rows = np.arange(SENSOR_HW[0])[:, None] < int(evaluation_rows)
    return finite & nonzero & rows


def authorize_real_evaluation(protocol_path: Path, manifest_path: Path, authorization_path: Path) -> None:
    """Fail closed; authorization must bind reviewed files by exact SHA-256."""
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if protocol.get("status") != "frozen_by_research_review" or protocol.get("real_scores_authorized") is not True:
        raise PermissionError("MVSEC real scoring is closed: protocol is not frozen and explicitly authorized")
    if manifest.get("status") != "frozen_by_research_review" or manifest.get("real_scores_authorized") is not True:
        raise PermissionError("MVSEC real scoring is closed: test manifest is not frozen and explicitly authorized")
    authorization = json.loads(authorization_path.read_text())
    expected = {
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    if authorization.get("authorized_real_test") is not True:
        raise PermissionError("authorization record does not authorize real test scoring")
    if authorization.get("authorization_id") in (None, "") or authorization.get("approved_by") in (None, ""):
        raise PermissionError("authorization record must identify the research approval")
    for key, digest in expected.items():
        if authorization.get(key) != digest:
            raise PermissionError(f"authorization {key} does not match reviewed protocol artifact")


def evaluate_pair_manifest(manifest_path: Path, protocol_path: Path, authorization_path: Path) -> dict[str, Any]:
    """Score registered paired artifact NPZs after the external approval gate."""
    authorize_real_evaluation(protocol_path, manifest_path, authorization_path)
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    expected_by_sequence = manifest.get("registered_aps_start_indices", {})
    seen_by_sequence: dict[str, set[int]] = {seq: set() for seq in protocol["sequences"]}
    rows_by_sequence: dict[str, list[dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    for row in manifest["windows"]:
        if int(row.get("dt", -1)) != 1:
            raise ValueError("primary evaluator accepts only explicitly registered dt1 windows")
        start_s = float(row["start_time_s"])
        end_s = float(row["end_time_s"])
        label_s = float(row["label_time_s"])
        if not np.isfinite([start_s, end_s, label_s]).all() or end_s <= start_s:
            raise ValueError("registered APS/event/label timestamps must be finite with end > start")
        if label_s != end_s or row.get("event_window_convention") != "(start_time_s, end_time_s]":
            raise ValueError("dt1 event right endpoint must equal label time under the registered (start,end] rule")
        sequence = str(row["sequence"])
        if sequence not in seen_by_sequence:
            raise ValueError(f"unregistered sequence in paired test manifest: {sequence}")
        sample_index = int(row["aps_start_index"])
        if sample_index in seen_by_sequence[sequence]:
            raise ValueError(f"duplicate registered sample {sequence} APS {sample_index}")
        seen_by_sequence[sequence].add(sample_index)
        if sample_index not in expected_by_sequence.get(sequence, []):
            raise ValueError(f"unregistered APS index {sample_index} for {sequence}")
        sequence_spec = protocol["sequence_specs"][sequence]
        arm_paths = row["artifacts"]
        required_arms = {"eventgenesis", "v2e"}
        if set(arm_paths) != required_arms:
            raise ValueError(f"each registered window must list exactly {sorted(required_arms)}")
        loaded = {}
        for arm, item in arm_paths.items():
            path = Path(item["path"])
            if not path.is_file() or sha256_file(path) != item.get("sha256"):
                raise ValueError(f"missing or changed {arm} artifact for {sequence} APS {row['aps_start_index']}")
            if item.get("checkpoint_sha256") != protocol["checkpoint_index"][arm].get("sha256"):
                raise ValueError(f"{arm} checkpoint hash differs from frozen synthetic-best index")
            with np.load(path, allow_pickle=False) as bundle:
                loaded[arm] = {key: np.asarray(bundle[key]) for key in (
                    "event_input_native_uint8", "prediction_px_per_interval", "gt_px_per_interval"
                )}
        eg, v2e = loaded["eventgenesis"], loaded["v2e"]
        if not np.array_equal(eg["gt_px_per_interval"], v2e["gt_px_per_interval"], equal_nan=True):
            raise ValueError(f"paired GT differs for {sequence} APS {row['aps_start_index']}")
        evaluation_rows = int(sequence_spec["evaluation_rows"])
        valid, counts = common_valid_mask(eg["gt_px_per_interval"], eg["event_input_native_uint8"],
                                          v2e["event_input_native_uint8"], evaluation_rows=evaluation_rows)
        all_gt_valid = all_valid_gt_mask(eg["gt_px_per_interval"], evaluation_rows)
        if not all_gt_valid.any():
            raise ValueError(f"registered frame has no finite nonzero GT in its evaluation crop: {sequence} APS {sample_index}")
        result_row = {"sequence": sequence, "aps_start_index": int(row["aps_start_index"]),
                      "start_time_s": start_s, "end_time_s": end_s,
                      "event_count": row["event_count"], "support_counts": counts,
                      "evaluation_rows": evaluation_rows,
                      "event_input_sha256": hashlib.sha256(np.ascontiguousarray(eg["event_input_native_uint8"]).view(np.uint8)).hexdigest(),
                      "eventgenesis": compute_window_metrics(eg["prediction_px_per_interval"], eg["gt_px_per_interval"], valid),
                      "eventgenesis_all_valid_gt_aee_px_per_interval": compute_window_metrics(
                          eg["prediction_px_per_interval"], eg["gt_px_per_interval"], all_gt_valid
                      )["aee_px_per_interval"],
                      "v2e": compute_window_metrics(v2e["prediction_px_per_interval"], eg["gt_px_per_interval"], valid),
                      "v2e_all_valid_gt_aee_px_per_interval": compute_window_metrics(
                          v2e["prediction_px_per_interval"], eg["gt_px_per_interval"], all_gt_valid
                      )["aee_px_per_interval"]}
        if valid.any():
            rows_by_sequence.setdefault(sequence, []).append(result_row)
        else:
            excluded.append({"sequence": sequence, "aps_start_index": int(row["aps_start_index"]),
                             "start_time_s": start_s, "end_time_s": end_s,
                             "reason": "empty common finite-nonzero-GT intersect event-active support"})
    required_sequences = protocol["sequences"]
    unprocessed = {
        seq: sorted(set(int(x) for x in expected_by_sequence.get(seq, [])) - seen_by_sequence.get(seq, set()))
        for seq in required_sequences
    }
    missing_registration = {seq: ids for seq, ids in unprocessed.items() if ids}
    if missing_registration:
        raise ValueError(f"registered APS windows were not evaluated: {missing_registration}")
    missing = [seq for seq in required_sequences if not rows_by_sequence.get(seq)]
    if missing:
        raise ValueError(f"cannot produce equal-sequence aggregate; no scored support for sequences {missing}")
    summary = {}
    for arm in ("eventgenesis", "v2e"):
        per_sequence = {}
        for seq in required_sequences:
            rows = rows_by_sequence[seq]
            per_sequence[seq] = {
                "scored_windows": len(rows),
                "aee_px_per_interval": float(np.mean([r[arm]["aee_px_per_interval"] for r in rows])),
                "outlier_3px_and_5pct_percent": float(np.mean([r[arm]["outlier_3px_and_5pct_percent"] for r in rows])),
                "legacy_epe_ge_3px_percent": float(np.mean([r[arm]["legacy_epe_ge_3px_percent"] for r in rows])),
            }
        summary[arm] = {
            "per_sequence": per_sequence,
            "equal_sequence_mean_aee_px_per_interval": float(np.mean([x["aee_px_per_interval"] for x in per_sequence.values()])),
            "equal_sequence_mean_outlier_3px_and_5pct_percent": float(np.mean([x["outlier_3px_and_5pct_percent"] for x in per_sequence.values()])),
            "equal_sequence_mean_all_valid_gt_aee_px_per_interval": float(np.mean([
                np.mean([r[f"{arm}_all_valid_gt_aee_px_per_interval"] for r in rows_by_sequence[seq]])
                for seq in required_sequences
            ])),
        }
    return {
        "status": "authorized_real_score",
        "scope": "MVSEC dt1 fixed test; evaluation region, masks, and weights bound by frozen manifests",
        "protocol_sha256": sha256_file(protocol_path),
        "test_manifest_sha256": sha256_file(manifest_path),
        "authorization_record": str(authorization_path.resolve()),
        "checkpoint_index": protocol["checkpoint_index"],
        "checkpoint_selection_or_generator_adjustment_from_real_scores": False,
        "dt": 1,
        "sequence_summary": summary,
        "scored_windows": {seq: len(rows) for seq, rows in rows_by_sequence.items()},
        "common_empty_windows_excluded": excluded,
        "windows": [r for rows in rows_by_sequence.values() for r in rows],
    }


def self_test() -> dict[str, Any]:
    """Small hand-calculated checks; no dataset, checkpoint, or model loaded."""
    gt = np.zeros((2, *SENSOR_HW), dtype=np.float32)
    pred = np.zeros_like(gt)
    event = np.zeros(EVENT_SHAPE, dtype=np.uint8)
    # Four hand-auditable valid pixels: EPE 0, 2, 4, 4; GT magnitudes 5, 1, 10, 100.
    xy = [(0, 0), (0, 1), (0, 2), (0, 3)]
    gt[0, 0, 0], gt[1, 0, 0] = 3, 4
    pred[0, 0, 0], pred[1, 0, 0] = 3, 4
    gt[0, 0, 1], pred[0, 0, 1], pred[1, 0, 1] = 1, 1, 2
    gt[0, 0, 2], pred[0, 0, 2] = 10, 14
    gt[0, 0, 3], pred[0, 0, 3] = 100, 104
    for y, x in xy:
        event[0, y, x, 0] = 1
    # A zero-flow event pixel is excluded; an active pixel with NaN GT is excluded.
    event[1, 0, 4, 1] = 1
    event[0, 0, 5, 2] = 1
    gt[0, 0, 5] = np.nan
    valid, counts = common_valid_mask(gt, event, event.copy())
    metrics = compute_window_metrics(pred, gt, valid)
    assert metrics["valid_pixels"] == 4
    assert metrics["aee_px_per_interval"] == 2.5
    assert metrics["outlier_3px_and_5pct_percent"] == 25.0
    assert metrics["legacy_epe_ge_3px_percent"] == 50.0
    assert metrics["reference_epe_lt_3px_percent"] == 50.0
    assert counts["common_event_active_pixels"] == 6
    assert counts["common_valid_pixels"] == 4
    empty = np.zeros(SENSOR_HW, dtype=bool)
    empty_metrics = compute_window_metrics(pred, gt, empty)
    assert empty_metrics["status"] == "undefined_empty_common_support"
    assert empty_metrics["aee_px_per_interval"] is None
    crop_gt = np.zeros_like(gt)
    crop_events = np.zeros_like(event)
    crop_gt[0, 189, 10] = 1.0
    crop_gt[0, 190, 10] = 1.0
    crop_events[0, 189, 10, 0] = 1
    crop_events[0, 190, 10, 0] = 1
    crop_valid, crop_counts = common_valid_mask(crop_gt, crop_events, crop_events.copy(), evaluation_rows=190)
    assert crop_valid[189, 10] and not crop_valid[190, 10]
    assert crop_counts["common_valid_pixels"] == 1
    crop_gt_valid = all_valid_gt_mask(crop_gt, 190)
    assert crop_gt_valid[189, 10] and not crop_gt_valid[190, 10] and int(crop_gt_valid.sum()) == 1
    mismatched = event.copy()
    mismatched[0, 1, 1, 0] = 1
    try:
        common_valid_mask(gt, event, mismatched)
    except ValueError as exc:
        assert "representations differ" in str(exc)
    else:
        raise AssertionError("arm-specific event inputs must fail closed")
    try:
        compute_window_metrics(pred[:1], gt, valid)
    except ValueError as exc:
        assert "both be [2,260,346]" in str(exc)
    else:
        raise AssertionError("one-component flow must not broadcast")
    try:
        compute_window_metrics(pred, gt, valid, prediction_unit="px_per_second")
    except ValueError as exc:
        assert "flow units" in str(exc)
    else:
        raise AssertionError("flow-unit mismatch must fail")
    try:
        authorize_real_evaluation(DEFAULT_PROTOCOL, DEFAULT_MANIFEST, Path("/nonexistent/approval.json"))
    except FileNotFoundError:
        pass
    except PermissionError:
        pass
    else:
        raise AssertionError("unapproved real scoring must remain closed")
    return {
        "status": "pass",
        "scope": "synthetic hand-calculated contract only; no real data or model artifacts read",
        "manual_epe_px": [0.0, 2.0, 4.0, 4.0],
        "manual_gt_magnitude_px": [5.0, 1.0, 10.0, 100.0],
        "expected_aee_px_per_interval": 2.5,
        "observed_aee_px_per_interval": metrics["aee_px_per_interval"],
        "expected_outlier_3px_and_5pct_percent": 25.0,
        "observed_outlier_3px_and_5pct_percent": metrics["outlier_3px_and_5pct_percent"],
        "legacy_ge3_percent_is_diagnostic_only": metrics["legacy_epe_ge_3px_percent"],
        "support_counts": counts,
        "outdoor_top190_crop_check": {
            "valid_pixels": crop_counts["common_valid_pixels"],
            "row_189_included": bool(crop_valid[189, 10]),
            "row_190_excluded": bool(not crop_valid[190, 10]),
        },
        "empty_common_support": empty_metrics,
        "arm_input_mismatch": "rejected",
        "component_broadcast": "rejected",
        "real_score_authorization": "closed_without_review-linked authorization record",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="hand-calculated CPU contract only")
    parser.add_argument("--manifest", type=Path, help="registered paired artifact manifest")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--authorization", type=Path, help="review authorization bound to protocol and manifest hashes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.manifest is None or args.authorization is None or args.output is None:
        parser.error("real scoring requires --manifest, --authorization, and --output; use --self-test for controlled examples")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation output: {args.output}")
    report = evaluate_pair_manifest(args.manifest, args.protocol, args.authorization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "scored_windows": report["scored_windows"]}, indent=2))


if __name__ == "__main__":
    main()
