#!/usr/bin/env python3
"""Summarize decoder failure modes and confidence signals from saved CSVs.

All labels are already post-inference evaluator fields in the inputs. This
script never changes model input or fits a predictor; confidence cutoffs are
calibrated only from the supplied native-validation heatmap outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def metric_rows(rows: list[dict[str, str]], value_key: str) -> dict[str, Any]:
    values = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
    return {"n": int(values.size), "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def auc_positive_high(scores: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() +
                  0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def gate_result(values: np.ndarray, errors: np.ndarray, threshold: float,
                reject_high: bool) -> dict[str, Any]:
    reject = values > threshold if reject_high else values < threshold
    bad = errors > 100.0
    accepted = ~reject
    return {
        "threshold": float(threshold),
        "rejected_count": int(reject.sum()),
        "rejected_fraction": float(reject.mean()),
        "large_errors_over_100mm": int(bad.sum()),
        "large_errors_caught": int((reject & bad).sum()),
        "large_error_recall": float((reject & bad).sum() / bad.sum()) if bad.any() else None,
        "accepted_count": int(accepted.sum()),
        "accepted_large_errors": int((accepted & bad).sum()),
        "accepted_large_error_fraction": float((accepted & bad).sum() / accepted.sum()) if accepted.any() else None,
        "accepted_error_mean_mm": float(errors[accepted].mean()) if accepted.any() else None,
        "accepted_error_p95_mm": float(np.percentile(errors[accepted], 95)) if accepted.any() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-csv", type=Path, required=True)
    parser.add_argument("--native-validation-dir", type=Path, required=True,
                        help="contains episode_004/005/006/heatmap_spatial_signature.csv")
    parser.add_argument("--scene-association-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--large-error-mm", type=float, default=100.0)
    parser.add_argument("--peak-far-px", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    direct_all = read_csv(args.direct_csv)
    direct = [row for row in direct_all
              if row["checkpoint"] == "ours_native_hdr" and row["decoder"] == "global_softargmax"]
    all_modes: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in direct_all:
        if row["checkpoint"] == "ours_native_hdr":
            all_modes[(row["episode_id"], row["cutoff_s"], row["checkpoint"])][row["decoder"]] = row
    errors = np.asarray([float(row["position_xy_error_mm"]) for row in direct], dtype=np.float64)
    far_peak = np.asarray([float(row["peak_pixel_distance_to_gt"]) > args.peak_far_px for row in direct])
    large = errors > args.large_error_mm
    decoder_by_checkpoint: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in direct_all:
        decoder_by_checkpoint[row["checkpoint"]][row["decoder"]] = None
    decoder_metrics = {}
    for checkpoint in decoder_by_checkpoint:
        decoder_metrics[checkpoint] = {}
        for decoder in decoder_by_checkpoint[checkpoint]:
            subset = [row for row in direct_all
                      if row["checkpoint"] == checkpoint and row["decoder"] == decoder]
            decoder_metrics[checkpoint][decoder] = metric_rows(subset, "position_xy_error_mm")

    peak_near = ~far_peak
    peak_condition = {
        "threshold_px": args.peak_far_px,
        "queries_with_peak_far_from_target": int(far_peak.sum()),
        "fraction_queries_with_peak_far_from_target": float(far_peak.mean()),
        "global_softargmax_errors_over_100mm": int(large.sum()),
        "large_errors_with_peak_far_from_target": int((large & far_peak).sum()),
        "large_errors_with_peak_near_target": int((large & peak_near).sum()),
        "among_peak_near_queries": {
            "n": int(peak_near.sum()),
            "global_errors_over_100mm": int(large[peak_near].sum()),
            "argmax_errors_over_100mm": sum(
                float(row["position_xy_error_mm"]) > args.large_error_mm
                for row in direct_all if row["checkpoint"] == "ours_native_hdr"
                and row["decoder"] == "heatmap_argmax"
                and float(row["peak_pixel_distance_to_gt"]) <= args.peak_far_px),
            "local_softargmax_errors_over_100mm": sum(
                float(row["position_xy_error_mm"]) > args.large_error_mm
                for row in direct_all if row["checkpoint"] == "ours_native_hdr"
                and row["decoder"] == "local_softargmax_r2"
                and float(row["peak_pixel_distance_to_gt"]) <= args.peak_far_px),
        },
        "among_peak_far_queries": {
            "n": int(far_peak.sum()),
            "global_errors_over_100mm": int(large[far_peak].sum()),
            "argmax_errors_over_100mm": sum(
                float(row["position_xy_error_mm"]) > args.large_error_mm
                for row in direct_all if row["checkpoint"] == "ours_native_hdr"
                and row["decoder"] == "heatmap_argmax"
                and float(row["peak_pixel_distance_to_gt"]) > args.peak_far_px),
            "local_softargmax_errors_over_100mm": sum(
                float(row["position_xy_error_mm"]) > args.large_error_mm
                for row in direct_all if row["checkpoint"] == "ours_native_hdr"
                and row["decoder"] == "local_softargmax_r2"
                and float(row["peak_pixel_distance_to_gt"]) > args.peak_far_px),
        },
        "target_heatmap_roi_mass_median_peak_near": float(np.median([
            float(row["heatmap_probability_in_target_roi"]) for row, is_far in zip(direct, far_peak) if not is_far
        ])),
        "target_heatmap_roi_mass_median_peak_far": float(np.median([
            float(row["heatmap_probability_in_target_roi"]) for row, is_far in zip(direct, far_peak) if is_far
        ])),
    }

    native_by_episode: dict[int, list[dict[str, str]]] = {}
    for eid in (4, 5, 6):
        path = args.native_validation_dir / f"episode_{eid:03d}" / "heatmap_spatial_signature.csv"
        native_by_episode[eid] = read_csv(path)
    native = [row for rows in native_by_episode.values() for row in rows]

    # These are matched task settings, not paired trajectories: the seeds differ.
    matched_specs = {
        4: {"direct_episode_id": 2, "speed_m_s": 0.30, "launch_y_m": 0.055,
            "native_seed": 2026104134, "direct_seed": 2026103422},
        5: {"direct_episode_id": 4, "speed_m_s": 0.40, "launch_y_m": 0.0,
            "native_seed": 2026104135, "direct_seed": 2026103424},
        6: {"direct_episode_id": 6, "speed_m_s": 0.50, "launch_y_m": -0.055,
            "native_seed": 2026104136, "direct_seed": 2026103426},
    }
    matched = []
    for native_eid, spec in matched_specs.items():
        native_rows = native_by_episode[native_eid]
        direct_rows = [row for row in direct if int(row["episode_id"]) == spec["direct_episode_id"]]
        matched.append({
            "native_episode_id": native_eid,
            **spec,
            "comparison_boundary": "same speed and launch y; different seed and physical trajectory",
            "native": {
                "position_xy_error_mm": metric_rows(native_rows, "position_xy_error_mm"),
                "heatmap_entropy_normalized": metric_rows(native_rows, "heatmap_entropy_normalized"),
                "heatmap_peak_probability": metric_rows(native_rows, "heatmap_peak_probability"),
                "heatmap_probability_in_target_roi": metric_rows(native_rows, "heatmap_probability_in_target_roi"),
                "event_count_25ms": metric_rows(native_rows, "event_count_25ms"),
            },
            "direct_hdr": {
                "position_xy_error_mm": metric_rows(direct_rows, "position_xy_error_mm"),
                "heatmap_entropy_normalized": metric_rows(direct_rows, "heatmap_entropy_normalized"),
                "heatmap_peak_probability": metric_rows(direct_rows, "heatmap_peak_probability"),
                "heatmap_probability_in_target_roi": metric_rows(direct_rows, "heatmap_probability_in_target_roi"),
                "event_count_25ms": metric_rows(direct_rows, "event_count_25ms"),
            },
        })

    entropy = np.asarray([float(row["heatmap_entropy_normalized"]) for row in direct])
    peak_prob = np.asarray([float(row["heatmap_peak_probability"]) for row in direct])
    peak_gap = np.asarray([np.linalg.norm([
        float(row["pred_u_px"]) - float(row["peak_u_px"]),
        float(row["pred_v_px"]) - float(row["peak_v_px"]),
    ]) for row in direct])
    native_entropy = np.asarray([float(row["heatmap_entropy_normalized"]) for row in native])
    native_peak = np.asarray([float(row["heatmap_peak_probability"]) for row in native])
    calibration = {
        "source": "three native-HDR validation episodes (194 correlated windows; checkpoint-selection validation set)",
        "entropy_p95_threshold": float(np.percentile(native_entropy, 95)),
        "peak_probability_p05_threshold": float(np.percentile(native_peak, 5)),
    }

    association = json.loads(args.scene_association_summary.read_text(encoding="utf-8"))
    result = {
        "status": "completed",
        "purpose": "offline failure localization; not a ranking, deployed decoder, or closed-loop result",
        "gt_boundary": "GT and robot trace used only after inference for scoring/projection diagnostics; no GT enters events or model inputs",
        "direct_reference": {
            "queries": len(direct),
            "event_archive_episodes": 9,
            "checkpoint": "ours_native_hdr",
            "current_decoder": "global_softargmax",
            "large_error_threshold_mm": args.large_error_mm,
            "global_error_over_100mm_count": int(large.sum()),
            "global_error_over_100mm_fraction": float(large.mean()),
            "decoder_metrics_all_models": decoder_metrics,
            "peak_failure_decomposition": peak_condition,
            "confidence_auc_for_global_error_over_100mm": {
                "normalized_entropy_high_is_failure": auc_positive_high(entropy, large),
                "peak_probability_low_is_failure": auc_positive_high(-peak_prob, large),
                "global_softargmax_to_peak_gap_high_is_failure": auc_positive_high(peak_gap, large),
            },
        },
        "confidence_gate_counterfactual": {
            "calibration": calibration,
            "entropy_threshold": gate_result(entropy, errors, calibration["entropy_p95_threshold"], True),
            "peak_probability_threshold": gate_result(peak_prob, errors, calibration["peak_probability_p05_threshold"], False),
            "boundary": "thresholds calibrated on selected native validation windows then evaluated on already-seen shared Direct-HDR development data; correlated, not independent, not closed loop",
        },
        "matched_task_configuration_domain_contrast": matched,
        "tcp_scene_association": association["counts"],
        "inputs": {
            "direct_csv_sha256": sha256(args.direct_csv),
            "native_validation_csv_sha256_by_episode": {
                str(eid): sha256(args.native_validation_dir / f"episode_{eid:03d}" / "heatmap_spatial_signature.csv")
                for eid in native_by_episode
            },
            "scene_association_summary_sha256": sha256(args.scene_association_summary),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "queries": len(direct),
        "large_errors": int(large.sum()),
        "peak_far_count": int(far_peak.sum()),
        "entropy_auc": result["direct_reference"]["confidence_auc_for_global_error_over_100mm"]["normalized_entropy_high_is_failure"],
        "native_p95_entropy_gate": result["confidence_gate_counterfactual"]["entropy_threshold"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
