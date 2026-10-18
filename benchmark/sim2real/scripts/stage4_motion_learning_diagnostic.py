#!/usr/bin/env python3
"""Fixed-sample, paired motion-learning diagnostic for Stage 4 checkpoints.

This is an evaluation-only synthetic diagnostic. It never reads training
samples or real data and does not affect checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch


SAMPLE_INDICES = (0, 9, 18, 27, 36, 45, 54, 65)
GT_MOTION_THRESHOLDS_PX_PER_30MS = (0.5, 1.0)
NEAR_ZERO_THRESHOLDS_PX_PER_30MS = (0.1, 0.5)
REGION_NAMES = ("full_valid", "shared_event_active_union", "gt_motion_ge_0_5", "gt_motion_ge_1_0")
STRATA = {
    0: "camera_motion_static_scene",
    1: "independent_rigid_motion_static_camera",
    2: "camera_and_independent_rigid_motion",
    3: "combined_motion_with_occlusion",
}
EXPECTED_UPDATE = 4554
EXPECTED_NEXT_EPOCH = 23
EXPECTED_RESOLUTION = (260, 346)
FLOW_UNIT = "px_per_30ms_interval"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_selected(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        out = {}
        for key in ("events", "gray_prev", "gray_next", "flow_gt", "flow_valid",
                    "event_counts", "event_start_s", "event_end_s", "sample_ids",
                    "camera_intrinsics_K_px", "resolution_hw", "label_sha256"):
            value = np.asarray(archive[key])
            if value.ndim > 0 and value.shape[0] == 66:
                value = value[np.asarray(SAMPLE_INDICES)]
            out[key] = value.copy()
    return out


def load_checkpoint(path: Path, generator: str, model_root: Path, data_manifest_sha: str,
                    config_sha: str, device: torch.device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "generator": (saved.get("generator"), generator),
        "seed": (saved.get("seed"), 0),
        "optimizer_update": (saved.get("optimizer_update"), EXPECTED_UPDATE),
        "next_epoch": (saved.get("next_epoch"), EXPECTED_NEXT_EPOCH),
        "next_position_samples": (saved.get("next_position_samples"), 0),
        "flow_unit": (saved.get("flow_unit"), FLOW_UNIT),
        "data_manifest_sha256": (saved.get("data_manifest_sha256"), data_manifest_sha),
        "config_sha256": (saved.get("config_sha256"), config_sha),
        "resume_phase": (saved.get("resume_phase"), "train"),
        "resume_allowed": (saved.get("resume_allowed"), True),
    }
    failed = {key: values for key, values in checks.items() if values[0] != values[1]}
    if failed:
        raise RuntimeError(f"{generator} checkpoint contract mismatch: {failed}")
    if saved.get("epoch") != EXPECTED_NEXT_EPOCH - 1:
        raise RuntimeError(f"{generator} checkpoint is not the expected completed epoch boundary")
    if not isinstance(saved.get("scheduler"), dict):
        raise RuntimeError(f"{generator} checkpoint lacks scheduler state")
    scheduler_epoch = int(saved["scheduler"].get("last_epoch", -1))
    if scheduler_epoch != EXPECTED_NEXT_EPOCH:
        raise RuntimeError(f"{generator} scheduler boundary mismatch: {scheduler_epoch}")

    sys.path.insert(0, str(model_root))
    script_dir = Path(__file__).resolve().parent
    compat_dir = script_dir.parent / "compat"
    sys.path.insert(0, str(compat_dir))
    from run_steflow_stage2_pilot import model_output_to_flow_hw, pure_torch_spatial_correlation
    from steflow_align_corners import patch_ste

    corr = importlib.import_module("models.corr")
    corr.spatial_correlation_sample = pure_torch_spatial_correlation
    import models
    import multiscaleloss
    upstream_hashes = patch_ste(models, multiscaleloss, model_root)
    model = models.steflow().to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    actual_state_sha = state_sha256(model)
    if actual_state_sha != saved.get("current_state_sha256"):
        raise RuntimeError(f"{generator} checkpoint model-state hash mismatch")
    model.eval()
    meta = {key: saved.get(key) for key in (
        "generator", "seed", "epoch", "optimizer_update", "next_epoch",
        "next_position_samples", "scheduler", "current_state_sha256",
        "initial_state_sha256", "best_epoch", "best_synthetic_validation_epe",
    )}
    meta["scheduler"] = {"last_epoch": scheduler_epoch}
    del saved
    return model, model_output_to_flow_hw, upstream_hashes, meta


def finite_array(name: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite or non-numeric values")


def fresh_accumulator() -> dict:
    return {
        "pixels": 0, "samples_with_pixels": 0, "epe_sum": 0.0, "zero_epe_sum": 0.0,
        "coverage_vs_full_valid_sum": 0.0, "coverage_vs_full_valid_samples": 0,
        "coverage_denominator_pixels": 0,
        "near_zero_counts": {str(t): 0 for t in NEAR_ZERO_THRESHOLDS_PX_PER_30MS},
        "gt_magnitude": [], "prediction_magnitude": [],
        "magnitude_ratio": [], "prediction_u_sum": 0.0, "prediction_v_sum": 0.0,
        "gt_u_sum": 0.0, "gt_v_sum": 0.0,
        "direction_cosine": [], "direction_angle_degrees": [],
        "pred_nonzero_direction_pixels": 0, "motion_region_pixels": 0,
    }


def update_region(acc: dict, prediction: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                  full_valid_count: int, gt_motion_region: bool = False) -> dict:
    pred_mag = np.hypot(prediction[0], prediction[1])
    gt_mag = np.hypot(gt[0], gt[1])
    count = int(mask.sum())
    acc["coverage_vs_full_valid_sum"] += count / max(1, full_valid_count)
    acc["coverage_vs_full_valid_samples"] += 1
    acc["coverage_denominator_pixels"] += full_valid_count
    row = {"pixels": count}
    if count == 0:
        return row
    error = np.hypot(prediction[0] - gt[0], prediction[1] - gt[1])
    selected_pred = pred_mag[mask]
    selected_gt = gt_mag[mask]
    acc["pixels"] += count
    acc["samples_with_pixels"] += 1
    acc["epe_sum"] += float(error[mask].sum(dtype=np.float64))
    acc["zero_epe_sum"] += float(selected_gt.sum(dtype=np.float64))
    for threshold in NEAR_ZERO_THRESHOLDS_PX_PER_30MS:
        acc["near_zero_counts"][str(threshold)] += int(np.count_nonzero(selected_pred < threshold))
    acc["gt_magnitude"].append(selected_gt.astype(np.float32, copy=False))
    acc["prediction_magnitude"].append(selected_pred.astype(np.float32, copy=False))
    acc["prediction_u_sum"] += float(prediction[0][mask].sum(dtype=np.float64))
    acc["prediction_v_sum"] += float(prediction[1][mask].sum(dtype=np.float64))
    acc["gt_u_sum"] += float(gt[0][mask].sum(dtype=np.float64))
    acc["gt_v_sum"] += float(gt[1][mask].sum(dtype=np.float64))
    nonzero_gt = mask & (gt_mag > 1e-6)
    if np.any(nonzero_gt):
        acc["magnitude_ratio"].append((pred_mag[nonzero_gt] / gt_mag[nonzero_gt]).astype(np.float32, copy=False))
    if gt_motion_region:
        acc["motion_region_pixels"] += count
        direction_mask = mask & (pred_mag >= NEAR_ZERO_THRESHOLDS_PX_PER_30MS[0])
        direction_count = int(direction_mask.sum())
        if direction_count:
            dot = prediction[0] * gt[0] + prediction[1] * gt[1]
            cosine = np.clip(dot[direction_mask] / (pred_mag[direction_mask] * gt_mag[direction_mask]), -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine))
            acc["direction_cosine"].append(cosine.astype(np.float32, copy=False))
            acc["direction_angle_degrees"].append(angle.astype(np.float32, copy=False))
            acc["pred_nonzero_direction_pixels"] += direction_count
    row.update({
        "epe_px_per_30ms": float(error[mask].mean(dtype=np.float64)),
        "zero_flow_epe_px_per_30ms": float(selected_gt.mean(dtype=np.float64)),
        "mean_prediction_magnitude_px_per_30ms": float(selected_pred.mean(dtype=np.float64)),
        "mean_gt_magnitude_px_per_30ms": float(selected_gt.mean(dtype=np.float64)),
        "near_zero_prediction_rate_lt_0_1_px": float(np.mean(selected_pred < 0.1)),
        "coverage_vs_full_valid": count / max(1, full_valid_count),
    })
    if gt_motion_region:
        row["prediction_direction_defined_fraction"] = int(np.count_nonzero(mask & (pred_mag >= 0.1))) / count
    return row


def summarize_accumulator(acc: dict) -> dict:
    pixels = acc["pixels"]
    result = {
        "pixel_count": pixels,
        "samples_with_pixels": acc["samples_with_pixels"],
        "epe_px_per_30ms": acc["epe_sum"] / pixels if pixels else None,
        "zero_flow_epe_px_per_30ms": acc["zero_epe_sum"] / pixels if pixels else None,
        "improvement_vs_zero_flow_fraction": (
            1.0 - acc["epe_sum"] / acc["zero_epe_sum"] if acc["zero_epe_sum"] else None
        ),
        "mean_per_sample_coverage_vs_full_valid": (
            acc["coverage_vs_full_valid_sum"] / acc["coverage_vs_full_valid_samples"]
            if acc["coverage_vs_full_valid_samples"] else None
        ),
        "pixel_pooled_coverage_vs_full_valid": (
            pixels / acc["coverage_denominator_pixels"] if acc["coverage_denominator_pixels"] else None
        ),
        "near_zero_prediction_rate": {
            threshold: count / pixels if pixels else None
            for threshold, count in acc["near_zero_counts"].items()
        },
        "prediction_direction_defined_fraction": (
            acc["pred_nonzero_direction_pixels"] / acc["motion_region_pixels"]
            if acc["motion_region_pixels"] else None
        ),
        "direction_cosine_mean": None,
        "direction_angle_degrees_median": None,
        "direction_within_30_degrees_fraction": None,
        "direction_within_60_degrees_fraction": None,
        "mean_prediction_u_px_per_30ms": acc["prediction_u_sum"] / pixels if pixels else None,
        "mean_gt_u_px_per_30ms": acc["gt_u_sum"] / pixels if pixels else None,
        "mean_prediction_v_px_per_30ms": acc["prediction_v_sum"] / pixels if pixels else None,
        "mean_gt_v_px_per_30ms": acc["gt_v_sum"] / pixels if pixels else None,
        "prediction_to_gt_magnitude_ratio_median": None,
        "gt_magnitude_quantiles_px_per_30ms": None,
        "prediction_magnitude_quantiles_px_per_30ms": None,
    }
    for key, values in (("gt_magnitude_quantiles_px_per_30ms", acc["gt_magnitude"]),
                         ("prediction_magnitude_quantiles_px_per_30ms", acc["prediction_magnitude"])):
        if values:
            flat = np.concatenate(values).astype(np.float64, copy=False)
            qs = np.percentile(flat, [0, 50, 90, 99, 100])
            result[key] = {name: float(value) for name, value in zip(("min", "p50", "p90", "p99", "max"), qs)}
    if acc["magnitude_ratio"]:
        result["prediction_to_gt_magnitude_ratio_median"] = float(np.median(np.concatenate(acc["magnitude_ratio"])))
    if acc["direction_angle_degrees"]:
        cosines = np.concatenate(acc["direction_cosine"]).astype(np.float64, copy=False)
        angles = np.concatenate(acc["direction_angle_degrees"]).astype(np.float64, copy=False)
        result["direction_cosine_mean"] = float(cosines.mean())
        result["direction_angle_degrees_median"] = float(np.median(angles))
        result["direction_within_30_degrees_fraction"] = float(np.mean(angles <= 30.0))
        result["direction_within_60_degrees_fraction"] = float(np.mean(angles <= 60.0))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--shared-bank", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eventgenesis-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.monotonic()
    data_root, model_root = args.data_root.resolve(), args.model_root.resolve()
    shared_bank = args.shared_bank.resolve()
    manifest_path = data_root / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    config_sha = sha256(args.config.resolve())
    if manifest.get("schema") != "stage4_steflow_30ms_windows_v1":
        raise RuntimeError(f"unexpected synthetic validation manifest schema: {manifest.get('schema')}")
    if manifest.get("flow_unit") != "pixels per 30ms endpoint interval":
        raise RuntimeError(f"unexpected physical flow unit: {manifest.get('flow_unit')}")
    if manifest.get("event_window") != "(t_i,t_i+3] on shared 100Hz endpoints; starts at i=0,3,...,195; the remaining 20ms of each 2s rollout is excluded by the registered 66-window count":
        raise RuntimeError("registered event-window contract changed")
    if (manifest.get("train_windows_per_generator"), manifest.get("validation_windows_per_generator")) != (1584, 528):
        raise RuntimeError("registered training/validation window census changed")
    if args.config.resolve().name != "stage4_corrected_v2_flowdt1_seed0.json":
        raise RuntimeError("diagnostic must use the frozen corrected-v2 seed-0 config")
    if not shared_bank.is_file():
        raise FileNotFoundError(f"shared physical bank is missing: {shared_bank}")
    bank_manifest_path = shared_bank.parent / "bank_manifest.json"
    bank_manifest = json.loads(bank_manifest_path.read_text())
    if bank_manifest.get("bank_sha256") != manifest.get("shared_bank_sha256"):
        raise RuntimeError("adapter and shared-bank manifests refer to different physical banks")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("motion diagnostic requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)

    models = {}
    model_meta = {}
    model_output_to_flow_hw = None
    patch_hashes = None
    checkpoint_paths = {"eventgenesis": args.eventgenesis_checkpoint.resolve(), "v2e": args.v2e_checkpoint.resolve()}
    checkpoint_meta = {}
    for generator in ("eventgenesis", "v2e"):
        model, decoder, hashes, meta = load_checkpoint(
            checkpoint_paths[generator], generator, model_root, manifest_sha, config_sha, device
        )
        models[generator] = model
        model_meta[generator] = meta
        checkpoint_meta[generator] = {
            **meta,
            "checkpoint_path": str(checkpoint_paths[generator]),
            "checkpoint_sha256": sha256(checkpoint_paths[generator]),
            "checkpoint_size_bytes": checkpoint_paths[generator].stat().st_size,
        }
        model_output_to_flow_hw = decoder
        patch_hashes = hashes
    if model_meta["eventgenesis"]["initial_state_sha256"] != model_meta["v2e"]["initial_state_sha256"]:
        raise RuntimeError("paired checkpoints do not share the registered initialization")
    if model_meta["eventgenesis"]["scheduler"] != model_meta["v2e"]["scheduler"]:
        raise RuntimeError("paired checkpoint scheduler boundary differs")

    entry_index = {(row["generator"], row["rollout_id"]): row for row in manifest["entries"]}
    accum = {
        arm: {name: {region: fresh_accumulator() for region in REGION_NAMES} for name in STRATA.values()}
        for arm in ("eventgenesis", "v2e")
    }
    activity = {name: {key: 0 for key in ("full_valid_pixels", "eventgenesis_active_pixels",
                                           "v2e_active_pixels", "common_union_active_pixels")}
                for name in STRATA.values()}
    consistency = {"paired_samples": 0, "paired_ids_equal": True, "paired_gt_equal": True, "paired_valid_equal": True,
                   "paired_gray_endpoints_equal": True, "paired_times_equal": True,
                   "paired_K_equal": True, "paired_label_sha_equal": True, "event_intervals_30ms": True,
                   "max_gray_endpoint_abs_difference": 0.0, "max_gt_abs_difference": 0.0,
                   "sample_ids": [], "samples": []}
    active_fractions = {"eventgenesis": [], "v2e": [], "common_union": []}
    validation_adapter_hashes = {}

    validation_rollouts = sorted((data_root / "eventgenesis" / "validation").glob("*.npz"))
    if len(validation_rollouts) != 8:
        raise RuntimeError(f"registered validation split must have 8 rollouts, got {len(validation_rollouts)}")
    stratum_counts = {index: 0 for index in STRATA}
    for rollout in validation_rollouts:
        rollout_id = rollout.stem
        stratum_index = int(rollout_id[1])
        if stratum_index not in STRATA:
            raise RuntimeError(f"unexpected validation rollout name: {rollout_id}")
        stratum_counts[stratum_index] += 1
        stratum = STRATA[stratum_index]
        paired_paths = {
            arm: data_root / arm / "validation" / f"{rollout_id}.npz"
            for arm in ("eventgenesis", "v2e")
        }
        if any(not path.is_file() for path in paired_paths.values()):
            raise FileNotFoundError(f"paired validation file missing for {rollout_id}")
        pair_data = {}
        for arm in ("eventgenesis", "v2e"):
            entry = entry_index.get((arm, rollout_id))
            if not entry or entry.get("split") != "validation":
                raise RuntimeError(f"training manifest lacks registered validation row: {arm}/{rollout_id}")
            actual_sha = sha256(paired_paths[arm])
            if actual_sha != entry.get("sha256"):
                raise RuntimeError(f"validation adapter hash mismatch: {paired_paths[arm]}")
            validation_adapter_hashes[str(paired_paths[arm])] = actual_sha
            pair_data[arm] = read_selected(paired_paths[arm])
        eg, v2 = pair_data["eventgenesis"], pair_data["v2e"]
        n = len(SAMPLE_INDICES)
        for arm, row in pair_data.items():
            if row["events"].shape != (n, 2, *EXPECTED_RESOLUTION, 5):
                raise RuntimeError(f"{arm}/{rollout_id}: event volume shape {row['events'].shape}")
            if row["gray_prev"].shape != (n, *EXPECTED_RESOLUTION) or row["gray_next"].shape != (n, *EXPECTED_RESOLUTION):
                raise RuntimeError(f"{arm}/{rollout_id}: grayscale endpoint shape mismatch")
            if row["flow_gt"].shape != (n, 2, *EXPECTED_RESOLUTION) or row["flow_valid"].shape != (n, *EXPECTED_RESOLUTION):
                raise RuntimeError(f"{arm}/{rollout_id}: flow/mask shape mismatch")
            for key in ("gray_prev", "gray_next", "flow_gt"):
                finite_array(f"{arm}/{rollout_id}/{key}", row[key])
            if not np.isfinite(row["event_start_s"]).all() or not np.isfinite(row["event_end_s"]).all():
                raise RuntimeError(f"{arm}/{rollout_id}: non-finite event window timestamps")
            if not np.isin(row["flow_valid"], (0, 1)).all():
                raise RuntimeError(f"{arm}/{rollout_id}: validity mask is not boolean 0/1")

        paired_ids = np.array_equal(eg["sample_ids"], v2["sample_ids"])
        paired_gt = np.array_equal(eg["flow_gt"], v2["flow_gt"])
        paired_valid = np.array_equal(eg["flow_valid"], v2["flow_valid"])
        paired_gray = np.array_equal(eg["gray_prev"], v2["gray_prev"]) and np.array_equal(eg["gray_next"], v2["gray_next"])
        paired_times = np.array_equal(eg["event_start_s"], v2["event_start_s"]) and np.array_equal(eg["event_end_s"], v2["event_end_s"])
        paired_K = np.array_equal(eg["camera_intrinsics_K_px"], v2["camera_intrinsics_K_px"])
        paired_label = np.array_equal(eg["label_sha256"], v2["label_sha256"])
        consistency["paired_ids_equal"] &= paired_ids
        consistency["paired_gt_equal"] &= paired_gt
        consistency["paired_valid_equal"] &= paired_valid
        consistency["paired_gray_endpoints_equal"] &= paired_gray
        consistency["paired_times_equal"] &= paired_times
        consistency["paired_K_equal"] &= paired_K
        consistency["paired_label_sha_equal"] &= paired_label
        if not all((paired_ids, paired_gt, paired_valid, paired_gray, paired_times, paired_K, paired_label)):
            raise RuntimeError(f"paired inputs/labels are inconsistent for {rollout_id}")
        dt = eg["event_end_s"] - eg["event_start_s"]
        if not np.allclose(dt, 0.03, rtol=0.0, atol=1e-8):
            consistency["event_intervals_30ms"] = False
            raise RuntimeError(f"{rollout_id}: selected windows do not all span 30ms: {dt}")
        if not np.array_equal(eg["sample_ids"].astype(str), v2["sample_ids"].astype(str)):
            raise RuntimeError(f"{rollout_id}: paired sample IDs differ")
        expected_sample_ids = np.asarray([f"{rollout_id}/dt1_{i * 3:03d}" for i in SAMPLE_INDICES])
        if not np.array_equal(eg["sample_ids"].astype(str), expected_sample_ids):
            raise RuntimeError(f"{rollout_id}: sample index order does not match preregistered selection")
        # Verify event timestamps, grayscale endpoints, labels, validity, and K
        # against the underlying shared physical bank for these exact windows.
        with h5py.File(shared_bank, "r") as source:
            group = source[f"episodes/{rollout_id}"]
            endpoints = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
            left_indices = np.asarray(SAMPLE_INDICES) * 3
            right_indices = left_indices + 3
            selected_indices = np.asarray(SAMPLE_INDICES)
            source_left = endpoints[left_indices]
            source_right = endpoints[right_indices]
            source_gray_prev = np.asarray(group["v2e_display_luma_u8"][left_indices], dtype=np.float32) / 255.0
            source_gray_next = np.asarray(group["v2e_display_luma_u8"][right_indices], dtype=np.float32) / 255.0
            source_flow = np.asarray(group["flow_forward_30ms_px"][selected_indices], dtype=np.float32)
            source_valid = np.asarray(group["flow_valid_30ms"][selected_indices], dtype=np.uint8)
            source_K = np.asarray(group["camera_intrinsics_K_px"], dtype=np.float64)
            timestamp_unit = group.attrs.get("timestamp_unit", "")
            flow_contract = group.attrs.get("flow_contract", "")
            source_split = group.attrs.get("split", "")
            source_stratum = group.attrs.get("stratum", "")
            if isinstance(timestamp_unit, bytes):
                timestamp_unit = timestamp_unit.decode()
            if isinstance(flow_contract, bytes):
                flow_contract = flow_contract.decode()
            if isinstance(source_split, bytes):
                source_split = source_split.decode()
            if isinstance(source_stratum, bytes):
                source_stratum = source_stratum.decode()
            checks = {
                "event_start_matches_source_endpoint": np.array_equal(eg["event_start_s"], source_left),
                "event_end_matches_label_endpoint": np.array_equal(eg["event_end_s"], source_right),
                "gray_prev_matches_start_endpoint": np.array_equal(eg["gray_prev"], source_gray_prev),
                "gray_next_matches_label_endpoint": np.array_equal(eg["gray_next"], source_gray_next),
                "flow_matches_forward_30ms_source_label": np.array_equal(eg["flow_gt"], source_flow),
                "validity_matches_shared_source_mask": np.array_equal(eg["flow_valid"], source_valid),
                "K_matches_shared_source": np.array_equal(eg["camera_intrinsics_K_px"], source_K),
                "timestamp_unit_is_seconds": str(timestamp_unit) == "second, relative to rollout start",
                "flow_contract_is_forward_30ms_px": str(flow_contract) == "forward source-pixel displacement from endpoint i to i+3; px per 30 ms interval",
                "rollout_is_registered_validation_stratum": str(source_split) == "validation" and str(source_stratum) == stratum,
            }
            if not all(checks.values()):
                raise RuntimeError(f"{rollout_id}: source-bank endpoint/label contract failed: {checks}")
        consistency.setdefault("shared_bank_sample_checks", {})[rollout_id] = checks
        consistency["paired_samples"] += n
        consistency["sample_ids"].extend(eg["sample_ids"].astype(str).tolist())

        active_eg = np.any(eg["events"] != 0, axis=(1, 4))
        active_v2 = np.any(v2["events"] != 0, axis=(1, 4))
        active_union = active_eg | active_v2
        for i, sample_index in enumerate(SAMPLE_INDICES):
            gt = eg["flow_gt"][i].astype(np.float32, copy=False)
            valid = eg["flow_valid"][i].astype(bool, copy=False)
            full_count = int(valid.sum())
            if full_count == 0:
                raise RuntimeError(f"{rollout_id}/sample{sample_index:03d}: empty registered flow_valid mask")
            gt_mag = np.hypot(gt[0], gt[1])
            masks = {
                "full_valid": valid,
                "shared_event_active_union": valid & active_union[i],
                "gt_motion_ge_0_5": valid & (gt_mag >= GT_MOTION_THRESHOLDS_PX_PER_30MS[0]),
                "gt_motion_ge_1_0": valid & (gt_mag >= GT_MOTION_THRESHOLDS_PX_PER_30MS[1]),
            }
            for key, mask in (("full_valid_pixels", valid), ("eventgenesis_active_pixels", valid & active_eg[i]),
                              ("v2e_active_pixels", valid & active_v2[i]),
                              ("common_union_active_pixels", valid & active_union[i])):
                activity[stratum][key] += int(mask.sum())
            sample_row = {
                "sample_id": str(eg["sample_ids"][i]),
                "rollout_id": rollout_id,
                "stratum": stratum,
                "sample_index": sample_index,
                "event_start_s": float(eg["event_start_s"][i]),
                "event_end_s": float(eg["event_end_s"][i]),
                "label_time_s": float(eg["event_end_s"][i]),
                "event_count_eventgenesis": int(eg["event_counts"][i]),
                "event_count_v2e": int(v2["event_counts"][i]),
                "valid_pixels": full_count,
                "event_active_union_valid_pixels": int(masks["shared_event_active_union"].sum()),
                "gt_motion_ge_0_5_valid_pixels": int(masks["gt_motion_ge_0_5"].sum()),
                "gt_motion_ge_1_0_valid_pixels": int(masks["gt_motion_ge_1_0"].sum()),
                "regions": {},
            }
            for arm in ("eventgenesis", "v2e"):
                event = pair_data[arm]["events"][i:i + 1].astype(np.float32, copy=False)
                padded_event = torch.from_numpy(event)
                padded_event = torch.nn.functional.pad(padded_event, (0, 0, 3, 3, 46, 46)).to(device)
                with torch.inference_mode():
                    raw_output = models[arm](padded_event, 352, use_dt4=False, flow_seq=False)
                    if tuple(raw_output.shape) != (1, 2, 352, 352):
                        raise RuntimeError(f"{arm}/{rollout_id}: model output must be [1,2,352,352], got {tuple(raw_output.shape)}")
                    decoded = model_output_to_flow_hw(raw_output)
                prediction = decoded[:, 46:306, 3:349]
                if prediction.shape != (2, *EXPECTED_RESOLUTION) or not np.isfinite(prediction).all():
                    raise RuntimeError(f"{arm}/{rollout_id}: decoded flow must be finite [2,260,346]")
                sample_regions = {}
                for region, mask in masks.items():
                    sample_regions[region] = update_region(
                        accum[arm][stratum][region], prediction, gt, mask, full_count,
                        gt_motion_region=region in ("gt_motion_ge_0_5", "gt_motion_ge_1_0"),
                    )
                sample_row["regions"][arm] = sample_regions
            consistency["samples"].append(sample_row)
            active_fractions["eventgenesis"].append(float((valid & active_eg[i]).sum() / max(1, full_count)))
            active_fractions["v2e"].append(float((valid & active_v2[i]).sum() / max(1, full_count)))
            active_fractions["common_union"].append(float((valid & active_union[i]).sum() / max(1, full_count)))

    if consistency["paired_samples"] != 64:
        raise RuntimeError(f"fixed diagnostic expected 64 paired samples, got {consistency['paired_samples']}")
    if stratum_counts != {index: 2 for index in STRATA}:
        raise RuntimeError(f"validation strata must have two held-out rollouts each, got {stratum_counts}")
    if len(consistency["samples"]) != 64:
        raise RuntimeError(f"sample evidence table must contain exactly 64 paired rows, got {len(consistency['samples'])}")
    summaries = {}
    for arm in ("eventgenesis", "v2e"):
        summaries[arm] = {}
        for stratum in STRATA.values():
            summaries[arm][stratum] = {
                region: summarize_accumulator(accum[arm][stratum][region]) for region in REGION_NAMES
            }

    # Recompute pooled aggregates from per-stratum accumulators without
    # changing the validation script's equal-rollout official checkpoint metric.
    summaries["all_strata_pixel_pooled"] = {}
    for arm in ("eventgenesis", "v2e"):
        summaries["all_strata_pixel_pooled"][arm] = {}
        for region in REGION_NAMES:
            merged = fresh_accumulator()
            for stratum in STRATA.values():
                current = accum[arm][stratum][region]
                for key in ("pixels", "samples_with_pixels", "epe_sum", "zero_epe_sum",
                            "coverage_vs_full_valid_sum", "coverage_vs_full_valid_samples",
                            "coverage_denominator_pixels", "prediction_u_sum", "prediction_v_sum",
                            "gt_u_sum", "gt_v_sum",
                            "pred_nonzero_direction_pixels", "motion_region_pixels"):
                    merged[key] += current[key]
                for threshold, count in current["near_zero_counts"].items():
                    merged["near_zero_counts"][threshold] += count
                for key in ("gt_magnitude", "prediction_magnitude", "direction_cosine", "direction_angle_degrees"):
                    merged[key].extend(current[key])
                merged["magnitude_ratio"].extend(current["magnitude_ratio"])
            summaries["all_strata_pixel_pooled"][arm][region] = summarize_accumulator(merged)

    activity_summary = {}
    for stratum, values in activity.items():
        base = max(1, values["full_valid_pixels"])
        activity_summary[stratum] = {
            **values,
            "eventgenesis_active_coverage_of_valid": values["eventgenesis_active_pixels"] / base,
            "v2e_active_coverage_of_valid": values["v2e_active_pixels"] / base,
            "common_union_active_coverage_of_valid": values["common_union_active_pixels"] / base,
        }
    activity_summary["all_strata"] = {
        key: sum(activity[s][key] for s in STRATA.values())
        for key in ("full_valid_pixels", "eventgenesis_active_pixels", "v2e_active_pixels", "common_union_active_pixels")
    }
    base = max(1, activity_summary["all_strata"]["full_valid_pixels"])
    for key, count_key in (("eventgenesis_active_coverage_of_valid", "eventgenesis_active_pixels"),
                           ("v2e_active_coverage_of_valid", "v2e_active_pixels"),
                           ("common_union_active_coverage_of_valid", "common_union_active_pixels")):
        activity_summary["all_strata"][key] = activity_summary["all_strata"][count_key] / base

    result = {
        "schema": "stage4_paired_motion_learning_diagnostic_v1",
        "status": "PASS",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - start_clock, 3),
        "real_data_used": False,
        "training_or_checkpoint_selection_changed": False,
        "protocol": {
            "checkpoint_selection": "same seed-0 epoch-boundary optimizer update 4554; both checkpoints after scheduler and synthetic validation for completed epoch 23",
            "validation_split_only": True,
            "rollouts": "all eight fixed held-out synthetic validation rollouts",
            "sample_indices_per_rollout": list(SAMPLE_INDICES),
            "paired_sample_count": consistency["paired_samples"],
            "regions": {
                "full_valid": "registered physical flow_valid mask",
                "shared_event_active_union": "flow_valid AND spatial pixels with any nonzero value in either generator's paired 5-bin event representation; one union mask for both predictions",
                "gt_motion_ge_0_5": "flow_valid AND GT vector magnitude >=0.5 px per 30ms interval",
                "gt_motion_ge_1_0": "flow_valid AND GT vector magnitude >=1.0 px per 30ms interval",
            },
            "near_zero_prediction_thresholds_px_per_30ms": list(NEAR_ZERO_THRESHOLDS_PX_PER_30MS),
            "direction_statistics": "on each fixed GT-motion region; direction is reported only where predicted magnitude >=0.1 px per 30ms",
            "metric_aggregation": "pixel pooled within each stratum and across the fixed diagnostic sample; official checkpoint selection metric is unchanged",
            "model_input": "event volume only; grayscale endpoint images are checked for pairing/timing but are not passed to the model",
            "output_contract": "STE output [1,2,352,352], remove registered padding y=46:306 and x=3:349, preserve u/v pixel displacement per 30ms interval without vector rescaling",
        },
        "paths": {
            "data_root": str(data_root), "training_bank_manifest": str(manifest_path),
            "training_bank_manifest_sha256": manifest_sha,
            "shared_bank": str(shared_bank),
            "shared_bank_size_bytes": shared_bank.stat().st_size,
            "shared_bank_sha256": manifest.get("shared_bank_sha256"),
            "shared_bank_manifest": str(bank_manifest_path),
            "validation_adapter_sha256": validation_adapter_hashes,
            "config": str(args.config.resolve()), "config_sha256": config_sha,
            "model_root": str(model_root), "model_source_hashes": patch_hashes,
            "device": str(device), "physical_gpu_visible_mapping": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoints": checkpoint_meta,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "numpy": np.__version__,
        },
        "dataset_and_label_checks": {
            "flow_unit": FLOW_UNIT,
            "shared_flow_label_source_field": "flow_forward_30ms_px",
            "flow_direction": "forward physical projection from event_start_s to label_time_s=event_end_s",
            "event_window_rule": manifest["event_window"],
            "interval_seconds": 0.03,
            "event_volume_shape": [2, *EXPECTED_RESOLUTION, 5],
            "gray_endpoint_shape": list(EXPECTED_RESOLUTION),
            "GT_shape": [2, *EXPECTED_RESOLUTION],
            "valid_mask_shape": list(EXPECTED_RESOLUTION),
            "checks": consistency,
        },
        "event_active_coverage": activity_summary,
        "active_coverage_sample_fraction_mean": {
            key: float(np.mean(values)) for key, values in active_fractions.items()
        },
        "metrics": summaries,
        "interpretation_limits": [
            "This is a fixed-sample synthetic diagnostic, not a real-data score or generator ranking.",
            "Event activity is measured after the accepted five-bin representation and may include its spatial Gaussian support.",
            "Direction metrics omit predictions below 0.1 px per interval because their angle is numerically uninformative; their share is reported separately.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".partial")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({
        "status": result["status"], "output": str(args.output.resolve()),
        "elapsed_seconds": result["elapsed_seconds"], "paired_samples": consistency["paired_samples"],
        "event_active_coverage": activity_summary,
        "metrics": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
