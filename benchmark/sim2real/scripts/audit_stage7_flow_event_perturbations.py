#!/usr/bin/env python3
"""Audit event-input sensitivity of fixed Stage 7 synthetic GT-fit checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def support(events: np.ndarray) -> np.ndarray:
    return np.any(events != 0, axis=(0, 3))


def mean_or_none(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return float(values[mask].mean(dtype=np.float64))


def load_rows(manifest: dict) -> dict[str, list[dict]]:
    """Load and independently hash-check every registered paired bank artifact."""
    entries = manifest["entries"]
    by_key = {(row["generator"], row["rollout_id"]): row for row in entries}
    train_ids = sorted({row["rollout_id"] for row in entries if row["split"] == "train"})
    validation_ids = sorted({row["rollout_id"] for row in entries if row["split"] == "validation"})
    if len(train_ids) != 4 or len(validation_ids) != 2:
        raise RuntimeError(f"expected 4 train and 2 validation rollouts, got {train_ids}/{validation_ids}")
    result: dict[str, list[dict]] = {"train": [], "validation": []}
    for split, rollout_ids in (("train", train_ids), ("validation", validation_ids)):
        for rollout_id in rollout_ids:
            arms = {}
            for arm in ("eventgenesis", "v2e"):
                row = by_key[(arm, rollout_id)]
                path = Path(row["path"])
                if row["split"] != split or not path.is_file() or file_sha256(path) != row["sha256"]:
                    raise RuntimeError(f"registered {split} artifact failed hash/split check: {path}")
                with np.load(path, allow_pickle=False) as archive:
                    sample = {key: np.asarray(archive[key]).copy() for key in
                              ("events", "flow_gt", "flow_valid", "sample_ids", "label_sha256")}
                    sample["event_count"] = int(np.asarray(archive["event_count"]))
                arms[arm] = sample
            a, b = arms["eventgenesis"], arms["v2e"]
            for key in ("flow_gt", "flow_valid", "sample_ids", "label_sha256"):
                if not np.array_equal(a[key], b[key]):
                    raise RuntimeError(f"paired shared field {key} differs for {rollout_id}")
            result[split].append({"rollout_id": rollout_id, "arms": arms})
    return result


def pad_inputs(sample: dict, device: torch.device, target_size: int = 352):
    event_np = np.asarray(sample["events"])
    flow_np = np.asarray(sample["flow_gt"], dtype=np.float32)
    valid_np = np.asarray(sample["flow_valid"], dtype=np.uint8)
    if event_np.ndim != 4 or event_np.shape[0] != 2 or event_np.shape[-1] != 5:
        raise ValueError(f"event tensor must be [2,H,W,5], got {event_np.shape}")
    h, w = event_np.shape[1:3]
    if flow_np.shape != (2, h, w) or valid_np.shape != (h, w):
        raise ValueError(f"sample tensor shape mismatch: {event_np.shape}/{flow_np.shape}/{valid_np.shape}")
    dy, dx = target_size - h, target_size - w
    if dy < 0 or dx < 0 or dy % 2 or dx % 2:
        raise ValueError(f"symmetric padding to {target_size} impossible for {h}x{w}")
    top, left = dy // 2, dx // 2
    events = torch.from_numpy(event_np.astype(np.float32, copy=False)).unsqueeze(0)
    events = torch.nn.functional.pad(events, (0, 0, left, dx - left, top, dy - top)).to(device)
    flow = torch.from_numpy(flow_np).unsqueeze(0)
    flow = torch.nn.functional.pad(flow, (left, dx - left, top, dy - top)).to(device)
    valid = torch.from_numpy(valid_np.astype(np.float32, copy=False)).view(1, 1, h, w)
    valid = torch.nn.functional.pad(valid, (left, dx - left, top, dy - top)).to(device)
    return events, flow, valid, (top, left, h, w)


def unpad(prediction: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
    top, left, h, w = pad
    result = prediction[:, top:top + h, left:left + w]
    if result.shape != (2, h, w) or not np.isfinite(result).all():
        raise ValueError(f"unpad returned invalid flow {result.shape}")
    return result


def summarize_prediction(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray,
                         common_active: np.ndarray, common_active_motion: np.ndarray) -> dict:
    error = np.hypot(prediction[0] - target[0], prediction[1] - target[1])
    gt_mag = np.hypot(target[0], target[1])
    pred_mag = np.hypot(prediction[0], prediction[1])
    regions = {
        "full_valid": valid,
        "common_event_active": valid & common_active,
        "active_motion": valid & common_active_motion,
    }
    result = {
        "prediction_sha256": array_sha256(prediction),
        "prediction_mean_magnitude_px_per_30ms": {
            name: mean_or_none(pred_mag, mask) for name, mask in regions.items()
        },
        "gt_mean_magnitude_px_per_30ms": {
            name: mean_or_none(gt_mag, mask) for name, mask in regions.items()
        },
        "epe_px_per_30ms": {},
        "zero_flow_epe_px_per_30ms": {},
        "direction_cosine_mean_active_motion": None,
    }
    zero_error = gt_mag
    for name, mask in regions.items():
        result["epe_px_per_30ms"][name] = mean_or_none(error, mask)
        result["zero_flow_epe_px_per_30ms"][name] = mean_or_none(zero_error, mask)
    if regions["active_motion"].any():
        mask = regions["active_motion"]
        dot = prediction[0] * target[0] + prediction[1] * target[1]
        cosine = dot / np.maximum(pred_mag * gt_mag, 1e-12)
        result["direction_cosine_mean_active_motion"] = float(cosine[mask].mean(dtype=np.float64))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo / "benchmark/sim2real/scripts"))
    sys.path.insert(0, str(repo / "benchmark/sim2real/compat"))
    from stage5_flow_supervised_pilot import (
        SOURCE_COMMIT, model_output_to_flow_hw, model_source, state_sha256,
    )
    from verify_pinned_source import verify_pinned_source

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("registered_before_execution") is not True:
        raise RuntimeError("protocol must be registered before execution")
    if protocol["data"]["real_data_used"] is not False:
        raise RuntimeError("real data must not be used")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this audit requires the registered CUDA execution path")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    config_path = args.config.resolve()
    data_root = args.data_root.resolve()
    manifest_path = data_root / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("flow_unit") != "pixels per 30ms endpoint interval":
        raise RuntimeError("unexpected GT flow unit")
    if manifest.get("real_data_used") is not False:
        raise RuntimeError("bank manifest is not synthetic-only")
    bank = load_rows(manifest)
    validation_ids = [row["rollout_id"] for row in bank["validation"]]
    if validation_ids != protocol["data"]["rollouts"]:
        raise RuntimeError(f"validation rollouts changed: {validation_ids}")
    source_verification = verify_pinned_source(
        args.model_root.resolve(), "STE-FlowNet", SOURCE_COMMIT,
        repo / "benchmark/sim2real/manifests/downstream_source_bundle.json",
    )
    checkpoint_audit = json.loads(args.audit.resolve().read_text())
    if checkpoint_audit.get("status") != "postrun_checkpoint_and_validation_audit_passed":
        raise RuntimeError("independent checkpoint audit is not PASS")
    if checkpoint_audit.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("checkpoint audit source commit mismatch")
    manifest_sha = file_sha256(manifest_path)
    config_sha = file_sha256(config_path)

    fixed_masks: dict[str, np.ndarray] = {}
    for row in bank["validation"]:
        eg = row["arms"]["eventgenesis"]["events"]
        v2e = row["arms"]["v2e"]["events"]
        if eg.shape != v2e.shape or eg.shape != (2, 240, 320, 5):
            raise RuntimeError(f"unexpected paired input shape for {row['rollout_id']}: {eg.shape}/{v2e.shape}")
        common = support(eg) | support(v2e)
        valid = np.asarray(row["arms"]["eventgenesis"]["flow_valid"], dtype=bool)
        target = np.asarray(row["arms"]["eventgenesis"]["flow_gt"], dtype=np.float32)
        motion = np.hypot(target[0], target[1]) >= 0.5
        fixed_masks[row["rollout_id"]] = common & valid
        if not np.array_equal(row["arms"]["eventgenesis"]["flow_valid"],
                              row["arms"]["v2e"]["flow_valid"]):
            raise RuntimeError("paired shared validity mask mismatch")

    result = {
        "schema": "stage7_flow_input_perturbation_audit_v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "config_path": str(config_path), "config_sha256": config_sha,
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
        "checkpoint_audit_path": str(args.audit.resolve()),
        "checkpoint_audit_sha256": file_sha256(args.audit.resolve()),
        "model_source_verification": source_verification,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "validation_rollouts": validation_ids,
        "frozen_masks": {}, "arms": {},
        "real_data_used": False, "checkpoint_selection": False,
    }
    for row in bank["validation"]:
        rollout = row["rollout_id"]
        valid = np.asarray(row["arms"]["eventgenesis"]["flow_valid"], dtype=bool)
        gt = np.asarray(row["arms"]["eventgenesis"]["flow_gt"], dtype=np.float32)
        common = fixed_masks[rollout]
        motion = np.hypot(gt[0], gt[1]) >= 0.5
        result["frozen_masks"][rollout] = {
            "full_valid_pixels": int(valid.sum()),
            "common_event_active_pixels": int((valid & common).sum()),
            "active_motion_pixels": int((valid & common & motion).sum()),
            "motion_threshold_px_per_30ms": 0.5,
            "event_active_union_sha256": array_sha256(common.astype(np.uint8)),
        }

    expected_arms = protocol["models"]["checkpoints"]
    result["arms"] = {}
    for arm in ("eventgenesis", "v2e"):
        checkpoint_path = Path(expected_arms[arm]["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_hash = file_sha256(checkpoint_path)
        if checkpoint_hash != expected_arms[arm]["sha256"]:
            raise RuntimeError(f"checkpoint SHA256 mismatch for {arm}")
        if checkpoint_audit["arms"][arm]["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError(f"checkpoint identity mismatch for {arm}")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("generator") != arm or saved.get("updates") != 100:
            raise RuntimeError(f"unexpected checkpoint metadata for {arm}")
        if saved.get("data_manifest_sha256") != manifest_sha or saved.get("config_sha256") != config_sha:
            raise RuntimeError(f"checkpoint is not bound to current bank/config for {arm}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA disappeared during audit")
        model, patch_info = model_source(args.model_root.resolve(), device)
        model.load_state_dict(saved["state_dict"], strict=True)
        loaded_state_sha = state_sha256(model)
        if loaded_state_sha != checkpoint_audit["arms"][arm]["state_sha256_loaded"]:
            raise RuntimeError(f"loaded model state mismatch for {arm}")
        model.eval()
        arm_records = []
        for row in bank["validation"]:
            rollout = row["rollout_id"]
            sample = row["arms"][arm]
            normal = np.asarray(sample["events"], dtype=np.float32)
            gt = np.asarray(sample["flow_gt"], dtype=np.float32)
            valid = np.asarray(sample["flow_valid"], dtype=bool)
            motion = np.hypot(gt[0], gt[1]) >= 0.5
            reversed_bins = normal[..., ::-1].copy()
            zeroed = np.zeros_like(normal)
            transformed = {
                "normal": normal,
                "zero_events": zeroed,
                "reverse_bins": reversed_bins,
            }
            if not np.allclose(normal.sum(axis=-1), reversed_bins.sum(axis=-1), atol=1e-6, rtol=0):
                raise RuntimeError("reverse_bins changed per-pixel/polarity aggregate representation")
            if not np.array_equal(support(normal), support(reversed_bins)):
                raise RuntimeError("reverse_bins changed spatial event support")
            if np.array_equal(normal, reversed_bins):
                raise RuntimeError(f"reverse_bins did not change the model input for {arm}/{rollout}")
            if np.any(zeroed) or support(zeroed).any():
                raise RuntimeError("zero_events perturbation did not zero the input")
            predictions = {}
            inputs = {}
            for condition, event_tensor in transformed.items():
                modified = dict(sample)
                modified["events"] = event_tensor
                events, _, _, pad = pad_inputs(modified, device)
                with torch.inference_mode():
                    output = model(events, 352, use_dt4=False, flow_seq=False)
                pred = unpad(model_output_to_flow_hw(output), pad)
                if pred.shape != (2, 240, 320) or not np.isfinite(pred).all():
                    raise RuntimeError(f"nonfinite or wrong-shape output: {arm}/{rollout}/{condition}")
                predictions[condition] = pred
                inputs[condition] = {
                    "input_sha256": array_sha256(event_tensor),
                    "nonzero_spatial_pixels": int(support(event_tensor).sum()),
                    "representation_sum": float(event_tensor.sum(dtype=np.float64)),
                    "changed_from_normal": not np.array_equal(event_tensor, normal),
                    "per_pixel_channel_sum_sha256": array_sha256(event_tensor.sum(axis=-1)),
                }
            fixed_active = fixed_masks[rollout]
            normal_summary = summarize_prediction(predictions["normal"], gt, valid, fixed_active,
                                                 fixed_active & motion)
            old_audit_rows = {
                item["rollout_id"]: item for item in checkpoint_audit["arms"][arm]["validation"]
            }
            prior = old_audit_rows[rollout]
            if normal_summary["prediction_sha256"] != prior["prediction_hash_recomputed"]:
                raise RuntimeError(f"normal prediction no longer matches independent checkpoint audit: {arm}/{rollout}")
            baseline_crosscheck = {}
            for metric_key, region in (
                ("zero_flow_full_valid_epe_px_per_30ms", "full_valid"),
                ("zero_flow_common_event_active_epe_px_per_30ms", "common_event_active"),
            ):
                current = normal_summary["zero_flow_epe_px_per_30ms"][region]
                expected = prior["metrics"][metric_key]["recomputed"]
                if current is None or abs(current - expected) > 2e-5:
                    raise RuntimeError(
                        f"GT/mask binding differs from independent audit for {arm}/{rollout}/{region}: "
                        f"{current} vs {expected}"
                    )
                baseline_crosscheck[metric_key] = {
                    "current": current, "prior_independent_audit": expected,
                    "absolute_difference": abs(current - expected), "pass": True,
                }
            for metric_key, region in (
                ("full_valid_epe_px_per_30ms", "full_valid"),
                ("common_event_active_epe_px_per_30ms", "common_event_active"),
            ):
                current = normal_summary["epe_px_per_30ms"][region]
                expected = prior["metrics"][metric_key]["recomputed"]
                if current is None or abs(current - expected) > 2e-5:
                    raise RuntimeError(f"normal EPE differs from independent audit: {arm}/{rollout}/{region}")
                baseline_crosscheck[metric_key] = {
                    "current": current, "prior_independent_audit": expected,
                    "absolute_difference": abs(current - expected), "pass": True,
                }
            condition_records = {}
            for condition in ("normal", "zero_events", "reverse_bins"):
                summary = summarize_prediction(predictions[condition], gt, valid, fixed_active,
                                               fixed_active & motion)
                delta = predictions[condition].astype(np.float64) - predictions["normal"].astype(np.float64)
                delta_mag = np.hypot(delta[0], delta[1])
                summary["prediction_delta_from_normal"] = {
                    "mean_vector_error_full_valid": mean_or_none(delta_mag, valid),
                    "mean_vector_error_fixed_common_active": mean_or_none(delta_mag, valid & fixed_active),
                    "max_abs_component": float(np.max(np.abs(delta))),
                }
                condition_records[condition] = summary
            arm_records.append({
                "rollout_id": rollout,
                "sample_id": str(sample["sample_ids"]),
                "fixed_mask_pixels": result["frozen_masks"][rollout],
                "input_diagnostics": inputs,
                "conditions": condition_records,
                "normal_prediction_sha256_matches_prior_independent_audit": True,
                "normal_metric_crosschecks": baseline_crosscheck,
                "note": "all conditions use exactly the same pre-frozen regions; no rollout was filtered",
            })
        result["arms"][arm] = {
            "checkpoint_path": str(checkpoint_path), "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": checkpoint_hash, "loaded_state_sha256": loaded_state_sha,
            "generator_training_arm": arm, "patch_info": patch_info,
            "updates_in_checkpoint": int(saved["updates"]), "rollouts": arm_records,
        }
        del model, saved
        torch.cuda.empty_cache()

    result["status"] = "complete"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(json.dumps({
        "status": result["status"], "gpu": result["gpu_name"], "gpu_uuid": result["gpu_uuid"],
        "output": str(args.output), "protocol_sha256": result["protocol_sha256"],
        "rollouts": validation_ids, "checkpoint_sha256": {k: v["checkpoint_sha256"] for k, v in result["arms"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
