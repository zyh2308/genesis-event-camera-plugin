#!/usr/bin/env python3
"""Evaluate fixed-sample STE-FlowNet checkpoints on one full held-out rollout.

The model receives only the event representation. Shared flow GT, validity,
and the union of the paired normal-input event supports are used for scoring.
This diagnostic does not select checkpoints or access real data.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/sim2real/scripts"))
from stage5_flow_supervised_pilot import (  # noqa: E402
    FLOW_UNIT,
    load_sample,
    model_source,
    predict_eval,
    sha256,
    state_sha256,
)


def masked_stats(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 2:
        raise ValueError(f"flow arrays must both be [2,H,W], got {prediction.shape} and {target.shape}")
    if mask.shape != target.shape[1:]:
        raise ValueError(f"mask {mask.shape} does not match flow raster {target.shape[1:]}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("non-finite prediction or GT flow")
    count = int(mask.sum())
    if not count:
        return {"pixels": 0, "epe_px_per_30ms": None, "pred_magnitude_p50": None,
                "gt_magnitude_p50": None, "direction_cosine_mean": None,
                "direction_agreement_fraction": None}
    pred_mag = np.hypot(prediction[0], prediction[1])
    gt_mag = np.hypot(target[0], target[1])
    dot = prediction[0] * target[0] + prediction[1] * target[1]
    denom = pred_mag * gt_mag
    moving = mask & (gt_mag >= 0.5)
    cos = dot[moving] / np.maximum(denom[moving], 1e-12)
    return {
        "pixels": count,
        "epe_px_per_30ms": float(np.hypot(prediction[0, mask] - target[0, mask],
                                          prediction[1, mask] - target[1, mask]).mean(dtype=np.float64)),
        "pred_magnitude_p50": float(np.median(pred_mag[mask])),
        "gt_magnitude_p50": float(np.median(gt_mag[mask])),
        "direction_pixels_gt_ge_0_5": int(moving.sum()),
        "direction_cosine_mean": float(cos.mean(dtype=np.float64)) if cos.size else None,
        "direction_agreement_fraction": float((cos > 0).mean(dtype=np.float64)) if cos.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eventgenesis-checkpoint", type=Path, required=True)
    parser.add_argument("--v2e-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("rollout evaluation requires CUDA")
    torch.cuda.set_device(device)

    config = json.loads(args.config.read_text())
    sample_cfg = config["sample"]
    validation_cfg = config["independent_validation"]
    manifest_path = args.data_root.resolve() / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash, config_hash = sha256(manifest_path), sha256(args.config.resolve())
    if manifest.get("flow_unit") != "pixels per 30ms endpoint interval":
        raise RuntimeError("unexpected shared flow-unit contract")
    entries = {(row["generator"], row["rollout_id"]): row for row in manifest["entries"]}
    rollout = validation_cfg["rollout_id"]
    if validation_cfg["split"] != "validation":
        raise RuntimeError("independent rollout must be registered as validation")
    pair = {}
    for arm in ("eventgenesis", "v2e"):
        row = entries[(arm, rollout)]
        if row["split"] != "validation":
            raise RuntimeError(f"{arm}/{rollout} is not a validation artifact")
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"registered validation artifact hash mismatch: {path}")
        pair[arm] = row
    if pair["eventgenesis"]["samples"] != 66 or pair["v2e"]["samples"] != 66:
        raise RuntimeError("expected all 66 registered validation windows")

    checkpoint_paths = {
        "eventgenesis": args.eventgenesis_checkpoint.resolve(),
        "v2e": args.v2e_checkpoint.resolve(),
    }
    models = {}
    checkpoints = {}
    for arm, path in checkpoint_paths.items():
        record = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "generator": arm,
            "sample_id": sample_cfg["sample_id"],
            "data_manifest_sha256": manifest_hash,
            "config_sha256": config_hash,
            "flow_unit": FLOW_UNIT,
            "real_data_used": False,
            "updates": int(config["diagnostic_budget"]["updates_per_arm"]),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise RuntimeError(f"{arm} checkpoint {key} mismatch: {record.get(key)!r} != {value!r}")
        model, _ = model_source(args.model_root.resolve(), device)
        model.load_state_dict(record["state_dict"], strict=True)
        if state_sha256(model) != record.get("state_sha256"):
            raise RuntimeError(f"{arm} checkpoint state hash mismatch")
        model.eval()
        models[arm], checkpoints[arm] = model, record
    initial_hashes = {checkpoints[arm].get("initial_state_sha256") for arm in checkpoints}
    if len(initial_hashes) != 1:
        raise RuntimeError("paired checkpoints do not share the same initialized model state")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    aggregate = {arm: {key: {"epe_sum": 0.0, "pixels": 0, "zero_epe_sum": 0.0}
                       for key in ("full_valid", "common_event_active", "gt_motion_ge_0_5")}
                 for arm in models}
    with torch.inference_mode():
        for index in range(66):
            samples = {arm: load_sample(Path(pair[arm]["path"]), index) for arm in models}
            event_counts = {}
            for arm in models:
                with np.load(Path(pair[arm]["path"]), allow_pickle=False) as archive:
                    event_counts[arm] = int(np.asarray(archive["event_counts"])[index])
            eg, v2 = samples["eventgenesis"], samples["v2e"]
            for key in ("flow_gt", "flow_valid", "sample_ids", "label_sha256"):
                if not np.array_equal(eg[key], v2[key]):
                    raise RuntimeError(f"paired physical label field {key} differs at validation index {index}")
            sample_id = str(eg["sample_ids"])
            if sample_id != str(v2["sample_ids"]):
                raise RuntimeError("paired validation sample IDs differ")
            target = eg["flow_gt"].astype(np.float32, copy=False)
            valid = eg["flow_valid"].astype(bool, copy=False)
            active = {arm: np.any(samples[arm]["events"] != 0, axis=(0, 3)) for arm in models}
            common_active = valid & (active["eventgenesis"] | active["v2e"])
            motion = np.hypot(target[0], target[1])
            moving = valid & (motion >= 0.5)
            masks = {
                "full_valid": valid,
                "common_event_active": common_active,
                "gt_motion_ge_0_5": moving,
            }
            arm_rows = {}
            for arm in models:
                model = models[arm]
                events = torch.from_numpy(samples[arm]["events"].astype(np.float32, copy=False)).unsqueeze(0)
                # Use the same padding and output unpadding as the supervised fit.
                events = F.pad(events, (0, 0, 3, 3, 46, 46)).to(device)
                pred = predict_eval(model, events, device)
                if pred.shape != target.shape:
                    raise RuntimeError(f"{arm}/{sample_id}: prediction {pred.shape} != GT {target.shape}")
                metrics = {name: masked_stats(pred, target, mask) for name, mask in masks.items()}
                zero = np.zeros_like(target)
                for name, mask in masks.items():
                    stats = metrics[name]
                    count = stats["pixels"]
                    if count:
                        epe_values = np.hypot(pred[0, mask] - target[0, mask], pred[1, mask] - target[1, mask])
                        zero_values = np.hypot(zero[0, mask] - target[0, mask], zero[1, mask] - target[1, mask])
                        aggregate[arm][name]["epe_sum"] += float(epe_values.sum(dtype=np.float64))
                        aggregate[arm][name]["zero_epe_sum"] += float(zero_values.sum(dtype=np.float64))
                        aggregate[arm][name]["pixels"] += count
                arm_rows[arm] = metrics
            rows.append({
                "sample_id": sample_id,
                "window_index": index,
                "event_counts": event_counts,
                "valid_pixels": int(valid.sum()),
                "common_event_active_pixels": int(common_active.sum()),
                "common_event_active_coverage_of_valid": float(common_active.sum() / max(int(valid.sum()), 1)),
                "generator_metrics": arm_rows,
            })

    summary = {}
    for arm in models:
        summary[arm] = {}
        for region, stats in aggregate[arm].items():
            count = stats["pixels"]
            summary[arm][region] = {
                "pixels": count,
                "pixel_weighted_epe_px_per_30ms": stats["epe_sum"] / count if count else None,
                "zero_flow_epe_px_per_30ms": stats["zero_epe_sum"] / count if count else None,
            }
    result = {
        "schema": "stage5_genesis_high_texture_gtfit_independent_rollout_eval_v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_rollout": rollout,
        "validation_windows": len(rows),
        "flow_unit": FLOW_UNIT,
        "model_input": "event representation only; GT is used only for masks/metrics",
        "common_active_mask": "shared valid GT AND union of normal EventGenesis/V2E event-active pixels",
        "real_data_used": False,
        "checkpoint_selection": False,
        "config_sha256": config_hash,
        "data_manifest_sha256": manifest_hash,
        "checkpoint_paths": {arm: str(path) for arm, path in checkpoint_paths.items()},
        "checkpoint_sha256": {arm: sha256(path) for arm, path in checkpoint_paths.items()},
        "initial_state_sha256": next(iter(initial_hashes)),
        "summary": summary,
        "windows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "validation_rollout": rollout,
                      "windows": len(rows), "summary": summary,
                      "output": str(args.output.resolve()), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
