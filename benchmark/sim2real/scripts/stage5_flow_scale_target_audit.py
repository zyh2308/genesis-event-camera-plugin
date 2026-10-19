#!/usr/bin/env python3
"""Audit STE-FlowNet's four supervised outputs against scaled shared GT.

This is a read-only synthetic diagnostic: it loads the fixed checkpoints,
computes the existing training objective at each native output scale, and
compares the first scale with ordinary eval-mode full-resolution flow. It
does not optimize weights or select a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/sim2real/scripts"))
from stage5_flow_supervised_pilot import (  # noqa: E402
    FLOW_UNIT,
    ORIGINAL_SCALE_WEIGHTS,
    epe_pair,
    load_sample,
    model_output_to_flow_hw,
    model_source,
    multiscale_supervised_epe,
    pad_sample,
    sha256,
    state_sha256,
)


def target_at_shape(gt: torch.Tensor, valid: torch.Tensor, shape: tuple[int, int]):
    h, w = shape
    height, width = gt.shape[-2:]
    finite = torch.isfinite(gt).all(dim=1, keepdim=True)
    valid = ((valid > 0.5) & finite).to(gt.dtype)
    safe_gt = torch.where(finite.expand_as(gt), gt, torch.zeros_like(gt))
    coverage = F.interpolate(valid, size=shape, mode="area")
    numerator = F.interpolate(safe_gt * valid, size=shape, mode="bilinear", align_corners=False)
    target = numerator / coverage.clamp_min(1e-6)
    target = target.clone()
    target[:, 0] *= w / width
    target[:, 1] *= h / height
    mask = coverage[:, 0] >= 0.999
    return target, mask


def strict_metadata(saved: dict, arm: str, manifest_sha: str, config_sha: str) -> None:
    expected = {
        "generator": arm,
        "seed": 0,
        "flow_unit": FLOW_UNIT,
        "data_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "real_data_used": False,
        "validation_used_for_selection": False,
    }
    mismatch = {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}
    if mismatch:
        raise RuntimeError(f"{arm} checkpoint metadata mismatch: {mismatch}")


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
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the fixed model-scale audit requires CUDA")
    torch.cuda.set_device(device)

    manifest_path = args.data_root / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha, config_sha = sha256(manifest_path), sha256(args.config)
    if manifest.get("flow_unit") != "pixels per 30ms endpoint interval":
        raise RuntimeError("unexpected shared-flow unit")
    by_arm = {(r["generator"], r["rollout_id"]): r for r in manifest["entries"]}
    train_rollouts = [f"s{i}_00" for i in range(4)]
    val_rollouts = sorted(
        rid for arm, rid in by_arm
        if arm == "eventgenesis" and by_arm[(arm, rid)]["split"] == "validation"
    )
    if len(val_rollouts) != 8:
        raise RuntimeError(f"expected 8 independent validation rollouts, got {val_rollouts}")
    sample_index_by_split = {"train": (16, 48), "validation": (0,)}
    samples = []
    for split, rollouts in (("train", train_rollouts), ("validation", val_rollouts)):
        for rollout in rollouts:
            row_pair = {}
            for arm in ("eventgenesis", "v2e"):
                row = by_arm[(arm, rollout)]
                if row["split"] != split:
                    raise RuntimeError(f"split mismatch for {arm}/{rollout}")
                path = Path(row["path"])
                if not path.is_file() or sha256(path) != row["sha256"]:
                    raise RuntimeError(f"registered adapter changed: {path}")
                row_pair[arm] = load_sample(path, sample_index_by_split[split][0] if split == "train" else 0)
            eg, v2 = row_pair["eventgenesis"], row_pair["v2e"]
            if not (
                np.array_equal(eg["flow_gt"], v2["flow_gt"])
                and np.array_equal(eg["flow_valid"], v2["flow_valid"])
                and np.array_equal(eg["sample_ids"], v2["sample_ids"])
                and np.array_equal(eg["label_sha256"], v2["label_sha256"])
            ):
                raise RuntimeError(f"paired physical labels or IDs differ for {rollout}")
            for arm, sample in row_pair.items():
                if sample["events"].shape != (2, 260, 346, 5):
                    raise RuntimeError(f"{arm} event shape mismatch: {sample['events'].shape}")
                if sample["flow_gt"].shape != (2, 260, 346) or sample["flow_valid"].shape != (260, 346):
                    raise RuntimeError(f"{arm} target/mask shape mismatch")
                if not np.isfinite(sample["events"]).all() or not np.isfinite(sample["flow_gt"]).all():
                    raise RuntimeError(f"{arm} sample contains non-finite inputs/targets")
            active = {
                arm: np.any(sample["events"] != 0, axis=(0, 3))
                for arm, sample in row_pair.items()
            }
            common_active = active["eventgenesis"] | active["v2e"]
            valid = eg["flow_valid"].astype(bool)
            samples.append({
                "split": split,
                "rollout_id": rollout,
                "sample_index": sample_index_by_split[split][0] if split == "train" else 0,
                "sample_id": str(eg["sample_ids"]),
                "label_sha256": str(eg["label_sha256"]),
                "arms": row_pair,
                "valid": valid,
                "active": valid & common_active,
            })
    # The training audit uses two distinct stratified indices per rollout.
    # Re-load the second index explicitly so the sample list is fixed/auditable.
    second_index_rows = []
    for rollout in train_rollouts:
        pair = {}
        for arm in ("eventgenesis", "v2e"):
            p = Path(by_arm[(arm, rollout)]["path"])
            pair[arm] = load_sample(p, 48)
        eg, v2 = pair["eventgenesis"], pair["v2e"]
        if not (np.array_equal(eg["flow_gt"], v2["flow_gt"]) and np.array_equal(eg["flow_valid"], v2["flow_valid"]) and np.array_equal(eg["sample_ids"], v2["sample_ids"])):
            raise RuntimeError(f"second paired training labels differ for {rollout}")
        active = {arm: np.any(sample["events"] != 0, axis=(0, 3)) for arm, sample in pair.items()}
        second_index_rows.append({
            "split": "train", "rollout_id": rollout, "sample_index": 48,
            "sample_id": str(eg["sample_ids"]), "label_sha256": str(eg["label_sha256"]),
            "arms": pair,
            "valid": eg["flow_valid"].astype(bool),
            "active": eg["flow_valid"].astype(bool) & (active["eventgenesis"] | active["v2e"]),
        })
    samples += second_index_rows

    out = {
        "schema": "stage5_flow_multiscale_target_audit_v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "data_manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "flow_unit": FLOW_UNIT,
        "objective": "same four-scale equal-weight robust endpoint function as the bounded diagnostic; per-scale target vectors are resized in pixels and scaled by W_out/W_in, H_out/H_in; validity area coverage >=0.999",
        "samples": [{k: row[k] for k in ("split", "rollout_id", "sample_index", "sample_id", "label_sha256")} for row in samples],
        "real_data_used": False,
        "arms": {},
    }
    for arm, checkpoint_path in (
        ("eventgenesis", args.eventgenesis_checkpoint.resolve()),
        ("v2e", args.v2e_checkpoint.resolve()),
    ):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        strict_metadata(saved, arm, manifest_sha, config_sha)
        model, source_patch = model_source(args.model_root.resolve(), device)
        model.load_state_dict(saved["state_dict"], strict=True)
        if state_sha256(model) != saved.get("state_sha256"):
            raise RuntimeError(f"{arm} model-state hash mismatch")
        arm_rows = []
        for row in samples:
            sample = row["arms"][arm]
            events, gt, valid = pad_sample(sample, device)
            model.eval()
            with torch.no_grad():
                eval_tensor = model(events, 352, use_dt4=False, flow_seq=False)
            eval_flow = model_output_to_flow_hw(eval_tensor)[:, 46:306, 3:349]
            checkpoint_state = copy.deepcopy(model.state_dict())
            model.train()
            with torch.no_grad():
                train_outputs = model(events, 352, use_dt4=False, flow_seq=False)
            if not isinstance(train_outputs, (tuple, list)) or len(train_outputs) != 4:
                raise RuntimeError(f"{arm} must return four training output scales")
            total_loss, scale_losses = multiscale_supervised_epe(train_outputs, gt, valid)
            zero_outputs = [torch.zeros_like(flow) for flow in train_outputs]
            zero_loss, zero_scale_losses = multiscale_supervised_epe(zero_outputs, gt, valid)
            first_train = model_output_to_flow_hw(train_outputs[0])[:, 46:306, 3:349]
            parity_max = float(np.max(np.abs(first_train - eval_flow)))
            model.load_state_dict(checkpoint_state, strict=True)
            model.eval()
            if not np.isfinite(eval_flow).all() or not np.isfinite(first_train).all():
                raise RuntimeError(f"{arm} has non-finite flow outputs")
            valid_np = row["valid"]
            active_np = row["active"]
            scale_rows = []
            for i, prediction in enumerate(train_outputs):
                h, w = prediction.shape[-2:]
                target, mask = target_at_shape(gt, valid, (h, w))
                pred = prediction.detach()
                magnitude = torch.linalg.vector_norm(pred, dim=1)
                target_mag = torch.linalg.vector_norm(target, dim=1)
                scale_rows.append({
                    "scale_index": i,
                    "shape": [int(h), int(w)],
                    "valid_pixels": int(mask.sum().item()),
                    "prediction_mean_magnitude_px_per_30ms_at_scale": float(magnitude[mask].mean().cpu()),
                    "target_mean_magnitude_px_per_30ms_at_scale": float(target_mag[mask].mean().cpu()),
                    "robust_endpoint_loss": scale_losses[i],
                    "zero_prediction_robust_endpoint_loss": zero_scale_losses[i],
                })
            arm_rows.append({
                "split": row["split"],
                "rollout_id": row["rollout_id"],
                "sample_index": row["sample_index"],
                "sample_id": row["sample_id"],
                "label_sha256": row["label_sha256"],
                "common_valid_pixels": int(valid_np.sum()),
                "common_active_pixels": int(active_np.sum()),
                "common_active_coverage": float(active_np.sum() / max(valid_np.sum(), 1)),
                "scale0_train_to_eval_max_abs_difference_px": parity_max,
                "multiscale_objective": float(total_loss.cpu()),
                "multiscale_zero_prediction_objective": float(zero_loss.cpu()),
                "eval_full_valid_epe_px_per_30ms": epe_pair(eval_flow, sample["flow_gt"], valid_np),
                "eval_full_valid_zero_epe_px_per_30ms": epe_pair(np.zeros_like(sample["flow_gt"], dtype=np.float32), sample["flow_gt"], valid_np),
                "eval_common_active_epe_px_per_30ms": epe_pair(eval_flow, sample["flow_gt"], active_np),
                "eval_common_active_zero_epe_px_per_30ms": epe_pair(np.zeros_like(sample["flow_gt"], dtype=np.float32), sample["flow_gt"], active_np),
                "scales": scale_rows,
            })
        out["arms"][arm] = {
            "checkpoint": {"path": str(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size, "sha256": sha256(checkpoint_path), "state_sha256": saved["state_sha256"]},
            "source_patch": source_patch,
            "scale_weights": list(ORIGINAL_SCALE_WEIGHTS),
            "samples": arm_rows,
        }
        del saved, model
        torch.cuda.empty_cache()
    out["status"] = "complete"
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "output": str(args.output), "samples": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
