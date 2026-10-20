#!/usr/bin/env python3
"""Fit the pinned STE-FlowNet on one preregistered physical window.

This bounded GT-supervised diagnostic checks actual network optimization. It
does not alter the formal self-supervised training target or select weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/sim2real/scripts"))
from stage5_flow_supervised_pilot import (  # noqa: E402
    FLOW_UNIT,
    load_sample,
    model_output_to_flow_hw,
    model_source,
    multiscale_supervised_epe,
    pad_sample,
    sha256,
    state_sha256,
)


def epe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    delta = pred.astype(np.float64, copy=False) - gt.astype(np.float64, copy=False)
    return float(np.sqrt(np.square(delta).sum(axis=0))[mask].mean())


def optimizer_steps(optimizer: torch.optim.Optimizer) -> list[int]:
    return sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})


def source_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest = ROOT / "SOURCE_FILE_HASHES.json"
        if not manifest.is_file():
            raise RuntimeError("execution source is neither a Git checkout nor a verified source archive")
        record = json.loads(manifest.read_text())
        if not isinstance(record.get("commit"), str) or len(record["commit"]) != 40:
            raise RuntimeError("source archive manifest has no valid commit ID")
        return record["commit"]


def model_parameter_delta(model: torch.nn.Module, initial: dict[str, torch.Tensor]) -> float:
    squared = 0.0
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            difference = value.detach().cpu().float() - initial[name].float()
            squared += float(difference.square().sum())
    return float(np.sqrt(squared))


def evaluate_fixed(model, sample, gt, valid, active, device) -> dict:
    # STE returns the four supervision scales in training mode. Freeze every
    # BN layer while keeping that return contract so evaluation does not
    # re-estimate statistics or contaminate training state.
    model.train()
    batchnorms = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    for module in batchnorms:
        module.eval()
    events, gt_tensor, valid_tensor = pad_sample(sample, device)
    with torch.inference_mode():
        outputs = model(events, 352, use_dt4=False, flow_seq=False)
        composite, scales = multiscale_supervised_epe(outputs, gt_tensor, valid_tensor)
        pred = model_output_to_flow_hw(outputs[0])[:, 46:306, 3:349][0].detach().cpu().numpy()
    if pred.shape != gt.shape or not np.isfinite(pred).all():
        raise RuntimeError(f"invalid fixed-window flow output: {pred.shape}")
    return {
        "four_scale_robust_objective": float(composite.cpu()),
        "scale_robust_endpoint_losses": scales,
        "full_valid_epe_px_per_30ms": epe(pred, gt, valid),
        "common_event_active_epe_px_per_30ms": epe(pred, gt, active),
        "zero_flow_full_valid_epe_px_per_30ms": epe(np.zeros_like(gt), gt, valid),
        "zero_flow_common_event_active_epe_px_per_30ms": epe(np.zeros_like(gt), gt, active),
        "prediction_mean_magnitude_full_valid": float(np.hypot(pred[0], pred[1])[valid].mean()),
        "prediction_mean_magnitude_active": float(np.hypot(pred[0], pred[1])[active].mean()) if active.any() else None,
        "batchnorm_layer_count_frozen_for_diagnostic": len(batchnorms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sample_cfg = config["sample"]
    run_cfg = config["diagnostic_budget"]
    out = args.output_root.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {out}")
    out.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fixed-network fit requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    manifest_path = args.data_root.resolve() / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha256(manifest_path)
    if manifest.get("flow_unit") != "pixels per 30ms endpoint interval":
        raise RuntimeError("expected shared 30 ms flow labels")
    rows = {(row["generator"], row["rollout_id"]): row for row in manifest["entries"]}
    rollout = sample_cfg["rollout_id"]
    paired = {}
    for arm in config["pairing"]["arms"]:
        row = rows[(arm, rollout)]
        if row["split"] != sample_cfg["split"]:
            raise RuntimeError(f"{arm}/{rollout} split mismatch: {row['split']}")
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"registered event/GT adapter failed hash check: {path}")
        paired[arm] = load_sample(path, int(sample_cfg["window_index"]))
    eg, v2e = paired["eventgenesis"], paired["v2e"]
    if str(eg["sample_ids"]) != sample_cfg["sample_id"] or str(v2e["sample_ids"]) != sample_cfg["sample_id"]:
        raise RuntimeError("fixed sample ID does not match the preregistered ID")
    for arm, sample in paired.items():
        if sample["events"].shape != (2, 260, 346, 5):
            raise RuntimeError(f"unexpected {arm} event tensor shape {sample['events'].shape}")
        if sample["flow_gt"].shape != (2, 260, 346) or sample["flow_valid"].shape != (260, 346):
            raise RuntimeError(f"unexpected {arm} flow/validity shapes")
        if not np.isfinite(sample["events"]).all() or not np.isfinite(sample["flow_gt"]).all():
            raise RuntimeError(f"non-finite {arm} input or label")
    for key in ("flow_gt", "flow_valid", "sample_ids", "label_sha256"):
        if not np.array_equal(eg[key], v2e[key]):
            raise RuntimeError(f"paired physical label field differs: {key}")
    gt = eg["flow_gt"].astype(np.float32, copy=False)
    valid = eg["flow_valid"].astype(bool, copy=False)
    event_active = {arm: np.any(sample["events"] != 0, axis=(0, 3)) for arm, sample in paired.items()}
    common_active = valid & (event_active["eventgenesis"] | event_active["v2e"])
    if valid.sum() == 0 or common_active.sum() == 0:
        raise RuntimeError("fixed sample has empty valid or shared event-active support")
    physical_magnitude = np.hypot(gt[0], gt[1])
    if not np.any(valid & (physical_magnitude > 1e-3)):
        raise RuntimeError("preregistered fixed sample has no nonzero physical motion")

    result = {
        "schema": "stage5_flow_fixed_sample_network_fit_v1",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit(),
        "config_sha256": sha256(args.config.resolve()),
        "data_manifest_sha256": manifest_hash,
        "flow_unit": FLOW_UNIT,
        "sample": {key: sample_cfg[key] for key in sample_cfg},
        "paired_labels_identical": True,
        "full_valid_pixels": int(valid.sum()),
        "common_event_active_pixels": int(common_active.sum()),
        "common_event_active_coverage": float(common_active.sum() / valid.sum()),
        "generator_active_pixels": {arm: int((valid & event_active[arm]).sum()) for arm in paired},
        "physical_motion_mean_magnitude_full_valid": float(physical_magnitude[valid].mean()),
        "physical_motion_nonzero_pixels": int((valid & (physical_magnitude > 1e-3)).sum()),
        "budget_updates_per_arm": int(run_cfg["updates_per_arm"]),
        "evaluation_updates": run_cfg["evaluate_updates"],
        "optimizer_contract": config["pairing"],
        "mask_contract": config["mask"],
        "real_data_used": False,
        "formal_checkpoint_selection": False,
        "arms": {},
    }
    initial_hashes = {}
    updates = int(run_cfg["updates_per_arm"])
    checkpoints = list(run_cfg["evaluate_updates"])
    result["sample"]["paired_label_sha256"] = str(eg["label_sha256"])
    result["sample"]["event_artifacts"] = {
        arm: {"path": str(Path(rows[(arm, rollout)]["path"])), "sha256": rows[(arm, rollout)]["sha256"]}
        for arm in paired
    }
    for arm in config["pairing"]["arms"]:
        random.seed(config["pairing"]["same_initialization_seed"])
        np.random.seed(config["pairing"]["same_initialization_seed"])
        torch.manual_seed(config["pairing"]["same_initialization_seed"])
        torch.cuda.manual_seed_all(config["pairing"]["same_initialization_seed"])
        model, source_patch = model_source(args.model_root.resolve(), device)
        initial_hash = state_sha256(model)
        initial_hashes[arm] = initial_hash
        if len(set(initial_hashes.values())) > 1:
            raise RuntimeError("paired initial model weights differ")
        initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.Adam([
            {"params": model.bias_parameters(), "weight_decay": 0.0},
            {"params": model.weight_parameters(), "weight_decay": 0.0004},
        ], lr=0.0004, betas=(0.9, 0.999))
        events, gt_tensor, valid_tensor = pad_sample(paired[arm], device)
        arm_result = {
            "initial_state_sha256": initial_hash,
            "source_patch": source_patch,
            "optimizer_param_group_sizes": [len(group["params"]) for group in optimizer.param_groups],
            "optimizer_steps": [],
            "history": [],
        }
        log_path = out / f"{arm}_fixed_sample_updates.jsonl"
        started = time.monotonic()
        with log_path.open("w") as log:
            for update in range(updates + 1):
                if update in checkpoints:
                    metrics = evaluate_fixed(model, paired[arm], gt, valid, common_active, device)
                    metrics.update({
                        "record_type": "checkpoint_evaluation",
                        "update": update,
                        "parameter_delta_l2_from_initial": model_parameter_delta(model, initial_state),
                        "optimizer_step_values": optimizer_steps(optimizer),
                        "recorded_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    arm_result["history"].append(metrics)
                    log.write(json.dumps(metrics, sort_keys=True) + "\n")
                    log.flush()
                    if update == updates:
                        break
                model.train()
                outputs = model(events, 352, use_dt4=False, flow_seq=False)
                loss, scale_losses = multiscale_supervised_epe(outputs, gt_tensor, valid_tensor)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{arm} update {update + 1}: non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not grads or any(not torch.isfinite(gradient).all() for gradient in grads):
                    raise RuntimeError(f"{arm} update {update + 1}: invalid gradient tensors")
                grad_norm = float(torch.sqrt(sum(g.detach().float().square().sum() for g in grads)).cpu())
                if not np.isfinite(grad_norm) or grad_norm <= 0:
                    raise RuntimeError(f"{arm} update {update + 1}: zero gradient norm")
                optimizer.step()
                step_values = optimizer_steps(optimizer)
                if step_values != [update + 1]:
                    raise RuntimeError(f"{arm} update {update + 1}: optimizer step did not advance ({step_values})")
                step_record = {
                    "record_type": "optimizer_update",
                    "update": update + 1,
                    "step_values": step_values,
                    "gradient_l2_norm": grad_norm,
                    "gradient_tensor_count": len(grads),
                    "multiscale_objective_before_step": float(loss.detach().cpu()),
                    "scale_losses_before_step": scale_losses,
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                }
                if update + 1 in checkpoints:
                    step_record["parameter_state_sha256_after_step"] = state_sha256(model)
                    parameter_delta = model_parameter_delta(model, initial_state)
                    step_record["parameter_delta_l2_from_initial"] = parameter_delta
                    if update + 1 == 20 and parameter_delta <= 0:
                        raise RuntimeError(f"{arm}: parameters did not change by update 20")
                arm_result["optimizer_steps"].append(step_record)
                log.write(json.dumps(step_record, sort_keys=True) + "\n")
                log.flush()
        arm_result["wall_seconds"] = time.monotonic() - started
        arm_result["final_state_sha256"] = state_sha256(model)
        checkpoint_path = out / f"{arm}_last.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": arm,
            "seed": 0,
            "updates": updates,
            "sample_id": sample_cfg["sample_id"],
            "initial_state_sha256": initial_hash,
            "state_sha256": arm_result["final_state_sha256"],
            "data_manifest_sha256": manifest_hash,
            "config_sha256": result["config_sha256"],
            "flow_unit": FLOW_UNIT,
            "real_data_used": False,
        }, checkpoint_path)
        restored, _ = model_source(args.model_root.resolve(), device)
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        restored.load_state_dict(saved["state_dict"], strict=True)
        restored_optimizer = torch.optim.Adam([
            {"params": restored.bias_parameters(), "weight_decay": 0.0},
            {"params": restored.weight_parameters(), "weight_decay": 0.0004},
        ], lr=0.0004, betas=(0.9, 0.999))
        restored_optimizer.load_state_dict(saved["optimizer"])
        restored.eval(); model.eval()
        with torch.inference_mode():
            expected = model_output_to_flow_hw(model(events, 352, use_dt4=False, flow_seq=False))
            actual = model_output_to_flow_hw(restored(events, 352, use_dt4=False, flow_seq=False))
        restore_max_abs = float((expected - actual).abs().max().cpu())
        restored_steps = optimizer_steps(restored_optimizer)
        if restore_max_abs != 0.0 or state_sha256(restored) != arm_result["final_state_sha256"] or restored_steps != [updates]:
            raise RuntimeError(f"{arm}: restored checkpoint prediction/state mismatch")
        arm_result["checkpoint"] = {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
            "restored_state_equal": True,
            "restored_prediction_max_abs": restore_max_abs,
            "restored_optimizer_step_values": restored_steps,
        }
        result["arms"][arm] = arm_result
        del restored_optimizer, restored, model, optimizer
        torch.cuda.empty_cache()

    criteria = run_cfg["memorization_pass"]
    for arm_result in result["arms"].values():
        history = arm_result["history"]
        first, final = history[0], history[-1]
        ratios = {
            "four_scale_objective_final_to_initial": final["four_scale_robust_objective"] / first["four_scale_robust_objective"],
            "full_valid_epe_final_to_initial": final["full_valid_epe_px_per_30ms"] / first["full_valid_epe_px_per_30ms"],
            "common_event_active_epe_final_to_initial": final["common_event_active_epe_px_per_30ms"] / first["common_event_active_epe_px_per_30ms"],
        }
        arm_result["memorization_ratios"] = ratios
        arm_result["memorization_pass"] = (
            ratios["four_scale_objective_final_to_initial"] <= criteria["four_scale_robust_objective_ratio_final_to_initial_max"]
            and ratios["full_valid_epe_final_to_initial"] <= criteria["full_valid_epe_ratio_final_to_initial_max"]
            and ratios["common_event_active_epe_final_to_initial"] <= criteria["common_event_active_epe_ratio_final_to_initial_max"]
        )
    result["status"] = "complete"
    result["memorization_gate"] = "pass" if all(row["memorization_pass"] for row in result["arms"].values()) else "not_passed"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary = out / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "memorization_gate": result["memorization_gate"],
        "sample_id": sample_cfg["sample_id"],
        "summary": {
            arm: {
                "final_to_initial": row["memorization_ratios"],
                "pass": row["memorization_pass"],
                "updates": len(row["optimizer_steps"]),
                "checkpoint": row["checkpoint"],
            }
            for arm, row in result["arms"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
