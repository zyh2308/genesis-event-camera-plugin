#!/usr/bin/env python3
"""Run a bounded paired STE-FlowNet fit on the registered ReplicaCAD pilot bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def epe(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not bool(mask.any()):
        return None
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 2:
        raise ValueError(f"prediction and target must share [2,H,W], got {prediction.shape}, {target.shape}")
    if mask.shape != target.shape[1:] or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("evaluation mask shape or flow finite check failed")
    delta = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.hypot(delta[0], delta[1])[mask].mean(dtype=np.float64))


def pad_inputs(sample: dict, device: torch.device, target_size: int = 352):
    event_np = np.asarray(sample["events"])
    flow_np = np.asarray(sample["flow_gt"], dtype=np.float32)
    valid_np = np.asarray(sample["flow_valid"], dtype=np.uint8)
    if event_np.ndim != 4 or event_np.shape[0] != 2 or event_np.shape[-1] != 5:
        raise ValueError(f"event tensor must be [2,H,W,5], got {event_np.shape}")
    h, w = event_np.shape[1:3]
    if flow_np.shape != (2, h, w) or valid_np.shape != (h, w):
        raise ValueError(f"sample tensors disagree: events={event_np.shape}, flow={flow_np.shape}, valid={valid_np.shape}")
    dy, dx = target_size - h, target_size - w
    if dy < 0 or dx < 0 or dy % 2 or dx % 2:
        raise ValueError(f"symmetric padding to {target_size} is impossible for {h}x{w}")
    top, left = dy // 2, dx // 2
    bottom, right = dy - top, dx - left
    events = torch.from_numpy(event_np.astype(np.float32, copy=False)).unsqueeze(0)
    events = F.pad(events, (0, 0, left, right, top, bottom)).to(device)
    flow = torch.from_numpy(flow_np).unsqueeze(0)
    flow = F.pad(flow, (left, right, top, bottom)).to(device)
    valid = torch.from_numpy(valid_np.astype(np.float32, copy=False)).view(1, 1, h, w)
    valid = F.pad(valid, (left, right, top, bottom)).to(device)
    return events, flow, valid, (top, left, h, w)


def unpad(prediction: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
    top, left, h, w = pad
    result = prediction[:, top:top+h, left:left+w]
    if result.shape != (2, h, w) or not np.isfinite(result).all():
        raise ValueError(f"unpad returned invalid flow array {result.shape}")
    return result


def evaluate(model, sample: dict, common_active: np.ndarray, device, helper):
    events, gt_tensor, valid_tensor, pad = pad_inputs(sample, device)
    model.eval()
    with torch.inference_mode():
        output = model(events, 352, use_dt4=False, flow_seq=False)
        prediction = unpad(helper["model_output_to_flow_hw"](output), pad)
    target = np.asarray(sample["flow_gt"], dtype=np.float32)
    valid = np.asarray(sample["flow_valid"], dtype=bool)
    active = valid & common_active
    return {
        "sample_id": str(sample["sample_ids"]),
        "full_valid_epe_px_per_30ms": epe(prediction, target, valid),
        "common_event_active_epe_px_per_30ms": epe(prediction, target, active),
        "zero_flow_full_valid_epe_px_per_30ms": epe(np.zeros_like(target), target, valid),
        "zero_flow_common_event_active_epe_px_per_30ms": epe(np.zeros_like(target), target, active),
        "full_valid_pixels": int(valid.sum()),
        "common_event_active_pixels": int(active.sum()),
        "prediction_mean_magnitude_valid_px_per_30ms": float(np.hypot(prediction[0], prediction[1])[valid].mean()) if valid.any() else None,
        "prediction_mean_magnitude_active_px_per_30ms": float(np.hypot(prediction[0], prediction[1])[active].mean()) if active.any() else None,
        "gt_mean_magnitude_valid_px_per_30ms": float(np.hypot(target[0], target[1])[valid].mean()) if valid.any() else None,
        "prediction_flow_sha256": array_sha256(prediction),
        "prediction": prediction,
        "gt": target,
        "valid": valid,
        "active": active,
    }


def optimizer_steps(optimizer):
    return sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})


def load_rows(manifest: dict):
    by_key = {(row["generator"], row["rollout_id"]): row for row in manifest["entries"]}
    train = sorted({row["rollout_id"] for row in manifest["entries"] if row["split"] == "train"})
    validation = sorted({row["rollout_id"] for row in manifest["entries"] if row["split"] == "validation"})
    if (len(train), len(validation)) != (4, 2):
        raise RuntimeError(f"registered bank must have 4 train and 2 validation rollouts; got {train}, {validation}")
    rows = {"train": [], "validation": []}
    for split, rollouts in (("train", train), ("validation", validation)):
        for rollout in rollouts:
            pair = {}
            for arm in ("eventgenesis", "v2e"):
                row = by_key[(arm, rollout)]
                path = Path(row["path"])
                if row["split"] != split or not path.is_file() or sha256(path) != row["sha256"]:
                    raise RuntimeError(f"registered {split} artifact failed path/hash/split check: {path}")
                with np.load(path, allow_pickle=False) as archive:
                    sample = {key: np.asarray(archive[key]).copy() for key in
                              ("events", "flow_gt", "flow_valid", "sample_ids", "label_sha256")}
                    sample["event_count"] = int(np.asarray(archive["event_count"]))
                pair[arm] = sample
            a, b = pair["eventgenesis"], pair["v2e"]
            for field in ("flow_gt", "flow_valid", "sample_ids", "label_sha256"):
                if not np.array_equal(a[field], b[field]):
                    raise RuntimeError(f"paired label field {field} differs for {rollout}")
            rows[split].append({"rollout_id": rollout, "arms": pair})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite pilot output: {args.output_root}")
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo / "benchmark/sim2real/scripts"))
    sys.path.insert(0, str(repo / "benchmark/sim2real/compat"))
    from stage5_flow_supervised_pilot import (
        SOURCE_COMMIT, model_output_to_flow_hw, model_source,
        multiscale_supervised_epe, sha256 as helper_sha256, state_sha256,
    )
    from verify_pinned_source import verify_pinned_source

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    updates = int(config["diagnostic_budget"]["updates_per_arm"])
    if updates < 1:
        raise ValueError("diagnostic update budget must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered network diagnostic requires CUDA")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    manifest_path = args.data_root.resolve() / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("flow_unit") != "pixels per 30ms endpoint interval":
        raise RuntimeError("training bank does not use the registered 30ms flow unit")
    if manifest.get("real_data_used") is not False:
        raise RuntimeError("real data must not enter the synthetic GT diagnostic")
    data_rows = load_rows(manifest)
    for split_name in ("train", "validation"):
        observed = [row["rollout_id"] for row in data_rows[split_name]]
        config_key = "training_rollouts" if split_name == "train" else "validation_rollouts"
        expected = config["data"][config_key]
        if observed != expected:
            raise RuntimeError(f"{split_name} rollout list differs from the frozen config: {observed} != {expected}")
    model_verification = verify_pinned_source(
        args.model_root.resolve(), "STE-FlowNet", SOURCE_COMMIT,
        repo / "benchmark/sim2real/manifests/downstream_source_bundle.json",
    )
    manifest_hash, config_hash = helper_sha256(manifest_path), helper_sha256(config_path)
    out = args.output_root.resolve()
    out.mkdir(parents=True)

    # Freeze the same event-active union from each normal paired input. The
    # mask is fixed before either arm is trained and is never arm-specific.
    masks = {}
    for split, rows in data_rows.items():
        for row in rows:
            eg = np.asarray(row["arms"]["eventgenesis"]["events"])
            v2 = np.asarray(row["arms"]["v2e"]["events"])
            if eg.shape != v2.shape or eg.shape[:3] != (2, 240, 320):
                raise RuntimeError(f"paired event representation shape mismatch in {row['rollout_id']}")
            active_union = np.any(eg != 0, axis=(0, 3)) | np.any(v2 != 0, axis=(0, 3))
            valid = np.asarray(row["arms"]["eventgenesis"]["flow_valid"], dtype=bool)
            masks[row["rollout_id"]] = active_union & valid

    result = {
        "schema": "stage7_replicad_multiaxis_paired_gtfit_v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repo_execution_checkout": str(repo),
        "model_source_verification": model_verification,
        "model_source_commit": SOURCE_COMMIT,
        "model_source_root": str(args.model_root.resolve()),
        "config_path": str(config_path), "config_sha256": config_hash,
        "data_manifest_path": str(manifest_path), "data_manifest_sha256": manifest_hash,
        "objective": config["objective"], "pairing": config["pairing"],
        "registered_updates_per_arm": updates,
        "train_rollouts": [row["rollout_id"] for row in data_rows["train"]],
        "validation_rollouts": [row["rollout_id"] for row in data_rows["validation"]],
        "common_masks_frozen_before_training": True,
        "common_active_pixels": {key: int(value.sum()) for key, value in masks.items()},
        "real_data_used": False, "formal_checkpoint_selection": False,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": torch.cuda.get_device_properties(device).uuid if hasattr(torch.cuda.get_device_properties(device), "uuid") else None,
        "arms": {},
    }
    seed = int(config["pairing"]["same_initialization_seed"])
    order = np.random.default_rng(20261031).permutation(len(data_rows["train"])).tolist()
    result["sample_order_indices_per_cycle"] = order
    update_records = list(range(1, updates + 1))
    eval_points = set(config["diagnostic_budget"]["evaluation_updates"])
    initial_hashes = set()

    for arm in config["pairing"]["arms"]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model, patch_info = model_source(args.model_root.resolve(), device)
        initial_hash = state_sha256(model)
        initial_hashes.add(initial_hash)
        if len(initial_hashes) > 1:
            raise RuntimeError("paired arms did not receive the same initialized weights")
        optimizer = torch.optim.Adam([
            {"params": model.bias_parameters(), "weight_decay": float(config["pairing"]["weight_decay_biases"])},
            {"params": model.weight_parameters(), "weight_decay": float(config["pairing"]["weight_decay_weights"])},
        ], lr=float(config["pairing"]["learning_rate"]), betas=tuple(config["pairing"]["betas"]))
        arm_dir = out / arm
        arm_dir.mkdir()
        log_path = arm_dir / "updates.jsonl"
        history = []
        started = time.monotonic()
        with log_path.open("w") as log:
            def record_evaluation(update: int):
                model.eval()
                values = []
                for split_name in ("train", "validation"):
                    for row in data_rows[split_name]:
                        sample = row["arms"][arm]
                        metrics = evaluate(model, sample, masks[row["rollout_id"]], device,
                                           {"model_output_to_flow_hw": model_output_to_flow_hw})
                        prediction = metrics.pop("prediction")
                        target = metrics.pop("gt")
                        active = metrics.pop("active")
                        metrics.pop("valid")
                        metrics.update({
                            "rollout_id": row["rollout_id"],
                            "split": split_name,
                            "update": update,
                            "prediction_direction_cosine_mean_active": None,
                        })
                        moving_active = active & (np.hypot(target[0], target[1]) >= 0.5)
                        if moving_active.any():
                            dot = prediction[0] * target[0] + prediction[1] * target[1]
                            denom = np.hypot(prediction[0], prediction[1]) * np.hypot(target[0], target[1])
                            metrics["prediction_direction_cosine_mean_active"] = float(
                                (dot[moving_active] / np.maximum(denom[moving_active], 1e-12)).mean(dtype=np.float64)
                            )
                        values.append(metrics)
                history.extend(values)
                for value in values:
                    log.write(json.dumps({"record_type": "evaluation", **value}, sort_keys=True) + "\n")
                log.flush()
                return values

            record_evaluation(0)
            for update in range(1, updates + 1):
                cycle = (update - 1) // len(order)
                train_index = order[(update - 1) % len(order)]
                sample_row = data_rows["train"][train_index]
                sample = sample_row["arms"][arm]
                events, gt, valid, _ = pad_inputs(sample, device)
                model.train()
                outputs = model(events, 352, use_dt4=False, flow_seq=False)
                loss, scale_losses = multiscale_supervised_epe(outputs, gt, valid)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{arm} update {update}: nonfinite four-scale loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError(f"{arm} update {update}: nonfinite or missing gradients")
                grad_norm = float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients)).cpu())
                if not np.isfinite(grad_norm) or grad_norm <= 0:
                    raise RuntimeError(f"{arm} update {update}: gradient norm is not positive and finite")
                optimizer.step()
                steps = optimizer_steps(optimizer)
                if steps != [update]:
                    raise RuntimeError(f"{arm} optimizer counters are {steps}, expected {update}")
                rec = {
                    "record_type": "optimizer_update", "update": update, "cycle": cycle,
                    "rollout_id": sample_row["rollout_id"],
                    "sample_id": str(sample["sample_ids"]),
                    "input_event_count": int(sample["event_count"]),
                    "loss_before_update": float(loss.detach().cpu()),
                    "four_scale_losses_before_update": scale_losses,
                    "gradient_l2_norm": grad_norm, "gradient_tensor_count": len(gradients),
                    "optimizer_step_values": steps,
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                }
                log.write(json.dumps(rec, sort_keys=True) + "\n")
                log.flush()
                if update in eval_points:
                    record_evaluation(update)
        checkpoint_path = arm_dir / "checkpoint_final.pt"
        state_hash = state_sha256(model)
        torch.save({
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "generator": arm, "seed": seed, "updates": updates,
            "initial_state_sha256": initial_hash, "state_sha256": state_hash,
            "config_sha256": config_hash, "data_manifest_sha256": manifest_hash,
            "flow_unit": manifest["flow_unit"], "real_data_used": False,
            "event_only_inference": True,
        }, checkpoint_path)
        restored, _ = model_source(args.model_root.resolve(), device)
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        restored.load_state_dict(saved["state_dict"], strict=True)
        if state_sha256(restored) != state_hash:
            raise RuntimeError(f"{arm} checkpoint restore state hash differs")
        restored_optimizer = torch.optim.Adam([
            {"params": restored.bias_parameters(), "weight_decay": float(config["pairing"]["weight_decay_biases"])},
            {"params": restored.weight_parameters(), "weight_decay": float(config["pairing"]["weight_decay_weights"])},
        ], lr=float(config["pairing"]["learning_rate"]), betas=tuple(config["pairing"]["betas"]))
        restored_optimizer.load_state_dict(saved["optimizer"])
        if optimizer_steps(restored_optimizer) != [updates]:
            raise RuntimeError(f"{arm} checkpoint did not restore all Adam steps")
        restored.eval(); model.eval()
        restore_comparisons = []
        with torch.inference_mode():
            for row in data_rows["validation"]:
                sample = row["arms"][arm]
                ev, _, _, pad = pad_inputs(sample, device)
                model_pred = unpad(model_output_to_flow_hw(model(ev, 352, use_dt4=False, flow_seq=False)), pad)
                restored_pred = unpad(model_output_to_flow_hw(restored(ev, 352, use_dt4=False, flow_seq=False)), pad)
                max_abs = float(np.max(np.abs(model_pred - restored_pred)))
                if max_abs > 1e-6:
                    raise RuntimeError(f"{arm} restored eval differs by {max_abs}")
                restore_comparisons.append({"rollout_id": row["rollout_id"], "max_abs": max_abs,
                                            "prediction_sha256": array_sha256(restored_pred)})
        arm_evaluations = [x for x in history if x["update"] in eval_points and x["split"] == "validation"]
        eval_by_update = {}
        for update in sorted(eval_points):
            subset = [x for x in arm_evaluations if x["update"] == update]
            if subset:
                full = [x["full_valid_epe_px_per_30ms"] for x in subset if x["full_valid_epe_px_per_30ms"] is not None]
                active = [x["common_event_active_epe_px_per_30ms"] for x in subset if x["common_event_active_epe_px_per_30ms"] is not None]
                zero_full = [x["zero_flow_full_valid_epe_px_per_30ms"] for x in subset if x["zero_flow_full_valid_epe_px_per_30ms"] is not None]
                zero_active = [x["zero_flow_common_event_active_epe_px_per_30ms"] for x in subset if x["zero_flow_common_event_active_epe_px_per_30ms"] is not None]
                eval_by_update[str(update)] = {
                    "validation_rollouts": len(subset),
                    "macro_full_valid_epe": float(np.mean(full)) if full else None,
                    "macro_common_active_epe": float(np.mean(active)) if active else None,
                    "macro_zero_flow_full_valid_epe": float(np.mean(zero_full)) if zero_full else None,
                    "macro_zero_flow_common_active_epe": float(np.mean(zero_active)) if zero_active else None,
                }
        result["arms"][arm] = {
            "initial_state_sha256": initial_hash, "final_state_sha256": state_hash,
            "source_patch": patch_info,
            "updates": updates, "optimizer_final_steps": optimizer_steps(optimizer),
            "checkpoint": {"path": str(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size,
                           "sha256": sha256(checkpoint_path), "restored_state_equal": True,
                           "restored_eval": restore_comparisons},
            "validation_by_update": eval_by_update,
            "validation_rollouts_final": [x for x in history if x["update"] == updates and x["split"] == "validation"],
            "wall_seconds": time.monotonic() - started,
        }
        del restored_optimizer, restored, model, optimizer
        torch.cuda.empty_cache()

    result["status"] = "complete"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"], "summary": str(summary_path),
        "gpu": result["gpu_name"], "updates_per_arm": updates,
        "paired_initial_state": len(initial_hashes) == 1,
        "validation_by_update": {arm: result["arms"][arm]["validation_by_update"] for arm in result["arms"]},
    }, indent=2))


if __name__ == "__main__":
    main()
