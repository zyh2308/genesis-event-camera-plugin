#!/usr/bin/env python3
"""Run a bounded paired self-supervised STE-FlowNet diagnostic on ReplicaCAD."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
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


def state_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def support(events: np.ndarray) -> np.ndarray:
    return np.any(events != 0, axis=(0, 3))


def epe(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 2:
        raise ValueError(f"flow must share strict [2,H,W] shape, got {prediction.shape}/{target.shape}")
    if mask.shape != prediction.shape[1:] or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"mask must be finite binary [H,W], got {mask.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and GT must be finite")
    mask = mask.astype(bool, copy=False)
    if not mask.any():
        return None
    delta = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.hypot(delta[0], delta[1])[mask].mean(dtype=np.float64))


def padded_tensors(events: np.ndarray, gray_prev: np.ndarray, gray_next: np.ndarray,
                   device: torch.device):
    if events.shape != (2, 240, 320, 5):
        raise ValueError(f"native event input must be [2,240,320,5], got {events.shape}")
    if gray_prev.shape != (240, 320) or gray_next.shape != (240, 320):
        raise ValueError(f"native grayscale endpoints must be [240,320], got {gray_prev.shape}/{gray_next.shape}")
    event_tensor = torch.from_numpy(events.astype(np.float32, copy=False)).unsqueeze(0)
    event_tensor = F.pad(event_tensor, (0, 0, 16, 16, 56, 56)).to(device)
    prev = torch.from_numpy(gray_prev.astype(np.float32, copy=False)).view(1, 1, 240, 320)
    next_ = torch.from_numpy(gray_next.astype(np.float32, copy=False)).view(1, 1, 240, 320)
    prev = F.pad(prev, (16, 16, 56, 56), mode="replicate").to(device)
    next_ = F.pad(next_, (16, 16, 56, 56), mode="replicate").to(device)
    return event_tensor, prev[:, 0], next_[:, 0]


def crop_native(prediction: np.ndarray) -> np.ndarray:
    if prediction.shape != (2, 352, 352) or not np.isfinite(prediction).all():
        raise ValueError(f"padded prediction must be finite [2,352,352], got {prediction.shape}")
    result = prediction[:, 56:296, 16:336]
    if result.shape != (2, 240, 320):
        raise RuntimeError("native output crop produced an incorrect shape")
    return result


def load_bank(manifest_path: Path) -> dict[str, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "stage7_replicad_multiaxis_flow_training_bank_v1"
            or manifest.get("status") != "complete" or manifest.get("real_data_used") is not False
            or manifest.get("flow_unit") != "pixels per 30ms endpoint interval"):
        raise RuntimeError("flow bank manifest contract failed")
    entries = {(row["generator"], row["rollout_id"]): row for row in manifest["entries"]}
    split_ids = {
        split: sorted({row["rollout_id"] for row in manifest["entries"] if row["split"] == split})
        for split in ("train", "validation")
    }
    if len(split_ids["train"]) != 4 or len(split_ids["validation"]) != 2:
        raise RuntimeError(f"expected a 4/2 rollout split, got {split_ids}")
    bank = {"manifest": manifest, "train": [], "validation": []}
    for split in ("train", "validation"):
        for rollout in split_ids[split]:
            arms = {}
            motion_hash = None
            gray = None
            for arm in ("eventgenesis", "v2e"):
                row = entries[(arm, rollout)]
                path = Path(row["path"])
                if row["split"] != split or not path.is_file() or sha256(path) != row["sha256"]:
                    raise RuntimeError(f"bank artifact hash/split mismatch: {path}")
                with np.load(path, allow_pickle=False) as archive:
                    sample = {key: np.asarray(archive[key]).copy() for key in
                              ("events", "flow_gt", "flow_valid", "sample_ids", "label_sha256")}
                    sample["event_count"] = int(np.asarray(archive["event_count"]))
                if (sample["events"].shape != (2, 240, 320, 5)
                        or sample["flow_gt"].shape != (2, 240, 320)
                        or sample["flow_valid"].shape != (240, 320)):
                    raise RuntimeError(f"bank tensor contract failed for {arm}/{rollout}")
                if not np.isfinite(sample["events"]).all() or not np.isfinite(sample["flow_gt"]).all():
                    raise RuntimeError(f"nonfinite bank tensor for {arm}/{rollout}")
                arms[arm] = sample
                source_dir = Path(row["source_motion_dir"])
                radiance_path = source_dir / "radiance_frames.npy"
                if not radiance_path.is_file() or sha256(radiance_path) != row["source_motion_sha256"]:
                    raise RuntimeError(f"radiance source hash mismatch for {rollout}: {radiance_path}")
                if motion_hash is not None and motion_hash != row["source_motion_sha256"]:
                    raise RuntimeError(f"paired generators use different radiance files for {rollout}")
                motion_hash = row["source_motion_sha256"]
                if gray is None:
                    radiance = np.load(radiance_path, allow_pickle=False)
                    if radiance.shape != (4, 240, 320, 3) or not np.isfinite(radiance).all():
                        raise RuntimeError(f"radiance contract failed for {radiance_path}: {radiance.shape}")
                    luma = (0.2126 * radiance[..., 0] + 0.7152 * radiance[..., 1]
                            + 0.0722 * radiance[..., 2]).astype(np.float32)
                    gray = {"prev": luma[0].copy(), "next": luma[-1].copy(),
                            "radiance_path": str(radiance_path), "radiance_sha256": motion_hash,
                            "radiance_min": float(radiance.min()), "radiance_max": float(radiance.max())}
            for key in ("flow_gt", "flow_valid", "sample_ids", "label_sha256"):
                if not np.array_equal(arms["eventgenesis"][key], arms["v2e"][key]):
                    raise RuntimeError(f"paired labels differ for {rollout}/{key}")
            bank[split].append({"rollout_id": rollout, "arms": arms, "gray": gray,
                                "split": split, "shared_label_sha256": str(arms["eventgenesis"]["label_sha256"])})
    return bank


def output_to_flow(output: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(output) or output.ndim != 4 or output.shape[:2] != (1, 2):
        raise ValueError(f"eval output must be [1,2,H,W], got {getattr(output, 'shape', None)}")
    array = output.detach().cpu().numpy()[0]
    if not np.isfinite(array).all():
        raise RuntimeError("model output is nonfinite")
    return array


def evaluate_rows(model, rows: list[dict], arm: str, device, output_to_flow, *, update: int) -> list[dict]:
    records = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            sample = row["arms"][arm]
            gray = row["gray"]
            normal = np.asarray(sample["events"], dtype=np.float32)
            paired = np.asarray(row["arms"]["v2e" if arm == "eventgenesis" else "eventgenesis"]["events"], dtype=np.float32)
            common = support(normal) | support(paired)
            valid = np.asarray(sample["flow_valid"], dtype=bool)
            target = np.asarray(sample["flow_gt"], dtype=np.float32)
            active = valid & common
            regions = {"full_valid": valid, "common_event_active": active}
            event_inputs = {"normal": normal, "zero_events": np.zeros_like(normal),
                            "reverse_bins": normal[..., ::-1].copy()}
            if not np.allclose(normal.sum(-1), event_inputs["reverse_bins"].sum(-1), atol=1e-6, rtol=0):
                raise RuntimeError("reverse-bin input altered event totals")
            if not np.array_equal(support(normal), support(event_inputs["reverse_bins"])):
                raise RuntimeError("reverse-bin input altered event support")
            prediction_by_condition = {}
            input_records = {}
            for condition, event_repr in event_inputs.items():
                event, prev, next_ = padded_tensors(event_repr, gray["prev"], gray["next"], device)
                output = model(event, 352, use_dt4=False, flow_seq=False)
                prediction = crop_native(output_to_flow(output))
                prediction_by_condition[condition] = prediction
                input_records[condition] = {
                    "sha256": array_sha256(event_repr), "changed_from_normal": not np.array_equal(event_repr, normal),
                    "active_pixels": int(support(event_repr).sum()),
                    "representation_sum": float(event_repr.sum(dtype=np.float64)),
                }
            zero_baselines = {
                name: epe(np.zeros_like(target), target, mask) for name, mask in regions.items()
            }
            condition_rows = {}
            for condition, prediction in prediction_by_condition.items():
                magnitude = np.hypot(prediction[0], prediction[1])
                gt_magnitude = np.hypot(target[0], target[1])
                motion_active = active & (gt_magnitude >= 0.5)
                direction = None
                if motion_active.any():
                    dot = prediction[0] * target[0] + prediction[1] * target[1]
                    cosine = dot / np.maximum(magnitude * gt_magnitude, 1e-12)
                    direction = float(cosine[motion_active].mean(dtype=np.float64))
                delta = prediction.astype(np.float64) - prediction_by_condition["normal"].astype(np.float64)
                delta_mag = np.hypot(delta[0], delta[1])
                condition_rows[condition] = {
                    "prediction_sha256": array_sha256(prediction),
                    "epe_px_per_30ms": {name: epe(prediction, target, mask) for name, mask in regions.items()},
                    "zero_flow_epe_px_per_30ms": zero_baselines,
                    "prediction_mean_magnitude_px_per_30ms": {
                        name: float(magnitude[mask].mean(dtype=np.float64)) if mask.any() else None
                        for name, mask in regions.items()
                    },
                    "gt_mean_magnitude_px_per_30ms": {
                        name: float(gt_magnitude[mask].mean(dtype=np.float64)) if mask.any() else None
                        for name, mask in regions.items()
                    },
                    "direction_cosine_mean_active_motion": direction,
                    "prediction_delta_from_normal": {
                        "mean_full_valid": float(delta_mag[valid].mean(dtype=np.float64)) if valid.any() else None,
                        "mean_common_event_active": float(delta_mag[active].mean(dtype=np.float64)) if active.any() else None,
                        "max_abs_component": float(np.max(np.abs(delta))),
                    },
                }
            records.append({
                "rollout_id": row["rollout_id"], "sample_id": str(sample["sample_ids"]),
                "update": update, "flow_unit": "pixels per 30ms endpoint interval",
                "valid_pixels": int(valid.sum()), "common_event_active_pixels": int(active.sum()),
                "active_motion_pixels": int(motion_active.sum()),
                "common_mask_sha256": array_sha256(active.astype(np.uint8)),
                "input": input_records, "conditions": condition_rows,
                "gray_radiance_sha256": gray["radiance_sha256"],
            })
    model.train()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite run output {args.output_root}")

    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo / "benchmark/sim2real/scripts"))
    sys.path.insert(0, str(repo / "benchmark/sim2real/compat"))
    from run_steflow_stage2_pilot import (
        objective_dt1, pure_torch_spatial_correlation, set_seed,
    )
    from steflow_align_corners import patch_ste
    from verify_pinned_source import verify_pinned_source

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("registered_before_execution") is not True or config.get("real_scoring_enabled") is not False:
        raise RuntimeError("configuration must be preregistered and keep real scoring closed")
    if sha256(Path(__file__).resolve()) != config["code"]["training_script_sha256"]:
        raise RuntimeError("training script hash differs from pre-registered config")
    for relative, expected_hash in config["code"]["pinned_helpers"].items():
        helper = repo / relative
        if not helper.is_file() or sha256(helper) != expected_hash:
            raise RuntimeError(f"pinned helper hash mismatch: {relative}")
    run_root = args.run_root.resolve()
    manifest_path = run_root / "flow_bank_retry01/training_bank_manifest.json"
    manifest_sha = sha256(manifest_path)
    if manifest_sha != config["data"]["manifest_sha256"]:
        raise RuntimeError("flow bank manifest hash differs from preregistered config")
    bank = load_bank(manifest_path)
    observed_train = [row["rollout_id"] for row in bank["train"]]
    observed_validation = [row["rollout_id"] for row in bank["validation"]]
    if observed_train != config["data"]["train_rollouts"] or observed_validation != config["data"]["validation_rollouts"]:
        raise RuntimeError("train/validation rollouts differ from the registered complete-family split")
    model_root = args.model_root.resolve()
    source = verify_pinned_source(model_root, "STE-FlowNet", config["model"]["source_commit"],
                                  repo / "benchmark/sim2real/manifests/downstream_source_bundle.json")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered self-supervised diagnostic requires CUDA")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(2)

    sys.path.insert(0, str(model_root))
    corr = importlib.import_module("models.corr")
    corr.spatial_correlation_sample = pure_torch_spatial_correlation
    import models
    import multiscaleloss
    patch_info = patch_ste(models, multiscaleloss, model_root)

    order_seed = int(config["training"]["sampler_seed"])
    cycles = int(config["training"]["updates_per_arm"]) // len(bank["train"])
    if cycles * len(bank["train"]) != int(config["training"]["updates_per_arm"]):
        raise RuntimeError("registered update budget must cover whole four-rollout sampling cycles")
    sample_schedule = []
    for cycle in range(cycles):
        permutation = np.random.default_rng(order_seed + cycle).permutation(len(bank["train"]))
        sample_schedule.extend([int(index) for index in permutation])

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True)
    paired_initial_hashes = set()
    overall = {
        "schema": "stage7_replicad_multiaxis_selfsup_diagnostic_result_v1",
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "bank_manifest_path": str(manifest_path), "bank_manifest_sha256": manifest_sha,
        "model_source_verification": source, "pinned_objective_patch": patch_info,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "train_rollouts": observed_train, "validation_rollouts": observed_validation,
        "sample_order_indices_per_cycle": [sample_schedule[i:i + len(bank["train"])] for i in range(0, len(sample_schedule), len(bank["train"]))],
        "real_data_used": False, "flow_gt_used_in_optimizer": False,
        "checkpoint_selection": False, "arms": {},
    }

    for arm in ("eventgenesis", "v2e"):
        seed = int(config["training"]["seed"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        set_seed(seed)
        model = models.steflow().to(device)
        initial_hash = state_sha256(model)
        paired_initial_hashes.add(initial_hash)
        if initial_hash != config["training"]["paired_initial_state_sha256"]:
            raise RuntimeError(f"fresh init hash differs from pre-registered paired hash: {initial_hash}")
        optimizer = torch.optim.Adam([
            {"params": model.bias_parameters(), "weight_decay": float(config["training"]["weight_decay_biases"])},
            {"params": model.weight_parameters(), "weight_decay": float(config["training"]["weight_decay_weights"])},
        ], lr=float(config["training"]["learning_rate"]), betas=tuple(config["training"]["betas"]))
        arm_dir = output_root / arm
        arm_dir.mkdir()
        metrics_path = arm_dir / "updates.jsonl"
        evaluations = {}
        started = time.monotonic()
        model.train()
        with metrics_path.open("w", buffering=1) as log:
            eval_rows = evaluate_rows(model, bank["validation"], arm, device,
                                      lambda out: out.detach().cpu().numpy()[0], update=0)
            evaluations["0"] = eval_rows
            log.write(json.dumps({"record_type": "validation", "update": 0,
                                  "rows": eval_rows, "real_data_used": False}, sort_keys=True) + "\n")
            for update, row_index in enumerate(sample_schedule, start=1):
                row = bank["train"][row_index]
                sample = row["arms"][arm]
                event, prev, next_ = padded_tensors(sample["events"], row["gray"]["prev"],
                                                    row["gray"]["next"], device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(event, 352, use_dt4=False, flow_seq=False)
                loss, photo, smooth = objective_dt1(
                    outputs, prev, next_, event, multiscaleloss
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite objective at {arm} update {update}")
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError(f"missing/nonfinite gradients at {arm} update {update}")
                grad_norm = float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients)).cpu())
                if not np.isfinite(grad_norm) or grad_norm <= 0:
                    raise RuntimeError(f"nonpositive gradient norm at {arm} update {update}")
                optimizer.step()
                steps = sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})
                if steps != [update]:
                    raise RuntimeError(f"Adam counter mismatch at {arm} update {update}: {steps}")
                record = {
                    "record_type": "optimizer_update", "update": update, "cycle": (update - 1) // 4,
                    "rollout_id": row["rollout_id"], "sample_id": str(sample["sample_ids"]),
                    "event_count": int(sample["event_count"]), "loss": float(loss.detach().cpu()),
                    "photometric_loss": float(photo.detach().cpu()),
                    "smoothness_loss": float(smooth.detach().cpu()),
                    "gradient_l2_norm": grad_norm, "gradient_tensor_count": len(gradients),
                    "optimizer_steps": steps, "gray_radiance_sha256": row["gray"]["radiance_sha256"],
                    "flow_gt_used_in_optimizer": False, "real_data_used": False,
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                }
                log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                if update in (20, 40, 60, 80, 100):
                    torch.cuda.synchronize(device)
                    eval_rows = evaluate_rows(model, bank["validation"], arm, device,
                                              lambda out: out.detach().cpu().numpy()[0], update=update)
                    evaluations[str(update)] = eval_rows
                    log.write(json.dumps({"record_type": "validation", "update": update,
                                          "rows": eval_rows, "real_data_used": False},
                                         sort_keys=True, allow_nan=False) + "\n")
                    log.flush()
                    model.train()

        if state_sha256(model) == initial_hash:
            raise RuntimeError(f"parameters did not change for {arm}")
        final_hash = state_sha256(model)
        checkpoint = arm_dir / "checkpoint_final.pt"
        torch.save({
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "generator": arm, "seed": seed, "updates": 100,
            "initial_state_sha256": initial_hash, "state_sha256": final_hash,
            "config_sha256": sha256(config_path), "data_manifest_sha256": manifest_sha,
            "flow_unit": "pixels per 30ms endpoint interval", "real_data_used": False,
            "flow_gt_used_in_optimizer": False, "event_only_inference": True,
        }, checkpoint)
        checkpoint_hash = sha256(checkpoint)
        restored = models.steflow().to(device)
        restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"], strict=True)
        if state_sha256(restored) != final_hash:
            raise RuntimeError(f"restored model hash mismatch for {arm}")
        restore_rows = evaluate_rows(restored, bank["validation"], arm, device,
                                     lambda out: out.detach().cpu().numpy()[0], update=100)
        final_rows = {row["rollout_id"]: row for row in evaluations["100"]}
        restore_rows_by_id = {row["rollout_id"]: row for row in restore_rows}
        for rollout in observed_validation:
            before = final_rows[rollout]["conditions"]["normal"]["prediction_sha256"]
            after = restore_rows_by_id[rollout]["conditions"]["normal"]["prediction_sha256"]
            if before != after:
                raise RuntimeError(f"checkpoint restore prediction differs for {arm}/{rollout}")
        overall["arms"][arm] = {
            "initial_state_sha256": initial_hash, "final_state_sha256": final_hash,
            "updates": 100, "optimizer_final_steps": [100],
            "checkpoint_path": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_hash, "restore_state_equal": True,
            "restore_validation_prediction_equal": True,
            "validation_by_update": evaluations,
            "first_20_updates_loss_mean": None,
            "last_20_updates_loss_mean": None,
            "wall_seconds": time.monotonic() - started,
        }
        update_rows = [json.loads(line) for line in metrics_path.read_text().splitlines()
                       if json.loads(line).get("record_type") == "optimizer_update"]
        overall["arms"][arm]["first_20_updates_loss_mean"] = float(np.mean([r["loss"] for r in update_rows[:20]]))
        overall["arms"][arm]["last_20_updates_loss_mean"] = float(np.mean([r["loss"] for r in update_rows[-20:]]))
        (arm_dir / "checkpoint.sha256").write_text(f"{checkpoint_hash}  {checkpoint}\n")
        del model, restored, optimizer
        torch.cuda.empty_cache()

    if len(paired_initial_hashes) != 1:
        raise RuntimeError("paired initial states were not identical")
    overall["status"] = "complete"
    overall["paired_initial_state_equal"] = True
    overall["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output_path = output_root / "summary.json"
    tmp = output_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(overall, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, output_path)
    print(json.dumps({"status": overall["status"], "gpu": overall["gpu_name"],
                      "gpu_uuid": overall["gpu_uuid"], "summary": str(output_path),
                      "updates_per_arm": 100, "initial_hash_equal": True}, indent=2))


if __name__ == "__main__":
    main()
