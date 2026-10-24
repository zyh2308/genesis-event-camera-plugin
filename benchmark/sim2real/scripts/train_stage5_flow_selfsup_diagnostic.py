#!/usr/bin/env python3
"""Run a fixed-budget paired STE-FlowNet self-supervision diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "compat"))
from run_steflow_stage2_pilot import (
    compute_epe, model_output_to_flow_hw, objective_dt1, pure_torch_spatial_correlation,
    set_seed, state_sha256,
)
from steflow_align_corners import patch_ste
from train_steflow_stage4 import apply_joint_transform, transform_matrix_and_grid


SIZE = 256
FLOW_UNIT = "px_per_interval"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def read_rollout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def initialize_model(model_root: Path, seed: int, device: torch.device):
    sys.path.insert(0, str(model_root))
    corr = importlib.import_module("models.corr")
    corr.spatial_correlation_sample = pure_torch_spatial_correlation
    import models
    import multiscaleloss

    upstream_hashes = patch_ste(models, multiscaleloss, model_root)
    set_seed(seed)
    model = models.steflow().to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.bias_parameters(), "weight_decay": 0.0},
            {"params": model.weight_parameters(), "weight_decay": 0.0004},
        ],
        lr=0.0004,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.7)
    return model, optimizer, scheduler, multiscaleloss, upstream_hashes


def strict_array_epe(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float | None, int]:
    count = int(np.asarray(mask, dtype=bool).sum())
    if count == 0:
        return None, 0
    value, _ = compute_epe(
        prediction, target, mask,
        prediction_unit=FLOW_UNIT, gt_unit=FLOW_UNIT,
    )
    return value, count


def prepare_paired_data(data_root: Path, generator: str):
    manifest_path = data_root / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "stage5_steflow_diagnostic_bank_v1"
            or manifest.get("status") != "paired_synthetic_diagnostic_data_ready_no_training_result"
            or manifest.get("real_data_used") is not False
            or manifest.get("flow_gt_used_for_self_supervised_optimizer") is not False
            or manifest.get("train_windows_per_generator") != 264
            or manifest.get("validation_windows_per_generator") != 264):
        raise RuntimeError("diagnostic dataset manifest is missing or not the registered 4/4 synthetic pilot")
    rows = [row for row in manifest["entries"] if row["generator"] == generator]
    train, validation = {}, {}
    for row in rows:
        path = Path(row["path"]).resolve()
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"rollout adapter path/hash mismatch: {path}")
        data = read_rollout(path)
        if data["events"].shape != (66, 2, 260, 346, 5):
            raise RuntimeError(f"{path}: expected event representation [66,2,260,346,5]")
        if data["flow_gt"].shape != (66, 2, 260, 346) or data["flow_valid"].shape != (66, 260, 346):
            raise RuntimeError(f"{path}: label/mask shape mismatch")
        if not np.isfinite(data["events"]).all() or not np.isfinite(data["gray_prev"]).all() or not np.isfinite(data["gray_next"]).all():
            raise RuntimeError(f"{path}: event or grayscale input contains non-finite values")
        if not np.isfinite(data["flow_gt"]).all() or not np.isin(data["flow_valid"], (0, 1)).all():
            raise RuntimeError(f"{path}: synthetic labels/masks invalid")
        (train if row["split"] == "train" else validation)[row["rollout_id"]] = data
    if len(train) != 4 or len(validation) != 4:
        raise RuntimeError(f"{generator}: expected four train and four held-out rollouts")
    return manifest_path, manifest, train, validation


def registered_code_fingerprints(repo_root: Path, model_root: Path) -> dict[str, str]:
    paths = {
        "benchmark/sim2real/scripts/train_stage5_flow_selfsup_diagnostic.py": repo_root / "benchmark/sim2real/scripts/train_stage5_flow_selfsup_diagnostic.py",
        "benchmark/sim2real/scripts/register_stage5_flow_selfsup_diagnostic_config.py": repo_root / "benchmark/sim2real/scripts/register_stage5_flow_selfsup_diagnostic_config.py",
        "benchmark/sim2real/scripts/prepare_stage5_flow_diagnostic_bank.py": repo_root / "benchmark/sim2real/scripts/prepare_stage5_flow_diagnostic_bank.py",
        "benchmark/sim2real/scripts/prepare_stage4_steflow_bank.py": repo_root / "benchmark/sim2real/scripts/prepare_stage4_steflow_bank.py",
        "benchmark/sim2real/scripts/prepare_steflow_stage0.py": repo_root / "benchmark/sim2real/scripts/prepare_steflow_stage0.py",
        "benchmark/sim2real/scripts/run_steflow_stage2_pilot.py": repo_root / "benchmark/sim2real/scripts/run_steflow_stage2_pilot.py",
        "benchmark/sim2real/scripts/train_steflow_stage4.py": repo_root / "benchmark/sim2real/scripts/train_steflow_stage4.py",
        "benchmark/sim2real/compat/steflow_align_corners.py": repo_root / "benchmark/sim2real/compat/steflow_align_corners.py",
        "@model/models/STEFlow.py": model_root / "models/STEFlow.py",
        "@model/models/corr.py": model_root / "models/corr.py",
        "@model/multiscaleloss.py": model_root / "multiscaleloss.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"registered training source file(s) missing: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def paired_common_masks(data_root: Path, split: str) -> dict[str, np.ndarray]:
    masks = {}
    arrays = {}
    for generator in ("eventgenesis", "v2e"):
        directory = data_root / generator / split
        arrays[generator] = {p.stem: read_rollout(p) for p in sorted(directory.glob("*.npz"))}
    if set(arrays["eventgenesis"]) != set(arrays["v2e"]):
        raise RuntimeError(f"paired {split} rollout IDs differ")
    for rollout in sorted(arrays["eventgenesis"]):
        left, right = arrays["eventgenesis"][rollout], arrays["v2e"][rollout]
        for key in ("flow_gt", "flow_valid", "sample_ids"):
            if not np.array_equal(left[key], right[key]):
                raise RuntimeError(f"paired labels/sample IDs differ: {split}/{rollout}/{key}")
        active_left = np.any(left["events"] != 0, axis=(1, 4))
        active_right = np.any(right["events"] != 0, axis=(1, 4))
        valid = np.asarray(left["flow_valid"], dtype=bool)
        masks[rollout] = {
            "union": active_left | active_right,
            "valid": valid,
            "active_union_valid": valid & (active_left | active_right),
        }
    return masks


@torch.no_grad()
def evaluate(model, data: dict[str, dict[str, np.ndarray]], common_masks: dict[str, dict[str, np.ndarray]],
             device: torch.device, *, perturbation: str = "normal",
             stratum_by_rollout: dict[str, str] | None = None) -> dict:
    model.eval()
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for rollout in sorted(data):
        sample = data[rollout]
        mask_record = common_masks[rollout]
        for index in range(66):
            event = torch.from_numpy(sample["events"][index:index + 1].astype(np.float32))
            if perturbation == "zero":
                event.zero_()
            elif perturbation == "reverse_time_bins":
                event = event.flip(-1)
            elif perturbation != "normal":
                raise ValueError(f"unknown event perturbation {perturbation}")
            event = F.pad(event, (0, 0, 3, 3, 46, 46)).to(device)
            output = model(event, 352, use_dt4=False, flow_seq=False)
            prediction = model_output_to_flow_hw(output)[:, 46:306, 3:349]
            target = np.asarray(sample["flow_gt"][index], dtype=np.float32)
            valid = mask_record["valid"][index]
            active = mask_record["active_union_valid"][index]
            full_epe, full_count = strict_array_epe(prediction, target, valid)
            active_epe, active_count = strict_array_epe(prediction, target, active)
            zero_full, _ = strict_array_epe(np.zeros_like(target), target, valid)
            zero_active, _ = strict_array_epe(np.zeros_like(target), target, active)
            gt_magnitude = np.linalg.norm(target, axis=0)
            pred_magnitude = np.linalg.norm(prediction, axis=0)
            direction_pixels = active & (gt_magnitude >= 0.1)
            if direction_pixels.any():
                dot = (prediction[0] * target[0] + prediction[1] * target[1])[direction_pixels]
                cosine = dot / np.maximum(pred_magnitude[direction_pixels] * gt_magnitude[direction_pixels], 1e-8)
                direction_cosine = float(np.mean(cosine))
            else:
                direction_cosine = None
            rows.append({
                "rollout_id": rollout,
                "stratum": stratum_by_rollout.get(rollout) if stratum_by_rollout else None,
                "sample_id": str(sample["sample_ids"][index]),
                "perturbation": perturbation,
                "full_valid_epe_px_per_30ms": full_epe,
                "full_valid_pixels": full_count,
                "active_union_valid_epe_px_per_30ms": active_epe,
                "active_union_valid_pixels": active_count,
                "zero_flow_full_valid_epe_px_per_30ms": zero_full,
                "zero_flow_active_union_valid_epe_px_per_30ms": zero_active,
                "gt_motion_magnitude_mean_full_valid": float(gt_magnitude[valid].mean()) if valid.any() else None,
                "prediction_magnitude_mean_full_valid": float(pred_magnitude[valid].mean()) if valid.any() else None,
                "gt_motion_magnitude_mean_active_union_valid": float(gt_magnitude[active].mean()) if active.any() else None,
                "prediction_magnitude_mean_active_union_valid": float(pred_magnitude[active].mean()) if active.any() else None,
                "direction_cosine_active_union_gt_ge_0_1px": direction_cosine,
            })
            predictions[f"{rollout}/dt1_{index:03d}"] = prediction
    model.train()
    return {"windows": rows, "predictions": predictions}


def summarize_evaluation(evaluation: dict) -> dict:
    rows = evaluation["windows"]
    summary = {}
    for metric in (
        "full_valid_epe_px_per_30ms",
        "active_union_valid_epe_px_per_30ms",
        "zero_flow_full_valid_epe_px_per_30ms",
        "zero_flow_active_union_valid_epe_px_per_30ms",
        "gt_motion_magnitude_mean_full_valid",
        "prediction_magnitude_mean_full_valid",
        "gt_motion_magnitude_mean_active_union_valid",
        "prediction_magnitude_mean_active_union_valid",
        "direction_cosine_active_union_gt_ge_0_1px",
    ):
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = float(np.mean(values)) if values else None
        summary[f"{metric}_window_count"] = len(values)
    summary["validation_windows"] = len(rows)
    summary["active_union_coverage_of_valid"] = float(
        np.mean([row["active_union_valid_pixels"] / max(row["full_valid_pixels"], 1) for row in rows])
    )
    return summary


def summarize_evaluation_by_stratum(evaluation: dict) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in evaluation["windows"]:
        if row.get("stratum") is None:
            continue
        groups.setdefault(str(row["stratum"]), []).append(row)
    metrics = (
        "full_valid_epe_px_per_30ms",
        "active_union_valid_epe_px_per_30ms",
        "zero_flow_full_valid_epe_px_per_30ms",
        "zero_flow_active_union_valid_epe_px_per_30ms",
        "gt_motion_magnitude_mean_active_union_valid",
        "prediction_magnitude_mean_active_union_valid",
        "direction_cosine_active_union_gt_ge_0_1px",
    )
    return {
        stratum: {
            "windows": len(rows),
            **{
                metric: float(np.mean(values)) if values else None
                for metric in metrics
                for values in [[row[metric] for row in rows if row[metric] is not None]]
            },
        }
        for stratum, rows in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=100)
    args = parser.parse_args()
    run = args.run_root.resolve() / args.generator
    if run.exists():
        raise FileExistsError(f"refusing to overwrite run: {run}")
    if args.max_updates != 100:
        raise RuntimeError("registered diagnostic budget is fixed at 100 optimizer updates per arm")

    config = json.loads(args.config.read_text())
    if (config.get("schema") != "stage5_flow_selfsup_diagnostic_v1"
            or config.get("real_scoring_enabled") is not False
            or config.get("checkpoint_selection_allowed") is not False
            or config.get("training", {}).get("optimizer_updates_per_arm") != args.max_updates):
        raise RuntimeError("self-supervised diagnostic config differs from the frozen 100-update pilot")
    repo_root = HERE.parents[2]
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    if source_commit != config.get("code", {}).get("source_commit"):
        raise RuntimeError(f"source commit mismatch: current={source_commit}, registered={config.get('code', {}).get('source_commit')}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
    if dirty:
        raise RuntimeError("training checkout is dirty; run from the registered clean immutable checkout")
    model_root = args.model_root.resolve()
    model_source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=model_root, text=True).strip()
    model_source_dirty_paths = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=model_root, text=True
    ).splitlines()
    if (model_source_commit != config.get("code", {}).get("model_source_commit")
            or model_source_dirty_paths != config.get("code", {}).get("model_source_dirty_paths")):
        raise RuntimeError("STE-FlowNet source checkout or dirty-file state differs from the registered config")
    code_fingerprints = registered_code_fingerprints(repo_root, model_root)
    if code_fingerprints != config.get("code", {}).get("file_sha256"):
        raise RuntimeError("training source file hashes differ from the registered immutable config")
    manifest_path, data_manifest, train, validation = prepare_paired_data(args.data_root.resolve(), args.generator)
    if (manifest_path.stat().st_size == 0
            or data_manifest["shared_physical_bank"]["sha256"] != config["data"]["expected_bank_sha256"]
            or sha256(manifest_path) != config["data"]["expected_adapter_manifest_sha256"]):
        raise RuntimeError("diagnostic code/data fingerprints are not those registered in config")
    common_masks = paired_common_masks(args.data_root.resolve(), "validation")
    stratum_by_rollout = {
        row["rollout_id"]: row["stratum"]
        for row in data_manifest["entries"]
        if row["generator"] == args.generator and row["split"] == "validation"
    }
    train_samples = [(rollout, index) for rollout in sorted(train) for index in range(66)]
    if len(train_samples) != 264:
        raise RuntimeError("paired training sampler must contain all 264 windows from four complete train rollouts")

    sys.path.insert(0, str(args.model_root.resolve()))
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; diagnostic refuses CPU training")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, optimizer, scheduler, loss_module, upstream_hashes = initialize_model(
        model_root, args.seed, device
    )
    initial_hash = state_sha256(model)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    paired_initial_hash = config["training"]["expected_paired_initial_state_sha256"]
    if initial_hash != paired_initial_hash:
        raise RuntimeError(f"fresh initialization differs from paired registered state: {initial_hash}")
    model.train()
    run.mkdir(parents=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    initial_eval = evaluate(model, validation, common_masks, device, stratum_by_rollout=stratum_by_rollout)
    records = [{
        "event": "initial_validation",
        "optimizer_updates": 0,
        "summary": summarize_evaluation(initial_eval),
        "real_data_used": False,
        "flow_gt_used_in_optimizer": False,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }]
    log_path = run / "training_metrics.jsonl"
    log_path.write_text(json.dumps(records[0], sort_keys=True) + "\n")
    log = log_path.open("a", buffering=1)
    global_update = 0
    start_time = time.monotonic()
    updates_per_epoch = len(train_samples) // 8
    epoch = 0
    for epoch in range((args.max_updates + updates_per_epoch - 1) // updates_per_epoch):
        permutation = np.random.default_rng(args.seed + 1000003 * epoch).permutation(len(train_samples))
        batches = [permutation[index:index + 8] for index in range(0, len(permutation), 8)]
        for batch_index, batch in enumerate(batches):
            if len(batch) != 8:
                raise RuntimeError("registered synthetic diagnostic expects drop_last with batch size eight")
            optimizer.zero_grad(set_to_none=True)
            event_batch, gray_batch, sample_ids, event_counts, augmentation_seeds = [], [], [], [], []
            for micro, sample_index in enumerate(batch):
                rollout, within = train_samples[int(sample_index)]
                sample = train[rollout]
                aug_seed = [args.seed, epoch, batch_index, micro]
                augmentation_seeds.append(aug_seed)
                aug_rng = np.random.default_rng(np.random.SeedSequence(aug_seed))
                transform_sample = {
                    key: sample[key][within:within + 1]
                    for key in ("events", "gray_prev", "gray_next", "flow_gt", "flow_valid")
                }
                event, gray, _, _ = apply_joint_transform(transform_sample, aug_rng, device)
                event_batch.append(event)
                gray_batch.append(gray)
                sample_ids.append(str(sample["sample_ids"][within]))
                event_counts.append(int(sample["event_counts"][within]))
            event_tensor = torch.cat(event_batch, dim=0)
            gray_tensor = torch.cat(gray_batch, dim=0)
            if tuple(event_tensor.shape) != (8, 2, SIZE, SIZE, 5) or tuple(gray_tensor.shape) != (8, 2, SIZE, SIZE):
                raise RuntimeError("augmented event/grayscale tensors differ from the pinned STE dt1 input contract")
            output = model(event_tensor, SIZE, use_dt4=False, flow_seq=False)
            loss, photo, smooth = objective_dt1(
                output, gray_tensor[:, 0:1], gray_tensor[:, 1:2], event_tensor, loss_module
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite self-supervised loss at update {global_update + 1}")
            loss.backward()
            grad_sq = 0.0
            grad_tensors = 0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"non-finite gradient at update {global_update + 1}")
                    grad_sq += float(parameter.grad.detach().float().square().sum().cpu())
                    grad_tensors += 1
            grad_norm = float(grad_sq ** 0.5)
            values = [float(loss.detach().cpu()), float(photo.detach().cpu()),
                      float(smooth.detach().cpu()), grad_norm]
            if grad_norm <= 0 or grad_tensors == 0 or not np.isfinite(values).all():
                raise RuntimeError(f"invalid loss or gradient at update {global_update + 1}")
            optimizer.step()
            global_update += 1
            log_record = {
                "event": "optimizer_update",
                "optimizer_update": global_update,
                "epoch": epoch,
                "batch_in_epoch": batch_index,
                "sample_ids": sample_ids,
                "augmentation_seed_sequences": augmentation_seeds,
                "event_counts_per_sample": event_counts,
                "loss": values[0],
                "photometric_loss": values[1],
                "smoothness_loss": values[2],
                "gradient_l2_norm": grad_norm,
                "finite_gradient_tensors": grad_tensors,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - start_time,
                "flow_gt_used_in_optimizer": False,
                "real_data_used": False,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            log.write(json.dumps(log_record, sort_keys=True) + "\n")
            if global_update % 20 == 0 or global_update == args.max_updates:
                print(json.dumps(log_record, sort_keys=True), flush=True)
            if global_update % updates_per_epoch == 0:
                scheduler.step()
            if global_update >= args.max_updates:
                break
        if global_update >= args.max_updates:
            break
    log.close()

    final_eval = evaluate(model, validation, common_masks, device)
    final_summary = summarize_evaluation(final_eval)
    if global_update != 100:
        raise RuntimeError(f"expected exactly 100 optimizer updates, got {global_update}")
    optimizer_steps = sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})
    if optimizer_steps != [100]:
        raise RuntimeError(f"Adam state step counters differ from the fixed budget: {optimizer_steps}")
    final_hash = state_sha256(model)
    delta_sq = sum(float((new.detach().float() - old.detach().float()).square().sum().cpu())
                   for new, old in zip(model.parameters(), initial_parameters))
    checkpoint_path = run / "final_100_updates.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "seed": args.seed,
        "generator": args.generator,
        "optimizer_update": global_update,
        "initial_state_sha256": initial_hash,
        "current_state_sha256": final_hash,
        "optimizer_state_steps": optimizer_steps,
        "config_sha256": sha256(args.config),
        "data_manifest_sha256": sha256(manifest_path),
        "shared_bank_sha256": data_manifest["shared_physical_bank"]["sha256"],
        "real_data_used": False,
        "flow_gt_used_in_optimizer": False,
        "resume_allowed": False,
        "checkpoint_selection_allowed": False,
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256(checkpoint_path)

    # Verify serialized weights through a fresh model instance and fixed event input.
    model.eval()
    first_rollout = sorted(validation)[0]
    probe_event = torch.from_numpy(validation[first_rollout]["events"][0:1].astype(np.float32))
    probe_event = F.pad(probe_event, (0, 0, 3, 3, 46, 46)).to(device)
    with torch.inference_mode():
        before_restore = model(probe_event, 352, use_dt4=False, flow_seq=False).detach().clone()
    restored, _, _, _, _ = initialize_model(model_root, args.seed + 9127, device)
    restored_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored.load_state_dict(restored_state["state_dict"], strict=True)
    restored.eval()
    with torch.inference_mode():
        after_restore = restored(probe_event, 352, use_dt4=False, flow_seq=False)
    restore_max_abs = float((before_restore - after_restore).abs().max().cpu())
    if restore_max_abs > 1e-5 or state_sha256(restored) != final_hash:
        raise RuntimeError(f"final checkpoint restore mismatch: max_abs={restore_max_abs}")

    perturbations = {}
    perturbation_windows = {}
    normal_result = evaluate(model, validation, common_masks, device, perturbation="normal",
                             stratum_by_rollout=stratum_by_rollout)
    normal_by_id = {row["sample_id"]: row for row in normal_result["windows"]}
    for name in ("zero", "reverse_time_bins"):
        result = evaluate(model, validation, common_masks, device, perturbation=name,
                          stratum_by_rollout=stratum_by_rollout)
        deltas, epe_values, zero_values = [], [], []
        per_window = []
        for row in result["windows"]:
            baseline = normal_by_id[row["sample_id"]]
            base_flow = normal_result["predictions"][row["sample_id"]]
            perturb_flow = result["predictions"][row["sample_id"]]
            flow_delta = float(np.mean(np.abs(perturb_flow - base_flow)))
            deltas.append(flow_delta)
            epe_values.append(row["active_union_valid_epe_px_per_30ms"])
            zero_values.append(row["zero_flow_active_union_valid_epe_px_per_30ms"])
            per_window.append({
                "rollout_id": row["rollout_id"],
                "stratum": row["stratum"],
                "sample_id": row["sample_id"],
                "normal_full_valid_epe_px_per_30ms": baseline["full_valid_epe_px_per_30ms"],
                "perturbed_full_valid_epe_px_per_30ms": row["full_valid_epe_px_per_30ms"],
                "normal_active_union_valid_epe_px_per_30ms": baseline["active_union_valid_epe_px_per_30ms"],
                "perturbed_active_union_valid_epe_px_per_30ms": row["active_union_valid_epe_px_per_30ms"],
                "zero_flow_active_union_valid_epe_px_per_30ms": row["zero_flow_active_union_valid_epe_px_per_30ms"],
                "active_union_valid_pixels": row["active_union_valid_pixels"],
                "mean_absolute_flow_change_vs_normal_px_per_30ms": flow_delta,
            })
        if name == "reverse_time_bins":
            normal_mass = sum(int(validation[rollout]["events"].sum(dtype=np.uint64)) for rollout in sorted(validation))
            reversed_mass = sum(int(validation[rollout]["events"][..., ::-1].sum(dtype=np.uint64))
                                for rollout in sorted(validation))
            input_difference = float(np.mean([
                np.mean(np.abs(validation[rollout]["events"] - validation[rollout]["events"][..., ::-1]))
                for rollout in sorted(validation)
            ]))
            active_before = float(np.mean([
                np.any(validation[rollout]["events"] != 0, axis=(1, 4)).mean()
                for rollout in sorted(validation)
            ]))
            active_after = float(np.mean([
                np.any(validation[rollout]["events"][..., ::-1] != 0, axis=(1, 4)).mean()
                for rollout in sorted(validation)
            ]))
            if normal_mass != reversed_mass or active_before != active_after:
                raise RuntimeError("time-bin reversal must preserve total event representation mass and spatial support")
        else:
            normal_mass = sum(int(validation[rollout]["events"].sum(dtype=np.uint64)) for rollout in sorted(validation))
            zero_mass = 0
            input_difference = float(np.mean([
                np.mean(np.abs(validation[rollout]["events"])) for rollout in sorted(validation)
            ]))
            active_before = float(np.mean([
                np.any(validation[rollout]["events"] != 0, axis=(1, 4)).mean()
                for rollout in sorted(validation)
            ]))
            active_after = 0.0
            if normal_mass == 0 or input_difference <= 0.0:
                raise RuntimeError("all-zero input perturbation did not change a nonempty normal validation tensor")
        numeric_epe = [float(value) for value in epe_values if value is not None]
        numeric_zero = [float(value) for value in zero_values if value is not None]
        perturbations[name] = {
            "mean_absolute_flow_change_vs_normal_px_per_30ms": float(np.mean(deltas)),
            "mean_active_union_epe_px_per_30ms": float(np.mean(numeric_epe)) if numeric_epe else None,
            "mean_zero_flow_active_union_epe_px_per_30ms": float(np.mean(numeric_zero)) if numeric_zero else None,
            "event_input_mean_absolute_change": input_difference,
            "normal_event_active_fraction": active_before,
            "perturbed_event_active_fraction": active_after,
            "normal_event_representation_mass": normal_mass,
            "perturbed_event_representation_mass": reversed_mass if name == "reverse_time_bins" else zero_mass,
            "time_reversal_preserved_mass_and_spatial_support": name == "reverse_time_bins",
            "same_frozen_normal_event_union_valid_mask": True,
            "window_count": len(epe_values),
            "time_order_perturbation": "reverse 5-bin temporal axis within each polarity; event count and per-pixel support are preserved" if name == "reverse_time_bins" else "replace the model input with zeros; evaluation mask remains the normal-input shared union",
        }
        perturbation_windows[name] = per_window

    finished_utc = datetime.now(timezone.utc).isoformat()
    result = {
        "schema": "stage5_flow_selfsup_diagnostic_result_v1",
        "status": "fixed_budget_synthetic_self_supervision_completed",
        "created_utc": started_utc,
        "finished_utc": finished_utc,
        "generator": args.generator,
        "seed": args.seed,
        "optimizer_updates": global_update,
        "effective_batch": 8,
        "shared_bank_sha256": data_manifest["shared_physical_bank"]["sha256"],
        "adapter_manifest_sha256": sha256(manifest_path),
        "config_sha256": sha256(args.config),
        "source_commit": source_commit,
        "source_file_sha256": code_fingerprints,
        "model_source_commit": model_source_commit,
        "model_source_dirty_paths": model_source_dirty_paths,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "parameter_delta_l2": float(delta_sq ** 0.5),
        "optimizer_state_steps": optimizer_steps,
        "final_checkpoint": {"path": str(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size,
                             "sha256": checkpoint_sha, "restore_max_absolute_output_difference": restore_max_abs,
                             "restore_state_sha256_matches": state_sha256(restored) == final_hash},
        "trainable_upstream_compatibility_hashes": upstream_hashes,
        "training_objective": "pinned STE-FlowNet official multiscale photometric loss plus 10x second-order smoothness; unchanged Stage-4 implementation",
        "flow_gt_used_in_optimizer": False,
        "real_data_used": False,
        "real_scoring_enabled": False,
        "checkpoint_selection_allowed": False,
        "validation_selection_performed": False,
        "initial_validation": summarize_evaluation(initial_eval),
        "final_validation": final_summary,
        "final_validation_per_window": normal_result["windows"],
        "final_validation_by_stratum": summarize_evaluation_by_stratum(normal_result),
        "input_perturbation_diagnostics": perturbations,
        "input_perturbation_per_window": perturbation_windows,
        "diagnostic_scope": "four independent synthetic validation rollouts; active-union masks frozen from normal paired inputs before perturbation; no real scores",
    }
    (run / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "generator": args.generator,
                      "optimizer_updates": global_update, "initial": result["initial_validation"],
                      "final": final_summary, "checkpoint_sha256": checkpoint_sha,
                      "restore_max_abs": restore_max_abs}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
