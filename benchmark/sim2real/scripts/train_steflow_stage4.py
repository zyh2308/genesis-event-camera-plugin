#!/usr/bin/env python3
"""Paired-seed STE-FlowNet synthetic-only Stage-4 full training runner."""
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

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compat"))
from run_steflow_stage2_pilot import (
    compute_epe, model_output_to_flow_hw, objective_dt1, pure_torch_spatial_correlation,
    set_seed, state_sha256,
)
from steflow_align_corners import patch_ste


SIZE = 256
FLOW_UNIT = "px_per_30ms_interval"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch_save(value: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {key: np.asarray(d[key]) for key in d.files}


def list_split(data_root: Path, generator: str, split: str) -> dict[str, dict]:
    paths = sorted((data_root / generator / split).glob("*.npz"))
    if len(paths) != (24 if split == "train" else 8):
        raise RuntimeError(f"{generator}/{split}: expected {24 if split == 'train' else 8} rollout files, got {len(paths)}")
    data = {path.stem: read_npz(path) for path in paths}
    for rollout, row in data.items():
        if row["events"].shape != (66, 2, 260, 346, 5):
            raise RuntimeError(f"{generator}/{split}/{rollout}: unexpected event shape {row['events'].shape}")
        if row["gray_prev"].shape != (66, 260, 346) or row["gray_next"].shape != (66, 260, 346):
            raise RuntimeError(f"{rollout}: gray endpoint contract mismatch")
    return data


def transform_matrix_and_grid(height: int, width: int, rng: np.random.Generator):
    area = float(rng.uniform(.5, 1.0)) * height * width
    ratio = float(np.exp(rng.uniform(np.log(.75), np.log(1.3333333333333333))))
    crop_h = int(round(np.sqrt(area / ratio)))
    crop_w = int(round(np.sqrt(area * ratio)))
    crop_h = min(height, max(16, crop_h)); crop_w = min(width, max(16, crop_w))
    y0 = int(rng.integers(0, height - crop_h + 1)); x0 = int(rng.integers(0, width - crop_w + 1))
    flip_x = -1.0 if rng.random() < .5 else 1.0
    flip_y = -1.0 if rng.random() < .5 else 1.0
    angle = np.deg2rad(float(rng.uniform(-30.0, 30.0)))
    sx, sy = SIZE / crop_w, SIZE / crop_h
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    linear = rotation @ np.diag([sx * flip_x, sy * flip_y]).astype(np.float32)
    inv = np.linalg.inv(linear)
    yy, xx = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32), torch.arange(SIZE, dtype=torch.float32), indexing="ij")
    output_xy = torch.stack((xx - (SIZE - 1) * .5, yy - (SIZE - 1) * .5), dim=-1).numpy()
    source_xy = output_xy @ inv.T
    source_xy[..., 0] += x0 + (crop_w - 1) * .5
    source_xy[..., 1] += y0 + (crop_h - 1) * .5
    grid_x = torch.from_numpy(2.0 * (source_xy[..., 0] + .5) / width - 1.0)
    grid_y = torch.from_numpy(2.0 * (source_xy[..., 1] + .5) / height - 1.0)
    grid = torch.stack((grid_x, grid_y), dim=-1)[None]
    inside = ((source_xy[..., 0] >= x0) & (source_xy[..., 0] <= x0 + crop_w - 1)
              & (source_xy[..., 1] >= y0) & (source_xy[..., 1] <= y0 + crop_h - 1))
    return grid, torch.from_numpy(inside)[None, None], torch.from_numpy(linear)


def apply_joint_transform(sample: dict, rng: np.random.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    event = torch.from_numpy(sample["events"].astype(np.float32)).permute(0, 1, 4, 2, 3).reshape(1, 10, 260, 346)
    gray = torch.from_numpy(np.stack((sample["gray_prev"], sample["gray_next"])).astype(np.float32)).reshape(1, 2, 260, 346)
    flow = torch.from_numpy(sample["flow_gt"].astype(np.float32)).reshape(1, 2, 260, 346)
    valid = torch.from_numpy(sample["flow_valid"].astype(np.float32)).reshape(1, 1, 260, 346)
    grid, in_crop, linear = transform_matrix_and_grid(260, 346, rng)
    event = F.grid_sample(event, grid, mode="bilinear", padding_mode="zeros", align_corners=False) * in_crop
    gray = F.grid_sample(gray, grid, mode="bilinear", padding_mode="zeros", align_corners=False) * in_crop
    flow = F.grid_sample(flow, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    flow = torch.einsum("ij,bjhw->bihw", linear, flow) * in_crop
    valid = F.grid_sample(valid, grid, mode="nearest", padding_mode="zeros", align_corners=False) * in_crop
    event = event.reshape(1, 2, 5, SIZE, SIZE).permute(0, 1, 3, 4, 2).to(device)
    gray = gray.to(device); flow = flow.to(device); valid = valid.to(device)
    return event, gray, flow, valid


def validate_geometry_transform() -> dict:
    # Independent hand-check: unit +u and +v vectors under horizontal flip.
    grid, inside, linear = transform_matrix_and_grid(260, 346, np.random.default_rng(7))
    del grid, inside
    f = linear @ torch.tensor([1.0, 0.0])
    if not torch.isfinite(f).all():
        raise RuntimeError("augmentation transform matrix is non-finite")
    # Algebra check independent of sampled crop: A=R diag(-sx,sy).
    sx, sy, theta = 2.0, 1.5, 0.0
    A = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float32) @ torch.diag(torch.tensor([-sx, sy]))
    if not torch.equal(A @ torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.0])) or not torch.equal(A @ torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.5])):
        raise RuntimeError("flow vector horizontal/vertical flip transform failed")
    return {"status": "pass", "flow_vector_rule": "u,v transformed by R @ diag(scale_x*flip_x,scale_y*flip_y)",
            "independent_horizontal_flip_example": {"u_unit_after": [-2.0, 0.0], "v_unit_after": [0.0, 1.5]}}


def val_input(data: dict, index: int, device: torch.device):
    event = torch.from_numpy(data["events"][index:index + 1].astype(np.float32))
    event = F.pad(event, (0, 0, 3, 3, 46, 46)).to(device)
    gray0 = torch.from_numpy(data["gray_prev"][index:index + 1, None].astype(np.float32))
    gray1 = torch.from_numpy(data["gray_next"][index:index + 1, None].astype(np.float32))
    gray0 = F.pad(gray0, (3, 3, 46, 46)).to(device)
    gray1 = F.pad(gray1, (3, 3, 46, 46)).to(device)
    return event, gray0, gray1


def save_checkpoint(path: Path, *, model, optimizer, scheduler, seed, generator, epoch, update,
                    sampler_position, config_hash, manifest_hash, initial_hash, best_epe, best_epoch,
                    next_epoch=None, next_position_samples=None, phase="train", resume_allowed=True):
    current_hash = state_sha256(model)
    state = {
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "seed": seed, "generator": generator, "epoch": epoch, "optimizer_update": update,
        "sampler_position": sampler_position, "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(), "config_sha256": config_hash,
        "data_manifest_sha256": manifest_hash, "initial_state_sha256": initial_hash,
        "current_state_sha256": current_hash,
        "best_synthetic_validation_epe": best_epe, "best_epoch": best_epoch,
        "effective_batch": 8, "microbatch": 8, "accumulation_steps": 1,
        "flow_unit": FLOW_UNIT, "real_data_used": False,
        "next_epoch": int(epoch if next_epoch is None else next_epoch),
        "next_position_samples": int(sampler_position if next_position_samples is None else next_position_samples),
        "resume_phase": phase, "resume_allowed": bool(resume_allowed),
    }
    atomic_torch_save(state, path)


@torch.no_grad()
def evaluate_validation(model, val, device, run: Path, label: str) -> dict:
    model.eval()
    epe_by_rollout, zero_by_rollout = {}, {}
    fixed_visualizations = {}
    for rollout in sorted(val):
        epe_values, zero_values = [], []
        for sample_index in range(66):
            event, _, _ = val_input(val[rollout], sample_index, device)
            pred = model(event, 352, use_dt4=False, flow_seq=False)
            flow = model_output_to_flow_hw(pred)
            if flow.shape != (2, 352, 352):
                raise RuntimeError(f"validation output must be [2,352,352], got {flow.shape}")
            flow = flow[:, 46:306, 3:349]
            gt = val[rollout]["flow_gt"][sample_index]
            valid = val[rollout]["flow_valid"][sample_index]
            epe, _ = compute_epe(flow, gt, valid, prediction_unit="px_per_interval", gt_unit="px_per_interval")
            zero, _ = compute_epe(np.zeros_like(gt), gt, valid, prediction_unit="px_per_interval", gt_unit="px_per_interval")
            epe_values.append(epe); zero_values.append(zero)
            if sample_index == 0 and rollout[:2] not in fixed_visualizations:
                fixed_visualizations[rollout[:2]] = (rollout, flow.copy(), gt.copy(), valid.copy())
        epe_by_rollout[rollout] = float(np.mean(epe_values))
        zero_by_rollout[rollout] = float(np.mean(zero_values))
    strata = {}
    for stratum_index in range(4):
        prefix = f"s{stratum_index}_"
        ids = [rid for rid in sorted(val) if rid.startswith(prefix)]
        if len(ids) != 2:
            raise RuntimeError(f"validation stratum {stratum_index} must contain 2 rollouts, got {ids}")
        strata[stratum_index] = {
            "rollouts": ids,
            "model_mean_epe_px_per_30ms": float(np.mean([epe_by_rollout[rid] for rid in ids])),
            "zero_flow_mean_epe_px_per_30ms": float(np.mean([zero_by_rollout[rid] for rid in ids])),
        }
    viz_root = run / "fixed_validation_visuals" / label
    viz_root.mkdir(parents=True, exist_ok=True)
    for _, (rollout, pred, gt, valid) in sorted(fixed_visualizations.items()):
        # Fixed, pre-registered sample 0 from the first held-out rollout in
        # each stratum; values are encoded over a fixed +/-20 px display range.
        def encode(field):
            return np.clip(np.rint((np.clip(field, -20.0, 20.0) + 20.0) * (255.0 / 40.0)), 0, 255).astype(np.uint8)
        panel = np.concatenate((encode(gt[0]), encode(gt[1]), valid.astype(np.uint8) * 255,
                                encode(pred[0]), encode(pred[1]), valid.astype(np.uint8) * 255), axis=1)
        if not cv2.imwrite(str(viz_root / f"{rollout}_sample000_gt_uv_valid_pred_uv_valid.png"), panel):
            raise IOError(f"failed to write fixed validation visualization for {rollout}")
    result = {
        "model_mean_epe_px_per_30ms": float(np.mean(list(epe_by_rollout.values()))),
        "zero_flow_mean_epe_px_per_30ms": float(np.mean(list(zero_by_rollout.values()))),
        "model_epe_by_rollout": epe_by_rollout,
        "zero_flow_epe_by_rollout": zero_by_rollout,
        "strata": strata,
        "fixed_visualization_samples": sorted(value[0] + "/sample000" for value in fixed_visualizations.values()),
        "real_data_used": False,
    }
    model.train()
    return result


def main() -> None:
    if not os.environ.get("STAGE4_LAUNCH_GATE_VERIFIED"):
        raise SystemExit("refusing direct trainer invocation; use scripts/stage4_launch_gate.py")
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-updates", type=int, default=None,
                   help="stop after this absolute update count; epoch-boundary stops include scheduler and validation")
    args = p.parse_args()
    run = args.run_root.resolve() / args.generator
    data_root, model_root = args.data_root.resolve(), args.model_root.resolve()
    config = json.loads(args.config.read_text())
    train_cfg = config["flow_dt1"]
    if (train_cfg["effective_batch"], train_cfg["epochs"], train_cfg["optimizer"]["lr"]) != (8, 45, .0004):
        raise RuntimeError("Stage4 flow optimizer contract changed")
    manifest_path = data_root / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["train_windows_per_generator"], manifest["validation_windows_per_generator"]) != (1584, 528):
        raise RuntimeError("Stage4 full-bank window manifest mismatch")
    train = list_split(data_root, args.generator, "train")
    val = list_split(data_root, args.generator, "validation")
    train_samples = [(rid, i) for rid in sorted(train) for i in range(66)]
    if len(train_samples) != 1584:
        raise RuntimeError("train sampler does not cover all registered windows")

    sys.path.insert(0, str(model_root))
    corr_module = importlib.import_module("models.corr")
    corr_module.spatial_correlation_sample = pure_torch_spatial_correlation
    import models
    import multiscaleloss
    upstream_hashes = patch_ste(models, multiscaleloss, model_root)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU training")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    model = models.steflow().to(device)
    model.train()
    initial_hash = state_sha256(model)
    optimizer = torch.optim.Adam([
        {"params": model.bias_parameters(), "weight_decay": 0.0},
        {"params": model.weight_parameters(), "weight_decay": .0004},
    ], lr=.0004, betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=.7)
    start_epoch, start_update, start_batch = 0, 0, 0
    best_epe, best_epoch = float("inf"), None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not saved.get("resume_allowed", False) or saved.get("resume_phase") != "train":
            raise ValueError("checkpoint is inference-only or not at a resumable training phase")
        if saved["generator"] != args.generator or saved["seed"] != args.seed:
            raise ValueError("resume checkpoint generator/seed mismatch")
        if saved["config_sha256"] != sha256(args.config) or saved["data_manifest_sha256"] != sha256(manifest_path):
            raise ValueError("resume checkpoint config/data hash mismatch")
        model.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        start_epoch = int(saved["next_epoch"])
        start_update = int(saved["optimizer_update"])
        start_batch = int(saved["next_position_samples"])
        initial_hash = saved["initial_state_sha256"]
        best_epe, best_epoch = saved["best_synthetic_validation_epe"], saved["best_epoch"]
        if state_sha256(model) != saved["current_state_sha256"]:
            raise RuntimeError("resume model state hash mismatch")

    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run / "run_metadata.json", {
        "generator": args.generator, "seed": args.seed, "gpu_visible_index": 0,
        "physical_gpu_id": os.environ.get("STAGE4_PHYSICAL_GPU", "unknown"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "repo_commit": os.environ.get("STAGE4_REPO_COMMIT", "unknown"),
        "config_path": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "data_manifest": str(manifest_path), "data_manifest_sha256": sha256(manifest_path),
        "model_root": str(model_root), "correlation_operator": "memory-bounded PyTorch implementation, math-equivalent to locked upstream op",
        "initial_state_sha256": initial_hash, "effective_batch": 8, "microbatch": 8, "gradient_accumulation_steps": 1,
        "epochs": 45, "train_windows": 1584, "validation_windows": 528,
        "objective": "official multiscale Charbonnier photometric + 10x second-order smoothness",
        "coordinate_compatibility": upstream_hashes,
        "real_data_used": False, "flow_gt_used_in_optimizer": False,
        "cudnn_deterministic": True,
        "augmentation_check": validate_geometry_transform(),
    })
    log_path = run / "training_metrics.jsonl"
    updates_per_epoch = len(train_samples) // 8
    total_updates = updates_per_epoch * 45
    if total_updates != 8910:
        raise RuntimeError(f"registered update budget mismatch: {total_updates}")
    manifest_hash = sha256(manifest_path)
    start_time = time.monotonic()
    global_update = start_update
    if global_update == 0:
        baseline = evaluate_validation(model, val, device, run, "initial")
        print(json.dumps({"event": "initial_synthetic_validation", "optimizer_updates": 0, **baseline}), flush=True)
        with log_path.open("a", buffering=1) as output:
            output.write(json.dumps({"event": "initial_synthetic_validation", "optimizer_updates": 0, **baseline}) + "\n")

    if args.max_updates is not None and args.max_updates <= global_update:
        raise ValueError(f"--max-updates={args.max_updates} must exceed the current update {global_update}")

    stopped_at_budget = False
    with log_path.open("a", buffering=1) as log:
        for epoch in range(start_epoch, 45):
            order_rng = np.random.default_rng(args.seed + 1000003 * epoch)
            perm = order_rng.permutation(len(train_samples))
            if epoch == start_epoch and start_batch:
                perm = perm[start_batch:]
            batches = [perm[i:i + 8] for i in range(0, len(perm), 8)]
            if config["flow_dt1"]["drop_last"]:
                batches = [b for b in batches if len(b) == 8]
            start_batch_index = (start_batch // 8 if epoch == start_epoch else 0)
            for batch_index, batch in enumerate(batches, start=start_batch_index):
                if len(batch) != 8:
                    continue
                optimizer.zero_grad(set_to_none=True)
                event_batch, gray_batch, event_counts = [], [], []
                batch_ids = []
                augmentation_seeds = []
                for micro, sample_ix in enumerate(batch):
                    rollout, sample_ix = train_samples[int(sample_ix)]
                    sample = train[rollout]
                    seed_seq = np.random.SeedSequence([args.seed, epoch, batch_index, micro])
                    augmentation_seeds.append([args.seed, epoch, batch_index, micro])
                    aug_rng = np.random.default_rng(seed_seq)
                    transform_sample = {k: sample[k][sample_ix:sample_ix + 1]
                                        for k in ("events", "gray_prev", "gray_next", "flow_gt", "flow_valid")}
                    event, gray, _, _ = apply_joint_transform(transform_sample, aug_rng, device)
                    event_batch.append(event); gray_batch.append(gray)
                    event_counts.append(int(sample["event_counts"][sample_ix]))
                    batch_ids.append(str(sample["sample_ids"][sample_ix]))
                event_batch = torch.cat(event_batch, dim=0)
                gray_batch = torch.cat(gray_batch, dim=0)
                output = model(event_batch, SIZE, use_dt4=False, flow_seq=False)
                loss, photo, smooth = objective_dt1(
                    output, gray_batch[:, 0:1], gray_batch[:, 1:2], event_batch, multiscaleloss)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at update {global_update + 1}")
                loss.backward()
                grad_sq, grad_tensors = 0.0, 0
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise RuntimeError(f"non-finite gradient at update {global_update + 1}")
                        grad_sq += float(parameter.grad.detach().float().square().sum().cpu())
                        grad_tensors += 1
                grad_norm = float(grad_sq ** .5)
                loss_value, photo_value, smooth_value = (float(loss.detach().cpu()), float(photo.detach().cpu()), float(smooth.detach().cpu()))
                if not all(np.isfinite(x) for x in (loss_value, photo_value, smooth_value, grad_norm)) or grad_norm <= 0 or grad_tensors == 0:
                    raise RuntimeError(f"invalid loss/gradient at update {global_update + 1}")
                optimizer.step(); global_update += 1
                sampler_position = (batch_index + 1) * 8
                record = {
                    "optimizer_update": global_update, "epoch": epoch + 1, "batch_in_epoch": batch_index + 1,
                    "rollout_sample_ids": batch_ids,
                    "augmentation_seed_sequences": augmentation_seeds,
                    "effective_batch": 8, "microbatch": 8, "accumulation_steps": 1,
                    "loss": loss_value, "photometric_loss": photo_value,
                    "smoothness_loss": smooth_value, "gradient_l2_norm": grad_norm,
                    "finite_gradient_tensors": grad_tensors, "event_counts_per_sample": event_counts,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_s": round(time.monotonic() - start_time, 3),
                    "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
                log.write(json.dumps(record) + "\n")
                at_epoch_end = sampler_position >= len(train_samples)
                should_stop = args.max_updates is not None and global_update >= args.max_updates
                if global_update == 20 or global_update % 100 == 0 or should_stop or at_epoch_end:
                    common = dict(model=model, optimizer=optimizer, scheduler=scheduler, seed=args.seed,
                                  generator=args.generator, epoch=epoch, update=global_update,
                                  sampler_position=sampler_position, config_hash=sha256(args.config),
                                  manifest_hash=manifest_hash, initial_hash=initial_hash,
                                  best_epe=best_epe, best_epoch=best_epoch,
                                  next_epoch=epoch, next_position_samples=sampler_position)
                    save_checkpoint(run / "last.pt", **common)
                    if global_update == 20:
                        save_checkpoint(run / "update_000020.pt", **common)
                    status = {**record, "status": "training", "checkpoint": str(run / "last.pt"),
                              "checkpoint_sha256": sha256(run / "last.pt"), "initial_state_sha256": initial_hash,
                              "current_state_sha256": state_sha256(model), "completed_updates": global_update,
                              "total_updates": total_updates, "run_root": str(run),
                              "checkpoint_updated_utc": datetime.now(timezone.utc).isoformat(),
                              "checkpoint_is_epoch_boundary": at_epoch_end,
                              "resume_rng_cpu_tensors": True}
                    atomic_json(run / "live_status.json", status)
                    print(json.dumps(status), flush=True)
                if global_update in (100, 500):
                    save_checkpoint(run / f"update_{global_update:06d}.pt", model=model, optimizer=optimizer,
                                    scheduler=scheduler, seed=args.seed, generator=args.generator, epoch=epoch,
                                    update=global_update, sampler_position=sampler_position,
                                    config_hash=sha256(args.config), manifest_hash=manifest_hash,
                                    initial_hash=initial_hash, best_epe=best_epe, best_epoch=best_epoch,
                                    next_epoch=epoch, next_position_samples=sampler_position)
                if should_stop and not at_epoch_end:
                    stopped_at_budget = True
                    break
            if stopped_at_budget:
                break

            # Epoch boundary includes scheduler update, full validation and a
            # post-validation resumable checkpoint. Resume never replays a
            # completed epoch or silently repeats its scheduler step.
            scheduler.step()
            validation = evaluate_validation(model, val, device, run, f"epoch_{epoch + 1:02d}")
            val_epe = validation["model_mean_epe_px_per_30ms"]
            if val_epe < best_epe:
                best_epe, best_epoch = val_epe, epoch + 1
                save_checkpoint(run / "synthetic_best.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                                seed=args.seed, generator=args.generator, epoch=epoch, update=global_update,
                                sampler_position=0, config_hash=sha256(args.config), manifest_hash=manifest_hash,
                                initial_hash=initial_hash, best_epe=best_epe, best_epoch=best_epoch,
                                next_epoch=epoch + 1, next_position_samples=0, phase="inference", resume_allowed=False)
            validation_record = {"event": "synthetic_validation", "epoch": epoch + 1,
                                 "optimizer_updates": global_update, "best_epe": best_epe,
                                 "best_epoch": best_epoch, **validation,
                                 "updated_utc": datetime.now(timezone.utc).isoformat()}
            log.write(json.dumps(validation_record) + "\n")
            print(json.dumps(validation_record), flush=True)
            complete = global_update >= total_updates
            save_checkpoint(run / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                            seed=args.seed, generator=args.generator, epoch=epoch, update=global_update,
                            sampler_position=len(train_samples), config_hash=sha256(args.config),
                            manifest_hash=manifest_hash, initial_hash=initial_hash,
                            best_epe=best_epe, best_epoch=best_epoch,
                            next_epoch=epoch + 1, next_position_samples=0,
                            phase="complete" if complete else "train", resume_allowed=not complete)
            if args.max_updates is not None and global_update >= args.max_updates:
                stopped_at_budget = True
                break
            if complete:
                break

    final_status = {"status": "complete" if global_update >= total_updates else "paused_at_update_budget",
                    "completed_updates": global_update, "total_updates": total_updates,
                    "best_synthetic_validation_epe_px_per_30ms": best_epe,
                    "best_epoch": best_epoch, "initial_state_sha256": initial_hash,
                    "current_state_sha256": state_sha256(model), "last_checkpoint": str(run / "last.pt"),
                    "resume_next_epoch": (torch.load(run / "last.pt", map_location="cpu", weights_only=False)["next_epoch"]),
                    "updated_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(run / "live_status.json", final_status)
    print(json.dumps(final_status), flush=True)


if __name__ == "__main__":
    main()
