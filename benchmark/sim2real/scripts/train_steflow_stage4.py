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

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_steflow_stage2_pilot import (
    compute_epe, model_output_to_flow_hw, objective_dt1, pure_torch_spatial_correlation,
    set_seed, state_sha256,
)


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
                    sampler_position, config_hash, manifest_hash, initial_hash, best_epe, best_epoch):
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
    }
    atomic_torch_save(state, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    run = args.run_root.resolve() / args.generator
    run.mkdir(parents=True, exist_ok=True)
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
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU training")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
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
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved["generator"] != args.generator or saved["seed"] != args.seed:
            raise ValueError("resume checkpoint generator/seed mismatch")
        if saved["config_sha256"] != sha256(args.config) or saved["data_manifest_sha256"] != sha256(manifest_path):
            raise ValueError("resume checkpoint config/data hash mismatch")
        model.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_epoch, start_update, start_batch = int(saved["epoch"]), int(saved["optimizer_update"]), int(saved["sampler_position"])
        best_epe, best_epoch = saved["best_synthetic_validation_epe"], saved["best_epoch"]
        if state_sha256(model) != saved["current_state_sha256"]:
            raise RuntimeError("resume model state hash mismatch")

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
        "real_data_used": False, "flow_gt_used_in_optimizer": False,
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
    with log_path.open("a", buffering=1) as log:
        for epoch in range(start_epoch, 45):
            order_rng = np.random.default_rng(args.seed + 1000003 * epoch)
            perm = order_rng.permutation(len(train_samples))
            if epoch == start_epoch and start_batch:
                perm = perm[start_batch:]
            batches = [perm[i:i + 8] for i in range(0, len(perm), 8)]
            if config["flow_dt1"]["drop_last"]:
                batches = [b for b in batches if len(b) == 8]
            for batch_index, batch in enumerate(batches, start=(start_batch // 8 if epoch == start_epoch else 0)):
                if len(batch) != 8:
                    continue
                optimizer.zero_grad(set_to_none=True)
                event_batch, gray_batch, event_counts = [], [], []
                for micro, sample_ix in enumerate(batch):
                    rollout, sample_ix = train_samples[int(sample_ix)]
                    sample = train[rollout]
                    seed_seq = np.random.SeedSequence([args.seed, epoch, batch_index, micro])
                    aug_rng = np.random.default_rng(seed_seq)
                    transform_sample = {k: sample[k][sample_ix:sample_ix + 1]
                                        for k in ("events", "gray_prev", "gray_next", "flow_gt", "flow_valid")}
                    event, gray, _, _ = apply_joint_transform(transform_sample, aug_rng, device)
                    event_batch.append(event); gray_batch.append(gray)
                    event_counts.append(int(sample["event_counts"][sample_ix]))
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
                record = {
                    "optimizer_update": global_update, "epoch": epoch + 1, "batch_in_epoch": batch_index + 1,
                    "rollout_sample_ids": [str(train[r]["sample_ids"][i]) for r, i in [train_samples[int(j)] for j in batch]],
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
                sampler_position = (batch_index + 1) * 8
                if global_update == 20 or global_update % 100 == 0:
                    current_hash = state_sha256(model)
                    common = dict(model=model, optimizer=optimizer, scheduler=scheduler, seed=args.seed,
                                  generator=args.generator, epoch=epoch, update=global_update,
                                  sampler_position=sampler_position, config_hash=sha256(args.config),
                                  manifest_hash=manifest_hash, initial_hash=initial_hash,
                                  best_epe=best_epe, best_epoch=best_epoch)
                    save_checkpoint(run / "last.pt", **common)
                    if global_update == 20:
                        save_checkpoint(run / "update_000020.pt", **common)
                        saved = torch.load(run / "update_000020.pt", map_location=device, weights_only=False)
                        restored = models.steflow().to(device)
                        restored.load_state_dict(saved["state_dict"])
                        restored_optimizer = torch.optim.Adam([
                            {"params": restored.bias_parameters(), "weight_decay": 0.0},
                            {"params": restored.weight_parameters(), "weight_decay": .0004},
                        ], lr=.0004, betas=(.9, .999))
                        restored_optimizer.load_state_dict(saved["optimizer"])
                        restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                            restored_optimizer, milestones=[5, 10, 20], gamma=.7)
                        restored_scheduler.load_state_dict(saved["scheduler"])
                        model.eval(); restored.eval()
                        probe, _, _ = val_input(val[sorted(val)[0]], 0, device)
                        with torch.no_grad():
                            live_probe = model(probe, 352, use_dt4=False, flow_seq=False)
                            restored_probe = restored(probe, 352, use_dt4=False, flow_seq=False)
                        restore_error = float((live_probe - restored_probe).abs().max().cpu())
                        restore_hash_match = state_sha256(model) == state_sha256(restored)
                        if not restore_hash_match or restore_error > 1e-6:
                            raise RuntimeError(f"update-20 checkpoint restore verification failed: hash={restore_hash_match}, max_abs={restore_error}")
                        model.train()
                        del restored, restored_optimizer, restored_scheduler, live_probe, restored_probe, probe
                        torch.cuda.empty_cache()
                    status = {**record, "status": "training", "checkpoint": str(run / "last.pt"),
                              "checkpoint_sha256": sha256(run / "last.pt"), "initial_state_sha256": initial_hash,
                              "physical_gpu_id": os.environ.get("STAGE4_PHYSICAL_GPU", "unknown"),
                              "current_state_sha256": current_hash, "completed_updates": global_update,
                              "total_updates": total_updates, "run_root": str(run),
                              "checkpoint_updated_utc": datetime.now(timezone.utc).isoformat(),
                              "update20_restore_verified": global_update == 20,
                              "restore_prediction_max_abs": restore_error if global_update == 20 else None}
                    atomic_json(run / "live_status.json", status)
                    print(json.dumps(status), flush=True)
                if global_update in (100, 500):
                    # Evidence snapshot is copied at a checked update count.
                    save_checkpoint(run / f"update_{global_update:06d}.pt", model=model, optimizer=optimizer,
                                    scheduler=scheduler, seed=args.seed, generator=args.generator, epoch=epoch,
                                    update=global_update, sampler_position=sampler_position,
                                    config_hash=sha256(args.config), manifest_hash=manifest_hash,
                                    initial_hash=initial_hash, best_epe=best_epe, best_epoch=best_epoch)
            start_batch = 0
            scheduler.step()
            model.eval()
            epe_by_rollout = {}
            with torch.no_grad():
                for rollout in sorted(val):
                    epe_values = []
                    for sample_index in range(66):
                        event, gray0, gray1 = val_input(val[rollout], sample_index, device)
                        pred = model(event, 352, use_dt4=False, flow_seq=False)
                        flow = model_output_to_flow_hw(pred)
                        if flow.shape != (2, 352, 352):
                            raise RuntimeError(f"validation output must be [2,352,352], got {flow.shape}")
                        flow = flow[:, 46:306, 3:349]
                        epe, _ = compute_epe(flow, val[rollout]["flow_gt"][sample_index], val[rollout]["flow_valid"][sample_index],
                                             prediction_unit="px_per_interval", gt_unit="px_per_interval")
                        epe_values.append(epe)
                    epe_by_rollout[rollout] = float(np.mean(epe_values))
            val_epe = float(np.mean(list(epe_by_rollout.values())))
            if val_epe < best_epe:
                best_epe, best_epoch = val_epe, epoch + 1
                save_checkpoint(run / "synthetic_best.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                                seed=args.seed, generator=args.generator, epoch=epoch, update=global_update,
                                sampler_position=0, config_hash=sha256(args.config), manifest_hash=manifest_hash,
                                initial_hash=initial_hash, best_epe=best_epe, best_epoch=best_epoch)
            model.train()
            validation_record = {"event": "synthetic_validation", "epoch": epoch + 1,
                                 "optimizer_updates": global_update, "rollout_equal_weight_epe_px_per_30ms": epe_by_rollout,
                                 "mean_epe_px_per_30ms": val_epe, "best_epe": best_epe,
                                 "best_epoch": best_epoch, "real_data_used": False,
                                 "updated_utc": datetime.now(timezone.utc).isoformat()}
            with log_path.open("a", buffering=1) as output:
                output.write(json.dumps(validation_record) + "\n")
            print(json.dumps(validation_record), flush=True)
            if global_update >= total_updates:
                break
    save_checkpoint(run / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                    seed=args.seed, generator=args.generator, epoch=44, update=global_update,
                    sampler_position=len(train_samples), config_hash=sha256(args.config),
                    manifest_hash=manifest_hash, initial_hash=initial_hash,
                    best_epe=best_epe, best_epoch=best_epoch)
    atomic_json(run / "live_status.json", {"status": "complete", "completed_updates": global_update,
                "total_updates": total_updates, "best_synthetic_validation_epe_px_per_30ms": best_epe,
                "best_epoch": best_epoch, "initial_state_sha256": initial_hash,
                "current_state_sha256": state_sha256(model),
                "last_checkpoint": str(run / "last.pt"), "updated_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
