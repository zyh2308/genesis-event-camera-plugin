#!/usr/bin/env python3
"""Paired synthetic-only STE-FlowNet dt=1/dt=4 short training pilot."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SIZE = 256
WEIGHTS = [1.0, 1.0, 1.0, 1.0]
FLOW_UNIT = "px_per_interval"


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _MemoryEfficientSpatialCorrelation(torch.autograd.Function):
    """Exact 1x1/9x9 correlation with an input-sized autograd context."""

    @staticmethod
    def forward(ctx, input1: torch.Tensor, input2: torch.Tensor):
        if input1.shape != input2.shape or input1.ndim != 4:
            raise ValueError(f"correlation inputs must share [B,C,H,W], got {input1.shape}, {input2.shape}")
        ctx.save_for_backward(input1, input2)
        batch, _, height, width = input1.shape
        radius = 4
        output = input1.new_zeros((batch, 81, height, width))
        channel = 0
        for dy in range(9):
            shift_y = dy - radius
            y0, y1 = max(0, -shift_y), min(height, height - shift_y)
            for dx in range(9):
                shift_x = dx - radius
                x0, x1 = max(0, -shift_x), min(width, width - shift_x)
                if y0 < y1 and x0 < x1:
                    output[:, channel, y0:y1, x0:x1] = (
                        input1[:, :, y0:y1, x0:x1]
                        * input2[:, :, y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
                    ).sum(dim=1)
                channel += 1
        return output.view(batch, 9, 9, height, width)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input1, input2 = ctx.saved_tensors
        batch, _, height, width = input1.shape
        grad1 = torch.zeros_like(input1) if ctx.needs_input_grad[0] else None
        grad2 = torch.zeros_like(input2) if ctx.needs_input_grad[1] else None
        channel = 0
        for dy in range(9):
            shift_y = dy - 4
            y0, y1 = max(0, -shift_y), min(height, height - shift_y)
            for dx in range(9):
                shift_x = dx - 4
                x0, x1 = max(0, -shift_x), min(width, width - shift_x)
                if y0 < y1 and x0 < x1:
                    grad = grad_output[:, dy, dx, y0:y1, x0:x1].unsqueeze(1)
                    input1_region = input1[:, :, y0:y1, x0:x1]
                    input2_region = input2[:, :, y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
                    if grad1 is not None:
                        grad1[:, :, y0:y1, x0:x1] += grad * input2_region
                    if grad2 is not None:
                        grad2[:, :, y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x] += grad * input1_region
                channel += 1
        return grad1, grad2


def pure_torch_spatial_correlation(input1, input2, kernel_size=1, patch_size=9,
                                   stride=1, padding=0, dilation_patch=1):
    """Memory-bounded equivalent of the pinned 1x1-kernel/9x9-patch call."""
    if kernel_size != 1 or patch_size != 9 or stride != 1 or padding != 0 or dilation_patch != 1:
        raise ValueError("pilot fallback only implements the official 1x1 kernel, 9x9 patch call")
    return _MemoryEfficientSpatialCorrelation.apply(input1, input2).contiguous()


def explicit_spatial_correlation(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = input1.shape
    padded = F.pad(input2, (4, 4, 4, 4))
    maps = []
    for dy in range(9):
        for dx in range(9):
            shifted = padded[:, :, dy:dy + height, dx:dx + width]
            maps.append((input1 * shifted).sum(dim=1))
    return torch.stack(maps, dim=1).view(batch, 9, 9, height, width)


def verify_correlation(device: torch.device) -> dict:
    # Input is 256x256, but the pinned first encoder stage has stride 2 and
    # produces the largest correlation tensor: [B=1, C=64, H=W=128].
    target_shape = (1, 64, SIZE // 2, SIZE // 2)
    x1 = torch.randn(target_shape, device=device, dtype=torch.float32)
    x2 = torch.randn(target_shape, device=device, dtype=torch.float32)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    actual = pure_torch_spatial_correlation(x1, x2)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    target_peak_allocated = int(torch.cuda.max_memory_allocated(device) - baseline_allocated)
    target_peak_reserved = int(torch.cuda.max_memory_reserved(device) - baseline_reserved)
    expected = explicit_spatial_correlation(x1, x2)
    max_abs = float((actual - expected).abs().max().cpu())
    if actual.shape != (1, 9, 9, SIZE // 2, SIZE // 2) or max_abs > 1e-5:
        raise RuntimeError(f"target-shape STE correlation check failed: shape={actual.shape}, max_abs={max_abs}")

    # Explicit forward/backward comparison at a small target for finite,
    # matched gradients without allocating the large patch matrix twice.
    a = torch.randn((1, 3, 8, 7), device=device, requires_grad=True)
    b = torch.randn((1, 3, 8, 7), device=device, requires_grad=True)
    pure_torch_spatial_correlation(a, b).square().mean().backward()
    grad_actual = (a.grad.detach().clone(), b.grad.detach().clone())
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    explicit_spatial_correlation(a_ref, b_ref).square().mean().backward()
    grad_error = max(float((grad_actual[0] - a_ref.grad).abs().max().cpu()),
                     float((grad_actual[1] - b_ref.grad).abs().max().cpu()))
    if grad_error > 1e-5:
        raise RuntimeError(f"STE correlation gradient mismatch {grad_error}")
    result = {
        "implementation": "benchmark-local memory-efficient pure PyTorch autograd function (81 spatial shifts)",
        "upstream_extension_installed": False,
        "target_shape": list(target_shape),
        "batch_size": 1,
        "dtype": "float32",
        "target_forward_max_abs_error": max_abs,
        "small_forward_backward_max_gradient_error": grad_error,
        "target_forward_seconds": elapsed,
        "target_peak_allocated_gpu_bytes": target_peak_allocated,
        "target_peak_reserved_gpu_bytes_above_baseline": target_peak_reserved,
        "output_shape": list(actual.shape),
    }
    del actual, expected, x1, x2, a, b, a_ref, b_ref
    torch.cuda.empty_cache()
    return result


def objective_dt1(output, gray_prev, gray_next, event_tensor, loss_module):
    # STEFlow returns a tuple of four scales in train mode but only the final
    # full-resolution tensor in eval mode. The pinned objective accepts a
    # sequence of scale tensors in both cases.
    if torch.is_tensor(output):
        output = (output,)
    # The pinned STE loss converts endpoint images to NumPy for cv2 resizing.
    # Keep image targets detached on CPU while preserving the CUDA flow graph.
    gray_prev_cpu = gray_prev.detach().cpu()
    gray_next_cpu = gray_next.detach().cpu()
    photo = loss_module.compute_photometric_loss(
        gray_prev_cpu, gray_next_cpu, torch.sum(event_tensor, dim=4), output, weights=WEIGHTS
    )
    smooth = loss_module.smooth_loss(output)
    return photo + 10.0 * smooth, photo, smooth


def objective_dt4(output, gray_frames, event_tensor, loss_module):
    flow1_list, flow2_list, flow3_list, flow4_list = output
    photos, smooths = [], []
    event_image = torch.sum(event_tensor, dim=4)
    gray_frames_cpu = [frame.detach().cpu() for frame in gray_frames]
    for k in range(4):
        current = (flow1_list[k], flow2_list[k], flow3_list[k], flow4_list[k])
        photos.append(loss_module.compute_photometric_loss(
            gray_frames_cpu[0], gray_frames_cpu[k + 1], event_image, current, weights=WEIGHTS
        ))
        smooths.append(loss_module.smooth_loss(current))
    photo = sum(photos) / 4.0
    smooth = sum(smooths) / 4.0
    return photo + 10.0 * smooth, photo, smooth


def read_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def select_two_indices(length: int) -> list[int]:
    if length < 1:
        raise ValueError("empty training interval set")
    if length == 1:
        return [0]
    return sorted(set(np.rint(np.linspace(0, length - 1, 2)).astype(int).tolist()))


def event_input(episode: dict, index: int, dt: int, device: torch.device):
    key = f"events_dt{dt}"
    events = np.asarray(episode[key][index:index + 1], dtype=np.float32)
    expected_slices = 5 * dt
    if events.shape != (1, 2, SIZE, SIZE, expected_slices):
        raise ValueError(f"unexpected {key} shape {events.shape}")
    return torch.from_numpy(events).to(device)


def gray_window(episode: dict, index: int, dt: int):
    frames = np.asarray(episode["gray_frames"][index:index + dt + 1], dtype=np.float32)
    if frames.shape != (dt + 1, SIZE, SIZE):
        raise ValueError(f"unexpected gray endpoint window {frames.shape}")
    return frames


def model_output_to_flow_hw(output: torch.Tensor) -> np.ndarray:
    """Extract one STE eval prediction while preserving both vector channels."""
    if not torch.is_tensor(output):
        raise TypeError(f"flow model output must be a torch.Tensor, got {type(output).__name__}")
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 2:
        raise ValueError(f"flow model output must have shape [1,2,H,W], got {tuple(output.shape)}")
    prediction = output.detach().cpu().numpy()[0]
    if not np.issubdtype(prediction.dtype, np.number) or not np.isfinite(prediction).all():
        raise ValueError("flow model output must contain only finite numeric values")
    return prediction


def compute_epe(
    prediction: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    *,
    prediction_unit: str,
    gt_unit: str,
) -> tuple[float, int]:
    """Strict endpoint error for [u,v] pixel displacement per endpoint interval."""
    prediction = np.asarray(prediction)
    gt = np.asarray(gt)
    valid = np.asarray(valid)
    if prediction.shape != gt.shape or prediction.ndim != 3 or prediction.shape[0] != 2:
        raise ValueError(
            "prediction and GT must have the same [2,H,W] vector-flow shape; "
            f"got prediction={prediction.shape}, GT={gt.shape}"
        )
    if valid.shape != prediction.shape[1:]:
        raise ValueError(f"validity mask must have shape [H,W]={prediction.shape[1:]}, got {valid.shape}")
    if prediction_unit != FLOW_UNIT or gt_unit != FLOW_UNIT:
        raise ValueError(
            f"EPE requires both prediction and GT in {FLOW_UNIT}; "
            f"got prediction={prediction_unit!r}, GT={gt_unit!r}"
        )
    for name, value in (("prediction", prediction), ("GT", gt)):
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite numeric values")
    if not (np.issubdtype(valid.dtype, np.bool_) or np.issubdtype(valid.dtype, np.number)):
        raise ValueError("validity mask must be boolean or numeric 0/1")
    if not np.isfinite(valid).all() or not np.isin(valid, (0, 1)).all():
        raise ValueError("validity mask must contain only finite 0/1 values")
    valid_mask = valid.astype(bool, copy=False)
    count = int(valid_mask.sum())
    if count == 0:
        raise ValueError("synthetic validation interval has no valid flow pixels")
    # Cast only after shape/unit/finite checks so NumPy can never silently
    # broadcast a missing vector channel during subtraction.
    delta = prediction.astype(np.float64, copy=False) - gt.astype(np.float64, copy=False)
    epe = np.hypot(delta[0], delta[1])
    value = float(epe[valid_mask].mean())
    if not np.isfinite(value):
        raise RuntimeError("synthetic EPE is not finite")
    return value, count


def train_one(generator: str, dt: int, data_root: Path, model_root: Path,
              output_root: Path, seed: int, epochs: int, models, multiscaleloss, device: torch.device):
    train_paths = sorted((data_root / generator / "train").glob("*.npz"))
    val_paths = sorted((data_root / generator / "validation").glob("*.npz"))
    if len(train_paths) < 2 or len(val_paths) < 2:
        raise RuntimeError(f"{generator}/dt{dt}: multi-rollout train/val files missing")
    train_data = {path.stem: read_episode(path) for path in train_paths}
    val_data = {path.stem: read_episode(path) for path in val_paths}
    first = next(iter(train_data.values()))
    interval_key = f"events_dt{dt}"
    flow_key = f"flow_gt_dt{dt}"
    valid_key = f"flow_valid_dt{dt}"
    train_indices = {
        episode_id: select_two_indices(len(episode[interval_key]))
        for episode_id, episode in train_data.items()
    }
    schedule = [
        (episode_id, index)
        for _epoch in range(epochs)
        for episode_id in sorted(train_data)
        for index in train_indices[episode_id]
    ]
    set_seed(seed)
    model = models.steflow().to(device)
    model.train()
    initial_hash = state_sha256(model)
    param_groups = [
        {"params": model.bias_parameters(), "weight_decay": 0.0},
        {"params": model.weight_parameters(), "weight_decay": 4e-4},
    ]
    optimizer = torch.optim.Adam(param_groups, lr=4e-4 if dt == 1 else 2e-4, betas=(0.9, 0.999))
    records = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for update, (episode_id, index) in enumerate(schedule, start=1):
        sample = train_data[episode_id]
        event_tensor = event_input(sample, index, dt, device)
        gray = gray_window(sample, index, dt)
        optimizer.zero_grad(set_to_none=True)
        if dt == 1:
            output = model(event_tensor, SIZE, use_dt4=False, flow_seq=False)
            loss, photo, smooth = objective_dt1(
                output, gray[0:1], gray[1:2], event_tensor, multiscaleloss
            )
        else:
            output = model(event_tensor, SIZE, use_dt4=True, flow_seq=False)
            gray_batch = [frame[None, ...] for frame in gray]
            loss, photo, smooth = objective_dt4(output, gray_batch, event_tensor, multiscaleloss)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {generator}/dt{dt} loss at update {update}")
        loss.backward()
        grad_sq = 0.0
        grad_tensors = 0
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("non-finite STE-FlowNet gradient")
                grad_sq += float(torch.sum(parameter.grad.detach().float() ** 2).cpu())
                grad_tensors += 1
        grad_norm = float(grad_sq**0.5)
        if not np.isfinite(grad_norm) or grad_norm <= 0 or grad_tensors == 0:
            raise RuntimeError("STE-FlowNet backward produced no finite nonzero gradient")
        optimizer.step()
        records.append({
            "update": update,
            "epoch": (update - 1) // max(1, len(train_data) * 2) + 1,
            "episode_id": episode_id,
            "interval_index": int(index),
            "event_count_in_physical_window": int(sample["interval_event_counts"][index:index + dt].sum()),
            "loss": float(loss.detach().cpu()),
            "photometric_loss": float(photo.detach().cpu()),
            "smoothness_loss": float(smooth.detach().cpu()),
            "gradient_l2_norm": grad_norm,
            "finite_gradient_tensors": grad_tensors,
        })
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    if state_sha256(model) == initial_hash:
        raise RuntimeError("STE-FlowNet parameters did not change during synthetic training")

    model.eval()
    validation = []
    first_validation_probe = None
    with torch.no_grad():
        for episode_id in sorted(val_data):
            sample = val_data[episode_id]
            for index in range(len(sample[interval_key])):
                tensor = event_input(sample, index, dt, device)
                if dt == 1:
                    pred = model(tensor, SIZE, use_dt4=False, flow_seq=False)
                else:
                    pred = model(tensor, SIZE, use_dt4=True, flow_seq=False)
                prediction = model_output_to_flow_hw(pred)
                epe, valid_pixels = compute_epe(
                    prediction,
                    sample[flow_key][index],
                    sample[valid_key][index],
                    prediction_unit=FLOW_UNIT,
                    gt_unit=FLOW_UNIT,
                )
                gray = gray_window(sample, index, dt)
                if dt == 1:
                    val_loss, val_photo, val_smooth = objective_dt1(
                        pred, gray[0:1], gray[1:2], tensor, multiscaleloss
                    )
                else:
                    # In eval mode the pinned model returns the final dt4 flow
                    # tensor rather than its four intermediate training flows.
                    # Score this output against the matching first/last image
                    # endpoints with the same official photometric/smoothness
                    # operators; validation EPE above uses the shared dt4 label.
                    val_loss, val_photo, val_smooth = objective_dt1(
                        pred, gray[0:1], gray[-1:], tensor, multiscaleloss
                    )
                validation.append({
                    "episode_id": episode_id,
                    "interval_index": int(index),
                    "valid_pixels": valid_pixels,
                    "epe_px_per_interval": epe,
                    "epe_unit": FLOW_UNIT,
                    "objective": float(val_loss.detach().cpu()),
                    "photometric_loss": float(val_photo.detach().cpu()),
                    "smoothness_loss": float(val_smooth.detach().cpu()),
                })
                if len(validation) == 1:
                    first_validation_probe = {
                        "episode_id": episode_id,
                        "interval_index": index,
                        "prediction": prediction.copy(),
                        "epe_px_per_interval": epe,
                    }

    final_hash = state_sha256(model)
    if first_validation_probe is None:
        raise RuntimeError(f"{generator}/dt{dt}: no synthetic validation intervals were evaluated")
    checkpoint = output_root / f"{generator}_dt{dt}_final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": generator,
        "dt": dt,
        "seed": seed,
        "updates": len(records),
        "source_model": "pinned official STE-FlowNet FlowNetS_spike",
        "final_state_sha256": final_hash,
    }, checkpoint)

    restored = models.steflow().to(device)
    restored_optimizer = torch.optim.Adam([
        {"params": restored.bias_parameters(), "weight_decay": 0.0},
        {"params": restored.weight_parameters(), "weight_decay": 4e-4},
    ], lr=4e-4 if dt == 1 else 2e-4, betas=(0.9, 0.999))
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    restored.load_state_dict(saved["state_dict"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    if state_sha256(restored) != final_hash:
        raise RuntimeError(f"checkpoint restore hash mismatch {generator}/dt{dt}")
    restored.eval()
    probe_sample = val_data[first_validation_probe["episode_id"]]
    probe_index = first_validation_probe["interval_index"]
    probe = event_input(probe_sample, probe_index, dt, device)
    with torch.no_grad():
        restored_pred = restored(probe, SIZE, use_dt4=(dt == 4), flow_seq=False)
    restored_flow = model_output_to_flow_hw(restored_pred)
    if restored_flow.shape != first_validation_probe["prediction"].shape:
        raise RuntimeError(f"checkpoint restore flow shape mismatch {generator}/dt{dt}")
    if not np.allclose(restored_flow, first_validation_probe["prediction"], rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"checkpoint restore prediction mismatch {generator}/dt{dt}")
    restored_epe, _ = compute_epe(
        restored_flow,
        probe_sample[flow_key][probe_index],
        probe_sample[valid_key][probe_index],
        prediction_unit=FLOW_UNIT,
        gt_unit=FLOW_UNIT,
    )
    if not np.isclose(restored_epe, first_validation_probe["epe_px_per_interval"], rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"checkpoint restore EPE mismatch {generator}/dt{dt}")

    return {
        "generator": generator,
        "dt": dt,
        "train_rollouts": sorted(train_data),
        "validation_rollouts": sorted(val_data),
        "seed": seed,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "updates": len(records),
        "epochs": epochs,
        "selected_intervals_per_train_rollout": train_indices,
        "train_objective": "official four-scale Charbonnier photometric loss + 10x second-order smoothness",
        "validation_objective": "eval-mode final flow tensor; dt1 adjacent endpoints or dt4 first/last endpoints, official photometric/smoothness operators",
        "flow_gt_use": "synthetic validation EPE only; never passed to the model or optimizer",
        "flow_vector_unit": FLOW_UNIT,
        "train_wall_seconds": train_seconds,
        "train_loss_mean": float(np.mean([row["loss"] for row in records])),
        "train_photometric_mean": float(np.mean([row["photometric_loss"] for row in records])),
        "train_smoothness_mean": float(np.mean([row["smoothness_loss"] for row in records])),
        "gradient_l2_norm_min": min(row["gradient_l2_norm"] for row in records),
        "gradient_l2_norm_max": max(row["gradient_l2_norm"] for row in records),
        "finite_gradient_tensors_min": min(row["finite_gradient_tensors"] for row in records),
        "validation_interval_count": len(validation),
        "synthetic_validation_epe_mean_px_per_interval": float(np.mean([row["epe_px_per_interval"] for row in validation])),
        "synthetic_validation_objective_mean": float(np.mean([row["objective"] for row in validation])),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_restore_state_hash_match": state_sha256(restored) == final_hash,
        "checkpoint_restore_synthetic_validation_epe": restored_epe,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        "training_updates_detail": records,
        "validation_detail": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.model_root.resolve()))
    corr_module = importlib.import_module("models.corr")
    corr_module.spatial_correlation_sample = pure_torch_spatial_correlation
    import models
    import multiscaleloss

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback is used for this pilot")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    corr_evidence = verify_correlation(device)

    results = []
    initial_hashes = {}
    for dt in (1, 4):
        for generator in ("eventgenesis", "v2e"):
            row = train_one(
                generator=generator,
                dt=dt,
                data_root=args.data_root,
                model_root=args.model_root,
                output_root=args.output_root,
                seed=args.seed,
                epochs=args.epochs,
                models=models,
                multiscaleloss=multiscaleloss,
                device=device,
            )
            results.append(row)
            initial_hashes[f"dt{dt}/{generator}"] = row["initial_state_sha256"]
            torch.cuda.empty_cache()
    for dt in (1, 4):
        pair = [initial_hashes[f"dt{dt}/{generator}"] for generator in ("eventgenesis", "v2e")]
        if len(set(pair)) != 1:
            raise RuntimeError(f"paired initialization mismatch at dt={dt}: {pair}")
    result = {
        "status": "pass",
        "scope": "paired, short synthetic-only STE-FlowNet training and independent synthetic validation on multiple physics rollouts",
        "real_data_accessed": False,
        "paired_initialization_hashes_match": True,
        "seed": args.seed,
        "epochs": args.epochs,
        "correlation_operator_verification": corr_evidence,
        "dt_semantics": "dt=1 spans one adjacent shared endpoint interval; dt=4 spans four adjacent endpoint intervals; labels are px per corresponding interval group",
        "results": results,
    }
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
