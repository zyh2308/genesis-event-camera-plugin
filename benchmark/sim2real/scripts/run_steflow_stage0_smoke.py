#!/usr/bin/env python3
"""One matched synthetic-only STE-FlowNet update per generator on g1."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SIZE = 256
TRAIN_INDEX = 9
VALIDATION_INDEX = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pure_torch_spatial_correlation(input1, input2, kernel_size=1, patch_size=9,
                                   stride=1, padding=0, dilation_patch=1):
    """Reference equivalent for spatial_correlation_sampler's 9x9 patch call."""
    if kernel_size != 1 or patch_size != 9 or stride != 1 or padding != 0 or dilation_patch != 1:
        raise ValueError("Stage-0 fallback supports only the official kernel_size=1, patch_size=9 call")
    batch, channels, height, width = input1.shape
    radius = patch_size // 2
    patches = F.unfold(input2, kernel_size=patch_size, padding=radius, stride=1)
    patches = patches.view(batch, channels, patch_size, patch_size, height, width)
    center = input1[:, :, None, None, :, :]
    return (center * patches).sum(dim=1).contiguous()


def photometric_objective(output, gray_prev, gray_next, event_tensor, loss_module):
    # Keep the upstream four-scale Charbonnier and second-order smoothness
    # formulas. The upstream function calls .cuda(), mapped to physical GPU 1
    # by CUDA_VISIBLE_DEVICES in the server launcher.
    photo = loss_module.compute_photometric_loss(
        gray_prev, gray_next,
        torch.sum(event_tensor, dim=4), output, weights=[1.0, 1.0, 1.0, 1.0],
    )
    smooth = loss_module.smooth_loss(output)
    return photo + 10.0 * smooth, photo, smooth


def load_sample(path: Path, index: int, device: torch.device):
    with np.load(path) as data:
        events = np.asarray(data["events"][index:index + 1], dtype=np.float32)
        prev = np.asarray(data["gray_prev"][index:index + 1], dtype=np.float32)
        nxt = np.asarray(data["gray_next"][index:index + 1], dtype=np.float32)
        flow = np.asarray(data["flow_gt"][index], dtype=np.float32)
        valid = np.asarray(data["flow_valid"][index], dtype=bool)
        count = int(data["interval_event_counts"][index])
    if events.shape != (1, 2, SIZE, SIZE, 5):
        raise ValueError(f"wrong STE event tensor shape in {path}: {events.shape}")
    if prev.shape != (1, SIZE, SIZE) or nxt.shape != (1, SIZE, SIZE):
        raise ValueError(f"wrong grayscale shape in {path}: {prev.shape}, {nxt.shape}")
    return (
        torch.from_numpy(events).to(device),
        torch.from_numpy(prev).to(device),
        torch.from_numpy(nxt).to(device),
        flow,
        valid,
        count,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model_root.resolve()))

    import importlib
    corr_module = importlib.import_module("models.corr")
    corr_module.spatial_correlation_sample = pure_torch_spatial_correlation
    import models
    import multiscaleloss

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback is used for the experiment")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    results = []
    initial_hashes = {}
    for generator in ("eventgenesis", "v2e"):
        train_path = args.data_root / generator / "train.npz"
        val_path = args.data_root / generator / "validation.npz"
        train_events, train_prev, train_next, _, _, train_count = load_sample(
            train_path, TRAIN_INDEX, device
        )
        val_events, val_prev, val_next, val_flow, val_valid, val_count = load_sample(
            val_path, VALIDATION_INDEX, device
        )
        if train_count < 1 or val_count < 1 or not val_valid.any():
            raise ValueError(f"empty event/label sample for {generator}")

        set_seed(args.seed)
        model = models.steflow().cuda()
        model.train()
        initial_hash = state_sha256(model)
        initial_hashes[generator] = initial_hash
        param_groups = [
            {"params": model.bias_parameters(), "weight_decay": 0.0},
            {"params": model.weight_parameters(), "weight_decay": 4e-4},
        ]
        optimizer = torch.optim.Adam(param_groups, lr=4e-4, betas=(0.9, 0.999))

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        train_outputs = model(train_events, SIZE, use_dt4=False, flow_seq=False)
        train_loss, train_photo, train_smooth = photometric_objective(
            train_outputs, train_prev.cpu().numpy(), train_next.cpu().numpy(),
            train_events, multiscaleloss,
        )
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"non-finite training loss for {generator}: {train_loss.item()}")
        train_loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started

        # Synthetic-only validation objective and flow labels are diagnostics;
        # no real MVSEC data or flow labels enter optimization/checkpoint choice.
        model.train()  # upstream returns all four scales in training mode
        with torch.no_grad():
            val_outputs = model(val_events, SIZE, use_dt4=False, flow_seq=False)
            val_loss, val_photo, val_smooth = photometric_objective(
                val_outputs, val_prev.cpu().numpy(), val_next.cpu().numpy(),
                val_events, multiscaleloss,
            )
        model.eval()
        with torch.no_grad():
            pred = model(val_events, SIZE, use_dt4=False, flow_seq=False)[0].detach().cpu().numpy()
        epe = np.sqrt(np.sum((pred - val_flow) ** 2, axis=0))
        synthetic_val_epe = float(epe[val_valid].mean())

        args.output_root.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output_root / f"{generator}_step000001.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator,
            "seed": args.seed,
            "updates": 1,
            "source_model": "official STE-FlowNet FlowNetS_spike",
            "train_objective": float(train_loss.detach().cpu()),
            "synthetic_validation_objective": float(val_loss.detach().cpu()),
        }, checkpoint)
        results.append({
            "generator": generator,
            "seed": args.seed,
            "updates": 1,
            "train_sample_interval_index": TRAIN_INDEX,
            "validation_sample_interval_index": VALIDATION_INDEX,
            "train_interval_event_count": train_count,
            "validation_interval_event_count": val_count,
            "representation_shape": list(train_events.shape),
            "representation_order": "[B, ON/OFF, H, W, 5 equal-event-count Gaussian slices]",
            "initial_state_sha256": initial_hash,
            "train_loss": float(train_loss.detach().cpu()),
            "train_photometric_loss": float(train_photo.detach().cpu()),
            "train_smoothness_loss": float(train_smooth.detach().cpu()),
            "synthetic_validation_loss": float(val_loss.detach().cpu()),
            "synthetic_validation_photometric_loss": float(val_photo.detach().cpu()),
            "synthetic_validation_smoothness_loss": float(val_smooth.detach().cpu()),
            "synthetic_validation_epe_px_per_interval": synthetic_val_epe,
            "train_step_wall_seconds": train_seconds,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        })
        del model, optimizer, train_events, val_events, train_outputs, val_outputs
        torch.cuda.empty_cache()

    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError(f"paired model initializations differ: {initial_hashes}")
    result = {
        "status": "pass",
        "scope": "one synthetic-only STE-FlowNet dt=1 update per generator plus one synthetic validation interval",
        "real_data_accessed": False,
        "initial_state_hashes_match": True,
        "optimizer": {"type": "Adam", "lr": 4e-4, "betas": [0.9, 0.999], "weight_decay": "4e-4 weights, 0 biases"},
        "objective": "official four-scale Charbonnier photometric loss + 10x official second-order smoothness",
        "correlation_adapter": "pure PyTorch F.unfold equivalent for upstream spatial_correlation_sampler 1x1 kernel, 9x9 patch; benchmark runtime injection only",
        "representation_adapter": "official polarity-separated Gaussian-weighted equal-event-count 5 slices; retain remainder; tied/empty times guarded; uint8 overflow fails closed; center-pad 160x128 to 256x256",
        "flow_ground_truth_use": "synthetic validation only; never passed to model or optimizer",
        "results": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
