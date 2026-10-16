#!/usr/bin/env python3
"""Bounded target-size diagnostic for the Stage-4 EventGenesis warp backend.

This does not generate training data. It exercises one registered 10 ms
interval at the registered 346x260 raster, including the locked sensor
sampler and DVS integration, then compares selected projected warps against
the NumPy reference implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator
from genesis_event_plugin.physics_warp import (
    depth_aware_bidirectional_warp_projected_diagnostics,
    interpolate_se3,
    project_endpoint_to_alpha,
)
from genesis_event_plugin.torch_physics import project_endpoint_to_alpha_stack_torch


def transform(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (x, y, z)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-4 diagnostic requires an available CUDA GPU")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("diagnostic device must be CUDA")
    torch.cuda.set_device(device)

    width, height = 346, 260
    fps, interval_s = 100, 0.01
    sensor_cutoff_hz, sensor_eps_target = 300.0, 0.05
    tau = 1.0 / (2.0 * math.pi * sensor_cutoff_hz)
    expected_uncapped = math.ceil(interval_s / tau / sensor_eps_target)
    expected_samples = min(300, expected_uncapped)
    if expected_samples != 300:
        raise RuntimeError(f"expected locked sampler to cap at 300, got {expected_samples}")

    K = np.asarray([[300.0, 0.0, (width - 1) / 2],
                    [0.0, 300.0, (height - 1) / 2],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    depth0 = np.full((height, width), 3.0, dtype=np.float32)
    depth1 = depth0.copy()
    seg0 = np.zeros((height, width), dtype=np.int32)
    seg1 = seg0.copy()
    seg0[48:208, 48:162] = seg1[50:210, 52:166] = 101
    seg0[88:192, 196:290] = seg1[86:190, 194:288] = 202
    depth0[seg0 == 101] = depth1[seg1 == 101] = 2.1
    depth0[seg0 == 202] = depth1[seg1 == 202] = 2.5
    frame0 = (0.08 + 0.35 * (xx / (width - 1)) + 0.11 * (yy / (height - 1))).astype(np.float32)
    frame1 = np.roll(frame0, 2, axis=1).copy()
    frame1[:, :2] = frame0[:, :2]

    camera0, camera1 = transform(0.0, 0.0, 0.0), transform(0.003, 0.001, 0.0)
    object0 = {101: transform(-0.15, -0.02, 2.1), 202: transform(0.18, 0.03, 2.5)}
    object1 = {101: transform(-0.13, -0.018, 2.1), 202: transform(0.175, 0.025, 2.5)}
    object_delta = {key: object1[key] @ np.linalg.inv(object0[key]) for key in object0}

    sampler = AdaptiveSamplingConfig(
        sensor_time_constant_s=tau,
        sensor_eps_target=sensor_eps_target,
    )
    interpolator = PhysicsInterpolator(
        K, sampler_config=sampler, disocclusion_policy="hold",
        composite="next_primary", device=str(device), warp_backend="torch",
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    interpolation_start = time.perf_counter()
    warped_t, timestamps = interpolator.interpolate(
        frame0, depth0, seg0, frame1, depth1, seg1,
        camera0, camera1, object_delta, 0.0, interval_s,
        input_space="linear", object_world0=object0, object_world1=object1,
        return_torch=True,
    )
    torch.cuda.synchronize(device)
    interpolation_seconds = time.perf_counter() - interpolation_start
    if interpolator.last_sampling.samples != expected_samples:
        raise RuntimeError(f"sampler mismatch: {interpolator.last_sampling.samples} vs {expected_samples}")
    if tuple(warped_t.shape) != (1, expected_samples, height, width):
        raise RuntimeError(f"unexpected warp tensor shape: {tuple(warped_t.shape)}")

    # Independently compare selected projections against the NumPy reference.
    sample_indices = sorted({0, expected_samples // 2, expected_samples - 1})
    alpha_values = torch.as_tensor(
        [(index + 1) / expected_samples for index in sample_indices],
        dtype=torch.float32, device=device,
    )
    d0_t, d1_t = torch.as_tensor(depth0, device=device), torch.as_tensor(depth1, device=device)
    seg0_t, seg1_t = torch.as_tensor(seg0, device=device), torch.as_tensor(seg1, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
    camera0_t, camera1_t = (torch.as_tensor(x, dtype=torch.float32, device=device)
                            for x in (camera0, camera1))
    obj0_t = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in object0.items()}
    obj1_t = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in object1.items()}
    uv0_t, z0_t, valid0_t = project_endpoint_to_alpha_stack_torch(
        d0_t, seg0_t, K_t, camera0_t, camera0_t, camera1_t, alpha_values,
        0, obj0_t, obj1_t,
    )
    uv1_t, z1_t, valid1_t = project_endpoint_to_alpha_stack_torch(
        d1_t, seg1_t, K_t, camera1_t, camera0_t, camera1_t, alpha_values,
        1, obj0_t, obj1_t,
    )
    max_uv_error, max_z_error, valid_mismatches = 0.0, 0.0, 0
    max_warp_error, warp_valid_mismatches = 0.0, 0
    for position, sample_index in enumerate(sample_indices):
        alpha = (sample_index + 1) / expected_samples
        camera_alpha = interpolate_se3(camera0, camera1, alpha)
        ref0 = project_endpoint_to_alpha(
            depth0, seg0, K, camera0, camera_alpha, alpha, 0, object0, object1,
        )
        ref1 = project_endpoint_to_alpha(
            depth1, seg1, K, camera1, camera_alpha, alpha, 1, object0, object1,
        )
        for ref, uv_t, z_t, valid_t in (
            (ref0, uv0_t, z0_t, valid0_t), (ref1, uv1_t, z1_t, valid1_t),
        ):
            observed_uv = uv_t[position].detach().cpu().numpy()
            observed_z = z_t[position].detach().cpu().numpy()
            observed_valid = valid_t[position].detach().cpu().numpy()
            max_uv_error = max(max_uv_error, float(np.max(np.abs(ref[0] - observed_uv))))
            finite_z = np.isfinite(ref[1]) & np.isfinite(observed_z)
            if finite_z.any():
                max_z_error = max(max_z_error, float(np.max(np.abs(ref[1][finite_z] - observed_z[finite_z]))))
            valid_mismatches += int(np.count_nonzero(ref[2] != observed_valid))
        warped_ref, valid_ref, _ = depth_aware_bidirectional_warp_projected_diagnostics(
            frame0, depth0, ref0[0], ref0[1], ref0[2],
            frame1, depth1, ref1[0], ref1[1], ref1[2],
            composite="next_primary",
        )
        alpha_image = warped_t[0, sample_index].detach().cpu().numpy()
        valid_np = valid_ref
        if np.any(valid_np):
            max_warp_error = max(
                max_warp_error,
                float(np.max(np.abs(warped_ref[valid_np] - alpha_image[valid_np]))),
            )
        warp_valid_mismatches += int(np.count_nonzero(valid_ref != np.isfinite(alpha_image)))

    dvs = DvsEmulator(
        res=(height, width),
        config=DvsNoiseConfig(
            pos_thres=0.2, neg_thres=0.2, sigma_thres=0.03,
            cutoff_hz=300.0, leak_rate_hz=0.01, leak_jitter_fraction=0.1,
            noise_rate_cov_decades=0.1, shot_noise_rate_hz=0.001,
            refractory_period_s=0.0005, seed=42, radiance_input=True,
            response_mode="physical_log", radiometric_calibration_status="uncalibrated",
        ),
        device=str(device),
    )
    dvs_start = time.perf_counter()
    dvs.initialize(frame0, 0.0)
    event_counts = []
    for index, timestamp in enumerate(timestamps):
        chunk = dvs.generate_events(warped_t[0, index].detach().cpu().numpy(), timestamp)
        event_counts.append(len(chunk))
    torch.cuda.synchronize(device)
    dvs_seconds = time.perf_counter() - dvs_start
    if valid_mismatches or warp_valid_mismatches:
        raise RuntimeError("Torch/NumPy projection or warp validity masks disagree")
    if max_uv_error > 2e-3 or max_z_error > 2e-4 or max_warp_error > 2e-3:
        raise RuntimeError(
            f"backend parity tolerance exceeded: uv={max_uv_error}, z={max_z_error}, warp={max_warp_error}"
        )

    result = {
        "schema": "stage4_eventgenesis_backend_diagnostic_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
        "physical_gpu_id": __import__("os").environ.get("STAGE4_PHYSICAL_GPU", "unrecorded"),
        "resolution_wh": [width, height],
        "interval_s": interval_s,
        "sampler": {
            "cutoff_hz": sensor_cutoff_hz,
            "tau_s": tau,
            "sensor_eps_target": sensor_eps_target,
            "uncapped_samples": expected_uncapped,
            "capped_samples": expected_samples,
        },
        "backend": "existing torch projected-stack implementation on CUDA",
        "projected_warp_seconds": interpolation_seconds,
        "dvs_integration_seconds": dvs_seconds,
        "combined_interval_seconds": interpolation_seconds + dvs_seconds,
        "projected_samples_per_second": expected_samples / interpolation_seconds,
        "event_count": int(sum(event_counts)),
        "nonempty_event_subframes": int(sum(count > 0 for count in event_counts)),
        "projection_reference": {
            "sample_indices": sample_indices,
            "max_abs_uv_px": max_uv_error,
            "max_abs_z_m": max_z_error,
            "valid_pixel_mismatches": valid_mismatches,
            "max_abs_warp_radiance": max_warp_error,
            "warp_valid_pixel_mismatches": warp_valid_mismatches,
        },
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "output_shape": list(warped_t.shape),
        "timestamp_range_s": [float(timestamps[0]), float(timestamps[-1])],
        "event_count_sha256": hashlib.sha256(np.asarray(event_counts, dtype=np.int64).tobytes()).hexdigest(),
        "status": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
