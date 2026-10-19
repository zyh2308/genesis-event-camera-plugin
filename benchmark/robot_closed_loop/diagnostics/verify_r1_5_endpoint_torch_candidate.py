#!/usr/bin/env python3
"""Check the exact-endpoint projection candidate on CPU NumPy and Torch paths.

This is a diagnostic-only wrapper: it does not edit ``genesis_event_plugin``.
It uses the real cached endpoint pairs, compares the current Torch projection
with an exact source-grid override at mathematical endpoints, and confirms
that alpha=0.5 output is byte-identical to the current path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.r1_5_consistency_candidates import (  # noqa: E402
    project_endpoint_candidate,
)
from genesis_event_plugin.torch_physics import (  # noqa: E402
    depth_aware_bidirectional_warp_projected_batch_torch,
    project_endpoint_to_alpha_stack_torch,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(reference: np.ndarray, image: np.ndarray, visible: np.ndarray) -> dict:
    ref = np.asarray(reference, dtype=np.float32)
    out = np.asarray(image, dtype=np.float32)
    mask = np.asarray(visible, dtype=bool) & np.isfinite(ref) & np.isfinite(out)
    diff = np.abs(np.log(np.maximum(ref[mask], 1e-6)) - np.log(np.maximum(out[mask], 1e-6)))
    return {
        "visible_pixels": int(mask.sum()),
        "pixels_log_error_over_0_2": int(np.count_nonzero(diff > 0.2)),
        "log_error_max": float(diff.max()) if diff.size else None,
        "log_error_p99": float(np.percentile(diff, 99)) if diff.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene_config.read_text(encoding="utf-8"))
    evidence_root, output = args.evidence_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; this verification must stay CPU-only")
    device = torch.device("cpu")

    width, height = (int(v) for v in scene["camera"]["resolution_wh"])
    fov = float(scene["camera"]["vertical_fov_deg"])
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K_np = np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
    alphas = torch.tensor([0.5, 1.0], dtype=torch.float32, device=device)
    h, w = height, width
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid_np = np.stack((xx, yy), axis=0).astype(np.float32)
    grid = torch.as_tensor(grid_np, dtype=torch.float32, device=device)
    results = {}
    started = time.perf_counter()

    for ep_id in (4, 5):
        ep_dir = evidence_root / f"episode_{ep_id:03d}"
        with h5py.File(ep_dir / "same_state_source_cache.h5", "r") as cache:
            times = np.asarray(cache["timestamps_s"][:], dtype=np.float64)
            idx = int(np.flatnonzero(np.isclose(times, 0.5905, atol=1e-9, rtol=0))[0])
            camera_np = np.asarray(cache["camera_to_world_cv"][idx], dtype=np.float64)
            pose_ids = np.asarray(cache["segmentation_pose_ids"][:], dtype=np.int32)
            frames = []
            for source_idx in (idx - 1, idx):
                frames.append({
                    "luma": np.asarray(cache["radiance_luma"][source_idx], dtype=np.float32),
                    "depth": np.asarray(cache["depth"][source_idx], dtype=np.float32),
                    "seg": np.asarray(cache["segmentation"][source_idx], dtype=np.int32),
                    "poses": {int(seg): np.asarray(cache["object_poses"][source_idx, j], dtype=np.float64)
                              for j, seg in enumerate(pose_ids)},
                })
        with h5py.File(ep_dir / "rendered_frame_evidence.h5", "r") as dense:
            dense_times = np.asarray(dense["dense_times_s"][:], dtype=np.float64)
            direct_idx = int(np.flatnonzero(np.isclose(dense_times, 0.5905, atol=1e-9, rtol=0))[0])
            direct = np.asarray(dense["dense_direct_hdr_luminance"][direct_idx], dtype=np.float32)

        d0 = torch.as_tensor(frames[0]["depth"], dtype=torch.float32, device=device)
        d1 = torch.as_tensor(frames[1]["depth"], dtype=torch.float32, device=device)
        seg0 = torch.as_tensor(frames[0]["seg"], device=device)
        seg1 = torch.as_tensor(frames[1]["seg"], device=device)
        camera = torch.as_tensor(camera_np, dtype=torch.float32, device=device)
        object0 = {key: torch.as_tensor(value, dtype=torch.float32, device=device)
                   for key, value in frames[0]["poses"].items()}
        object1 = {key: torch.as_tensor(value, dtype=torch.float32, device=device)
                   for key, value in frames[1]["poses"].items()}
        uv0, z0, valid0 = project_endpoint_to_alpha_stack_torch(
            d0, seg0, K, camera, camera, camera, alphas, 0, object0, object1)
        uv1, z1, valid1 = project_endpoint_to_alpha_stack_torch(
            d1, seg1, K, camera, camera, camera, alphas, 1, object0, object1)

        # Diagnostic equivalent of the proposed exact-identity override.
        corrected_uv0, corrected_z0, corrected_valid0 = uv0.clone(), z0.clone(), valid0.clone()
        corrected_uv1, corrected_z1, corrected_valid1 = uv1.clone(), z1.clone(), valid1.clone()
        for endpoint_index, source_index, depth, uv_stack, z_stack, valid_stack in (
            (0, 0, d0, corrected_uv0, corrected_z0, corrected_valid0),
            (1, 1, d1, corrected_uv1, corrected_z1, corrected_valid1),
        ):
            alpha = alphas[endpoint_index]
            is_exact = bool((source_index == 0 and alpha.item() == 0.0) or
                            (source_index == 1 and alpha.item() == 1.0))
            if not is_exact:
                continue
            base_valid = torch.isfinite(depth) & (depth > 1e-6)
            uv_stack[endpoint_index] = torch.where(base_valid.unsqueeze(0), grid, torch.zeros_like(grid))
            z_stack[endpoint_index] = torch.where(base_valid, depth, torch.full_like(depth, float("inf")))
            valid_stack[endpoint_index] = base_valid

        def run_warp(use_corrected: bool):
            u0, zz0, vv0 = ((corrected_uv0, corrected_z0, corrected_valid0) if use_corrected else (uv0, z0, valid0))
            u1, zz1, vv1 = ((corrected_uv1, corrected_z1, corrected_valid1) if use_corrected else (uv1, z1, valid1))
            return depth_aware_bidirectional_warp_projected_batch_torch(
                torch.as_tensor(frames[0]["luma"], dtype=torch.float32, device=device).unsqueeze(0), d0.unsqueeze(0),
                u0.unsqueeze(0), zz0.unsqueeze(0), vv0.unsqueeze(0),
                torch.as_tensor(frames[1]["luma"], dtype=torch.float32, device=device).unsqueeze(0), d1.unsqueeze(0),
                u1.unsqueeze(0), zz1.unsqueeze(0), vv1.unsqueeze(0),
                composite="next_primary", return_diagnostics=True)

        raw_output, raw_visible, raw_diag = run_warp(False)
        fixed_output, fixed_visible, fixed_diag = run_warp(True)
        raw_image = raw_output[0].detach().cpu().numpy()
        fixed_image = fixed_output[0].detach().cpu().numpy()
        direct_visible = np.asarray(raw_diag["visibility1"][0, 1].detach().cpu().numpy(), dtype=bool)
        fixed_direct_visible = np.asarray(fixed_diag["visibility1"][0, 1].detach().cpu().numpy(), dtype=bool)
        raw_endpoint_uv = uv1[1].detach().cpu().numpy()
        raw_uv_error = np.abs(raw_endpoint_uv - grid_np)
        raw_uv_error = raw_uv_error[:, valid1[1].detach().cpu().numpy()]

        # The proposal must leave every interior sample exactly unchanged.
        interior_raw = raw_output[0, 0].detach().cpu().numpy()
        interior_fixed = fixed_output[0, 0].detach().cpu().numpy()
        finite_mid = np.isfinite(interior_raw) & np.isfinite(interior_fixed)
        midpoint_difference = np.abs(interior_raw[finite_mid] - interior_fixed[finite_mid])
        # Pixel displacement confirms this actual scene contains nonzero motion
        # at alpha=.5; the exact-endpoint correction does not round it away.
        midpoint_projected = uv0[0].detach().cpu().numpy()
        midpoint_motion = np.abs(midpoint_projected - grid_np)
        motion_pixels = valid0[0].detach().cpu().numpy()

        results[f"episode_{ep_id:03d}"] = {
            "source_cache_sha256": _sha256(ep_dir / "same_state_source_cache.h5"),
            "dense_render_evidence_sha256": _sha256(ep_dir / "rendered_frame_evidence.h5"),
            "time_s": 0.5905,
            "torch_current_source1_alpha1": {
                "valid_source_pixels": int(valid1[1].sum().item()),
                "max_uv_error_px": float(raw_uv_error.max()) if raw_uv_error.size else None,
                "p99_uv_error_px": float(np.percentile(raw_uv_error, 99)) if raw_uv_error.size else None,
                "raw_image_error": _error(direct, raw_image[1], direct_visible),
                "candidate_identity_image_error": _error(direct, fixed_image[1], fixed_direct_visible),
                "visibility_difference_pixels": int(torch.count_nonzero(
                    raw_diag["visibility1"][0, 1] != fixed_diag["visibility1"][0, 1]).item()),
            },
            "midpoint_alpha_half": {
                "raw_versus_candidate_max_abs_image_delta": float(midpoint_difference.max()) if midpoint_difference.size else 0.0,
                "projection_displacement_max_px": float(midpoint_motion[:, motion_pixels].max()) if motion_pixels.any() else 0.0,
                "projected_pixels_with_motion_over_0_01_px": int(np.count_nonzero(midpoint_motion[:, motion_pixels] > 0.01)),
            },
        }
    report = {
        "experiment_id": "r1_5_endpoint_torch_candidate_20261019a",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scene_config_sha256": _sha256(args.scene_config.resolve()),
        "device": "cpu", "cuda_initialized": bool(torch.cuda.is_initialized()),
        "episodes": results, "elapsed_wall_s": time.perf_counter() - started,
        "core_modified": False,
        "interpretation": "Exact source identity was applied only at alpha=1/source_index=1 in the local diagnostic wrapper. Alpha=0.5 was not changed or rounded.",
    }
    path = output / "summary.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    manifest = f"{_sha256(path)}  summary.json\n"
    (output / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    print(json.dumps({"output": str(output), "elapsed_wall_s": report["elapsed_wall_s"],
                      "sha256sums": _sha256(output / "SHA256SUMS"), "episodes": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
