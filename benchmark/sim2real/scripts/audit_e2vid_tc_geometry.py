#!/usr/bin/env python3
"""Numerical Stage-5 regression checks for the E2VID TC coordinate contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from benchmark.sim2real.compat.e2vid_temporal_consistency import (  # noqa: E402
    COMPAT_VERSION,
    temporal_consistency_loss_align_corners_true,
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def independent_warp(image: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Small explicit oracle; deliberately does not call compatibility helpers."""
    b, _, h, w = image.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=displacement.dtype),
        torch.arange(w, device=image.device, dtype=displacement.dtype),
        indexing="ij",
    )
    source_x = x.unsqueeze(0) + displacement[:, 0]
    source_y = y.unsqueeze(0) + displacement[:, 1]
    normalized = torch.stack(
        (2 * source_x / (w - 1) - 1, 2 * source_y / (h - 1) - 1), dim=-1
    )
    return F.grid_sample(
        image, normalized, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def _assert_close(name: str, error: float, tolerance: float, results: dict) -> None:
    passed = torch.isfinite(torch.tensor(error)).item() and error <= tolerance
    results[name] = {"max_abs_error": error, "tolerance": tolerance, "status": "pass" if passed else "fail"}
    if not passed:
        raise AssertionError(f"{name}: max_abs_error {error} > tolerance {tolerance}")


def run(device: torch.device) -> dict:
    h, w = 9, 17  # deliberately non-square; normalized integer grids are binary-exact
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    xy = (1.0 + yy.float() * 100.0 + xx.float()).to(device)[None, None]
    zeros = torch.zeros((1, 2, h, w), dtype=torch.float32, device=device)
    out: dict = {"device": str(device), "shape_hw": [h, w], "checks": {}}

    # Zero displacement must be an exact identity on a nontrivial pixel field.
    loss0, maps0 = temporal_consistency_loss_align_corners_true(
        xy, xy, xy, xy, zeros, output_images=True
    )
    identity_error = float((maps0["image0_warped_to1"] - xy).abs().max().cpu())
    _assert_close("zero_flow_identity", identity_error, 1e-6, out["checks"])
    _assert_close("zero_flow_tc_loss", float(loss0.cpu()), 1e-6, out["checks"])
    oracle_error = float((maps0["image0_warped_to1"] - independent_warp(xy, zeros)).abs().max().cpu())
    _assert_close("independent_sampling_oracle", oracle_error, 1e-6, out["checks"])

    # A +2,-1 forward image translation requires a -2,+1 backward sampling map.
    backward = torch.zeros_like(zeros)
    backward[:, 0].fill_(-2.0)
    backward[:, 1].fill_(1.0)
    target = torch.zeros_like(xy)
    target[:, :, : h - 1, 2:] = xy[:, :, 1:, : w - 2]
    translated_loss, translated_maps = temporal_consistency_loss_align_corners_true(
        xy, target, xy, target, backward, output_images=True
    )
    translated_warp = translated_maps["image0_warped_to1"]
    translation_region = translated_warp[:, :, : h - 1, 2:]
    target_region = target[:, :, : h - 1, 2:]
    translation_error = float((translation_region - target_region).abs().max().cpu())
    _assert_close("known_integer_translation_and_sign", translation_error, 1e-5, out["checks"])
    _assert_close("translated_tc_loss", float(translated_loss.cpu()), 1e-6, out["checks"])

    # Bilinear subpixel samples of a planar ramp have a closed-form solution.
    ramp = (0.2 + 0.3 * xx.float() + 0.5 * yy.float()).to(device)[None, None]
    subpixel = torch.zeros_like(zeros)
    subpixel[:, 0].fill_(0.25)
    subpixel[:, 1].fill_(-0.375)
    subpixel_warp = independent_warp(ramp, subpixel)
    expected = ramp + 0.3 * 0.25 + 0.5 * (-0.375)
    subpixel_error = float((subpixel_warp[:, :, 1:-1, 1:-1] - expected[:, :, 1:-1, 1:-1]).abs().max().cpu())
    _assert_close("subpixel_displacement", subpixel_error, 2e-6, out["checks"])

    # Integer displacement at the right border defines exact in-bounds support.
    right = torch.zeros_like(zeros)
    right[:, 0].fill_(2.0)
    boundary_maps = temporal_consistency_loss_align_corners_true(
        xy, xy, xy, xy, right, output_images=True
    )[1]
    in_bounds = torch.zeros((h, w), dtype=torch.bool, device=device)
    in_bounds[:, : w - 2] = True  # source x = output x + 2 <= W-1
    boundary_warp = boundary_maps["image0_warped_to1"][0, 0]
    expected_boundary = torch.zeros_like(boundary_warp)
    expected_boundary[:, : w - 2] = xy[0, 0, :, 2:]
    boundary_error = float((boundary_warp[in_bounds] - expected_boundary[in_bounds]).abs().max().cpu())
    # The expression above is width-aligned but includes all rows; check the two
    # known boundary samples independently to make the support rule explicit.
    left_of_edge_error = float((boundary_warp[:, w - 3] - xy[0, 0, :, w - 1]).abs().max().cpu())
    outside_max = float(boundary_warp[:, w - 2:].abs().max().cpu())
    support_count = int(in_bounds.sum().cpu())
    _assert_close("valid_boundary_last_in_bounds_pixel", left_of_edge_error, 1e-6, out["checks"])
    _assert_close("invalid_boundary_zero_padding", outside_max, 1e-6, out["checks"])
    out["checks"]["boundary_support"] = {
        "expected_valid_pixels": support_count,
        "expected_valid_rule": "x + dx <= W-1; y + dy in [0,H-1]",
        "invalid_right_columns": 2,
        "all_pixels_finite": bool(torch.isfinite(boundary_warp).all().cpu()),
        "status": "pass" if boundary_error < 1e-6 else "fail",
    }
    if not torch.isfinite(boundary_warp).all():
        raise AssertionError("boundary warp contains non-finite values")

    # Recreate the former implicit-default path to prove the fixture detects it.
    checker = ((xx + yy) % 2).float().to(device)[None, None]
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32), indexing="ij",
    )
    old_grid = torch.stack((2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1), dim=-1)[None]
    old_default = F.grid_sample(checker, old_grid)  # intentionally exercises PyTorch default
    old_default_error = float((old_default - checker).abs().max().cpu())
    out["old_default_false_identity_error"] = old_default_error
    if old_default_error <= 0.1:
        raise AssertionError("regression fixture failed to distinguish the old implicit default")

    out["status"] = "pass"
    out["compat_version"] = COMPAT_VERSION
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    result = {
        "schema": "e2vid_tc_geometry_preflight_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "compat_sha256": file_sha256(REPO / "benchmark/sim2real/compat/e2vid_temporal_consistency.py"),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        **run(device),
    }
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
