#!/usr/bin/env python3
"""Run model-independent CPU correctness controls for the R1.5 candidate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_warp import interpolate_se3, project_endpoint_to_alpha
from benchmark.robot_closed_loop.diagnostics.r1_5_surface_raster_candidate import (
    compose_surface_warps,
    surface_raster_warp,
)


def identity_projection(height: int, width: int, depth: np.ndarray):
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    uv = np.stack((xx, yy), axis=0).astype(np.float32)
    z = np.asarray(depth, dtype=np.float32).copy()
    valid = np.isfinite(z) & (z > 1e-6)
    return uv, z, valid


def ideal_dvs(height: int, width: int) -> DvsEmulator:
    cfg = DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0, noise_rate_cov_decades=0.0,
        refractory_period_s=0.0, radiance_input=False, log_input=True,
        response_mode="physical_log", readout_model="ideal_readout", seed=1,
    )
    return DvsEmulator(res=(height, width), config=cfg, device="cpu")


def check_static_zero_events() -> dict:
    height, width = 32, 48
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    K = np.array([[55.0, 0, width / 2], [0, 55.0, height / 2], [0, 0, 1.0]])
    depth = (1.0 / (1.0 + 0.003 * xx)).astype(np.float32)
    labels = np.ones((height, width), dtype=np.int64)
    image = (0.7 + 0.2 * np.sin(xx * 0.17 + yy * 0.11)).astype(np.float32)
    uv, z, valid = identity_projection(height, width, depth)
    warped, wz, wl, visible, diag = surface_raster_warp(image, depth, labels, uv, z, valid, K)
    composed, common, request, compose_diag = compose_surface_warps(
        warped, wz, wl, visible, warped, wz, wl, visible, 0.5
    )
    compose_diag = {key: value for key, value in compose_diag.items() if key != "target_labels"}
    emu = ideal_dvs(height, width)
    emu.initialize(np.log(image), 0.0)
    events = [emu.generate_events(np.log(composed), 0.01),
              emu.generate_events(np.log(composed), 0.02)]
    count = sum(len(x) for x in events)
    return {
        "passed": count == 0 and bool(common.all()) and not bool(request.any()),
        "event_count": count,
        "max_abs_radiance_error": float(np.nanmax(np.abs(composed - image))),
        "render_request_fraction": float(request.mean()),
        "raster": diag,
        "compose": compose_diag,
    }


def check_bidirectional_micro_motion() -> dict:
    height, width = 32, 64
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    K = np.array([[50.0, 0, width / 2], [0, 50.0, height / 2], [0, 0, 1.0]])
    image = (0.5 + 0.5 * (xx % 2)).astype(np.float32)
    records = []
    for slope_name, depth in (
        ("decreasing", (1.0 / (1.0 + 0.005 * xx)).astype(np.float32)),
        ("increasing", (1.0 / (1.0 - 0.005 * xx)).astype(np.float32)),
    ):
        for dx in (-1e-5, 1e-5):
            uv = np.stack((xx + dx, yy), axis=0).astype(np.float32)
            labels = np.ones((height, width), dtype=np.int64)
            valid = np.ones((height, width), dtype=bool)
            got, _, _, visible, diag = surface_raster_warp(image, depth, labels, uv,
                                                            depth, valid, K)
            expected = ((1.0 - abs(dx)) * image[:, 1:-1]
                        + abs(dx) * (image[:, 2:] if dx < 0 else image[:, :-2]))
            actual = got[:, 1:-1]
            mask = visible[:, 1:-1]
            log_error = np.abs(np.log(np.maximum(actual, 1e-6))
                               - np.log(np.maximum(expected, 1e-6)))
            interior = log_error[mask]
            records.append({
                "depth_slope": slope_name,
                "projected_shift_px": dx,
                "valid_interior_pixels": int(mask.sum()),
                "max_abs_log_error": float(interior.max()) if interior.size else None,
                "pixels_over_log_0_2": int(np.count_nonzero(interior > 0.2)),
                "max_radiance_change": float(np.max(np.abs(actual[mask] - image[:, 1:-1][mask]))) if np.any(mask) else None,
                "raster": diag,
            })
    return {
        "passed": all(x["pixels_over_log_0_2"] == 0
                       and x["max_abs_log_error"] is not None
                       and x["max_abs_log_error"] < 1e-3 for x in records),
        "cases": records,
    }


def analytic_frame(camera_x: float, direction: int, xx: np.ndarray,
                   yy: np.ndarray, fx: float, plane_a: float):
    height, width = xx.shape
    cx, cy = width / 2, height / 2
    xnorm = (xx - cx) / fx
    xcam = direction * camera_x
    depth = (1.0 - plane_a * xcam) / (1.0 + plane_a * xnorm)
    source_u = xx + fx * xcam / depth
    image = (0.75 + 0.25 * np.sin(2.0 * np.pi * source_u / 4.0)).astype(np.float32)
    camera = np.eye(4, dtype=np.float64)
    camera[0, 3] = xcam
    return image, depth.astype(np.float32), camera


def check_analytic_weak_change() -> dict:
    height, width, fx, plane_a = 32, 64, 50.0, 0.25
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])
    labels = np.ones((height, width), dtype=np.int64)
    records = []
    for direction in (-1, 1):
        times = np.linspace(0.0, 0.04, 5)
        states = [analytic_frame(0.001 * t / 0.04, direction, xx, yy, fx, plane_a)
                  for t in times]
        emu = ideal_dvs(height, width)
        emu.initialize(np.log(states[0][0]), 0.0)
        all_events = []
        render_pixels = 0
        child_log_errors = []
        for idx in range(1, len(times)):
            i0, d0, T0 = states[idx - 1]
            i1, d1, T1 = states[idx]
            # First child at alpha=1/2 uses both endpoint surfaces.
            alpha = 0.5
            Talpha = interpolate_se3(T0, T1, alpha)
            p0 = project_endpoint_to_alpha(d0, labels, K, T0, Talpha, alpha, 0)
            p1 = project_endpoint_to_alpha(d1, labels, K, T1, Talpha, alpha, 1)
            a0 = surface_raster_warp(i0, d0, labels, *p0, K)
            a1 = surface_raster_warp(i1, d1, labels, *p1, K)
            child, visible, requests, cdiag = compose_surface_warps(
                a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3], alpha
            )
            # In this analytic control the exact intermediate radiance is
            # available in closed form; use it only to satisfy flagged border
            # render requests and count the requested pixels.
            exact_mid = analytic_frame(0.001 * ((idx - 0.5) / 4.0), direction,
                                       xx, yy, fx, plane_a)[0]
            render_pixels += int(requests.sum())
            child[requests] = exact_mid[requests]
            child_log_errors.append(float(np.nanmax(np.abs(np.log(child) - np.log(exact_mid)))))
            all_events.append(emu.generate_events(np.log(child), float((times[idx - 1] + times[idx]) / 2.0)))
            # Exact endpoint image is an observed keyframe, not reconstructed.
            all_events.append(emu.generate_events(np.log(i1), float(times[idx])))
        event_count = sum(len(x) for x in all_events)
        max_true_change = max(float(np.max(np.abs(np.log(states[k][0]) - np.log(states[0][0]))))
                              for k in range(1, len(states)))
        records.append({
            "camera_motion_sign": direction,
            "absolute_translation_m": 0.001,
            "max_true_log_change": max_true_change,
            "threshold": 0.2,
            "event_count": event_count,
            "render_requested_pixel_samples": render_pixels,
            "child_max_abs_log_error_vs_analytic": max(child_log_errors),
        })
    return {
        "passed": all(x["event_count"] == 0 and x["max_true_log_change"] < x["threshold"]
                       for x in records),
        "plane": "0.25*X+Z=1",
        "duration_s": 0.04,
        "source_cadence_hz": 100,
        "subsamples_per_interval": 2,
        "directions": records,
    }


def check_luminance_ramps() -> dict:
    height, width = 16, 20
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    K = np.array([[40.0, 0, width / 2], [0, 40.0, height / 2], [0, 0, 1.0]])
    depth = np.ones((height, width), dtype=np.float32)
    labels = np.ones((height, width), dtype=np.int64)
    uv, z, valid = identity_projection(height, width, depth)
    rows = []
    for delta in (0.5, -0.5):
        start = np.ones((height, width), dtype=np.float32)
        end = np.full((height, width), np.exp(delta), dtype=np.float32)
        w0 = surface_raster_warp(start, depth, labels, uv, z, valid, K)
        w1 = surface_raster_warp(end, depth, labels, uv, z, valid, K)
        emu = ideal_dvs(height, width)
        emu.initialize(np.zeros((height, width), dtype=np.float32), 0.0)
        events = []
        for k, t in enumerate((0.002, 0.004, 0.006, 0.008, 0.010), start=1):
            frame, _, request, _ = compose_surface_warps(
                w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3], k / 5.0
            )
            if request.any():
                raise AssertionError("static ramp unexpectedly requested a render")
            events.append(emu.generate_events(np.log(frame), t))
        concatenated = np.concatenate([x for x in events if len(x)], axis=0)
        expected_polarity = 1 if delta > 0 else -1
        expected_times = np.array([0.004, 0.008])
        observed_times = np.sort(concatenated[:, 0])
        rows.append({
            "log_radiance_delta": delta,
            "event_count": int(len(concatenated)),
            "unique_polarities": np.unique(concatenated[:, 3]).astype(int).tolist(),
            "expected_polarity": expected_polarity,
            "event_time_min_s": float(observed_times.min()),
            "event_time_max_s": float(observed_times.max()),
            "expected_threshold_times_s": expected_times.tolist(),
            "time_match_max_abs_s": float(np.max(np.abs(observed_times - np.repeat(expected_times, height * width)))),
        })
    return {
        "passed": all(x["event_count"] == height * width * 2
                       and x["unique_polarities"] == [x["expected_polarity"]]
                       and x["time_match_max_abs_s"] < 2e-7 for x in rows),
        "threshold": 0.2,
        "ramp_duration_s": 0.010,
        "ramps": rows,
    }


def check_foreground_occlusion() -> dict:
    height, width = 32, 48
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    K = np.array([[48.0, 0, width / 2], [0, 48.0, height / 2], [0, 0, 1.0]])
    depth_bg = np.full((height, width), 2.0, dtype=np.float32)
    labels_bg = np.ones((height, width), dtype=np.int64)
    image_bg = np.full((height, width), 0.1, dtype=np.float32)
    fg = (slice(8, 24), slice(12, 28))
    uv_bg, z_bg, valid_bg = identity_projection(height, width, depth_bg)
    bg = surface_raster_warp(image_bg, depth_bg, labels_bg, uv_bg, z_bg, valid_bg, K)
    depth_fg = np.ones((height, width), dtype=np.float32)
    labels_fg = np.full((height, width), 2, dtype=np.int64)
    image_fg = np.full((height, width), 0.9, dtype=np.float32)
    uv_fg, z_fg, valid_fg = identity_projection(height, width, depth_fg)
    valid_fg[:] = False
    valid_fg[fg] = True
    uv_fg[0][fg] += 4.0
    foreground = surface_raster_warp(
        image_fg, depth_fg, labels_fg, uv_fg, z_fg, valid_fg, K
    )
    out, out_valid, request, diagnostics = compose_surface_warps(
        bg[0], bg[1], bg[2], bg[3],
        foreground[0], foreground[1], foreground[2], foreground[3], 0.5,
    )
    out_labels = diagnostics["target_labels"]
    # The translated foreground overlaps the background at this pixel and
    # must win by actual interpolated surface depth.
    point = (16, 20)
    passed = bool(out_valid[point] and bg[3][point] and foreground[3][point]
                  and out_labels[point] == 2 and foreground[1][point] < bg[1][point]
                  and out[point] > 0.8)
    return {
        "passed": passed,
        "overlap_pixel_yx": list(point),
        "winner_label": int(out_labels[point]),
        "winner_depth": float(foreground[1][point]),
        "winner_radiance": float(out[point]),
        "depth_background_m": float(bg[1][point]),
        "raster_background": bg[4],
        "raster_foreground": foreground[4],
        "composition": {key: value for key, value in diagnostics.items() if key != "target_labels"},
        "render_request_fraction": float(request.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    results = {
        "candidate": "diagnostic_continuous_surface_triangles_v1",
        "implementation_file": "benchmark/robot_closed_loop/diagnostics/r1_5_surface_raster_candidate.py",
        "frozen_core_modified": False,
        "device": "cpu",
    }
    controls = [
        ("static_zero_events", check_static_zero_events),
        ("bidirectional_micro_motion", check_bidirectional_micro_motion),
        ("analytic_weak_change_plane", check_analytic_weak_change),
        ("smooth_luminance_ramp", check_luminance_ramps),
        ("foreground_background_occlusion", check_foreground_occlusion),
    ]
    for name, function in controls:
        tick = time.perf_counter()
        result = function()
        result["wall_time_s"] = time.perf_counter() - tick
        results[name] = result
    results["passed"] = all(results[name]["passed"] for name, _ in controls)
    results["total_wall_time_s"] = time.perf_counter() - start
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
