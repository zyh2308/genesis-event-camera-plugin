#!/usr/bin/env python3
"""Trace two active warp rules against the archived center/+55 mm clips.

CPU-only diagnostic. It reads cached endpoint HDR/depth/segmentation/poses and
the already saved 1 kHz Direct-HDR frames. It does not initialize Genesis,
advance physics, run learned inference, or modify the frozen core.

The alternatives are deliberately local diagnostic interventions:
  1. compare endpoint z-buffers before `next_primary` selects a source;
  2. resolve each splat front-to-back using bilinear footprint as coverage,
     so an arbitrarily small near-surface weight does not erase farther
     coverage before normalization.

Neither alternative is promoted as a production renderer without the recorded
static, occlusion, and subpixel controls and follow-up review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.dvs_emulator import DvsEmulator
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator
from genesis_event_plugin.physics_warp import (
    _splat_projected_diagnostics,
    depth_aware_bidirectional_warp_projected_diagnostics,
    project_endpoint_to_alpha,
)


DEFAULT_CACHE = ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
DEFAULT_OUTPUT = ROOT / "benchmark/robot_closed_loop/results/r1_5_pixel_rule_attribution_20261030a"
DEPTH_TIE_M = 1e-5  # Match the active splat z-buffer tie rule for layer grouping.
LOG_EPS = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime_module_record(module_name: str) -> dict:
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    return {"imported_from": str(module_path), "sha256": sha256(module_path)}


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def log_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.log(np.maximum(np.asarray(a, dtype=np.float32), LOG_EPS)) -
                  np.log(np.maximum(np.asarray(b, dtype=np.float32), LOG_EPS)))


def collect_contributions(image, source_depth, target_uv, target_z, source_valid=None):
    """Return all in-bounds bilinear taps, retaining source provenance."""
    image = np.asarray(image, dtype=np.float32)
    source_depth = np.asarray(source_depth, dtype=np.float32)
    target_uv = np.asarray(target_uv, dtype=np.float32)
    target_z = np.asarray(target_z, dtype=np.float32)
    H, W = source_depth.shape
    src = image[..., None] if image.ndim == 2 else image
    valid = np.isfinite(src).all(axis=-1) & np.isfinite(source_depth) & (source_depth > 1e-6)
    valid &= np.isfinite(target_z) & (target_z > 1e-6)
    if source_valid is not None:
        valid &= np.asarray(source_valid, dtype=bool)
    tx, ty = target_uv[0], target_uv[1]
    x0, y0 = np.floor(tx).astype(np.int32), np.floor(ty).astype(np.int32)
    wx, wy = tx - x0, ty - y0
    source_index = np.arange(H * W, dtype=np.int32).reshape(H, W)
    pieces = []
    for ox, oy, weight in ((0, 0, (1 - wx) * (1 - wy)),
                           (1, 0, wx * (1 - wy)),
                           (0, 1, (1 - wx) * wy),
                           (1, 1, wx * wy)):
        x, y = x0 + ox, y0 + oy
        keep = valid & (weight > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if np.any(keep):
            pieces.append({
                "target": (y[keep] * W + x[keep]).astype(np.int32),
                "source": source_index[keep],
                "weight": weight[keep].astype(np.float32),
                "z": target_z[keep].astype(np.float32),
                "value": src[keep].astype(np.float32),
            })
    if not pieces:
        return {k: np.empty((0,), dtype=np.int32 if k in ("target", "source") else np.float32)
                for k in ("target", "source", "weight", "z", "value")}
    return {k: np.concatenate([p[k] for p in pieces], axis=0) for k in pieces[0]}


def coverage_splat(image, source_depth, target_uv, target_z, source_valid=None):
    """Front-to-back footprint resolve; unlike the legacy z-buffer it retains
    farther coverage behind subpixel near-surface contributions.

    Contributions within the active 1e-5 m z tie band are one layer. Layer
    coverage is the sum of bilinear weights capped by remaining pixel coverage.
    If total support is below one, values are normalized over observed support,
    matching the old helper's no-background assumption.
    """
    image = np.asarray(image, dtype=np.float32)
    scalar = image.ndim == 2
    H, W = np.asarray(source_depth).shape
    contributions = collect_contributions(image, source_depth, target_uv, target_z, source_valid)
    out = np.zeros((H * W, 1 if scalar else image.shape[-1]), dtype=np.float32)
    visible = np.zeros(H * W, dtype=bool)
    weight_total = np.zeros(H * W, dtype=np.float32)
    if not contributions["target"].size:
        return (out.reshape(H, W) if scalar else out.reshape(H, W, -1)), visible.reshape(H, W), weight_total.reshape(H, W)
    pix, z = contributions["target"], contributions["z"]
    # Stable pixel then near-to-far ordering. Source order resolves equal z
    # deterministically; equal-depth values are averaged as one coverage layer.
    order = np.lexsort((contributions["source"], z, pix))
    sorted_pix = pix[order]
    starts = np.r_[0, np.flatnonzero(sorted_pix[1:] != sorted_pix[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    for begin, end in zip(starts, ends):
        idx = order[begin:end]
        local_z = contributions["z"][idx]
        local_w = contributions["weight"][idx]
        local_v = contributions["value"][idx]
        loc = int(sorted_pix[begin])
        remaining = 1.0
        total_coverage = 0.0
        accum = np.zeros(out.shape[1], dtype=np.float64)
        k = 0
        while k < len(idx) and remaining > 1e-8:
            layer_end = k + 1
            while layer_end < len(idx) and float(local_z[layer_end] - local_z[k]) <= DEPTH_TIE_M:
                layer_end += 1
            layer_w = local_w[k:layer_end].astype(np.float64)
            layer_values = local_v[k:layer_end].astype(np.float64)
            raw_coverage = float(layer_w.sum())
            if raw_coverage > 0:
                alpha = min(raw_coverage, remaining)
                layer_mean = (layer_values * layer_w[:, None]).sum(axis=0) / raw_coverage
                accum += alpha * layer_mean
                total_coverage += alpha
                remaining -= alpha
            k = layer_end
        if total_coverage > 1e-8:
            out[loc] = accum / total_coverage
            weight_total[loc] = total_coverage
            visible[loc] = True
    shaped = out.reshape(H, W, out.shape[-1])
    return (shaped[..., 0] if scalar else shaped), visible.reshape(H, W), weight_total.reshape(H, W)


def hold_fill(frame, valid, alpha, I0, I1):
    out = np.asarray(frame, dtype=np.float32).copy()
    invalid = ~np.asarray(valid, dtype=bool)
    out[invalid] = (I0 if alpha < 0.5 else I1)[invalid]
    return out


def nearest_endpoint_composite(a0, d0, a1, d1):
    v0, v1 = d0["visibility"], d1["visibility"]
    valid = v0 | v1
    out = np.full_like(a0, np.nan, dtype=np.float32)
    choose1 = v1 & (~v0 | (d1["z_buffer"] <= d0["z_buffer"]))
    out[choose1] = a1[choose1]
    out[~choose1 & v0] = a0[~choose1 & v0]
    return out, valid, choose1


def endpoint_diagnostics(image, depth, uv, z, valid):
    output, diag = _splat_projected_diagnostics(image, depth, uv, z, valid)
    contrib = collect_contributions(image, depth, uv, z, valid)
    H, W = depth.shape
    winner_coverage = np.zeros(H * W, dtype=np.float32)
    next_coverage = np.zeros(H * W, dtype=np.float32)
    min_z = np.full(H * W, np.inf, dtype=np.float32)
    pix = contrib["target"]
    if pix.size:
        np.minimum.at(min_z, pix, contrib["z"])
        near = contrib["z"] <= min_z[pix] + DEPTH_TIE_M
        np.add.at(winner_coverage, pix[near], contrib["weight"][near])
        np.add.at(next_coverage, pix[~near], contrib["weight"][~near])
    return output, diag, contrib, winner_coverage.reshape(H, W), next_coverage.reshape(H, W)


def contributors_at(contrib, target_index, zbuf_value, seg, depth_source):
    pick = np.flatnonzero(contrib["target"] == target_index)
    rows = []
    for i in pick:
        source = int(contrib["source"][i])
        sy, sx = divmod(source, seg.shape[1])
        rows.append({
            "source_xy": [sx, sy],
            "source_seg_id": int(seg[sy, sx]),
            "source_depth_m": float(depth_source[sy, sx]),
            "projected_z_m": float(contrib["z"][i]),
            "bilinear_weight": float(contrib["weight"][i]),
            "source_luma": float(contrib["value"][i, 0]),
            "survives_active_zbuffer": bool(contrib["z"][i] <= zbuf_value + DEPTH_TIE_M),
        })
    rows.sort(key=lambda r: (r["projected_z_m"], -r["bilinear_weight"]))
    return rows[:16]


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_candidate(base: np.ndarray, candidate: np.ndarray, reference: np.ndarray) -> dict:
    base_err, cand_err = log_error(base, reference), log_error(candidate, reference)
    bad = base_err > 0.2
    resolved = bad & (cand_err <= 0.2)
    return {
        "bad_pixel_time_baseline_over_0p2": int(bad.sum()),
        "resolved_bad_pixel_time": int(resolved.sum()),
        "resolved_fraction_of_baseline_bad": float(resolved.sum() / max(int(bad.sum()), 1)),
        "improved_bad_pixel_time": int((bad & (cand_err < base_err)).sum()),
        "worsened_bad_pixel_time": int((bad & (cand_err > base_err)).sum()),
        "absolute_log_error_sum_on_baseline_bad": float(base_err[bad].sum()),
        "absolute_log_error_sum_after_on_same_pixels": float(cand_err[bad].sum()),
        "bad_pixel_log_error_reduction_fraction": float(1.0 - cand_err[bad].sum() / max(float(base_err[bad].sum()), 1e-12)),
        "all_pixel_time_mae_log": float(cand_err.mean()),
        "all_pixel_time_p95_log": float(np.percentile(cand_err, 95)),
        "all_pixel_time_max_log": float(cand_err.max()),
        "bad_pixel_time_candidate_over_0p2": int((cand_err > 0.2).sum()),
    }


def run_episode(ep: str, cache_root: Path):
    episode_dir = cache_root / f"episode_{ep}"
    cache_path = episode_dir / "same_state_source_cache.h5"
    evidence_path = episode_dir / "rendered_frame_evidence.h5"
    with h5py.File(cache_path, "r") as f:
        times = np.asarray(f["timestamps_s"][:], dtype=np.float64)
        src = {name: np.asarray(f[name][:]) for name in (
            "radiance_luma", "depth", "segmentation", "camera_to_world_cv", "object_poses",
            "segmentation_pose_ids")}
        width, height = map(int, f.attrs["camera_resolution_wh"])
        fov = float(f.attrs["vertical_fov_deg"])
    with h5py.File(evidence_path, "r") as f:
        direct_times = np.asarray(f["dense_times_s"][:], dtype=np.float64)
        direct = np.asarray(f["dense_direct_hdr_luminance"][:], dtype=np.float32)
        direct_seg = np.asarray(f["dense_direct_segmentation"][:], dtype=np.int32)
        stored = np.asarray(f["ours_intermediate_luminance"][:], dtype=np.float32)
        stored_unknown = np.asarray(f["ours_intermediate_unknown_mask"][:], dtype=bool)

    left = int(np.flatnonzero(np.isclose(times, 0.5805, atol=1e-9, rtol=0))[0])
    right = int(np.flatnonzero(np.isclose(times, 0.5905, atol=1e-9, rtol=0))[0])
    if right != left + 1:
        raise RuntimeError("expected adjacent cached endpoints at 0.5805/0.5905 s")
    focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    object0 = {int(seg_id): src["object_poses"][left, j]
               for j, seg_id in enumerate(src["segmentation_pose_ids"])}
    object1 = {int(seg_id): src["object_poses"][right, j]
               for j, seg_id in enumerate(src["segmentation_pose_ids"])}
    T0, T1 = src["camera_to_world_cv"][left], src["camera_to_world_cv"][right]
    I0, I1 = src["radiance_luma"][left], src["radiance_luma"][right]

    interpolator = PhysicsInterpolator(
        K, sampler_config=AdaptiveSamplingConfig(min_samples=10, max_samples=10),
        disocclusion_policy="hold", composite="next_primary", warp_backend="numpy")
    reproduced, sample_times = interpolator.interpolate(
        I0, src["depth"][left], src["segmentation"][left],
        I1, src["depth"][right], src["segmentation"][right],
        T0, T1, None, float(times[left]), float(times[right]),
        input_space="linear", object_world0=object0, object_world1=object1)
    reproduced = np.asarray(reproduced, dtype=np.float32)
    replay_delta = np.abs(reproduced - stored)
    if replay_delta.max() > 1e-6:
        raise RuntimeError(f"{ep}: active NumPy path differs from archived intermediate by {replay_delta.max()}")
    if len(sample_times) != len(direct_times) or not np.allclose(sample_times, direct_times, atol=1e-8, rtol=0):
        raise RuntimeError(f"{ep}: the fixed 10-sample diagnostic must align with saved direct times")

    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    grid = np.stack((grid_x, grid_y), axis=0).astype(np.float32)
    baseline_frames, cross_frames, coverage_frames, exact_end_frames = [], [], [], []
    per_frame_rows, trace_context = [], []
    for qi, t in enumerate(direct_times):
        alpha = float((t - times[left]) / (times[right] - times[left]))
        p0 = project_endpoint_to_alpha(
            src["depth"][left], src["segmentation"][left], K, T0,
            # Camera is fixed in these clips, but use the endpoint contract.
            T0, alpha, 0, object0, object1, None)
        p1 = project_endpoint_to_alpha(
            src["depth"][right], src["segmentation"][right], K, T1,
            T0, alpha, 1, object0, object1, None)
        a0, d0, contrib0, c0, f0 = endpoint_diagnostics(I0, src["depth"][left], *p0)
        a1, d1, contrib1, c1, f1 = endpoint_diagnostics(I1, src["depth"][right], *p1)
        base, base_valid, base_diag = depth_aware_bidirectional_warp_projected_diagnostics(
            I0, src["depth"][left], p0[0], p0[1], p0[2],
            I1, src["depth"][right], p1[0], p1[1], p1[2], composite="next_primary")
        cross, cross_valid, choose1_nearest = nearest_endpoint_composite(a0, d0, a1, d1)
        cov0, covv0, _ = coverage_splat(I0, src["depth"][left], *p0)
        cov1, covv1, _ = coverage_splat(I1, src["depth"][right], *p1)
        cov_valid = covv0 | covv1
        cov = np.full_like(base, np.nan, dtype=np.float32)
        cov[ covv1] = cov1[ covv1]
        cov[~covv1 & covv0] = cov0[~covv1 & covv0]
        base = hold_fill(base, base_valid, alpha, I0, I1)
        cross = hold_fill(cross, cross_valid, alpha, I0, I1)
        cov = hold_fill(cov, cov_valid, alpha, I0, I1)

        exact_end = base.copy()
        if qi == len(direct_times) - 1 and abs(alpha - 1.0) <= 1e-8:
            # Algebraic endpoint invariant: t=t1 must use the observed t1 image
            # and its own source-grid depth/validity, without a float projection.
            p1_exact = (grid.copy(), src["depth"][right].astype(np.float32, copy=True),
                        np.isfinite(src["depth"][right]) & (src["depth"][right] > 1e-6))
            a1_exact, d1_exact = _splat_projected_diagnostics(
                I1, src["depth"][right], *p1_exact)
            v0, v1 = d0["visibility"], d1_exact["visibility"]
            exact_valid = v0 | v1
            exact_end = np.full_like(base, np.nan, dtype=np.float32)
            exact_end[v1] = a1_exact[v1]
            exact_end[~v1 & v0] = a0[~v1 & v0]
            exact_end = hold_fill(exact_end, exact_valid, alpha, I0, I1)
        baseline_frames.append(base)
        cross_frames.append(cross)
        coverage_frames.append(cov)
        exact_end_frames.append(exact_end)

        ref = direct[qi]
        err = log_error(base, ref)
        bad = err > 0.2
        both = d0["visibility"] & d1["visibility"]
        cross_order = both & (d0["z_buffer"] < d1["z_buffer"])
        choose_wrong_nearer = cross_order & d1["visibility"]
        # For next_primary, endpoint 1 is active wherever its own splat is
        # visible. Attribute tiny-weight competition only to that exact branch.
        active_uses_1 = d1["visibility"]
        active_winner_coverage = np.where(active_uses_1, c1, c0)
        active_next_layer_coverage = np.where(active_uses_1, f1, f0)
        cross_err = log_error(cross, ref)
        coverage_err = log_error(cov, ref)
        tiny_coverage_sensitivity = {}
        for threshold in (1e-6, 1e-4, 1e-3, 1e-2, 5e-2):
            tiny_signal = bad & (active_winner_coverage <= threshold) & (active_next_layer_coverage > 0.05)
            tiny_coverage_sensitivity[f"winner_coverage_le_{threshold:g}"] = {
                "baseline_bad_pixel_count": int(tiny_signal.sum()),
                "resolved_by_front_to_back_candidate": int((tiny_signal & (coverage_err <= 0.2)).sum()),
                "resolved_fraction": float((tiny_signal & (coverage_err <= 0.2)).sum() / max(int(tiny_signal.sum()), 1)),
            }
        branch_bad = bad & choose_wrong_nearer
        per_frame_rows.append({
            "episode": ep, "query_time_s": float(t), "alpha": alpha,
            "baseline_over_0p2_pixels": int(bad.sum()),
            "cross_endpoint_both_visible_pixels": int(both.sum()),
            "cross_endpoint_nearer_0_contributes_but_1_selected": int(choose_wrong_nearer.sum()),
            "baseline_bad_in_that_branch": int((bad & choose_wrong_nearer).sum()),
            "cross_branch_bad_resolved_by_nearer_z_intervention": int((branch_bad & (cross_err <= 0.2)).sum()),
            "cross_branch_bad_improved_by_nearer_z_intervention": int((branch_bad & (cross_err < err)).sum()),
            "cross_branch_bad_worsened_by_nearer_z_intervention": int((branch_bad & (cross_err > err)).sum()),
            "tiny_winner_coverage_sensitivity": tiny_coverage_sensitivity,
            "unknown_pixel_count_before_hold": int((~base_valid).sum()),
            "stored_unknown_pixel_count": int(stored_unknown[qi].sum()),
            "cross_intervention": summarize_candidate(base, cross, ref),
            "coverage_intervention": summarize_candidate(base, cov, ref),
        })
        trace_context.append({
            "t": float(t), "alpha": alpha, "base": base, "cross": cross,
            "coverage": cov, "exact_end": exact_end, "ref": ref, "err": err,
            "p0": p0, "p1": p1, "a0": a0, "a1": a1, "d0": d0, "d1": d1,
            "c0": c0, "c1": c1, "f0": f0, "f1": f1,
            "contrib0": contrib0,
            "contrib1": contrib1,
            "seg0": src["segmentation"][left], "seg1": src["segmentation"][right],
        })

    baseline = np.stack(baseline_frames)
    cross = np.stack(cross_frames)
    coverage = np.stack(coverage_frames)
    exact_end = np.stack(exact_end_frames)
    ref = direct
    candidate_summaries = {
        "cross_endpoint_nearest_z_before_next_primary": summarize_candidate(baseline, cross, ref),
        "front_to_back_coverage_splat_per_endpoint": summarize_candidate(baseline, coverage, ref),
        "exact_t1_identity_projection_control": summarize_candidate(
            baseline[-1:], exact_end[-1:], ref[-1:]),
        "interior_times_only_0p5815_to_0p5895": {
            "cross_endpoint_nearest_z_before_next_primary": summarize_candidate(baseline[:-1], cross[:-1], ref[:-1]),
            "front_to_back_coverage_splat_per_endpoint": summarize_candidate(baseline[:-1], coverage[:-1], ref[:-1]),
        },
    }

    # Trace a small set of spatial pixels with the largest baseline errors,
    # then retain their full 10 ms trajectory across all saved query times.
    max_err_per_pixel = np.max(log_error(baseline, ref), axis=0)
    order = np.argsort(max_err_per_pixel.ravel())[::-1]
    selected = []
    for flat in order:
        y, x = divmod(int(flat), width)
        if max_err_per_pixel[y, x] <= 0.2:
            break
        if all((x - sx) ** 2 + (y - sy) ** 2 >= 4 for sx, sy in selected):
            selected.append((x, y))
        if len(selected) >= 12:
            break
    traces = []
    for x, y in selected:
        for qi, ctx in enumerate(trace_context):
            index = y * width + x
            z0, z1 = float(ctx["d0"]["z_buffer"][y, x]), float(ctx["d1"]["z_buffer"][y, x])
            vis0, vis1 = bool(ctx["d0"]["visibility"][y, x]), bool(ctx["d1"]["visibility"][y, x])
            chosen = 1 if vis1 else (0 if vis0 else -1)
            coverage_chosen = float(ctx[f"c{chosen}"][y, x]) if chosen >= 0 else None
            traces.append({
                "episode": ep, "time_s": ctx["t"], "alpha": ctx["alpha"],
                "target_xy": [x, y], "direct_luma": float(ctx["ref"][y, x]),
                "legacy_frame_luma": float(ctx["base"][y, x]),
                "abs_log_error": float(ctx["err"][y, x]),
                "cross_endpoint_candidate_luma": float(ctx["cross"][y, x]),
                "coverage_candidate_luma": float(ctx["coverage"][y, x]),
                "endpoint0_visible": vis0, "endpoint1_visible": vis1,
                "endpoint0_z_m": z0 if np.isfinite(z0) else None,
                "endpoint1_z_m": z1 if np.isfinite(z1) else None,
                "legacy_next_primary_selected_endpoint": chosen,
                "selected_endpoint_near_coverage": coverage_chosen,
                "endpoint0_source_contributors": json.dumps(contributors_at(
                    ctx["contrib0"], index, z0, ctx["seg0"], src["depth"][left]), separators=(",", ":")),
                "endpoint1_source_contributors": json.dumps(contributors_at(
                    ctx["contrib1"], index, z1, ctx["seg1"], src["depth"][right]), separators=(",", ":")),
            })

    # Re-run the deterministic ideal DVS on this 10 ms window, keeping an
    # independent continuous per-pixel history for every image intervention.
    # The dynamic segmentation mask is used only for post-hoc localization of
    # the prior robot/nonrobot counterfactual, never to alter the warp outputs.
    trace_rows = [json.loads(line) for line in (episode_dir / "trace_window.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    trace_by_time = {round(float(row["time_s"]), 9): row for row in trace_rows}
    pose_ids = set(map(int, src["segmentation_pose_ids"].tolist()))
    target_distances = {seg_id: [] for seg_id in pose_ids}
    for t in direct_times:
        row = trace_by_time.get(round(float(t), 9))
        if row is None:
            continue
        target_xyz = np.asarray(row["object"]["position_m"], dtype=np.float64)
        for j, seg_id in enumerate(src["segmentation_pose_ids"]):
            target_distances[int(seg_id)].append(float(np.linalg.norm(
                src["object_poses"][right, j][:3, 3] - target_xyz)))
    target_seg = min(target_distances, key=lambda k: np.median(target_distances[k]) if target_distances[k] else np.inf)
    robot_ids = pose_ids - {target_seg}
    dynamic_robot_mask = np.isin(direct_seg, list(robot_ids))
    mask_switches_by_step = [int(np.count_nonzero(dynamic_robot_mask[k] != dynamic_robot_mask[k - 1]))
                             for k in range(1, len(dynamic_robot_mask))]
    variants = {
        "native_reconstruction": stored,
        "front_to_back_coverage_candidate": coverage,
        "direct_hdr_exact_states": direct,
        "robot_pixels_replaced": np.where(dynamic_robot_mask, direct, stored).astype(np.float32),
        "nonrobot_pixels_replaced": np.where(~dynamic_robot_mask, direct, stored).astype(np.float32),
    }
    dvs_cfg = DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0, cutoff_hz=0.0,
        leak_rate_hz=0.0, leak_jitter_fraction=0.0, shot_noise_rate_hz=0.0,
        noise_rate_cov_decades=0.0, refractory_period_s=0.0,
        log_input=False, radiance_input=True, response_mode="physical_log",
        radiometric_gain=1.0, radiometric_offset=0.0, radiometric_floor=LOG_EPS,
        radiometric_calibration_status="uncalibrated", readout_model="ideal_readout", seed=42)
    event_variant_summary = {}
    for name, frames in variants.items():
        dvs = DvsEmulator(res=(height, width), config=dvs_cfg, device="cpu")
        dvs.initialize(I0, t=float(times[left]))
        counts = np.zeros((height, width), dtype=np.int32)
        per_step = []
        threshold_ref_trace = {f"{x},{y}": [] for x, y in selected}
        for qi, frame in enumerate(frames):
            events = np.asarray(dvs.generate_events(frame, float(direct_times[qi])))
            per_step.append(int(len(events)))
            if events.size:
                ex, ey = events[:, 1].astype(np.int64), events[:, 2].astype(np.int64)
                np.add.at(counts, (ey, ex), 1)
            ref_state = dvs.base_log_frame.detach().cpu().numpy()
            for x, y in selected:
                threshold_ref_trace[f"{x},{y}"].append(float(ref_state[y, x]))
        event_variant_summary[name] = {
            "event_count": int(counts.sum()), "event_pixels": int(np.count_nonzero(counts)),
            "events_by_saved_1ms_frame": per_step,
            "events_by_traced_pixel": {f"{x},{y}": int(counts[y, x]) for x, y in selected},
            "per_pixel_log_reference_after_each_frame": threshold_ref_trace,
        }
    pixel_intervention_delta = (event_variant_summary["native_reconstruction"]["event_count"] -
                                event_variant_summary["direct_hdr_exact_states"]["event_count"])
    robot_delta = (event_variant_summary["native_reconstruction"]["event_count"] -
                   event_variant_summary["robot_pixels_replaced"]["event_count"])
    nonrobot_delta = (event_variant_summary["native_reconstruction"]["event_count"] -
                      event_variant_summary["nonrobot_pixels_replaced"]["event_count"])
    temporal_intervention_note = {
        "target_segmentation_id_posthoc_only": int(target_seg),
        "robot_segmentation_ids_posthoc_only": sorted(map(int, robot_ids)),
        "robot_mask_pixel_switches_between_1ms_frames": mask_switches_by_step,
        "pixels_ever_switching_robot_mask": int(np.count_nonzero(np.any(dynamic_robot_mask[1:] != dynamic_robot_mask[:-1], axis=0))),
        "event_counts": event_variant_summary,
        "region_intervention_nonadditivity_on_this_10ms_window": {
            "full_native_minus_direct_events": int(pixel_intervention_delta),
            "robot_only_replacement_reduction": int(robot_delta),
            "nonrobot_only_replacement_reduction": int(nonrobot_delta),
            "difference_full_minus_sum_of_region_reductions": int(pixel_intervention_delta - robot_delta - nonrobot_delta),
            "interpretation": "DVS response state is per pixel; a time-varying segmentation mask can make one pixel alternate between the two image sources, changing that same pixel's later threshold crossings. This is not evidence for cross-pixel DVS coupling.",
        },
    }

    return {
        "episode": ep,
        "source_cache_sha256": sha256(cache_path),
        "direct_frame_cache_sha256": sha256(evidence_path),
        "runtime_modules": {
            "physics_interpolator": sha256(ROOT / "genesis_event_plugin/physics_interpolator.py"),
            "physics_warp": sha256(ROOT / "genesis_event_plugin/physics_warp.py"),
            "physics_plugin": sha256(ROOT / "genesis_event_plugin/physics_plugin.py"),
        },
        "active_path_replay": {
            "module": "genesis_event_plugin.physics_interpolator.PhysicsInterpolator -> genesis_event_plugin.physics_warp.depth_aware_bidirectional_warp_projected_diagnostics",
            "sample_count": int(len(sample_times)),
            "sample_times_s": [float(x) for x in sample_times],
            "archived_intermediate_max_abs_delta": float(replay_delta.max()),
            "archived_intermediate_nonzero_delta_pixels": int(np.count_nonzero(replay_delta)),
            "sampler_summary": interpolator.last_sampling.__dict__,
        },
        "endpoint_reference_check": {
            "t1_endpoint_log_abs_max": float(log_error(I1, direct[-1]).max()),
            "t1_endpoint_luma_abs_max": float(np.abs(I1 - direct[-1]).max()),
        },
        "comparison_boundary": "Saved Direct-HDR exact-state images are a diagnostic reference only; the 10 ms interpolation is not a physical event truth or newly rendered hidden state.",
        "candidate_summary": candidate_summaries,
        "per_sample": per_frame_rows,
        "traced_target_xy": [[int(x), int(y)] for x, y in selected],
        "pixel_traces": traces,
        "temporal_pixel_history_and_mask_switches": temporal_intervention_note,
    }


def run_controls():
    # Static identical frame through active and candidate warp paths.
    image = np.asarray([[0.3, 0.5, 0.7], [0.4, 0.6, 0.8]], dtype=np.float32)
    depth = np.full_like(image, 1.0)
    y, x = np.meshgrid(np.arange(2), np.arange(3), indexing="ij")
    uv = np.stack((x, y), axis=0).astype(np.float32)
    valid = np.ones_like(depth, dtype=bool)
    base_static, v_static, _ = depth_aware_bidirectional_warp_projected_diagnostics(
        image, depth, uv, depth, valid, image, depth, uv, depth, valid, "next_primary")
    cov_static, covv_static, _ = coverage_splat(image, depth, uv, depth, valid)
    static_err = float(np.max(np.abs(base_static - image)))
    static_cov_err = float(np.max(np.abs(cov_static - image)))

    # DVS no-event control on an exactly static radiance sequence.
    cfg = DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0, cutoff_hz=0.0,
        leak_rate_hz=0.0, leak_jitter_fraction=0.0, shot_noise_rate_hz=0.0,
        noise_rate_cov_decades=0.0, refractory_period_s=0.0,
        log_input=False, radiance_input=True, response_mode="physical_log",
        radiometric_gain=1.0, radiometric_offset=0.0, radiometric_floor=LOG_EPS,
        radiometric_calibration_status="uncalibrated", readout_model="ideal_readout", seed=42)
    dvs = DvsEmulator(res=image.shape, config=cfg, device="cpu")
    dvs.initialize(image, t=0.0)
    static_events = []
    for k in range(1, 5):
        static_events.append(dvs.generate_events(image.copy(), k * 0.001))
    static_event_count = int(sum(len(x) for x in static_events))

    # Cross-endpoint occlusion: nearer endpoint 0 must beat farther endpoint 1.
    one = np.ones((1, 1), dtype=np.float32)
    uv1 = np.zeros((2, 1, 1), dtype=np.float32)
    near = np.asarray([[1.0]], dtype=np.float32)
    far = np.asarray([[2.0]], dtype=np.float32)
    a0, d0 = _splat_projected_diagnostics(one, near, uv1, near, np.ones_like(near, dtype=bool))
    a1, d1 = _splat_projected_diagnostics(np.asarray([[0.01]], dtype=np.float32), far, uv1, far,
                                           np.ones_like(far, dtype=bool))
    old_occlusion, old_valid, _ = depth_aware_bidirectional_warp_projected_diagnostics(
        one, near, uv1, near, np.ones_like(near, dtype=bool),
        np.asarray([[0.01]], dtype=np.float32), far, uv1, far, np.ones_like(far, dtype=bool), "next_primary")
    fixed_occlusion, _, _ = nearest_endpoint_composite(a0, d0, a1, d1)

    # Same-endpoint surface coverage control: full foreground coverage must
    # fully occlude; a fractional silhouette footprint leaves only uncovered
    # area for the background contribution.
    layered_image = np.asarray([[1.0, 0.01, 0.01]], dtype=np.float32)
    layered_depth = np.asarray([[1.0, 2.0, 2.0]], dtype=np.float32)
    layered_uv = np.asarray([[[1.0, 1.0, 2.0]], [[0.0, 0.0, 0.0]]], dtype=np.float32)
    full_layered, _, _ = coverage_splat(layered_image, layered_depth, layered_uv, layered_depth)
    partial_uv = layered_uv.copy()
    partial_uv[0, 0, 0] = 1.75  # 25% near-surface coverage at target pixel 1.
    partial_legacy, _ = _splat_projected_diagnostics(layered_image, layered_depth, partial_uv, layered_depth)
    partial_coverage, _, _ = coverage_splat(layered_image, layered_depth, partial_uv, layered_depth)

    # Tiny near-surface footprint at pixel 1 competing with a full far surface.
    sample_image = np.asarray([[1.0, 0.01]], dtype=np.float32)
    sample_depth = np.asarray([[1.0, 2.0]], dtype=np.float32)
    grid = np.asarray([[[0.0, 1.0]], [[0.0, 0.0]]], dtype=np.float32)
    tiny_rows = []
    for shift in (-1e-6, 0.0, 1e-6):
        shifted = grid.copy()
        shifted[0, 0, 0] += shift
        legacy, _ = _splat_projected_diagnostics(sample_image, sample_depth, shifted, sample_depth)
        coverage, _, _ = coverage_splat(sample_image, sample_depth, shifted, sample_depth)
        tiny_rows.append({"near_pixel_shift": shift, "legacy_pixel1": float(legacy[0, 1]),
                          "coverage_pixel1": float(coverage[0, 1])})
    legacy_jump = max(abs(tiny_rows[0]["legacy_pixel1"] - tiny_rows[1]["legacy_pixel1"]),
                      abs(tiny_rows[2]["legacy_pixel1"] - tiny_rows[1]["legacy_pixel1"]))
    coverage_jump = max(abs(tiny_rows[0]["coverage_pixel1"] - tiny_rows[1]["coverage_pixel1"]),
                        abs(tiny_rows[2]["coverage_pixel1"] - tiny_rows[1]["coverage_pixel1"]))

    return {
        "static_identity_control": {
            "active_visibility_all": bool(v_static.all()), "coverage_visibility_all": bool(covv_static.all()),
            "active_max_abs_image_error": static_err,
            "coverage_candidate_max_abs_image_error": static_cov_err,
            "ideal_dvs_events_over_4_static_ms": static_event_count,
        },
        "cross_endpoint_occlusion_control": {
            "current_next_primary_luma": float(old_occlusion[0, 0]),
            "nearest_z_candidate_luma": float(fixed_occlusion[0, 0]),
            "near_z_m": float(d0["z_buffer"][0, 0]),
            "far_z_m": float(d1["z_buffer"][0, 0]),
        },
        "same_endpoint_coverage_occlusion_control": {
            "full_foreground_coverage_candidate_luma": float(full_layered[0, 1]),
            "partial_foreground_coverage_legacy_luma": float(partial_legacy[0, 1]),
            "partial_foreground_coverage_candidate_luma": float(partial_coverage[0, 1]),
            "expected_partial_coverage_luma": float(0.25 * 1.0 + 0.75 * 0.01),
            "near_surface_z_m": 1.0,
            "background_z_m": 2.0,
        },
        "subpixel_weight_continuity_control": {
            "samples": tiny_rows,
            "legacy_max_change_from_zero": float(legacy_jump),
            "coverage_candidate_max_change_from_zero": float(coverage_jump),
        },
    }


def write_manifest(root: Path):
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cache_root, output = args.cache_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    output.mkdir(parents=True)
    episodes = {ep: run_episode(ep, cache_root) for ep in ("004", "005")}
    controls = run_controls()
    summary = {
        "purpose": "single-rule pixel attribution on the existing center and +55 mm cached episodes",
        "scope": "CPU-only cached replay; no renderer, physics, GPU, model inference, or training",
        "repository_head_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "runtime_modules": {
            name.rsplit(".", 1)[-1]: runtime_module_record(name)
            for name in (
                "genesis_event_plugin.physics_plugin",
                "genesis_event_plugin.physics_interpolator",
                "genesis_event_plugin.physics_warp",
                "genesis_event_plugin.dvs_emulator",
            )
        },
        "runtime_environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "h5py": h5py.__version__, "device": "CPU",
        },
        "effective_config": {
            "source_pair_s": [0.5805, 0.5905], "saved_direct_query_count": 10,
            "sampler_min_samples": 10, "sampler_max_samples": 10,
            "interpolation_input": "scene-linear HDR luminance",
            "DVS": {"response_mode": "physical_log", "positive_threshold": 0.2,
                    "negative_threshold": 0.2, "threshold_noise": 0.0,
                    "cutoff_hz": 0.0, "leak_hz": 0.0, "shot_noise_hz": 0.0,
                    "refractory_period_s": 0.0, "radiometric_gain": 1.0,
                    "radiometric_offset": 0.0, "radiometric_floor": 1e-6,
                    "seed": 42},
            "splat_depth_tie_band_m": DEPTH_TIE_M,
            "coverage_candidate": "front-to-back bilinear footprint compositing, unit-area cap, normalize observed support when coverage is incomplete",
        },
        "frozen_core_modified": False,
        "direct_hdr_reference_limit": "exact-state cached frames are diagnostic comparisons, not physical ground truth at every intermediate time",
        "research_reproduction_boundary": "Research-side independent native-event replay differed by +3/-5 events and did not retrieve Direct-HDR camera attributes; this run reports its own cached-frame/path hashes and does not claim cross-environment elementwise event reproduction.",
        "rules": {
            "cross_endpoint": "physics_warp.py depth_aware_bidirectional_warp_projected_diagnostics: next_primary always chooses endpoint 1 where visible, without comparing endpoint z-buffers",
            "tiny_weight": "physics_warp.py _splat_projected_diagnostics: every positive bilinear weight participates in min-depth competition before weighted normalization",
            "coverage_candidate": "Diagnostic-only front-to-back bilinear footprint resolve; uses existing 1e-5 m depth tie band, no GT masks or event manipulation",
        },
        "episodes": episodes,
        "controls": controls,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=jsonable) + "\n", encoding="utf-8")
    save_csv(output / "per_sample_metrics.csv", [row for ep in episodes.values() for row in ep["per_sample"]])
    save_csv(output / "dominant_pixel_traces.csv", [row for ep in episodes.values() for row in ep["pixel_traces"]])
    write_manifest(output)
    print(json.dumps({
        "output": str(output),
        "episodes": {ep: {"active_replay_delta": d["active_path_replay"]["archived_intermediate_max_abs_delta"],
                          "candidates": d["candidate_summary"]} for ep, d in episodes.items()},
        "controls": controls,
        "sha256_manifest": sha256(output / "SHA256SUMS"),
    }, indent=2, default=jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
