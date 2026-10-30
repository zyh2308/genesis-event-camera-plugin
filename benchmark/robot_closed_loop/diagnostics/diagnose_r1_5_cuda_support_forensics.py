#!/usr/bin/env python3
"""Trace CPU/CUDA projection, splat support, and event-path parity on cached clips."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_5_core_candidate_cuda_parity import (  # noqa: E402
    event_metrics,
    run_dvs,
    run_threshold_probes,
)
from benchmark.robot_closed_loop.diagnostics.validate_r1_5_core_candidate import (  # noqa: E402
    load_cached_episode,
    strict_cached_replay,
)
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig  # noqa: E402
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator  # noqa: E402
from genesis_event_plugin.physics_warp import (  # noqa: E402
    _splat_projected_coverage_diagnostics,
    depth_aware_bidirectional_warp_projected_diagnostics,
    project_endpoint_to_alpha,
)
from genesis_event_plugin.torch_physics import (  # noqa: E402
    _splat_projected_coverage_batch_torch,
    depth_aware_bidirectional_warp_projected_batch_torch,
    project_endpoint_to_alpha_stack_torch,
)


T0, T1, ALPHA = 0.5805, 0.5905, 0.5
MIN_COVERAGE = 0.995
DEPTH_TIE_ABS = 1e-7
DEPTH_TIE_RTOL = 8.0 * np.finfo(np.float32).eps
SURFACE_DEPTH_ABS = 1e-3
SURFACE_DEPTH_RTOL = 1e-3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _as_numpy_diag(diag):
    return {k: v.detach().cpu().numpy()[0] for k, v in diag.items()}


def project_numpy_pair(src, K, left, right, objects0, objects1):
    T0_cam, T1_cam = src["camera_to_world_cv"][left], src["camera_to_world_cv"][right]
    alpha_cam = T0_cam if np.array_equal(T0_cam, T1_cam) else None
    if alpha_cam is None:
        raise RuntimeError("forensic projection requires the archived fixed-camera clips")
    return [
        project_endpoint_to_alpha(
            src["depth"][left], src["segmentation"][left], K, T0_cam, alpha_cam,
            ALPHA, 0, objects0, objects1, None,
        ),
        project_endpoint_to_alpha(
            src["depth"][right], src["segmentation"][right], K, T1_cam, alpha_cam,
            ALPHA, 1, objects0, objects1, None,
        ),
    ]


def project_cuda_pair(src, K, left, right, objects0, objects1):
    device = torch.device("cuda:0")
    alphas = torch.tensor([ALPHA], dtype=torch.float64, device=device)
    K_t = torch.as_tensor(K, dtype=torch.float64, device=device)
    cam0 = torch.as_tensor(src["camera_to_world_cv"][left], dtype=torch.float64, device=device)
    cam1 = torch.as_tensor(src["camera_to_world_cv"][right], dtype=torch.float64, device=device)
    out = []
    with torch.inference_mode():
        for side, frame_index, source_camera in ((0, left, cam0), (1, right, cam1)):
            d = torch.as_tensor(src["depth"][frame_index], dtype=torch.float64, device=device)
            seg = torch.as_tensor(src["segmentation"][frame_index], device=device)
            o0 = {int(k): torch.as_tensor(v, dtype=torch.float64, device=device) for k, v in objects0.items()}
            o1 = {int(k): torch.as_tensor(v, dtype=torch.float64, device=device) for k, v in objects1.items()}
            uv, z, valid = project_endpoint_to_alpha_stack_torch(
                d, seg, K_t, source_camera, cam0, cam1, alphas, side, o0, o1, None,
            )
            out.append((uv[0].cpu().numpy(), z[0].cpu().numpy(), valid[0].cpu().numpy()))
    return out


def splat_numpy(image, depth, projection, seg):
    uv, z, valid = projection
    return _splat_projected_coverage_diagnostics(
        image, depth, uv, z, valid, seg, MIN_COVERAGE, DEPTH_TIE_ABS, DEPTH_TIE_RTOL,
    )


def splat_cuda(image, depth, projection, seg):
    device = torch.device("cuda:0")
    uv, z, valid = projection
    with torch.inference_mode():
        out, visible, diag = _splat_projected_coverage_batch_torch(
            torch.as_tensor(image[None], dtype=torch.float32, device=device),
            torch.as_tensor(depth[None], dtype=torch.float32, device=device),
            torch.as_tensor(uv[None], dtype=torch.float32, device=device),
            torch.as_tensor(z[None], dtype=torch.float32, device=device),
            torch.as_tensor(valid[None], dtype=torch.bool, device=device),
            torch.as_tensor(seg[None], dtype=torch.long, device=device),
            MIN_COVERAGE, DEPTH_TIE_ABS, DEPTH_TIE_RTOL,
        )
    out = out[0].cpu().numpy()
    d = _as_numpy_diag(diag)
    d["visibility"] = visible[0].cpu().numpy()
    return out, d


def composite_numpy(frames, depths, segs, projections):
    p0, p1 = projections
    return depth_aware_bidirectional_warp_projected_diagnostics(
        frames[0], depths[0], *p0,
        frames[1], depths[1], *p1,
        composite="same_surface_log_linear", alpha=ALPHA,
        surface_id0=segs[0], surface_id1=segs[1],
        min_coverage_fraction=MIN_COVERAGE,
        depth_tie_abs_m=DEPTH_TIE_ABS, depth_tie_rtol=DEPTH_TIE_RTOL,
        surface_depth_abs_m=SURFACE_DEPTH_ABS, surface_depth_rtol=SURFACE_DEPTH_RTOL,
    )


def run_production_interpolator_pair(src, K, left, right, objects0, objects1, width, event_rows, array_store, episode):
    """Exercise the public PhysicsInterpolator CPU and CUDA paths end to end."""
    common = {
        "sampler_config": AdaptiveSamplingConfig(min_samples=2, max_samples=10),
        "disocclusion_policy": "hold",
        "composite": "same_surface_log_linear",
        "strict_fidelity": False,
        "max_photometric_interval_s": 0.005,
    }
    outputs, masks, times, wall_s, details = {}, {}, {}, {}, {}
    t0, t1 = T0, T1
    initial = src["radiance_luma"][left]
    for name, backend, device in (("numpy", "numpy", "cpu"), ("cuda", "torch", "cuda:0")):
        interpolator = PhysicsInterpolator(K, warp_backend=backend, device=device, **common)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        frames, frame_times = interpolator.interpolate(
            src["radiance_luma"][left], src["depth"][left], src["segmentation"][left],
            src["radiance_luma"][right], src["depth"][right], src["segmentation"][right],
            src["camera_to_world_cv"][left], src["camera_to_world_cv"][right], None,
            t0, t1, input_space="linear", object_world0=objects0, object_world1=objects1,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        wall_s[name] = float(time.perf_counter() - started)
        outputs[name] = np.stack([np.asarray(frame, dtype=np.float32) for frame in frames])
        times[name] = np.asarray(frame_times, dtype=np.float64)
        masks[name] = np.asarray(interpolator.last_visibility["intermediate_visibility"], dtype=bool)
        details[name] = {
            "sample_count": int(len(frame_times)),
            "uncertainty": interpolator.last_uncertainty,
            "unknown_fraction_by_frame": [float(value) for value in (~masks[name]).mean(axis=(1, 2))],
        }

    if not np.array_equal(times["numpy"], times["cuda"]):
        raise RuntimeError("production CPU/CUDA PhysicsInterpolator timestamps differ")
    if outputs["numpy"].shape != outputs["cuda"].shape or masks["numpy"].shape != masks["cuda"].shape:
        raise RuntimeError("production CPU/CUDA PhysicsInterpolator output shapes differ")
    common_support = masks["numpy"] & masks["cuda"]
    log_delta = np.abs(
        np.log(np.maximum(outputs["numpy"][common_support], 1e-6))
        - np.log(np.maximum(outputs["cuda"][common_support], 1e-6))
    ) if common_support.any() else np.asarray([])
    mask_xor = int(np.count_nonzero(masks["numpy"] != masks["cuda"]))

    events = {}
    event_comparisons = []
    for reconstruction in ("numpy", "cuda"):
        for dvs_backend in ("cpu", "cuda:0"):
            key = f"{reconstruction}_{dvs_backend.replace(':', '')}"
            event_stream = run_dvs(
                list(outputs[reconstruction]),
                [float(value) for value in times[reconstruction]],
                initial, t0, dvs_backend,
            )
            events[key] = event_stream
            array_store[f"{episode}_production_{key}_events"] = event_stream
            event_rows.append({
                "episode": episode,
                "projection": "public_PhysicsInterpolator",
                "compositor": reconstruction,
                "dvs_backend": dvs_backend.replace(":", ""),
                "event_count": int(len(event_stream)),
                "on_count": int(np.count_nonzero(event_stream[:, 3] > 0)),
                "off_count": int(np.count_nonzero(event_stream[:, 3] < 0)),
            })
        pair = event_metrics(events[f"{reconstruction}_cpu"], events[f"{reconstruction}_cuda0"], width)
        pair["event_count_relative_delta"] = abs(pair["candidate_event_count"] - pair["reference_event_count"]) / max(1, pair["reference_event_count"])
        event_comparisons.append({"reconstruction": reconstruction, **pair})
    cross_reconstruction = {
        backend: event_metrics(events[f"numpy_{backend}"], events[f"cuda_{backend}"], width)
        for backend in ("cpu", "cuda0")
    }
    return {
        "sample_times_s": [float(value) for value in times["numpy"]],
        "sample_count": int(len(times["numpy"])),
        "numpy_unknown_fraction_by_frame": details["numpy"]["unknown_fraction_by_frame"],
        "cuda_unknown_fraction_by_frame": details["cuda"]["unknown_fraction_by_frame"],
        "timestamp_max_abs_delta_s": float(np.max(np.abs(times["numpy"] - times["cuda"]))),
        "support_mask_xor_pixels_all_frames": mask_xor,
        "common_supported_log_luma_abs_max_all_frames": float(log_delta.max()) if log_delta.size else None,
        "wall_s": wall_s,
        "dvs_backend_comparisons_per_reconstruction": event_comparisons,
        "cpu_dvs_numpy_vs_cuda_reconstruction": cross_reconstruction["cpu"],
        "cuda_dvs_numpy_vs_cuda_reconstruction": cross_reconstruction["cuda0"],
        "path_details": details,
    }


def composite_cuda(frames, depths, segs, projections):
    device = torch.device("cuda:0")
    p0, p1 = projections

    def tensor(value, dtype=None):
        return torch.as_tensor(value, dtype=dtype, device=device)

    with torch.inference_mode():
        result = depth_aware_bidirectional_warp_projected_batch_torch(
            tensor(frames[0][None], torch.float32), tensor(depths[0][None], torch.float32),
            tensor(p0[0][None, None], torch.float32), tensor(p0[1][None, None], torch.float32),
            tensor(p0[2][None, None], torch.bool),
            tensor(frames[1][None], torch.float32), tensor(depths[1][None], torch.float32),
            tensor(p1[0][None, None], torch.float32), tensor(p1[1][None, None], torch.float32),
            tensor(p1[2][None, None], torch.bool),
            composite="same_surface_log_linear", return_diagnostics=True,
            alpha=torch.tensor([ALPHA], dtype=torch.float32, device=device),
            surface_id0=tensor(segs[0][None], torch.long),
            surface_id1=tensor(segs[1][None], torch.long),
            min_coverage_fraction=MIN_COVERAGE,
            depth_tie_abs_m=DEPTH_TIE_ABS, depth_tie_rtol=DEPTH_TIE_RTOL,
            surface_depth_abs_m=SURFACE_DEPTH_ABS, surface_depth_rtol=SURFACE_DEPTH_RTOL,
        )
    image, valid, diag = result
    return image[0, 0].cpu().numpy(), valid[0, 0].cpu().numpy(), {
        k: v[0, 0].cpu().numpy() for k, v in diag.items()
    }


def projection_stats(reference, candidate):
    rows = []
    for side, (a, b) in enumerate(zip(reference, candidate)):
        uv_a, z_a, v_a = a
        uv_b, z_b, v_b = b
        duv = np.abs(uv_a.astype(np.float32) - uv_b.astype(np.float32))
        dz = np.abs(z_a.astype(np.float32) - z_b.astype(np.float32))
        valid_xor = int(np.count_nonzero(v_a != v_b))
        both = v_a & v_b
        rows.append({
            "endpoint": side,
            "valid_xor_pixels": valid_xor,
            "uv_abs_max_px_both_valid": float(duv[:, both].max()) if both.any() else None,
            "uv_abs_p99_px_both_valid": float(np.quantile(duv[:, both], 0.99)) if both.any() else None,
            "z_abs_max_m_both_valid": float(dz[both].max()) if both.any() else None,
            "uv_differing_pixels": int(np.count_nonzero(np.any(duv > 0, axis=0) & both)),
        })
    return rows


def rejection_reasons(diag, y, x):
    reasons = []
    for i in (0, 1):
        if not bool(diag[f"support{i}"][y, x]):
            reasons.append(f"coverage{i}")
        if not bool(diag[f"surface_consistent{i}"][y, x]):
            reasons.append(f"surface_consistency{i}")
    if int(diag["surface_id0"][y, x]) != int(diag["surface_id1"][y, x]):
        reasons.append("surface_id")
    z0, z1 = float(diag["z_buffer0"][y, x]), float(diag["z_buffer1"][y, x])
    ztol = max(SURFACE_DEPTH_ABS, SURFACE_DEPTH_RTOL * max(abs(z0), abs(z1)))
    if not np.isfinite(z0) or not np.isfinite(z1) or abs(z0 - z1) > ztol:
        reasons.append("depth_match")
    return reasons


def trace_target_layers(projection, source_valid, seg, targets):
    """Collect the actual projected samples/weights competing at each target."""
    uv, z, valid = projection
    H, W = valid.shape
    tx, ty = uv[0], uv[1]
    x0, y0 = np.floor(tx).astype(np.int32), np.floor(ty).astype(np.int32)
    wx, wy = tx - x0, ty - y0
    target_ids = {(y * W + x): (x, y) for x, y in targets}
    target_mask = np.zeros((H, W), dtype=bool)
    for x, y in targets:
        target_mask[y, x] = True
    buckets = {tid: [] for tid in target_ids}
    for ox, oy, weight in (
        (0, 0, (1.0 - wx) * (1.0 - wy)),
        (1, 0, wx * (1.0 - wy)),
        (0, 1, (1.0 - wx) * wy),
        (1, 1, wx * wy),
    ):
        xx, yy = x0 + ox, y0 + oy
        take = valid & source_valid & (weight > 0) & (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        take &= target_mask[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)]
        for sy, sx in zip(*np.nonzero(take)):
            tid = int(yy[sy, sx] * W + xx[sy, sx])
            buckets[tid].append({
                "source_xy": [int(sx), int(sy)],
                "projected_uv": [float(tx[sy, sx]), float(ty[sy, sx])],
                "target_z_m": float(z[sy, sx]),
                "footprint_weight": float(weight[sy, sx]),
                "surface_id": int(seg[sy, sx]),
            })
    traces = {}
    for tid, contributions in buckets.items():
        contributions.sort(key=lambda item: item["target_z_m"])
        layers = []
        remaining = list(contributions)
        while remaining:
            nearest = remaining[0]["target_z_m"]
            tolerance = max(DEPTH_TIE_ABS, DEPTH_TIE_RTOL * abs(nearest))
            layer = [item for item in remaining if item["target_z_m"] <= nearest + tolerance]
            layer_ids = [item["surface_id"] for item in layer]
            layers.append({
                "nearest_z_m": nearest,
                "tie_tolerance_m": tolerance,
                "max_z_m": max(item["target_z_m"] for item in layer),
                "contributor_count": len(layer),
                "coverage_weight_sum_float64_diagnostic": float(sum(item["footprint_weight"] for item in layer)),
                "surface_id_min": min(layer_ids),
                "surface_id_max": max(layer_ids),
                "contributors": layer,
            })
            selected = {id(item) for item in layer}
            remaining = [item for item in remaining if id(item) not in selected]
        x, y = target_ids[tid]
        traces[(x, y)] = layers
    return traces


def _device_control(image, depth, uv, z, valid, seg, device):
    if device == "numpy":
        return _splat_projected_coverage_diagnostics(
            image, depth, uv, z, valid, seg, MIN_COVERAGE, DEPTH_TIE_ABS, DEPTH_TIE_RTOL,
        )
    out, diag = splat_cuda(image, depth, (uv, z, valid), seg)
    return out, diag


def run_minimal_controls():
    """Threshold-adjacent coverage, unsupported sliver, and front/back occlusion."""
    H, W = 8, 10
    depth = np.ones((H, W), dtype=np.float32)
    image = np.zeros((H, W), dtype=np.float32)
    seg = np.ones((H, W), dtype=np.int32)
    valid = np.zeros((H, W), dtype=bool)
    image[3, 3] = 0.8
    valid[3, 3] = True
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    uv0 = np.stack((xx, yy)).astype(np.float32)
    adjacent = []
    center = np.float32(1.005)
    values = [np.nextafter(center, np.float32(-np.inf)), center, np.nextafter(center, np.float32(np.inf))]
    for value in values:
        uv = uv0.copy()
        uv[0, 3, 3] = value
        row = {"projected_u": float(value), "coverage_threshold": MIN_COVERAGE}
        for backend in ("numpy", "cuda"):
            out, d = _device_control(image, depth, uv, depth, valid, seg, backend)
            row[f"{backend}_coverage_at_target_x1_y3"] = float(d["coverage"][3, 1])
            row[f"{backend}_support_at_target_x1_y3"] = bool(d["support"][3, 1])
            row[f"{backend}_visible_at_target_x2_y3"] = bool(d["visibility"][3, 2])
            row[f"{backend}_support_at_target_x2_y3"] = bool(d["support"][3, 2])
        adjacent.append(row)

    # Tiny footprint must remain visible-but-unsupported on both backends.
    tiny_uv = uv0.copy()
    tiny_uv[0, 3, 3] = np.float32(3.000001)
    tiny = {"expected": "visible but unsupported"}
    for backend in ("numpy", "cuda"):
        _out, d = _device_control(image, depth, tiny_uv, depth, valid, seg, backend)
        tiny[f"{backend}_neighbor_coverage"] = float(d["coverage"][3, 4])
        tiny[f"{backend}_neighbor_visible"] = bool(d["visibility"][3, 4])
        tiny[f"{backend}_neighbor_supported"] = bool(d["support"][3, 4])

    occ_img = np.full((H, W), 0.02, dtype=np.float32)
    occ_depth = np.full((H, W), 2.0, dtype=np.float32)
    occ_seg = np.ones((H, W), dtype=np.int32)
    occ_valid = np.ones((H, W), dtype=bool)
    occ_img[2:6, 2:6], occ_depth[2:6, 2:6], occ_seg[2:6, 2:6] = 0.8, 1.0, 2
    occ_uv = np.stack((xx + 0.25, yy)).astype(np.float32)
    occ_rows = {}
    for backend in ("numpy", "cuda"):
        out, d = _device_control(occ_img, occ_depth, occ_uv, occ_depth, occ_valid, occ_seg, backend)
        occ_rows[backend] = {
            "support_count": int(d["support"].sum()),
            "visibility_count": int(d["visibility"].sum()),
            "edge_luma_high": float(out[3, 2]),
            "edge_luma_low": float(out[3, 6]),
            "edge_surface_consistent": [bool(d["surface_consistent"][3, x]) for x in (2, 6)],
        }
    near_threshold_equal = all(
        row["numpy_support_at_target_x1_y3"] == row["cuda_support_at_target_x1_y3"]
        for row in adjacent
    )
    tiny_valid = all(
        tiny[f"{backend}_neighbor_visible"] and not tiny[f"{backend}_neighbor_supported"]
        for backend in ("numpy", "cuda")
    )
    occlusion_valid = all(
        occ_rows[backend]["edge_luma_high"] > occ_rows[backend]["edge_luma_low"]
        and occ_rows[backend]["edge_surface_consistent"] == [True, True]
        for backend in ("numpy", "cuda")
    ) and occ_rows["numpy"]["support_count"] == occ_rows["cuda"]["support_count"]
    return {
        "near_threshold": adjacent,
        "tiny_footprint": tiny,
        "front_back_occlusion": occ_rows,
        "control_gates": {
            "near_threshold_cpu_cuda_support_equal": near_threshold_equal,
            "tiny_positive_footprint_visible_but_unsupported_both": tiny_valid,
            "front_back_occlusion_preserved_both": occlusion_valid,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "benchmark/robot_closed_loop/configs/r1_5_cuda_support_forensics_20261031a.json")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059")
    parser.add_argument("--prior-arrays", type=Path, default=ROOT / "benchmark/robot_closed_loop/results/r1_5_core_cuda_parity_20261030c/cuda_parity_arrays.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this forensic run")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    prior = np.load(args.prior_arrays)
    device_info = torch.cuda.get_device_properties(0)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    point_rows, layer_records, event_rows = [], [], []
    import platform
    import subprocess
    try:
        source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        source_commit = __import__("os").environ.get("GENESIS_EVENT_PLUGIN_SOURCE_COMMIT", "source_commit_unavailable")
    summary = {"run_id": config["run_id"],
               "source_commit": source_commit,
               "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
               "cuda_runtime": torch.version.cuda,
               "gpu": {"name": device_info.name, "capability": list(torch.cuda.get_device_capability(0))},
               "inputs": {}, "episodes": [], "controls": None}
    summary["source_sha256"] = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in (Path(__file__).resolve(), ROOT / "genesis_event_plugin/physics_warp.py",
                     ROOT / "genesis_event_plugin/torch_physics.py",
                     ROOT / "genesis_event_plugin/dvs_emulator.py")
    }
    summary["config_sha256"] = sha256(args.config.resolve())
    summary["prior_result_sha256"] = sha256(args.prior_arrays.resolve())
    summary["prior_acceptance"] = config["acceptance"]
    array_store = {}

    for ep in ("004", "005"):
        src, dense_times, dense_luma, _rgb, _seg, width, height, fov, cache_path, evidence_path = load_cached_episode(args.cache_root.resolve(), ep)
        left = int(np.flatnonzero(np.isclose(src["timestamps_s"], T0, atol=1e-9, rtol=0))[0])
        right = int(np.flatnonzero(np.isclose(src["timestamps_s"], T1, atol=1e-9, rtol=0))[0])
        focal = height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        K = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
        ids = src["segmentation_pose_ids"]
        objects0 = {int(sid): src["object_poses"][left, j] for j, sid in enumerate(ids)}
        objects1 = {int(sid): src["object_poses"][right, j] for j, sid in enumerate(ids)}
        images = [src["radiance_luma"][left], src["radiance_luma"][right]]
        depths = [src["depth"][left], src["depth"][right]]
        segs = [src["segmentation"][left], src["segmentation"][right]]
        projs = {"numpy_projection": project_numpy_pair(src, K, left, right, objects0, objects1),
                 "cuda_projection": project_cuda_pair(src, K, left, right, objects0, objects1)}
        xor = prior[f"ep{ep}_numpy_support"].astype(bool) ^ prior[f"ep{ep}_cuda_support"].astype(bool)
        targets = [(int(x), int(y)) for y, x in np.argwhere(xor)]
        episode = {"episode": ep, "state_cache_sha256": sha256(cache_path), "direct_hdr_evidence_sha256": sha256(evidence_path),
                   "original_xor_target_count": len(targets), "projection_delta": projection_stats(projs["numpy_projection"], projs["cuda_projection"]),
                   "cases": []}
        summary["inputs"][ep] = {"state_cache_sha256": episode["state_cache_sha256"], "direct_hdr_evidence_sha256": episode["direct_hdr_evidence_sha256"]}

        episode["dvs_backend_comparisons"] = []
        projection_outputs = {}
        for projection_name, projections in projs.items():
            for side, (uv, z, valid_proj) in enumerate(projections):
                array_store[f"{ep}_{projection_name}_uv{side}"] = np.asarray(uv, dtype=np.float32)
                array_store[f"{ep}_{projection_name}_z{side}"] = np.asarray(z, dtype=np.float32)
                array_store[f"{ep}_{projection_name}_valid{side}"] = np.asarray(valid_proj, dtype=np.uint8)
            source_traces = [trace_target_layers(projections[i], np.asarray(projections[i][2]), segs[i], targets) for i in (0, 1)]
            for side in (0, 1):
                for x, y in targets:
                    layer_records.append({"episode": ep, "projection": projection_name, "endpoint": side, "x": x, "y": y,
                                          "layers": source_traces[side].get((x, y), [])})
            outputs = {}
            diags = {}
            for compositor in ("numpy", "cuda"):
                if compositor == "numpy":
                    out, valid, diag = composite_numpy(images, depths, segs, projections)
                else:
                    out, valid, diag = composite_cuda(images, depths, segs, projections)
                out = np.asarray(out, dtype=np.float32)
                valid = np.asarray(valid, dtype=bool)
                filled = out.copy()
                filled[~valid] = images[1][~valid]  # same alpha=.5 hold policy as the archived replay
                outputs[compositor] = (out, valid, filled)
                diags[compositor] = diag
                array_store[f"{ep}_{projection_name}_{compositor}_midpoint"] = out
                array_store[f"{ep}_{projection_name}_{compositor}_valid"] = valid.astype(np.uint8)
                case = {"projection": projection_name, "compositor": compositor,
                        "final_mask_xor_vs_other_compositor": None,
                        "valid_fraction": float(valid.mean()),
                        "invalid_reason_counts": {"coverage0": int((~diag["support0"]).sum()), "coverage1": int((~diag["support1"]).sum()),
                                                   "surface_consistency0": int((~diag["surface_consistent0"]).sum()),
                                                   "surface_consistency1": int((~diag["surface_consistent1"]).sum()),
                                                   "surface_id": int((diag["surface_id0"] != diag["surface_id1"]).sum()),
                                                   "depth_match": int((~np.isfinite(diag["z_buffer0"]) | ~np.isfinite(diag["z_buffer1"]) | (np.abs(diag["z_buffer0"] - diag["z_buffer1"]) > np.maximum(SURFACE_DEPTH_ABS, SURFACE_DEPTH_RTOL * np.maximum(np.abs(diag["z_buffer0"]), np.abs(diag["z_buffer1"]))))).sum())}}
                target_reason_counts = {}
                for x, y in targets:
                    reasons = rejection_reasons(diag, y, x)
                    row = {"episode": ep, "projection": projection_name, "compositor": compositor, "x": x, "y": y,
                           "prior_numpy_final_valid": bool(prior[f"ep{ep}_numpy_support"][y, x]),
                           "prior_cuda_final_valid": bool(prior[f"ep{ep}_cuda_support"][y, x]),
                           "final_valid": bool(valid[y, x]), "reject_reasons": reasons}
                    for reason in reasons:
                        target_reason_counts[reason] = target_reason_counts.get(reason, 0) + 1
                    for side in (0, 1):
                        for key in ("coverage", "support", "visibility", "z_buffer", "surface_id", "surface_consistent"):
                            value = diag[f"{key}{side}"][y, x]
                            row[f"{key}{side}"] = float(value) if key in {"coverage", "z_buffer"} else (int(value) if key == "surface_id" else bool(value))
                        uv, z, valid_proj = projections[side]
                        row[f"uv{side}_at_source_xy"] = [float(uv[0, y, x]), float(uv[1, y, x])]
                        row[f"zproj{side}_at_source_xy"] = float(z[y, x])
                        row[f"projection_valid{side}_at_source_xy"] = bool(valid_proj[y, x])
                        row[f"depth_layers{side}"] = source_traces[side].get((x, y), [])
                    z0, z1 = float(diag["z_buffer0"][y, x]), float(diag["z_buffer1"][y, x])
                    row["cross_endpoint_depth_abs_delta_m"] = abs(z0 - z1)
                    row["cross_endpoint_depth_tolerance_m"] = max(SURFACE_DEPTH_ABS, SURFACE_DEPTH_RTOL * max(abs(z0), abs(z1)))
                    point_rows.append({k: json.dumps(v, separators=(",", ":"), default=json_default) if isinstance(v, (list, dict)) else v for k, v in row.items()})
                case["original_xor_target_rejection_counts"] = target_reason_counts
                episode["cases"].append(case)

            projection_outputs[projection_name] = outputs

            cpu_out, cpu_valid, cpu_filled = outputs["numpy"]
            cuda_out, cuda_valid, cuda_filled = outputs["cuda"]
            common = cpu_valid & cuda_valid
            log_delta = np.abs(np.log(np.maximum(cpu_out[common], 1e-6)) - np.log(np.maximum(cuda_out[common], 1e-6))) if common.any() else np.asarray([])
            cpu_diag, cuda_diag = diags["numpy"], diags["cuda"]
            finite_z_deltas = []
            for i in (0, 1):
                finite = np.isfinite(cpu_diag[f"z_buffer{i}"]) & np.isfinite(cuda_diag[f"z_buffer{i}"])
                finite_z_deltas.append(float(np.max(np.abs(cpu_diag[f"z_buffer{i}"][finite] - cuda_diag[f"z_buffer{i}"][finite]))) if finite.any() else None)
            pair_summary = {
                "projection": projection_name,
                "same_projection_input_compositor_mask_xor_pixels": int(np.count_nonzero(cpu_valid != cuda_valid)),
                "same_projection_input_component_mismatches": {
                    key: int(np.count_nonzero(cpu_diag[key] != cuda_diag[key]))
                    for key in ("support0", "support1", "surface_consistent0", "surface_consistent1", "surface_id0", "surface_id1")
                },
                "coverage_abs_max_by_endpoint": [float(np.max(np.abs(cpu_diag[f"coverage{i}"] - cuda_diag[f"coverage{i}"]))) for i in (0, 1)],
                "coverage_abs_p99_by_endpoint": [float(np.quantile(np.abs(cpu_diag[f"coverage{i}"] - cuda_diag[f"coverage{i}"]), 0.99)) for i in (0, 1)],
                "zbuffer_abs_max_m_by_endpoint": finite_z_deltas,
                "same_projection_input_supported_log_luma_abs_max": float(log_delta.max()) if log_delta.size else None,
            }
            episode["cases"][-2]["final_mask_xor_vs_other_compositor"] = pair_summary["same_projection_input_compositor_mask_xor_pixels"]
            episode["cases"][-1]["final_mask_xor_vs_other_compositor"] = pair_summary["same_projection_input_compositor_mask_xor_pixels"]

            # Four complete reconstruction -> DVS paths: 2 projection sources × 2 warp backends × 2 DVS backends.
            for compositor, (_out, _valid, midpoint) in outputs.items():
                frames = [midpoint, images[1]]
                frame_times = [0.5855, T1]
                event_streams = {}
                for dvs_backend in ("cpu", "cuda:0"):
                    events = run_dvs(frames, frame_times, images[0], T0, dvs_backend)
                    key = f"{compositor}_{dvs_backend.replace(':', '')}"
                    event_streams[key] = events
                    array_store[f"{ep}_{projection_name}_{key}_events"] = events
                for key, events in event_streams.items():
                    event_rows.append({"episode": ep, "projection": projection_name, "compositor": compositor,
                                       "dvs_backend": key.rsplit("_", 1)[-1], "event_count": int(len(events)),
                                       "on_count": int(np.count_nonzero(events[:, 3] > 0)), "off_count": int(np.count_nonzero(events[:, 3] < 0))})
                comparison = event_metrics(event_streams[f"{compositor}_cpu"], event_streams[f"{compositor}_cuda0"], width)
                comparison["event_count_relative_delta"] = abs(comparison["candidate_event_count"] - comparison["reference_event_count"]) / max(1, comparison["reference_event_count"])
                comparison_row = {"projection": projection_name, "compositor": compositor, **comparison}
                episode["dvs_backend_comparisons"].append(comparison_row)
                event_rows.append({"episode": ep, "projection": projection_name, "compositor": compositor,
                                   "comparison": "cpu_vs_cuda_dvs", **comparison})
            episode.setdefault("same_projection_compositor_pair", []).append(pair_summary)
        episode["projector_effects_same_compositor"] = {}
        for compositor in ("numpy", "cuda"):
            np_out, np_valid, _ = projection_outputs["numpy_projection"][compositor]
            cu_out, cu_valid, _ = projection_outputs["cuda_projection"][compositor]
            common_projection = np_valid & cu_valid
            delta = np.abs(np.log(np.maximum(np_out[common_projection], 1e-6)) - np.log(np.maximum(cu_out[common_projection], 1e-6))) if common_projection.any() else np.asarray([])
            episode["projector_effects_same_compositor"][compositor] = {
                "final_mask_xor_pixels": int(np.count_nonzero(np_valid != cu_valid)),
                "common_supported_log_luma_abs_max": float(delta.max()) if delta.size else None,
            }
        native_numpy_out, native_numpy_valid, _ = projection_outputs["numpy_projection"]["numpy"]
        native_cuda_out, native_cuda_valid, _ = projection_outputs["cuda_projection"]["cuda"]
        native_common = native_numpy_valid & native_cuda_valid
        native_delta = np.abs(
            np.log(np.maximum(native_numpy_out[native_common], 1e-6))
            - np.log(np.maximum(native_cuda_out[native_common], 1e-6))
        ) if native_common.any() else np.asarray([])
        native_event_pair = event_metrics(
            array_store[f"{ep}_numpy_projection_numpy_cpu_events"],
            array_store[f"{ep}_cuda_projection_cuda_cuda0_events"], width,
        )
        episode["native_end_to_end"] = {
            "support_mask_xor_pixels": int(np.count_nonzero(native_numpy_valid != native_cuda_valid)),
            "common_supported_log_luma_abs_max": float(native_delta.max()) if native_delta.size else None,
            "numpy_projection_numpy_compositor_cpu_dvs_vs_cuda_projection_cuda_compositor_cuda_dvs": native_event_pair,
        }
        episode["production_interpolator"] = run_production_interpolator_pair(
            src, K, left, right, objects0, objects1, width, event_rows, array_store, ep,
        )
        strict_sample_check = strict_cached_replay(args.cache_root.resolve(), ep)
        strict_sample_arrays = strict_sample_check.pop("event_arrays")
        episode["strict_actual_intermediate_radiance"] = strict_sample_check
        for key, value in strict_sample_arrays.items():
            array_store[f"{ep}_strict_actual_{key}"] = value
        summary["episodes"].append(episode)

    renderer_cost_path = ROOT / "benchmark/robot_closed_loop/results/r1_5_multiref_surface_cached_replay_20261021d/summary.json"
    renderer_cost_summary = json.loads(renderer_cost_path.read_text(encoding="utf-8"))
    summary["photometric_sampling_and_render_cost"] = {
        "exact_cached_direct_hdr_samples": {
            item["episode"]: {
                "sample_times_s": item["timestamps_s"],
                "exact_image_sequence_match": item["exact_image_sequence_match"],
                "raw_events_exact_match": item["raw_events_exact_match"],
                "maximum_anchor_gap_s": item["max_anchor_gap_s"],
                "output_available_at_s": item["output_available_at_s"],
                "default_motion_sampler_requests_beyond_one_midpoint": item["default_motion_sampler_midpoint_only_probe"]["required_additional_sample_count"],
                "one_midpoint_5ms_event_count": item["one_midpoint_5ms_sampling_ablation"]["midpoint_sample_events"],
                "one_midpoint_5ms_reference_event_count": item["one_midpoint_5ms_sampling_ablation"]["direct_reference_events_1ms"],
                "one_midpoint_5ms_event_pixel_polarity_l1": item["one_midpoint_5ms_sampling_ablation"]["event_metrics_vs_dense_1ms_direct"]["pixel_polarity_count_l1"],
                "one_midpoint_5ms_event_time_p95_error_s": item["one_midpoint_5ms_sampling_ablation"]["event_metrics_vs_dense_1ms_direct"]["matched_event_time_abs_error_p95_s"],
                "cached_api_wall_s_cpu": item["strict_api_wall_s_cpu_using_cached_frames"],
                "new_renderer_cost_measured": item["new_renderer_cost_measured"],
            }
            for item in summary["episodes"]
        },
        "measured_full_frame_fallback_reference": {
            "source_summary_path": renderer_cost_path.relative_to(ROOT).as_posix(),
            "source_summary_sha256": sha256(renderer_cost_path),
            "source_code_commit": renderer_cost_summary["code_commit"],
            "episodes": {
                key: {
                    "fallback_rendered_frames": value["candidate"]["full_frame_fallback_count"],
                    "fallback_render_wall_s": value["candidate"]["fallback_render_wall_s"],
                    "candidate_total_wall_s": value["candidate"]["total_production_wall_s_estimate"],
                    "direct_render_wall_s_47_states": value["direct_render"]["wall_s_all_for_audit"],
                    "direct_dvs_wall_s": value["direct_logged_state_comparator"]["dvs_wall_s"],
                    "direct_render_plus_dvs_wall_s": value["direct_render"]["wall_s_all_for_audit"] + value["direct_logged_state_comparator"]["dvs_wall_s"],
                    "candidate_over_direct_pipeline_ratio": value["candidate"]["total_production_wall_s_estimate"] / max(
                        value["direct_render"]["wall_s_all_for_audit"] + value["direct_logged_state_comparator"]["dvs_wall_s"], 1e-12,
                    ),
                }
                for key, value in renderer_cost_summary["episodes"].items()
            },
            "scope_note": "Historical actual Genesis rerender measurement of the multi-reference candidate on 42 intermediate states per 40 ms clip; it bounds the cost of full-frame fallback, not the exact cost of the current six-request 10 ms schedule.",
        },
        "current_strict_provider_cost_measured": False,
        "freeze_recommendation": "do_not_freeze_default_core_until exact-time intermediate render-provider wall cost and online availability are measured; strict cached frames prove temporal fidelity but do not measure a live provider",
    }

    summary["controls"] = run_minimal_controls()
    threshold_rows = run_threshold_probes()
    summary["threshold_probe_cpu_cuda_arrays_exact"] = [bool(row["raw_event_arrays_exact"]) for row in threshold_rows]
    summary["threshold_probe_results"] = []
    for index, row in enumerate(threshold_rows):
        summary["threshold_probe_results"].append({k: v for k, v in row.items() if k not in {"cpu_events", "cuda_events"}})
        array_store[f"threshold_{index:02d}_cpu_events"] = row["cpu_events"]
        array_store[f"threshold_{index:02d}_cuda_events"] = row["cuda_events"]
    acceptance = config["acceptance"]
    threshold_exact = all(summary["threshold_probe_cpu_cuda_arrays_exact"])
    episode_gates = {}
    for episode in summary["episodes"]:
        shared_projection_gates = {
            row["projection"]: {
                "support_mask_exact": row["same_projection_input_compositor_mask_xor_pixels"] == 0,
                "supported_log_luma_max_within_limit": row["same_projection_input_supported_log_luma_abs_max"] is not None
                    and row["same_projection_input_supported_log_luma_abs_max"] <= acceptance["max_abs_log_luminance_delta_supported_pixels"],
            }
            for row in episode["same_projection_compositor_pair"]
        }
        event_gates = {
            f"{row['projection']}:{row['compositor']}": row["event_count_relative_delta"] <= acceptance["cached_clip_event_count_relative_delta_max"]
            for row in episode["dvs_backend_comparisons"]
        }
        episode_gates[episode["episode"]] = {
            "same_projection_compositor": shared_projection_gates,
            "cpu_cuda_dvs_event_count_within_limit": event_gates,
            "native_end_to_end": {
                "support_mask_exact": episode["native_end_to_end"]["support_mask_xor_pixels"] == 0,
                "supported_log_luma_max_within_limit": episode["native_end_to_end"]["common_supported_log_luma_abs_max"] is not None
                    and episode["native_end_to_end"]["common_supported_log_luma_abs_max"] <= acceptance["max_abs_log_luminance_delta_supported_pixels"],
            },
            "public_PhysicsInterpolator_end_to_end": {
                "support_mask_exact_all_samples": episode["production_interpolator"]["support_mask_xor_pixels_all_frames"] == 0,
                "supported_log_luma_max_within_limit_all_samples": episode["production_interpolator"]["common_supported_log_luma_abs_max_all_frames"] is not None
                    and episode["production_interpolator"]["common_supported_log_luma_abs_max_all_frames"] <= acceptance["max_abs_log_luminance_delta_supported_pixels"],
                "cpu_cuda_dvs_event_counts_within_limit": all(
                    row["event_count_relative_delta"] <= acceptance["cached_clip_event_count_relative_delta_max"]
                    for row in episode["production_interpolator"]["dvs_backend_comparisons_per_reconstruction"]
                ),
            },
            "all_pass": all(all(gates.values()) for gates in shared_projection_gates.values())
                and all(event_gates.values())
                and episode["native_end_to_end"]["support_mask_xor_pixels"] == 0
                and episode["native_end_to_end"]["common_supported_log_luma_abs_max"] is not None
                and episode["native_end_to_end"]["common_supported_log_luma_abs_max"] <= acceptance["max_abs_log_luminance_delta_supported_pixels"]
                and episode["production_interpolator"]["support_mask_xor_pixels_all_frames"] == 0
                and episode["production_interpolator"]["common_supported_log_luma_abs_max_all_frames"] is not None
                and episode["production_interpolator"]["common_supported_log_luma_abs_max_all_frames"] <= acceptance["max_abs_log_luminance_delta_supported_pixels"]
                and all(row["event_count_relative_delta"] <= acceptance["cached_clip_event_count_relative_delta_max"]
                        for row in episode["production_interpolator"]["dvs_backend_comparisons_per_reconstruction"]),
        }
    control_gates = summary["controls"]["control_gates"]
    summary["acceptance_result"] = {
        "threshold_dvs_cpu_cuda_exact": threshold_exact,
        "controls": control_gates,
        "episodes": episode_gates,
        "all_gates_pass": threshold_exact and all(control_gates.values()) and all(row["all_pass"] for row in episode_gates.values()),
        "criteria": acceptance,
    }
    summary["wall_s_total"] = float(time.perf_counter() - start)
    summary["scope"] = "forensic parity only; renderer, physics advancement, training and ranking excluded"
    summary["replay_command"] = (
        "python -u benchmark/robot_closed_loop/diagnostics/diagnose_r1_5_cuda_support_forensics.py "
        "--cache-root <verified-cache-root> --prior-arrays "
        "benchmark/robot_closed_loop/results/r1_5_core_cuda_parity_20261030c/cuda_parity_arrays.npz "
        f"--config {args.config.as_posix()} "
        f"--output {args.output.as_posix()}"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8",
    )
    with (args.output / "point_components.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(point_rows[0]))
        writer.writeheader(); writer.writerows(point_rows)
    with (args.output / "event_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in event_rows for key in row}))
        writer.writeheader(); writer.writerows(event_rows)
    with (args.output / "layer_contributors.jsonl").open("w", encoding="utf-8") as f:
        for row in layer_records:
            f.write(json.dumps(row, separators=(",", ":"), default=json_default) + "\n")
    np.savez_compressed(args.output / "forensic_arrays.npz", **array_store)
    (args.output / "README.md").write_text(
        f"# {args.output.name}\n\nSame-projection-input and independently projected CPU/CUDA support trace on cached episodes 004/005. "
        "Results do not validate renderer fidelity or closed-loop perception. See `summary.json`, `point_components.csv`, `layer_contributors.jsonl`, `event_metrics.csv`, and `forensic_arrays.npz`.\n",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            manifest.append(f"{sha256(path)}  {path.name}\n")
    (args.output / "SHA256SUMS").write_text("".join(manifest), encoding="utf-8")
    print(json.dumps({"run_id": args.output.name, "episodes": [
        {"episode": e["episode"], "projection_delta": e["projection_delta"], "same_projection_pairs": e["same_projection_compositor_pair"]}
        for e in summary["episodes"]], "controls": summary["controls"], "wall_s_total": summary["wall_s_total"]}, indent=2))


if __name__ == "__main__":
    main()
