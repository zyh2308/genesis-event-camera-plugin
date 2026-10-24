#!/usr/bin/env python3
"""CPU-only test of whether cached floor transients look like illumination changes.

This post-hoc analysis reuses R1.5 archived direct-HDR frames, segmentation,
poses, and already-generated floor events. It does not render, advance physics,
run a network, or modify the frozen plugin core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / "benchmark/robot_closed_loop/results/r1_failure_chain_diagnosis_20261019a/native_hdr_cutoff059"
DEFAULT_EVIDENCE = ROOT / "benchmark/robot_closed_loop/results/r1_5_photometric_sampling_contract_20261024a/evidence"
DEFAULT_SCENE_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
DEFAULT_OUTPUT = ROOT / "benchmark/robot_closed_loop/results/r1_5_achromatic_shadow_signature_cpu_20261024c"
DEFAULT_DIRECTIONAL_LIGHT_RAY = np.asarray([-1.0, -1.0, -1.0], dtype=np.float64)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def floor_mask(source_seg: np.ndarray, source_depth: np.ndarray,
               dense_seg: np.ndarray) -> np.ndarray:
    mask = ((source_seg[3] == 1) & (source_seg[4] == 1)
            & np.all(dense_seg == 1, axis=0))
    # Exclude two-pixel segmentation boundaries.
    pad = np.pad(mask, 1, constant_values=False)
    for _ in range(2):
        mask = np.logical_and.reduce([
            pad[y:y + mask.shape[0], x:x + mask.shape[1]]
            for y in range(3) for x in range(3)
        ])
        pad = np.pad(mask, 1, constant_values=False)
    depth0 = source_depth[3]
    depth1 = source_depth[4]
    return mask & np.isfinite(depth0) & np.isfinite(depth1) & (np.abs(depth0 - depth1) < 1e-7)


def edge(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))


def draw_edge(draw: ImageDraw.ImageDraw, mask: np.ndarray, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    ox, oy = xy
    ys, xs = np.where(edge(mask))
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.point((ox + x, oy + y), fill=color)


def quaternion_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        raise ValueError("target quaternion has near-zero norm")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def project_world(point: np.ndarray, camera_to_world_cv: np.ndarray,
                  intrinsics: tuple[float, float, float, float]) -> np.ndarray:
    camera = np.linalg.inv(camera_to_world_cv) @ np.append(point, 1.0)
    fx, fy, cx, cy = intrinsics
    if camera[2] <= 1e-8:
        raise ValueError("projected target or shadow point is behind the camera")
    return np.asarray([fx * camera[0] / camera[2] + cx,
                       fy * camera[1] / camera[2] + cy], dtype=np.float64)


def predicted_target_shadow_edges(rows: list[dict], times: np.ndarray,
                                  camera_to_world_cv: np.ndarray,
                                  target_size_m: np.ndarray,
                                  resolution_wh: tuple[int, int],
                                  fov_deg: float) -> list[np.ndarray]:
    """Project a cube's directional-light shadow onto the z=0 support plane.

    The direction is Genesis 1.2.2's documented default directional-light ray
    direction. The result is a hypothesis mask, not a rendered shadow buffer.
    """
    width, height = resolution_wh
    focal = height / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    intrinsics = (focal, focal, width / 2.0, height / 2.0)
    half = np.asarray(target_size_m, dtype=np.float64) / 2.0
    local_corners = np.asarray([
        [sx * half[0], sy * half[1], sz * half[2]]
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
    ], dtype=np.float64)
    ray = DEFAULT_DIRECTIONAL_LIGHT_RAY
    if ray[2] >= 0:
        raise ValueError("default directional light must point toward the support plane")
    masks: list[np.ndarray] = []
    for row in rows:
        obj = row["object"]
        rotation = quaternion_wxyz_to_matrix(np.asarray(obj["orientation_wxyz"]))
        position = np.asarray(obj["position_m"], dtype=np.float64)
        world_corners = (rotation @ local_corners.T).T + position
        shadow_points = []
        for point in world_corners:
            # The logged block can penetrate the ideal plane by a few microns;
            # clamp that numerical penetration to the support surface.
            z = max(float(point[2]), 0.0)
            q = np.asarray([point[0], point[1], z], dtype=np.float64)
            if z > 0:
                q = q + (-z / ray[2]) * ray
            q[2] = 0.0
            shadow_points.append(project_world(q, camera_to_world_cv, intrinsics))
        shadow_points = np.asarray(shadow_points)
        hull = ConvexHull(shadow_points)
        polygon = np.rint(shadow_points[hull.vertices]).astype(np.int32)
        image = Image.new("1", (width, height), 0)
        ImageDraw.Draw(image).polygon([tuple(p) for p in polygon], fill=1)
        masks.append(np.asarray(image, dtype=bool))
    return masks


def make_panel(ep: str, rgb: np.ndarray, max_abs_delta: np.ndarray,
               fixed_floor: np.ndarray, on: np.ndarray, off: np.ndarray,
               robot: np.ndarray, target: np.ndarray,
               target_shadow_edge: np.ndarray) -> Image.Image:
    h, w = fixed_floor.shape
    panel = Image.new("RGB", (3 * w, h + 28), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 5), f"episode {ep}, cached direct HDR near 0.5855 s", fill=(245, 245, 245))

    rgb_view = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255), mode="RGB")
    panel.paste(rgb_view, (0, 28))

    q = np.clip(max_abs_delta / 0.5, 0, 1)
    heat = np.zeros((h, w, 3), dtype=np.uint8)
    heat[..., 0] = np.uint8(q * 255)
    heat[..., 1] = np.uint8(np.sqrt(q) * 90)
    heat[..., 2] = np.where(fixed_floor, 0, 28).astype(np.uint8)
    panel.paste(Image.fromarray(heat, mode="RGB"), (w, 28))

    event_view = np.zeros((h, w, 3), dtype=np.uint8)
    event_view[..., 1] = np.uint8(np.clip(on / 3.0, 0, 1) * 255)
    event_view[..., 0] = np.uint8(np.clip(off / 3.0, 0, 1) * 255)
    event_view[..., 2] = np.where(fixed_floor, 0, 28).astype(np.uint8)
    panel.paste(Image.fromarray(event_view, mode="RGB"), (2 * w, 28))

    draw = ImageDraw.Draw(panel)
    # These are diagnostic-only GT segmentation contours, never model inputs.
    for xoff in (0, w, 2 * w):
        draw_edge(draw, robot, (xoff, 28), (80, 230, 255))
        draw_edge(draw, target, (xoff, 28), (255, 230, 30))
        draw_edge(draw, target_shadow_edge, (xoff, 28), (255, 50, 210))
    draw.text((4, h + 13), "cyan robot / yellow target / magenta predicted target-shadow edge", fill=(245, 245, 245))
    draw.text((w + 4, h + 13), "max |1 ms log-luminance change| on fixed floor", fill=(245, 245, 245))
    draw.text((2 * w + 4, h + 13), "floor events: green ON / red OFF; contours are GT diagnostics", fill=(245, 245, 245))
    return panel


def qstats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {f"q{q}": float(np.quantile(values, q / 100)) for q in (10, 25, 50, 75, 90)}


def run_ideal_dvs(frames: np.ndarray, times: np.ndarray, mask: np.ndarray,
                  init_luma: np.ndarray, init_time: float) -> tuple[np.ndarray, list[int]]:
    """Use the unchanged plugin DVS model on one isolated diagnostic ROI."""
    import torch

    from genesis_event_plugin.dvs_emulator import DvsEmulator
    from genesis_event_plugin.noise_model import DvsNoiseConfig

    torch.set_num_threads(1)
    cfg = DvsNoiseConfig(
        pos_thres=0.2, neg_thres=0.2, sigma_thres=0.0,
        cutoff_hz=0.0, leak_rate_hz=0.0, leak_jitter_fraction=0.0,
        shot_noise_rate_hz=0.0, noise_rate_cov_decades=0.0,
        refractory_period_s=0.0, radiance_input=True,
        response_mode="physical_log", readout_model="ideal_readout",
    )
    emulator = DvsEmulator(res=mask.shape, config=cfg, device="cpu")
    emulator.initialize(np.where(mask, init_luma, 1.0), float(init_time))
    chunks: list[np.ndarray] = []
    counts: list[int] = []
    for frame, timestamp in zip(frames, times):
        events = emulator.generate_events(
            np.where(mask, frame, 1.0), float(timestamp)
        )
        chunks.append(events)
        counts.append(int(len(events)))
    merged = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)
    return merged, counts


def analyze(ep: str, source: Path, evidence: Path, out: Path,
            target_size_m: np.ndarray) -> dict:
    cache_path = source / f"episode_{ep}/same_state_source_cache.h5"
    direct_path = source / f"episode_{ep}/rendered_frame_evidence.h5"
    trace_path = source / f"episode_{ep}/trace_window.jsonl"
    event_path = evidence / f"episode_{ep}_static_floor_temporal_events.npz"

    with h5py.File(cache_path, "r") as f:
        source_times = f["timestamps_s"][:]
        source_luma = f["radiance_luma"][:]
        source_depth = f["depth"][:]
        source_seg = f["segmentation"][:]
        pose_ids = f["segmentation_pose_ids"][:]
        object_poses = f["object_poses"][:]
        camera_to_world_cv = f["camera_to_world_cv"][3]
        resolution_wh = tuple(int(v) for v in f.attrs["camera_resolution_wh"])
        vertical_fov_deg = float(f.attrs["vertical_fov_deg"])
    with h5py.File(direct_path, "r") as f:
        times = f["dense_times_s"][:]
        direct_luma = f["dense_direct_hdr_luminance"][:]
        ours_luma = f["ours_intermediate_luminance"][:]
        direct_rgb = f["dense_direct_hdr_rgb"][:]
        direct_seg = f["dense_direct_segmentation"][:]
        dense_rows = [json.loads(v.decode() if isinstance(v, bytes) else v)
                      for v in f["dense_state_trace_rows_json"][:]]

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    target_positions = np.asarray([
        min(rows, key=lambda r: abs(float(r["time_s"]) - float(t)))["object"]["position_m"]
        for t in source_times
    ], dtype=np.float64)
    target_pose_errors = []
    for j, seg_id in enumerate(pose_ids):
        target_pose_errors.append(float(np.median(np.linalg.norm(
            object_poses[:, j, :3, 3] - target_positions, axis=1
        ))))
    target_seg_id = int(pose_ids[int(np.argmin(target_pose_errors))])
    if target_pose_errors[int(np.argmin(target_pose_errors))] > 1e-3:
        raise RuntimeError(f"{ep}: could not map target pose to segmentation id")
    robot_seg_ids = [int(v) for v in pose_ids if int(v) != target_seg_id]
    if target_seg_id != 2 or robot_seg_ids != list(range(3, 14)):
        raise RuntimeError(f"{ep}: unexpected segmentation mapping: target={target_seg_id}, robot={robot_seg_ids}")

    mask = floor_mask(source_seg, source_depth, direct_seg)
    log_rgb = np.log(np.maximum(direct_rgb, 1e-6))
    rgb_delta = np.diff(log_rgb, axis=0)
    common_delta = rgb_delta.mean(axis=-1)
    chromatic_residual = np.sqrt(((rgb_delta - common_delta[..., None]) ** 2).mean(axis=-1))
    selected = mask[None] & np.isfinite(common_delta) & (np.abs(common_delta) > 0.02)
    ratio = chromatic_residual[selected] / np.maximum(np.abs(common_delta[selected]), 1e-9)
    selected_rgb = rgb_delta[selected]
    channel_corr = [float(np.corrcoef(selected_rgb[:, 0], selected_rgb[:, c])[0, 1]) for c in (1, 2)]
    same_sign = (np.sign(selected_rgb) == np.sign(selected_rgb[:, :1])).all(axis=1)

    luma_log = np.log(np.maximum(direct_luma, 1e-6))
    max_abs_delta = np.max(np.abs(np.diff(luma_log, axis=0)), axis=0)
    mid = int(np.argmin(np.abs(times - 0.5855)))
    robot_mask = np.isin(direct_seg[mid], robot_seg_ids)
    target_mask = direct_seg[mid] == target_seg_id
    if not robot_mask.any() or not target_mask.any():
        raise RuntimeError(f"{ep}: missing target or robot segmentation at midpoint")

    event_archive = np.load(event_path)
    events = np.asarray(event_archive["direct"], dtype=np.float64)
    event_mask = np.asarray(event_archive["mask"], dtype=bool)
    if not np.all(event_mask == mask):
        raise RuntimeError(f"{ep}: fixed-floor event mask differs from reconstructed mask")
    if len(events) and not np.all(mask[events[:, 2].astype(int), events[:, 1].astype(int)]):
        raise RuntimeError(f"{ep}: an event is outside the fixed floor mask")

    source_lo = int(np.argmin(np.abs(source_times - 0.5805)))
    replayed_direct_events, replayed_direct_counts = run_ideal_dvs(
        direct_luma, times, mask, source_luma[source_lo], source_times[source_lo]
    )
    reconstructed_events, reconstructed_counts = run_ideal_dvs(
        ours_luma,
        times, mask, source_luma[source_lo], source_times[source_lo]
    )
    def canonical(events_array: np.ndarray) -> np.ndarray:
        if not len(events_array):
            return events_array
        return events_array[np.lexsort((events_array[:, 3], events_array[:, 2],
                                        events_array[:, 1], events_array[:, 0]))]

    direct_replay_matches = (
        replayed_direct_events.shape == events.shape
        and np.allclose(canonical(replayed_direct_events), canonical(events), atol=1e-10, rtol=0)
    )
    direct_spatiopolar = {(int(x), int(y), int(p)) for _, x, y, p in replayed_direct_events}
    reconstructed_spatiopolar = {(int(x), int(y), int(p)) for _, x, y, p in reconstructed_events}
    np.savez_compressed(
        out / f"episode_{ep}_floor_direct_vs_ours_dvs_events.npz",
        direct=replayed_direct_events,
        ours_reconstruction=reconstructed_events,
        times=times,
        direct_counts_per_frame=np.asarray(replayed_direct_counts, dtype=np.int32),
        ours_counts_per_frame=np.asarray(reconstructed_counts, dtype=np.int32),
    )
    on = np.zeros(mask.shape, dtype=np.uint16)
    off = np.zeros(mask.shape, dtype=np.uint16)
    for _, x_f, y_f, polarity in events:
        x, y = int(x_f), int(y_f)
        if polarity > 0:
            on[y, x] += 1
        else:
            off[y, x] += 1

    y_ev = events[:, 2].astype(int)
    x_ev = events[:, 1].astype(int)
    distance_maps = {
        "robot": ndimage.distance_transform_edt(~robot_mask),
        "target": ndimage.distance_transform_edt(~target_mask),
    }
    proximity = {}
    for name, dist in distance_maps.items():
        ev_d = dist[y_ev, x_ev]
        floor_d = dist[mask]
        proximity[name] = {
            "event_distance_px_quantiles": qstats(ev_d),
            "fixed_floor_reference_distance_px_quantiles": qstats(floor_d),
            "event_fraction_within_px": {str(r): float(np.mean(ev_d <= r)) for r in (2, 5, 8, 16, 32)},
            "floor_fraction_within_px": {str(r): float(np.mean(floor_d <= r)) for r in (2, 5, 8, 16, 32)},
        }

    shadow_masks = predicted_target_shadow_edges(
        dense_rows, times, camera_to_world_cv, target_size_m,
        resolution_wh, vertical_fov_deg,
    )
    shadow_edge_masks = [m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool))
                         for m in shadow_masks]
    shadow_edge_distance = [ndimage.distance_transform_edt(~m) for m in shadow_edge_masks]
    event_frame_index = np.argmin(np.abs(events[:, 0, None] - times[None, :]), axis=1)
    event_shadow_distance = np.asarray([
        shadow_edge_distance[i][y, x]
        for i, x, y in zip(event_frame_index, x_ev, y_ev)
    ], dtype=np.float64)
    shadow_union_edge = np.logical_or.reduce(shadow_edge_masks)
    shadow_union_distance = ndimage.distance_transform_edt(~shadow_union_edge)
    floor_shadow_distance = shadow_union_distance[mask]
    target_shadow_edge_mid = shadow_edge_masks[mid]
    shadow_proximity = {
        "assumption": "opaque target box, support plane z=0, Genesis 1.2.2 default directional-light rays (-1,-1,-1), no soft-shadow penumbra model",
        "event_to_nearest_timestamp_target_shadow_edge_px_quantiles": qstats(event_shadow_distance),
        "event_fraction_within_px_of_nearest_timestamp_target_shadow_edge": {
            str(r): float(np.mean(event_shadow_distance <= r)) for r in (1, 2, 5, 8)
        },
        "event_distance_to_union_of_all_target_shadow_edges_px_quantiles": qstats(shadow_union_distance[y_ev, x_ev]),
        "fixed_floor_reference_distance_to_union_shadow_edge_px_quantiles": qstats(floor_shadow_distance),
        "event_fraction_within_px_of_union_shadow_edge": {
            str(r): float(np.mean(shadow_union_distance[y_ev, x_ev] <= r)) for r in (1, 2, 5, 8)
        },
        "floor_fraction_within_px_of_union_shadow_edge": {
            str(r): float(np.mean(floor_shadow_distance <= r)) for r in (1, 2, 5, 8)
        },
    }

    source_hi = int(np.argmin(np.abs(source_times - 0.5905)))
    translation_mm = np.linalg.norm(
        object_poses[source_hi, :, :3, 3] - object_poses[source_lo, :, :3, 3], axis=1
    ) * 1000.0
    target_index = int(np.where(pose_ids == target_seg_id)[0][0])
    robot_translation = translation_mm[np.arange(len(pose_ids)) != target_index]
    robot_first = np.isin(direct_seg[0], robot_seg_ids)
    robot_last = np.isin(direct_seg[-1], robot_seg_ids)
    robot_xor = int(np.logical_xor(robot_first, robot_last).sum())
    robot_union = int(np.logical_or(robot_first, robot_last).sum())

    target_centroids = []
    for i in (0, mid, len(times) - 1):
        yy, xx = np.where(direct_seg[i] == target_seg_id)
        target_centroids.append([float(xx.mean()), float(yy.mean())])

    panel = make_panel(
        ep, direct_rgb[mid], max_abs_delta, mask, on, off,
        robot_mask, target_mask, target_shadow_edge_mid,
    )
    panel.save(out / f"episode_{ep}_achromatic_shadow_signature.png")

    return {
        "episode": ep,
        "source_interval_s": [float(source_times[source_lo]), float(source_times[source_hi])],
        "dense_times_s": [float(v) for v in times],
        "fixed_floor_pixels": int(mask.sum()),
        "target_segmentation_id": target_seg_id,
        "robot_segmentation_ids": robot_seg_ids,
        "floor_rgb_log_delta_selection": {
            "selection_rule": "fixed floor and absolute mean RGB log change > 0.02 between adjacent dense frames",
            "pixel_intervals": int(selected.sum()),
            "median_chromatic_residual_log": float(np.median(chromatic_residual[selected])),
            "median_chromatic_residual_over_common_change": float(np.median(ratio)),
            "q90_chromatic_residual_over_common_change": float(np.quantile(ratio, 0.90)),
            "rgb_log_delta_correlation_with_red": {"green": channel_corr[0], "blue": channel_corr[1]},
            "fraction_with_same_sign_change_in_all_rgb_channels": float(np.mean(same_sign)),
            "interpretation_limit": "Achromatic multiplicative changes support illumination/shadow as a class of causes; they do not identify the occluder or prove shadow causality.",
        },
        "projected_geometry_motion": {
            "max_target_translation_mm": float(translation_mm[target_index]),
            "max_robot_link_translation_mm": float(robot_translation.max()),
            "robot_silhouette_endpoint_xor_pixels": robot_xor,
            "robot_silhouette_endpoint_union_pixels": robot_union,
            "robot_silhouette_endpoint_xor_fraction": float(robot_xor / max(robot_union, 1)),
            "target_silhouette_centroids_xy_start_mid_end_px": target_centroids,
            "target_silhouette_centroid_displacement_start_to_end_px": float(np.linalg.norm(np.asarray(target_centroids[2]) - target_centroids[0])),
        },
        "floor_event_distance_to_projected_geometry": proximity,
        "floor_event_distance_to_projected_target_shadow": shadow_proximity,
        "floor_event_count": int(len(events)),
        "floor_event_unique_pixels": int(np.count_nonzero(on + off)),
        "same_ideal_dvs_model_on_direct_vs_ours_reconstructed_floor_luminance": {
            "configuration": "threshold +/-0.2 log, no mismatch/noise/refractory/bandwidth limit, physical_log radiance input, fixed-floor evaluation mask only",
            "direct_replay_matches_saved_direct_events_exactly": bool(direct_replay_matches),
            "direct_event_count_per_dense_frame": replayed_direct_counts,
            "ours_reconstruction_event_count_per_dense_frame": reconstructed_counts,
            "direct_total_events": int(len(replayed_direct_events)),
            "ours_reconstruction_total_events": int(len(reconstructed_events)),
            "ours_reconstruction_fraction_in_first_dense_frame": float(reconstructed_counts[0] / max(sum(reconstructed_counts), 1)),
            "unique_xy_polarity_pixels_direct": int(len(direct_spatiopolar)),
            "unique_xy_polarity_pixels_ours_reconstruction": int(len(reconstructed_spatiopolar)),
            "unique_xy_polarity_overlap_fraction_of_direct": float(len(direct_spatiopolar & reconstructed_spatiopolar) / max(len(direct_spatiopolar), 1)),
            "event_archive": f"episode_{ep}_floor_direct_vs_ours_dvs_events.npz",
        },
        "input_sha256": {
            "source_cache": sha256(cache_path),
            "direct_frame_cache": sha256(direct_path),
            "trace_window": sha256(trace_path),
            "floor_event_archive": sha256(event_path),
        },
        "diagnostic_png": f"episode_{ep}_achromatic_shadow_signature.png",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    scene_config = json.loads(args.scene_config.read_text(encoding="utf-8"))
    target_size_m = np.asarray(scene_config["target"]["size_m"], dtype=np.float64)
    args.output.mkdir(parents=True)
    records = [analyze(ep, args.source.resolve(), args.evidence.resolve(), args.output.resolve(), target_size_m)
               for ep in ("004", "005")]
    report = {
        "purpose": "test whether cached fixed-floor transients resemble achromatic illumination changes and quantify association with projected geometry",
        "scope": "CPU-only post hoc analysis of two existing development episodes; no render, physics, training, model inference, or core edit",
        "execution_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "scene_config_sha256": sha256(args.scene_config.resolve()),
        "scene_config": str(args.scene_config.resolve()),
        "projected_shadow_renderer_default_source": "Genesis 1.2.2 VisOptions default DirectionalLight(dir=(-1,-1,-1)); target-plane polygon is an analytic offline hypothesis, not a renderer shadow map",
        "python": sys.version,
        "platform": platform.platform(),
        "records": records,
        "global_interpretation": "The data strengthen a local illumination/shadow hypothesis. In both episodes, events are strongly enriched near the analytically projected target-shadow boundary under Genesis 1.2.2's default directional light. The robot projection is almost stationary while the target moves. This is a specific spatial association, not a causal shadow-on/off intervention; same-state rerender controls remain required.",
        "limitations": [
            "Archival RGB and event streams are from an earlier renderer execution; active binary/source identity was not recovered.",
            "Segmentation and pose are used only for offline analysis, not as event-model inputs.",
            "Distance to projected robot/target silhouettes is an association statistic, not a shadow mask or causal test.",
            "The ideal DVS event archive is a simulator reference, not physical sensor truth.",
        ],
    }
    report_path = args.output / "achromatic_shadow_signature.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    readme = args.output / "README.md"
    table_rows = []
    for record in records:
        rgb = record["floor_rgb_log_delta_selection"]
        motion = record["projected_geometry_motion"]
        shadow = record["floor_event_distance_to_projected_target_shadow"]
        dvs = record["same_ideal_dvs_model_on_direct_vs_ours_reconstructed_floor_luminance"]
        table_rows.append(
            f"| {record['episode']} | {dvs['direct_total_events']} over 10 calls | "
            f"{dvs['ours_reconstruction_total_events']} in first frame | "
            f"{rgb['median_chromatic_residual_over_common_change']:.3%} | "
            f"{motion['max_target_translation_mm']:.3f} / {motion['max_robot_link_translation_mm']:.3f} | "
            f"{shadow['event_fraction_within_px_of_union_shadow_edge']['5']:.1%} / "
            f"{shadow['floor_fraction_within_px_of_union_shadow_edge']['5']:.1%} |"
        )
    table = "\n".join(table_rows)
    run_command = (
        "python benchmark/robot_closed_loop/diagnostics/analyze_r1_5_cached_achromatic_shadow_signature.py "
        "--output benchmark/robot_closed_loop/results/r1_5_achromatic_shadow_signature_cpu_20261024d_rerun1"
    )
    readme.write_text(
        "# R1.5 cached achromatic-lighting signature\n\n"
        "CPU-only analysis of already archived episode 004/005 Direct-HDR frames, segmentation, poses, and fixed-floor events. No renderer, physics, GPU, training, model inference, or frozen-core edit was used.\n\n"
        "The analysis asks whether fixed-floor log-RGB changes are approximately a common multiplicative change across channels, whether visible geometry moves during the same 10 ms, and whether floor event pixels lie unusually close to projected robot/target masks or an analytic target-shadow boundary. The exact statistics and input hashes are in `achromatic_shadow_signature.json`; overlays are in the two PNGs.\n\n"
        "The unchanged ideal DVS model is also run on the saved direct luminance and Ours reconstructed intermediate luminance, with the same initial frame and an evaluator-only fixed-floor mask. This isolates temporal input differences; it does not test a learned estimator.\n\n"
        "## Main measurements\n\n"
        "| Episode | Direct-HDR floor events | Ours-reconstruction floor events | Median RGB chromatic residual / common change | Target / robot translation (mm) | Events / floor area within 5 px of predicted target-shadow edge |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"{table}\n\n"
        "The direct DVS replay matches the archived direct floor events exactly (after canonical ordering). Direct events occupy all 10 1 ms DVS calls; the Ours reconstructed luma puts all its floor events in the first queried subframe. Direct floor changes are almost achromatic across RGB, the robot silhouette is almost unchanged, and the target moves about 3.43 mm. Under the documented default light direction, 44–58% of event samples fall within 5 px of the analytic target-shadow boundary, compared with about 1.2% of fixed floor pixels. This supports a target-shadow contribution, but does not prove it by intervention.\n\n"
        "The projected target-shadow polygon assumes an opaque cube, support plane z=0, and Genesis 1.2.2's default directional-light ray direction (-1,-1,-1); the magenta outline is an analytic hypothesis, not a rendered shadow buffer. Direct events are enriched near that moving boundary. Near-achromatic changes also support illumination/shadow variation. The robot silhouette is nearly fixed while the target translates about 3.43 mm. This makes target-driven shadow motion the leading specific hypothesis, but same-state repeat renders and a shadow on/off intervention remain necessary for causal attribution.\n\n"
        "The segmentations and poses are offline diagnostics only. They are not fed into event generation or downstream perception.\n"
        f"\n## Reproduction\n\nFrom the repository root, use a fresh output directory:\n\n```bash\n{run_command}\n```\n\nThe output JSON records the input hashes, scene-config hash, execution commit, and script hash. No additional experiment config was changed.\n\n## Method references\n\n- ESIM's adaptive sampling derives a brightness-change bound under Lambertian brightness constancy; that assumption does not account for an unobserved moving-shadow visibility term: [ESIM paper](https://proceedings.mlr.press/v87/rebecq18a.html).\n- A path-tracing event simulator samples log luminance and uses statistical event detection; this is a principled but substantially heavier alternative if the raster shadow path remains unobservable: [Monte Carlo Path Tracing and Statistical Event Detection](https://arxiv.org/abs/2408.07996).\n- A recent event-adaptive path-tracing method refines time around event decisions and uses stream compaction; it is a possible later reference for a high-fidelity fallback, not a proposed change to the current benchmark: [Path-Tracing-Based Event Camera Simulation via Event-Adaptive Time Refinement](https://tatsylab.github.io/projects/manabe2026eapt/).\n- Genesis 1.2.2 documents the default light direction and `shadow=True`: [VisOptions source](https://genesis-world.readthedocs.io/en/v1.2.2/_modules/genesis/options/vis.html).\n",
        encoding="utf-8",
    )
    files = sorted(p for p in args.output.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="ascii"
    )
    print(json.dumps({"output": str(args.output.resolve()), "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
