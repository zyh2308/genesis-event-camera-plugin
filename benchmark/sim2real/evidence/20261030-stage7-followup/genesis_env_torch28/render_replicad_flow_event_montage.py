#!/usr/bin/env python3
"""Render a fixed-scale visual audit of one paired ReplicaCAD motion pilot.

The RGB-looking panels are a display-only sRGB encoding of saved scene-linear
radiance. They are not fed to either event generator and are not a claim of
physical radiometric calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[5]
import sys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scene_linear_to_srgb_display(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x.astype(np.float32), 0.0, 1.0)
    encoded = np.where(clipped <= 0.0031308, 12.92 * clipped,
                       1.055 * np.power(clipped, 1.0 / 2.4) - 0.055)
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255).astype(np.uint8)


def depth_gray(depth: np.ndarray, max_m: float) -> np.ndarray:
    z = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(z) & (z > 0)
    gray = np.zeros(z.shape, dtype=np.uint8)
    gray[valid] = np.rint(np.clip(z[valid] / max_m, 0.0, 1.0) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def flow_color(flow: np.ndarray, valid: np.ndarray, clip_px: float) -> np.ndarray:
    u, v = flow[0].astype(np.float32), flow[1].astype(np.float32)
    angle = np.mod(np.arctan2(v, u) / (2 * np.pi), 1.0)
    h6 = angle * 6.0
    sector = np.floor(h6).astype(np.int8) % 6
    frac = h6 - np.floor(h6)
    q, t = 1.0 - frac, frac
    rgb = np.zeros((*u.shape, 3), dtype=np.float32)
    choices = (
        (1.0, t, 0.0), (q, 1.0, 0.0), (0.0, 1.0, t),
        (0.0, q, 1.0), (t, 0.0, 1.0), (1.0, 0.0, q),
    )
    mag = np.hypot(u, v)
    value = np.clip(mag / clip_px, 0.0, 1.0)
    for idx, (r, g, b) in enumerate(choices):
        m = sector == idx
        rgb[m, 0] = np.asarray(r)[m] if isinstance(r, np.ndarray) else r
        rgb[m, 1] = np.asarray(g)[m] if isinstance(g, np.ndarray) else g
        rgb[m, 2] = np.asarray(b)[m] if isinstance(b, np.ndarray) else b
    rgb *= value[..., None]
    rgb[~valid] = 0.0
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def polarity_image(events: np.ndarray, t0: float, t1: float, time_scale: float,
                   height: int, width: int) -> tuple[np.ndarray, int, int, int]:
    times = events[:, 0].astype(np.float64) * time_scale
    keep = (times > t0) & (times <= t1)
    e = events[keep]
    on = np.zeros((height, width), dtype=np.uint16)
    off = np.zeros((height, width), dtype=np.uint16)
    if len(e):
        x = np.rint(e[:, 1]).astype(np.int64)
        y = np.rint(e[:, 2]).astype(np.int64)
        if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
            raise RuntimeError("event coordinates fall outside the registered image")
        p = e[:, 3] > 0
        np.add.at(on, (y[p], x[p]), 1)
        np.add.at(off, (y[~p], x[~p]), 1)
    # Fixed display scale: event counts above four saturate; no per-window
    # normalization is used, so the rows remain visually comparable.
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.minimum(on, 4).astype(np.uint8) * 63
    rgb[..., 2] = np.minimum(off, 4).astype(np.uint8) * 63
    active = int(np.count_nonzero((on + off) > 0))
    return rgb, len(e), active, int(on.sum() + off.sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-dir", type=Path, required=True)
    p.add_argument("--v2e-h5", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--flow-clip-px-per-10ms", type=float, default=8.0)
    p.add_argument("--depth-display-max-m", type=float, default=8.0)
    a = p.parse_args()
    motion = a.motion_dir.resolve()
    rad_path = motion / "radiance_frames.npy"
    depth_path = motion / "depth_frames.npy"
    seg_path = motion / "renderer_segmentation_raw.npy"
    pose_path = motion / "camera_poses_cv.npy"
    result_path = motion / "motion_result.json"
    radiance = np.load(rad_path, allow_pickle=False)
    depth = np.load(depth_path, allow_pickle=False)
    seg = np.load(seg_path, allow_pickle=False)
    poses = np.load(pose_path, allow_pickle=False)
    result = json.loads(result_path.read_text())
    h, w = depth.shape[1:]
    if radiance.shape != (4, h, w, 3) or seg.shape != depth.shape or poses.shape != (4, 4, 4):
        raise RuntimeError("pilot arrays do not satisfy the four-frame shared-motion schema")
    warp_path = REPO_ROOT / "genesis_event_plugin/physics_warp.py"
    spec = importlib.util.spec_from_file_location("stage7_physics_warp", warp_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load physical flow implementation: {warp_path}")
    warp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warp_module)
    compute_se3_flow = warp_module.compute_se3_flow
    K = np.asarray(result["camera_pose_check"]["K_pixel_index"], dtype=np.float64)
    eg = np.load(motion / "eventgenesis_events_seconds_xy_p.npy", allow_pickle=False)
    with h5py.File(a.v2e_h5.resolve(), "r") as f:
        v2e = np.asarray(f["events"], dtype=np.uint32)
    if eg.ndim != 2 or eg.shape[1] != 4 or v2e.ndim != 2 or v2e.shape[1] != 4:
        raise RuntimeError("event data must have [timestamp,x,y,polarity] shape")

    # Use one fixed flow color scale and one fixed event-count display scale.
    gap, title_h, row_h = 8, 32, h + 28
    panel_w = w
    canvas = Image.new("RGB", (5 * panel_w + 6 * gap, title_h + 3 * row_h + 52), (18, 20, 25))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    panel_names = ["Scene-linear radiance (display)", "Renderer depth", "GT flow, px/10 ms",
                   "EventGenesis ON/OFF", "V2E ON/OFF"]
    for j, label in enumerate(panel_names):
        x = gap + j * (panel_w + gap)
        draw.text((x, 8), label, fill=(240, 240, 240), font=font)
    metadata_rows = []
    for i in range(3):
        t0, t1 = i * 0.01, (i + 1) * 0.01
        flow, valid0 = compute_se3_flow(depth[i], seg[i], K, poses[i], poses[i + 1])
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        inside = ((xx + flow[0] >= 0) & (xx + flow[0] < w)
                  & (yy + flow[1] >= 0) & (yy + flow[1] < h))
        valid = valid0 & inside & np.isfinite(flow).all(axis=0)
        eg_img, eg_count, eg_active, eg_total = polarity_image(eg, t0, t1, 1.0, h, w)
        v2e_img, v2e_count, v2e_active, v2e_total = polarity_image(v2e, t0, t1, 1e-6, h, w)
        panels = [
            scene_linear_to_srgb_display(radiance[i]),
            depth_gray(depth[i], a.depth_display_max_m),
            flow_color(flow, valid, a.flow_clip_px_per_10ms), eg_img, v2e_img,
        ]
        y0 = title_h + i * row_h
        draw.text((gap, y0 + h + 3), f"window {i}: ({t0*1000:.0f}, {t1*1000:.0f}] ms",
                  fill=(220, 220, 220), font=font)
        for j, pixels in enumerate(panels):
            x0 = gap + j * (panel_w + gap)
            canvas.paste(Image.fromarray(pixels, mode="RGB"), (x0, y0))
        metadata_rows.append({
            "window_s_left_open_right_closed": [t0, t1],
            "flow_valid_pixels": int(valid.sum()),
            "flow_magnitude_px_quantiles_0_50_90_99_100": np.percentile(
                np.hypot(flow[0], flow[1])[valid], [0, 50, 90, 99, 100]).tolist(),
            "eventgenesis": {"event_count": eg_count, "active_pixels": eg_active, "polarity_count_sum": eg_total},
            "v2e": {"event_count": v2e_count, "active_pixels": v2e_active, "polarity_count_sum": v2e_total},
        })
    footer_y = title_h + 3 * row_h + 4
    trajectory = result["trajectory"]
    delta = trajectory.get("registered_translation_habitat_xyz_m",
                           [trajectory.get("registered_translation_habitat_x_m", 0.0), 0.0, 0.0])
    draw.text((gap, footer_y),
              f"{a.name}; registered camera path: Habitat xyz Δ=({delta[0]:.2f},{delta[1]:.2f},{delta[2]:.2f}) m / {trajectory['duration_s']*1000:.0f} ms; static scene",
              fill=(240, 240, 240), font=font)
    draw.text((gap, footer_y + 16),
              f"Display only: radiance clipped at 1.0 then sRGB encoded; flow magnitude clipped at {a.flow_clip_px_per_10ms:g} px/10 ms; event channel count clips at 4/pixel.",
              fill=(190, 190, 190), font=font)
    out = a.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    metadata = {
        "schema": "stage7_replicad_paired_motion_montage_v1",
        "name": a.name,
        "scope": "visualization of one registered synthetic camera-only path; not a model result or generator ranking",
        "inputs": {str(x): sha256(x) for x in [rad_path, depth_path, seg_path, pose_path, result_path,
                                                motion / "eventgenesis_events_seconds_xy_p.npy", a.v2e_h5.resolve()]},
        "rendered_display": {
            "radiance_encoding": "clip scene-linear float32 to [0,1], then sRGB transfer function; display only",
            "depth_gray_range_m": [0.0, a.depth_display_max_m],
            "flow_hsv_value_clip_px_per_10ms": a.flow_clip_px_per_10ms,
            "event_count_clip_per_polarity_per_pixel": 4,
            "timestamp_window": "left-open, right-closed; three fixed 10ms intervals",
        },
        "trajectory": trajectory,
        "intervals": metadata_rows,
        "output": str(out),
        "output_sha256": sha256(out),
    }
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "sha256": metadata["output_sha256"], "intervals": metadata_rows}, indent=2))


if __name__ == "__main__":
    main()
