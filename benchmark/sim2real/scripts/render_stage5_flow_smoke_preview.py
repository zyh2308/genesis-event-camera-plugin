#!/usr/bin/env python3
"""Render a fixed, registered first-window preview from an existing Stage5 smoke.

This reads stored g1 artifacts only. It does not run Genesis, either event
generator, a model, or a score. The preview deliberately uses dt1_000 from the
registered one-rollout audit and is not a rich-room scene result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_u8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def depth_rgb(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8), 0.0, 0.0
    lo, hi = float(depth[valid].min()), float(depth[valid].max())
    span = max(hi - lo, 1e-12)
    x = np.clip((depth - lo) / span, 0, 1)
    stops = np.asarray([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=np.float32)
    pos = x * (len(stops) - 1)
    left = np.floor(pos).astype(np.int32)
    right = np.minimum(left + 1, len(stops) - 1)
    frac = (pos - left)[..., None]
    out = stops[left] * (1 - frac) + stops[right] * frac
    out[~valid] = 0
    return np.rint(out).astype(np.uint8), lo, hi


def flow_rgb(flow: np.ndarray, valid: np.ndarray, max_mag_px: float = 2.0) -> tuple[np.ndarray, float]:
    u, v = flow[0], flow[1]
    mag = np.sqrt(u * u + v * v)
    hue = (np.arctan2(v, u) / (2 * np.pi)) % 1.0
    sat = np.ones_like(hue)
    val = np.clip(mag / max_mag_px, 0, 1)
    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    channels = np.zeros((*mag.shape, 3), dtype=np.float32)
    choices = (
        (val, t, p), (q, val, p), (p, val, t),
        (p, q, val), (t, p, val), (val, p, q),
    )
    for idx, values in enumerate(choices):
        choose = sector == idx
        for channel in range(3):
            channels[..., channel][choose] = values[channel][choose]
    channels[~valid] = 0
    return np.rint(channels * 255).astype(np.uint8), float(np.percentile(mag[valid], 99.0)) if valid.any() else 0.0


def event_rgb(events_xypt: np.ndarray, left_s: float, right_s: float, height: int, width: int,
              time_scale: float = 1.0) -> tuple[np.ndarray, int, int, int]:
    # Shared bank format: [t_seconds, x, y, p].
    t = events_xypt[:, 0] * time_scale
    selected = events_xypt[(t > left_s) & (t <= right_s)]
    if not len(selected):
        return np.zeros((height, width, 3), dtype=np.uint8), 0, 0, 0
    xy = np.rint(selected[:, 1:3]).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    xy = xy[inside]
    polarity = selected[inside, 3]
    on = np.zeros((height, width), dtype=np.float32)
    off = np.zeros((height, width), dtype=np.float32)
    np.add.at(on, (xy[polarity > 0, 1], xy[polarity > 0, 0]), 1)
    np.add.at(off, (xy[polarity < 0, 1], xy[polarity < 0, 0]), 1)
    scale = max(float(np.percentile(np.concatenate([on.ravel(), off.ravel()]), 99.5)), 1.0)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.rint(np.clip(on / scale, 0, 1) * 255).astype(np.uint8)
    image[..., 2] = np.rint(np.clip(off / scale, 0, 1) * 255).astype(np.uint8)
    image[..., 1] = np.rint(np.clip(np.minimum(on, off) / scale, 0, 1) * 80).astype(np.uint8)
    return image, int(len(selected)), int(np.count_nonzero(on)), int(np.count_nonzero(off))


def draw_path(points: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.new("RGB", (width, height), (20, 24, 31))
    draw = ImageDraw.Draw(image)
    pad = 28
    x = points[:, 0]
    z = points[:, 2]
    xmin, xmax = float(x.min()), float(x.max())
    zmin, zmax = float(z.min()), float(z.max())
    xspan, zspan = max(xmax - xmin, 1e-6), max(zmax - zmin, 1e-6)
    px = pad + (x - xmin) / xspan * (width - 2 * pad)
    py = height - pad - (z - zmin) / zspan * (height - 2 * pad)
    draw.rectangle((pad, pad, width - pad, height - pad), outline=(100, 110, 125), width=1)
    coords = list(zip(px.tolist(), py.tolist()))
    if len(coords) > 1:
        draw.line(coords, fill=(80, 185, 235), width=3)
    draw.ellipse((px[0]-5, py[0]-5, px[0]+5, py[0]+5), fill=(50, 230, 110))
    draw.ellipse((px[-1]-5, py[-1]-5, px[-1]+5, py[-1]+5), fill=(255, 110, 70))
    draw.text((pad, 4), f"x-z path; dx={x[-1]-x[0]:.4g} m, dz={z[-1]-z[0]:.4g} m", fill=(235, 235, 235))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--v2e-events", type=Path, required=True)
    parser.add_argument("--v2e-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.manifest_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite preview artifact: {path}")

    bank_manifest = json.loads(args.bank_manifest.read_text())
    split = json.loads(args.split.read_text())
    v2e_manifest = json.loads(args.v2e_manifest.read_text())
    rollout = "s5q_02_train"
    if len(split) != 1 or split[0].get("id") != rollout:
        raise RuntimeError("preview is fixed to the one registered s5q_02_train smoke rollout")
    if sha256(args.bank) != bank_manifest.get("bank_sha256"):
        raise RuntimeError("physical bank SHA256 does not match its manifest")
    if bank_manifest.get("generated_episode_ids") != [rollout]:
        raise RuntimeError("physical bank manifest does not identify the fixed rollout")
    v2e_row = next((row for row in v2e_manifest.get("episodes", []) if row.get("rollout_id") == rollout), None)
    if v2e_row is None or sha256(args.v2e_events) != v2e_row.get("raw_events_sha256"):
        raise RuntimeError("V2E raw event file does not match its manifest")

    with h5py.File(args.bank, "r") as shared, h5py.File(args.v2e_events, "r") as raw_v2e:
        group = shared[f"episodes/{rollout}"]
        times = np.asarray(group["frame_timestamps_s"], dtype=np.float64)
        rgb = np.asarray(group["linearized_display_rgb_proxy"], dtype=np.float32)
        depth = np.asarray(group["depth_m"], dtype=np.float32)
        flow = np.asarray(group["flow_forward_30ms_px"], dtype=np.float32)
        valid = np.asarray(group["flow_valid_30ms"], dtype=np.uint8).astype(bool)
        eg = np.asarray(group["eventgenesis_events_txyp"], dtype=np.float64)
        camera = np.asarray(group["camera_to_world_cv"], dtype=np.float64)
        K = np.asarray(group["camera_intrinsics_K_px"], dtype=np.float64)
        v2e = np.asarray(raw_v2e["events"], dtype=np.float64)
    if times.shape != (201,) or not np.allclose(np.diff(times), 0.01, atol=1e-7, rtol=0):
        raise RuntimeError("registered 100 Hz endpoint contract failed")
    if rgb.shape != (201, 260, 346, 3) or depth.shape != (201, 260, 346):
        raise RuntimeError("renderer RGB/depth shape contract failed")
    if flow.shape != (66, 2, 260, 346) or valid.shape != (66, 260, 346):
        raise RuntimeError("shared 30 ms flow shape contract failed")
    left_s, right_s = float(times[0]), float(times[3])
    window = flow[0]
    flow_image, flow_p99 = flow_rgb(window, valid[0])
    depth_image, depth_min, depth_max = depth_rgb(depth[0])
    eg_image, eg_count, eg_on_px, eg_off_px = event_rgb(eg, left_s, right_s, 260, 346)
    # V2E raw schema is [timestamp_us, x, y, polarity]. Convert to the shared
    # [t_seconds, x, y, p] representation before applying the same (t0,t1] rule.
    v2e_shared = np.column_stack((v2e[:, 0] * 1e-6, v2e[:, 1], v2e[:, 2], np.where(v2e[:, 3] > 0, 1.0, -1.0)))
    v2e_image, v2e_count, v2e_on_px, v2e_off_px = event_rgb(v2e_shared, left_s, right_s, 260, 346)
    camera_image = draw_path(camera[:, :3, 3], 346, 260)

    panels = [
        (rgb_u8(rgb[0]), "Genesis RGB proxy, t=0 ms"),
        (rgb_u8(rgb[3]), "Genesis RGB proxy, t=30 ms"),
        (depth_image, f"Depth at t0, shared range {depth_min:.2f}-{depth_max:.2f} m"),
        (flow_image, f"GT flow; fixed HSV scale 0-2 px/30 ms; p99={flow_p99:.3f}"),
        (eg_image, f"EventGenesis ON/OFF count; N={eg_count}; (t0,t1]"),
        (v2e_image, f"V2E ON/OFF count; N={v2e_count}; (t0,t1]"),
        (camera_image, "Camera trajectory, x-z, full 2 s"),
    ]
    tile_w, tile_h, title_h, gap = 346, 260, 34, 12
    canvas_w = 4 * tile_w + 5 * gap
    canvas_h = 2 * (tile_h + title_h) + 3 * gap + 140
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 246, 248))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small_font = font
    for index, (image, title) in enumerate(panels):
        row, col = divmod(index, 4)
        x = gap + col * (tile_w + gap)
        y = gap + row * (tile_h + title_h + gap)
        draw.text((x, y), title[:58], fill=(20, 24, 31), font=font)
        canvas.paste(Image.fromarray(image), (x, y + title_h))
        if "EventGenesis" in title or "V2E ON/OFF" in title:
            draw.rectangle((x+5, y+title_h+5, x+74, y+title_h+21), fill=(15, 15, 15))
            draw.text((x+9, y+title_h+7), "ON=red OFF=blue", fill=(255, 255, 255), font=small_font)
    info_x = gap
    info_y = gap + 2 * (tile_h + title_h + gap)
    summary = [
        "FIXED HISTORICAL SMOKE PREVIEW — not ReplicaCAD, not the requested rich-room pilot, not training evidence.",
        f"rollout={rollout}; scene=independent_rigid_motion_static_camera; t=({left_s:.3f},{right_s:.3f}] s; label=forward displacement px/30ms",
        f"event pixels: EventGenesis ON/OFF={eg_on_px}/{eg_off_px}; V2E ON/OFF={v2e_on_px}/{v2e_off_px}; K fx/fy/cx/cy={K[0,0]:.3f}/{K[1,1]:.3f}/{K[0,2]:.3f}/{K[1,2]:.3f}",
        "Renderer: Genesis uint8 display RGB decoded by gamma 2.2; 8-bit proxy, not native HDR/radiometric calibration.",
        f"camera center displacement over 2 s: dx={camera[-1,0,3]-camera[0,0,3]:.6f} m, dy={camera[-1,1,3]-camera[0,1,3]:.6f} m, dz={camera[-1,2,3]-camera[0,2,3]:.6f} m (static-camera stratum).",
    ]
    for line_index, line in enumerate(summary):
        draw.text((info_x, info_y + line_index * 19), line, fill=(24, 30, 38), font=small_font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    result = {
        "schema": "stage5_historical_smoke_preview_v1",
        "status": "preview_of_existing_one_rollout_toy_scene_only",
        "real_data_used": False,
        "model_inference": False,
        "generator_settings_changed": False,
        "fixed_rollout_id": rollout,
        "fixed_window_seconds_open_left_closed_right": [left_s, right_s],
        "window_flow_shape": list(window.shape),
        "valid_pixels": int(valid[0].sum()),
        "event_counts": {
            "eventgenesis": {"total": eg_count, "on_pixels": eg_on_px, "off_pixels": eg_off_px},
            "v2e": {"total": v2e_count, "on_pixels": v2e_on_px, "off_pixels": v2e_off_px},
        },
        "camera_translation_start_m": camera[0, :3, 3].tolist(),
        "camera_translation_end_m": camera[-1, :3, 3].tolist(),
        "K_px": K.tolist(),
        "depth_display_range_m": [depth_min, depth_max],
        "flow_display_fixed_max_magnitude_px_per_30ms": 2.0,
        "flow_valid_magnitude_p99_px_per_30ms": flow_p99,
        "renderer_signal": "Genesis uint8 display RGB inverse-gamma-2.2 proxy; not native HDR/radiance",
        "bank": {"path": str(args.bank), "size_bytes": args.bank.stat().st_size, "sha256": sha256(args.bank)},
        "v2e_raw_events": {"path": str(args.v2e_events), "size_bytes": args.v2e_events.stat().st_size, "sha256": sha256(args.v2e_events)},
        "preview_sha256": sha256(args.output),
        "note": "The first time window is selected from the registered single-rollout smoke audit, not from model output or either generator's score. This visualization does not meet the mature indoor scene requirement.",
    }
    args.manifest_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
