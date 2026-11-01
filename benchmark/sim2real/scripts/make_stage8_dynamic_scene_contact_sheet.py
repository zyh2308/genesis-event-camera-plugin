#!/usr/bin/env python3
"""Make a compact visual audit of one saved Stage 8 Genesis motion rollout."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def event_image(events: np.ndarray, height: int, width: int, xcol: int, ycol: int,
                polarity_col: int, positive_when_gt_zero: bool) -> Image.Image:
    positive = np.zeros((height, width), dtype=np.float32)
    negative = np.zeros_like(positive)
    if len(events):
        x = np.rint(events[:, xcol]).astype(np.int64)
        y = np.rint(events[:, ycol]).astype(np.int64)
        p = events[:, polarity_col]
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        is_pos = p > 0 if positive_when_gt_zero else p >= 1
        np.add.at(positive, (y[inside & is_pos], x[inside & is_pos]), 1)
        np.add.at(negative, (y[inside & ~is_pos], x[inside & ~is_pos]), 1)
    scale = max(1.0, float(np.percentile(np.maximum(positive, negative), 99.5)))
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(positive / scale * 255, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(negative / scale * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def flow_image(flow: np.ndarray, valid: np.ndarray) -> Image.Image:
    if flow.shape != (2, *valid.shape):
        raise ValueError(f"expected flow [2,H,W] and mask [H,W], got {flow.shape}, {valid.shape}")
    u, v = flow.astype(np.float32)
    mag = np.hypot(u, v)
    scale = max(1e-6, float(np.percentile(mag[valid.astype(bool)], 95)))
    hue = (np.arctan2(v, u) / (2 * np.pi) + 1.0) % 1.0
    hsv = np.stack((hue, np.ones_like(hue), np.clip(mag / scale, 0, 1)), axis=-1)
    rgb = np.array(Image.fromarray(np.uint8(np.clip(hsv * 255, 0, 255)), "HSV").convert("RGB"), copy=True)
    rgb[~valid.astype(bool)] = 0
    return Image.fromarray(rgb, "RGB")


def depth_image(depth: np.ndarray) -> Image.Image:
    values = depth[np.isfinite(depth) & (depth > 0)]
    if not len(values):
        raise ValueError("depth frame has no finite positive samples")
    low, high = np.percentile(values, [2, 98])
    norm = np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1)
    gray = (255 * norm).astype(np.uint8)
    gray[~(np.isfinite(depth) & (depth > 0))] = 0
    return Image.fromarray(gray, "L").convert("RGB")


def load_v2e(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        events = np.asarray(h5["events"])
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"expected V2E events [N,4], got {events.shape}")
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True,
                        help="one copied Stage 8 episode root containing layout files, camera_motion_30ms, and v2e/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.episode_root.resolve()
    motion = root / "camera_motion_30ms"
    layout = np.asarray(Image.open(root / "layout_rgb.png").convert("RGB"))
    endpoint = np.asarray(Image.open(motion / "camera_motion_endpoint_rgb.png").convert("RGB"))
    depth = np.load(root / "layout_depth.npy", allow_pickle=False)
    flow = np.load(motion / "gt_flow_px_t0_to_t30ms.npy", allow_pickle=False)
    valid = np.load(motion / "gt_flow_valid.npy", allow_pickle=False)
    genesis_events = np.load(motion / "eventgenesis_events_seconds_xy_p.npy", allow_pickle=False).reshape(-1, 4)
    v2e_events = load_v2e(root / "v2e/v2e_events.h5")
    panels = [
        ("Genesis RGB: start", Image.fromarray(layout, "RGB")),
        ("Genesis RGB: 30 ms", Image.fromarray(endpoint, "RGB")),
        ("Renderer depth", depth_image(depth[0] if depth.ndim == 3 else depth)),
        ("Shared GT flow (px/30 ms)", flow_image(flow, valid)),
        (f"EventGenesis ({len(genesis_events):,} events)", event_image(genesis_events, layout.shape[0], layout.shape[1], 1, 2, 3, True)),
        (f"V2E ({len(v2e_events):,} events)", event_image(v2e_events, layout.shape[0], layout.shape[1], 1, 2, 3, False)),
    ]
    label_h = 38
    canvas = Image.new("RGB", (layout.shape[1] * len(panels), layout.shape[0] + label_h), (24, 27, 33))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, image) in enumerate(panels):
        if image.size != (layout.shape[1], layout.shape[0]):
            image = image.resize((layout.shape[1], layout.shape[0]), Image.Resampling.NEAREST)
        x = index * layout.shape[1]
        draw.text((x + 5, 4), label, fill=(245, 245, 245), font=font)
        canvas.paste(image, (x, label_h))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
