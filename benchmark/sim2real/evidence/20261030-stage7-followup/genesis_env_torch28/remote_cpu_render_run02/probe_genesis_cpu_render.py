#!/usr/bin/env python3
"""Bounded test of Genesis 1.2.2 camera rendering with all CUDA devices hidden."""
from __future__ import annotations

import json
import argparse
import time
from pathlib import Path

import genesis as gs
import numpy as np


def main(output_dir: Path) -> None:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, -9.81)),
            vis_options=gs.options.VisOptions(ambient_light=(0.38, 0.38, 0.38)),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=False,
        )
        plane = scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Default(color=(0.38, 0.40, 0.43, 1.0)),
        )
        camera = scene.add_camera(
            res=(96, 72),
            pos=(1.05, 1.12, 0.78),
            lookat=(0.0, 0.0, 0.24),
            fov=58,
            GUI=False,
            near=0.03,
            far=8.0,
        )
        scene.build()
        rgb, depth, segmentation, _ = camera.render(
            rgb=True, depth=True, segmentation=True
        )
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        segmentation = np.asarray(segmentation)
        rgb_min = float(rgb.min())
        rgb_max = float(rgb.max())
        if rgb.dtype == np.uint8:
            rgb_u8 = rgb
        else:
            rgb_u8 = np.clip(rgb, 0.0, 1.0)
            rgb_u8 = np.rint(rgb_u8 * 255.0).astype(np.uint8)
        from PIL import Image

        Image.fromarray(rgb_u8, mode="RGB").save(output_dir / "plane_preview.png")
        np.save(output_dir / "depth.npy", depth)
        np.save(output_dir / "segmentation.npy", segmentation)
        result = {
            "status": "passed_cpu_camera_render_smoke",
            "backend": str(gs.backend),
            "device": str(gs.device),
            "cuda_visible_devices": "empty",
            "scene_entities": 1,
            "camera_hw": [72, 96],
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "segmentation_shape": list(segmentation.shape),
            "rgb_finite": bool(np.isfinite(rgb).all()),
            "rgb_min": rgb_min,
            "rgb_max": rgb_max,
            "depth_finite_fraction": float(np.isfinite(depth).mean()),
            "depth_positive_fraction": float((depth > 0).mean()),
            "segmentation_ids": np.unique(segmentation).tolist(),
            "elapsed_seconds": time.monotonic() - started,
            "interpretation": "One synthetic plane only; no ReplicaCAD asset, physical trajectory, flow label, event generator, or training was involved.",
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        if not result["rgb_finite"] or result["depth_positive_fraction"] <= 0:
            raise RuntimeError("CPU render returned invalid RGB/depth")
    finally:
        gs.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args().output_dir)
