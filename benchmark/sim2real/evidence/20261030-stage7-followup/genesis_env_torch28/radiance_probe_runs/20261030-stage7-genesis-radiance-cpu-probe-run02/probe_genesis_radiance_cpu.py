#!/usr/bin/env python3
"""Probe Genesis 1.2.2 radiance=True on a tiny CPU-rendered scene."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
import traceback
from pathlib import Path

import genesis as gs
import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_info(value) -> dict:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite_fraction": float(np.isfinite(a).mean()) if a.size else 1.0,
        "min": float(np.nanmin(a)) if a.size else None,
        "max": float(np.nanmax(a)) if a.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    result = {
        "schema": "genesis_radiance_cpu_probe_v1",
        "genesis_version": gs.__version__,
        "backend": str(gs.backend),
        "device": str(gs.device),
        "cuda_visible_devices": "empty",
        "probe_scene": "one colored box over plane, 96x72 camera",
        "radiance_units": "unverified; this probe tests the API path, not physical calibration",
    }
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, -9.81)),
            vis_options=gs.options.VisOptions(ambient_light=(0.4, 0.4, 0.4)),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.35, 0.35, 0.35, 1.0)))
        scene.add_entity(
            gs.morphs.Box(pos=(0.0, 0.0, 0.4), size=(0.4, 0.3, 0.5)),
            surface=gs.surfaces.Default(color=(0.9, 0.2, 0.1, 1.0)),
        )
        camera = scene.add_camera(
            res=(96, 72), pos=(1.2, 1.2, 1.0), lookat=(0.0, 0.0, 0.25),
            fov=60, GUI=False, near=0.03, far=8.0,
        )
        scene.build()
        render_fn = camera.render
        try:
            source_path = Path(inspect.getsourcefile(render_fn) or "").resolve()
            result["camera_render_signature"] = str(inspect.signature(render_fn))
            result["camera_render_source"] = str(source_path)
            if source_path.is_file():
                result["camera_render_source_sha256"] = digest(source_path)
        except (OSError, TypeError, ValueError) as exc:
            result["camera_render_introspection_error"] = f"{type(exc).__name__}: {exc}"

        try:
            radiance, _, _, _ = camera.render(
                rgb=True, depth=False, segmentation=False, radiance=True, force_render=True
            )
            if radiance is None:
                raise RuntimeError("radiance=True returned None for the RGB payload")
            if hasattr(radiance, "detach"):
                radiance = radiance.detach().cpu().numpy()
            radiance = np.asarray(radiance)
            result["radiance_request"] = "accepted"
            result["radiance_payload"] = array_info(radiance)
            np.save(args.output_dir / "radiance_payload.npy", radiance)
        except Exception as exc:
            result["radiance_request"] = "failed"
            result["radiance_error"] = f"{type(exc).__name__}: {exc}"
            result["radiance_traceback"] = traceback.format_exc()

        try:
            rgb, _, _, _ = camera.render(
                rgb=True, depth=False, segmentation=False, force_render=True
            )
            if rgb is None:
                raise RuntimeError("ordinary RGB render returned None")
            if hasattr(rgb, "detach"):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.asarray(rgb)
            result["ordinary_rgb"] = array_info(rgb)
            np.save(args.output_dir / "ordinary_rgb.npy", rgb)
        except Exception as exc:
            result["ordinary_rgb_error"] = f"{type(exc).__name__}: {exc}"
            result["ordinary_rgb_traceback"] = traceback.format_exc()
    except Exception as exc:
        result["scene_probe_error"] = f"{type(exc).__name__}: {exc}"
        result["scene_probe_traceback"] = traceback.format_exc()
    finally:
        gs.destroy()

    result["elapsed_seconds"] = time.monotonic() - started
    result["status"] = (
        "probe_completed_radiance_api_accepted" if result.get("radiance_request") == "accepted"
        else "probe_completed_radiance_api_failed"
    )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
