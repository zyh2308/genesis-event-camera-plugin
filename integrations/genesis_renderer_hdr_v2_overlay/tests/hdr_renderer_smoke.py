#!/usr/bin/env python3
"""Minimal smoke test for the independent Genesis scene-linear renderer.

The test deliberately exercises both contracts:
* ``camera.render()`` remains uint8 (legacy compatibility).
* ``camera.render(radiance=True)`` returns float32 linear values and retains
  values above one when the scene contains a bright emissive patch.
"""

from __future__ import annotations

import json
import time

import numpy as np


def main() -> None:
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)),
        vis_options=gs.options.VisOptions(ambient_light=(0.2, 0.2, 0.2)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.2, 0.2, 0.2, 1.0)))
    scene.add_entity(
        gs.morphs.Box(pos=(0, 0, 0.15), size=(0.3, 0.3, 0.3)),
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.1, 1.0), emissive=(20.0, 20.0, 20.0)),
    )
    camera = scene.add_camera(
        res=(64, 48), pos=(0.8, 0.8, 0.6), lookat=(0, 0, 0.15), fov=50,
        GUI=False, near=0.05, far=10.0,
    )
    scene.build()

    # Genesis' public VisOptions intentionally restricts ambient light to a
    # display-oriented interval.  For this renderer contract test only, make
    # the internal scene-linear ambient term bright enough to prove that the
    # float framebuffer preserves values above one; this does not alter the
    # renderer implementation or any task configuration.
    context = camera._rasterizer._context
    context.ambient_light = np.asarray((10.0, 10.0, 10.0), dtype=np.float32)
    context._scene.ambient_light = context.ambient_light
    context.jit.set_light(context._scene, context._scene.light_nodes, context.ambient_light)

    t0 = time.perf_counter()
    rgb, _, _, _ = camera.render(rgb=True, depth=False, segmentation=False)
    legacy_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    hdr, depth, _, _ = camera.render(rgb=True, depth=True, segmentation=False, radiance=True)
    hdr_ms = (time.perf_counter() - t0) * 1000.0

    rgb = np.asarray(rgb)
    hdr = np.asarray(hdr)
    depth = np.asarray(depth)
    result = {
        "genesis_version": getattr(gs, "__version__", None),
        "legacy": {"dtype": str(rgb.dtype), "shape": list(rgb.shape), "min": int(rgb.min()), "max": int(rgb.max())},
        "radiance": {
            "dtype": str(hdr.dtype), "shape": list(hdr.shape),
            "min": float(np.nanmin(hdr)), "max": float(np.nanmax(hdr)),
            "finite": bool(np.isfinite(hdr).all()), "positive_pixels": int((hdr > 0).any(axis=-1).sum()),
            "above_one_pixels": int((hdr > 1.0).any(axis=-1).sum()),
        },
        "depth": {"dtype": str(depth.dtype), "shape": list(depth.shape)},
        "timing_ms": {"legacy_rgb": legacy_ms, "radiance_rgbd": hdr_ms},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert rgb.dtype == np.uint8, rgb.dtype
    assert hdr.dtype == np.float32, hdr.dtype
    assert hdr.shape == (48, 64, 3), hdr.shape
    assert np.isfinite(hdr).all()
    assert (hdr > 1.0).any(), "radiance path did not retain a value above one"
    gs.destroy()


if __name__ == "__main__":
    main()
