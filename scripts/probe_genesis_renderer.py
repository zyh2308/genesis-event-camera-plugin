"""Probe the installed Genesis renderer without changing any scene/task files.

Run this inside the target Genesis environment.  It records the public output
types and likely internal color-buffer owners, but never patches or writes the
Genesis installation.
"""

import argparse
import time

import numpy as np


def _interesting(obj):
    if obj is None:
        return []
    keys = ("color", "rgb", "rgba", "hdr", "radiance", "render", "buffer", "target")
    return [name for name in dir(obj) if any(key in name.lower() for key in keys)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    args = parser.parse_args()

    import genesis as gs

    backend = gs.cpu if args.backend == "cpu" else gs.gpu
    gs.init(backend=backend, logging_level="error")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05)))
    cam = scene.add_camera(
        res=(args.width, args.height), pos=(1, 1, 0.8), lookat=(0, 0, 0.05),
        fov=50, GUI=False,
    )
    scene.build()
    t0 = time.perf_counter()
    rgb, depth, seg, normal = cam.render(
        rgb=True, depth=True, segmentation=True, normal=True
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    print("genesis_version", getattr(gs, "__version__", None))
    print("render_ms", round(elapsed, 3))
    for name, value in (("rgb", rgb), ("depth", depth), ("seg", seg), ("normal", normal)):
        arr = np.asarray(value)
        print(name, "dtype", arr.dtype, "shape", arr.shape,
              "min", float(np.nanmin(arr)), "max", float(np.nanmax(arr)))
    print("camera_transform", np.asarray(cam.transform).tolist())
    print("rasterizer_attrs", _interesting(getattr(cam, "_rasterizer", None)))
    rasterizer = getattr(cam, "_rasterizer", None)
    print("rasterizer_context_attrs", _interesting(getattr(rasterizer, "_context", None)))
    print("raytracer_attrs", _interesting(getattr(cam, "_raytracer", None)))


if __name__ == "__main__":
    main()
