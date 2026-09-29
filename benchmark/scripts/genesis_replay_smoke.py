#!/usr/bin/env python3
"""Build and execute the v3 checkerboard replay in Genesis.

The command is a readiness smoke, not a benchmark runner.  It deliberately
archives the native HDR/depth/segmentation/pose/timestamp contract and runs
the frozen Ours v3.1 plugin, but never evaluates a real stream or computes a
formal metric.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _matrix_from_x(x: float, z: float = 0.10) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = float(x)
    matrix[2, 3] = float(z)
    return matrix


def _smoothstep(start: float, end: float, t: float, duration: float) -> float:
    # The endpoint holds in the short acceleration/deceleration segments.
    u = min(max(float(t) / float(duration), 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    return start + (end - start) * u


def _set_scene_light(camera, values=(10.0, 10.0, 10.0)):
    context = camera._rasterizer._context
    context.ambient_light = np.asarray(values, dtype=np.float32)
    context._scene.ambient_light = context.ambient_light
    context.jit.set_light(context._scene, context._scene.light_nodes, context.ambient_light)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ModuleNotFoundError:
        import imageio.v3 as iio
        iio.imwrite(path, rgb)


def _build_scene(gs, spec):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0)),
        # Genesis validates display-oriented VisOptions in [0, 1].  The
        # scene-linear HDR stress value is applied inside _set_scene_light()
        # after build, exactly as in the renderer contract smoke.
        vis_options=gs.options.VisOptions(ambient_light=(0.2, 0.2, 0.2)),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.04, 0.04, 0.04, 1.0)),
    )
    square = float(spec["board"]["square_m"])
    cols, rows = (int(x) for x in spec["board"]["squares_xy"])
    z = float(spec["board_pose"]["initial_position_m"][2])
    squares = []
    for row in range(rows):
        for col in range(cols):
            x = (col - (cols - 1) / 2.0) * square
            y = (row - (rows - 1) / 2.0) * square
            color = (0.98, 0.98, 0.98, 1.0) if (row + col) % 2 == 0 else (0.02, 0.02, 0.02, 1.0)
            squares.append(scene.add_entity(
                gs.morphs.Box(pos=(x - 0.10, y, z), size=(square, square, 0.004)),
                surface=gs.surfaces.Default(color=color),
            ))
    camera_spec = spec["camera"]
    camera = scene.add_camera(
        res=tuple(int(x) for x in camera_spec["resolution_xy"]),
        pos=tuple(camera_spec["pose_world"]["position_m"]),
        lookat=tuple(camera_spec["pose_world"]["look_at_m"]),
        fov=50,
        GUI=False,
        near=0.05,
        far=5.0,
    )
    scene.build()
    _set_scene_light(camera)
    return scene, camera, squares


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("LR", "RL"), default="LR")
    parser.add_argument("--duration", type=float, default=0.10, help="short smoke duration, seconds")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    replay = yaml.safe_load((ROOT / "benchmark/replays/checkerboard_translation_v3.yaml").read_text())
    direction_spec = replay["motion"][args.direction]
    duration = min(float(args.duration), float(direction_spec["duration_s"]))
    if duration <= 0:
        raise SystemExit("duration must be positive")

    try:
        import genesis as gs
        from genesis_event_plugin import GenesisPhysicsEventPlugin
        from genesis_event_plugin.genesis_adapter import make_frame_provider, camera_to_world_cv
    except Exception as exc:
        print(json.dumps({"status": "STOP", "stage": "Genesis import", "error": f"{type(exc).__name__}: {exc}"}))
        return 2

    output = args.output_dir.resolve()
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    backend_name = os.environ.get("GENESIS_BACKEND", "gpu").lower()
    backend = getattr(gs, backend_name)
    gs.init(backend=backend, logging_level="warning")
    try:
        scene, camera, squares = _build_scene(gs, replay)
        frame_provider = make_frame_provider(camera, native_radiance=True)
        current_x = float(direction_spec["x_start_m"])

        def move_board(x):
            nonlocal current_x
            delta = float(x) - current_x
            for entity in squares:
                # Entity.set_pos is the public Genesis kinematic pose API.
                pos = np.asarray(entity.get_pos(), dtype=np.float64)
                if pos.ndim > 1:
                    pos = pos[0]
                entity.set_pos(pos + np.asarray((delta, 0.0, 0.0)))
            current_x = float(x)

        def motion_provider(scene_obj, cam_obj):
            return {
                "camera_to_world": camera_to_world_cv(cam_obj),
                "object_to_world": {1: _matrix_from_x(current_x)},
            }

        plugin = GenesisPhysicsEventPlugin(
            preset="real_v2",
            input_space="linear",
            frame_provider=frame_provider,
            motion_state_provider=motion_provider,
            device=os.environ.get("GENESIS_EVENT_DEVICE", "cpu"),
        )
        plugin.attach(scene, camera)
        plugin.start_episode()
        source_rate = float(replay["render"]["source_frame_rate_hz"])
        count = max(2, int(round(duration * source_rate)) + 1)
        timestamps = np.linspace(0.0, duration, count)
        total_events = 0
        hdr_max = -np.inf
        required_shapes = {}
        for index, t in enumerate(timestamps):
            x = _smoothstep(
                float(direction_spec["x_start_m"]), float(direction_spec["x_end_m"]),
                float(t), duration,
            )
            move_board(x)
            if index:
                scene.step()
            payload = frame_provider(scene, camera)
            radiance = np.asarray(payload["radiance"])
            depth = np.asarray(payload["depth"])
            seg = np.asarray(payload["seg"])
            if radiance.dtype != np.float32 or not np.isfinite(radiance).all():
                raise RuntimeError("native radiance is not finite float32")
            hdr_max = max(hdr_max, float(radiance.max()))
            required_shapes = {
                "radiance": list(radiance.shape),
                "depth": list(depth.shape),
                "segmentation": list(seg.shape),
            }
            _write_png(frames_dir / f"frame_{index:06d}.png", payload["rgb"])
            np.savez_compressed(
                output / f"state_{index:06d}.npz",
                radiance=radiance.astype(np.float32),
                depth=depth.astype(np.float32),
                segmentation=seg,
                camera_to_world=camera_to_world_cv(camera),
                checkerboard_to_world=_matrix_from_x(x),
                timestamp_s=float(t),
            )
            events = plugin.capture()
            total_events += int(len(events))
        plugin.end_episode()
        plugin.close()
        metadata = {
            "status": "PASS",
            "formal_metrics_executed": False,
            "direction": args.direction,
            "duration_s": float(duration),
            "source_frame_rate_hz": source_rate,
            "frame_count": count,
            "native_radiance": True,
            "native_radiance_max": hdr_max,
            "required_outputs": required_shapes,
            "ours_core_commit": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
            "ours_event_count_smoke_only": total_events,
            "v2e_not_run_by_this_script": True,
        }
        (output / "replay_smoke.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        return 0
    finally:
        gs.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
