#!/usr/bin/env python3
"""Build and execute the v3 checkerboard replay in Genesis.

This is a readiness smoke, not a benchmark runner. It archives the native
HDR/depth/segmentation/pose/timestamp contract and runs the frozen Ours v3.1
plugin, but never evaluates a real stream or computes a formal metric.
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


def evaluate_keyframed_trajectory(t: float, keyframes, easing: str) -> float:
    """Evaluate the explicit keyframes from the replay YAML."""

    points = [(float(point[0]), float(point[1])) for point in keyframes]
    if len(points) < 2 or any(points[i][0] >= points[i + 1][0] for i in range(len(points) - 1)):
        raise ValueError("trajectory keyframes must have strictly increasing times")
    time = float(t)
    if time <= points[0][0]:
        return points[0][1]
    if time >= points[-1][0]:
        return points[-1][1]
    for (t0, x0), (t1, x1) in zip(points, points[1:]):
        if time <= t1:
            if x0 == x1:
                return x0
            u = (time - t0) / (t1 - t0)
            if easing in {"smoothstep", "smoothstep_with_explicit_keyframes"}:
                u = u * u * (3.0 - 2.0 * u)
            elif easing != "linear":
                raise ValueError(f"unsupported trajectory easing: {easing}")
            return x0 + (x1 - x0) * u
    return points[-1][1]


def _trajectory_parity(direction_spec):
    """Check representative times against the explicit YAML keyframes."""

    keyframes = direction_spec["keyframes"]
    easing = direction_spec["easing"]
    motion_end_t = float(keyframes[-2][0])
    final_t = float(keyframes[-1][0])
    midpoint_t = 0.5 * (float(keyframes[1][0]) + motion_end_t)
    samples = {
        "t_0": float(keyframes[0][0]),
        "t_0_10": 0.10,
        "t_0_15": float(keyframes[1][0]),
        "midpoint": midpoint_t,
        "motion_end": motion_end_t,
        "final_time": final_t,
    }
    values = {
        name: evaluate_keyframed_trajectory(time, keyframes, easing)
        for name, time in samples.items()
    }
    expected = {
        "t_0": float(keyframes[0][1]),
        "t_0_10": float(keyframes[0][1]),
        "t_0_15": float(keyframes[1][1]),
        "midpoint": values["midpoint"],
        "motion_end": float(keyframes[-2][1]),
        "final_time": float(keyframes[-1][1]),
    }
    errors = {name: abs(values[name] - expected[name]) for name in samples}
    if max(errors.values()) > 1e-12:
        raise RuntimeError(f"YAML/runtime trajectory parity failed: {errors}")
    return {
        "status": "PASS",
        "easing": easing,
        "samples_s": samples,
        "runtime_x_m": values,
        "yaml_expected_x_m": expected,
        "max_abs_error_m": max(errors.values()),
    }


def _set_scene_light(camera, values=(10.0, 10.0, 10.0)):
    context = camera._rasterizer._context
    context.ambient_light = np.asarray(values, dtype=np.float32)
    context._scene.ambient_light = context.ambient_light
    context.jit.set_light(context._scene, context._scene.light_nodes, context.ambient_light)


def _display_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert an already-rendered display-domain frame for PNG/V2E only."""

    array = np.asarray(rgb)
    if array.dtype == np.uint8:
        return array.copy()
    values = array.astype(np.float32, copy=False)
    if float(values.max()) <= 1.0001:
        values = values * 255.0
    return np.clip(values, 0.0, 255.0).round().astype(np.uint8)


def _render_two_domains(camera):
    """Render native radiance and normal display RGB separately, no scene step."""

    radiance, depth, seg, _ = camera.render(
        rgb=True, depth=True, segmentation=True, radiance=True
    )
    display_rgb, _, _, _ = camera.render(
        rgb=True, depth=False, segmentation=False
    )
    radiance = np.asarray(radiance).copy()
    display_rgb = np.asarray(display_rgb).copy()
    depth = np.asarray(depth).copy()
    seg = np.asarray(seg).copy()
    if radiance.dtype != np.float32 or not np.isfinite(radiance).all():
        raise RuntimeError("native radiance is not finite float32")
    if np.shares_memory(radiance, display_rgb):
        raise RuntimeError("radiance and display RGB share storage")
    display_uint8 = _display_rgb_uint8(display_rgb)
    if np.array_equal(display_uint8, _display_rgb_uint8(radiance)):
        raise RuntimeError("display RGB appears to be a clip/convert of radiance")
    return {
        "rgb": display_rgb,
        "radiance": radiance,
        "depth": depth,
        "seg": seg,
        "display_rgb_uint8": display_uint8,
        "same_scene_state": True,
        "ours_input_domain": "scene_linear_radiance",
        "v2e_input_domain": "display_rgb",
    }


def _write_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = _display_rgb_uint8(rgb)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ModuleNotFoundError:
        import imageio.v3 as iio
        iio.imwrite(path, rgb)


def _descriptor_entity_idx(value):
    """Extract an entity index from Genesis' segmentation descriptor."""

    if hasattr(value, "idx"):
        return int(value.idx)
    if hasattr(value, "entity"):
        return _descriptor_entity_idx(value.entity)
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _descriptor_entity_idx(item)
            except (TypeError, ValueError, AttributeError):
                continue
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"cannot extract Genesis entity index from {value!r}")


def _resolve_segmentation_motion(scene, segmentation, squares, background_plane):
    """Map the actual rendered segmentation IDs to moving square entities."""

    index_dict = getattr(scene, "segmentation_idx_dict", None)
    if not index_dict:
        raise RuntimeError("Genesis did not expose segmentation_idx_dict")
    square_by_entity_idx = {int(square.idx): square for square in squares}
    expected_entity_indices = set(square_by_entity_idx)
    background_entity_idx = int(background_plane.idx)
    seg_to_square = {}
    entity_to_seg_ids = {}
    background_seg_ids = set()
    for raw_seg_id, descriptor in index_dict.items():
        try:
            entity_idx = _descriptor_entity_idx(descriptor)
        except (TypeError, ValueError, AttributeError):
            continue
        if entity_idx == background_entity_idx:
            background_seg_ids.add(int(raw_seg_id))
        if entity_idx in square_by_entity_idx:
            seg_to_square[int(raw_seg_id)] = square_by_entity_idx[entity_idx]
            entity_to_seg_ids.setdefault(entity_idx, set()).add(int(raw_seg_id))

    resolved_entity_indices = set(entity_to_seg_ids)
    unresolved_expected_entities = sorted(expected_entity_indices - resolved_entity_indices)
    if unresolved_expected_entities:
        raise RuntimeError(
            "Genesis segmentation_idx_dict did not resolve every expected "
            f"checkerboard square entity; unresolved={unresolved_expected_entities}"
        )
    expected_entity_resolution_coverage = (
        len(resolved_entity_indices) / len(expected_entity_indices)
        if expected_entity_indices else 0.0
    )
    if expected_entity_resolution_coverage != 1.0:
        raise RuntimeError(
            "checkerboard entity-resolution coverage is not 100%; "
            f"resolved={len(resolved_entity_indices)}/{len(expected_entity_indices)}"
        )
    if set(seg_to_square).intersection(background_seg_ids):
        raise RuntimeError("background segmentation ID entered the checkerboard motion map")

    visible = {int(value) for value in np.unique(segmentation) if int(value) > 0}
    visible_board_ids = sorted(visible.intersection(seg_to_square))
    if not visible_board_ids:
        raise RuntimeError(
            "no visible checkerboard segmentation ID mapped to a square entity; "
            f"segmentation values={sorted(visible)[:20]}, index_dict={index_dict}"
        )
    visible_unknown_ids = sorted(visible.difference(set(index_dict)))
    visible_background_ids = sorted(visible.intersection(background_seg_ids))
    board_pixels = int(np.isin(segmentation, visible_board_ids).sum())
    mapped_pixels = int(np.isin(segmentation, list(seg_to_square)).sum())
    if visible_background_ids and set(visible_background_ids).intersection(seg_to_square):
        raise RuntimeError("visible background segmentation ID overlaps the motion map")
    return {
        "seg_to_square": seg_to_square,
        "visible_board_ids": visible_board_ids,
        "expected_entity_count": len(expected_entity_indices),
        "resolved_entity_count": len(resolved_entity_indices),
        "unresolved_expected_entities": unresolved_expected_entities,
        "expected_entity_resolution_coverage": expected_entity_resolution_coverage,
        "all_resolved_segmentation_ids": sorted(seg_to_square),
        "background_entity_idx": background_entity_idx,
        "background_segmentation_ids": sorted(background_seg_ids),
        "motion_map_background_overlap": sorted(set(seg_to_square).intersection(background_seg_ids)),
        "visible_unknown_segmentation_ids": visible_unknown_ids,
        "visible_background_segmentation_ids": visible_background_ids,
        "visible_board_pixels": board_pixels,
        "mapped_board_pixels": mapped_pixels,
        "all_visible_segmentation_ids": sorted(visible),
    }


def _entity_pose(entity) -> np.ndarray:
    pos = np.asarray(entity.get_pos(relative=False), dtype=np.float64)
    if pos.ndim > 1:
        pos = pos[0]
    if pos.shape != (3,):
        raise RuntimeError(f"unexpected checkerboard entity position shape: {pos.shape}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = pos
    return pose


def _build_scene(gs, spec, initial_x=None):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0)),
        vis_options=gs.options.VisOptions(ambient_light=(0.2, 0.2, 0.2)),
        show_viewer=False,
    )
    background_plane = scene.add_entity(
        gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.04, 0.04, 0.04, 1.0)),
        name="background_plane",
    )
    square = float(spec["board"]["square_m"])
    cols, rows = (int(x) for x in spec["board"]["squares_xy"])
    z = float(spec["board_pose"]["initial_position_m"][2])
    board_center_x = (
        float(spec["board_pose"]["initial_position_m"][0])
        if initial_x is None else float(initial_x)
    )
    squares = []
    for row in range(rows):
        for col in range(cols):
            x = (col - (cols - 1) / 2.0) * square
            y = (row - (rows - 1) / 2.0) * square
            color = (0.98, 0.98, 0.98, 1.0) if (row + col) % 2 == 0 else (0.02, 0.02, 0.02, 1.0)
            squares.append(scene.add_entity(
                gs.morphs.Box(pos=(x + board_center_x, y, z), size=(square, square, 0.004)),
                surface=gs.surfaces.Default(color=color),
                name=f"checkerboard_square_{row:02d}_{col:02d}",
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
    return scene, camera, squares, background_plane


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("LR", "RL"), default="LR")
    parser.add_argument("--duration", type=float, default=0.30, help="fixed active-readiness smoke duration, seconds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", default=None, help="readiness-only WxH override; formal replay stays 1280x720")
    args = parser.parse_args()

    replay = yaml.safe_load((ROOT / "benchmark/replays/checkerboard_translation_v3.yaml").read_text())
    if args.resolution:
        try:
            width, height = (int(value) for value in args.resolution.lower().split("x", 1))
        except Exception as exc:
            raise SystemExit("--resolution must be formatted WxH") from exc
        if width <= 0 or height <= 0:
            raise SystemExit("--resolution must be positive")
        replay["camera"]["resolution_xy"] = [width, height]
    direction_spec = replay["motion"][args.direction]
    duration = float(args.duration)
    if abs(duration - 0.30) > 1e-12:
        raise SystemExit("active readiness smoke duration is frozen at exactly 0.30 s")
    if duration > float(direction_spec["duration_s"]):
        raise SystemExit("duration exceeds the declared direction replay interval")

    try:
        import genesis as gs
        from genesis_event_plugin import GenesisPhysicsEventPlugin
        from genesis_event_plugin.genesis_adapter import camera_to_world_cv
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
        scene, camera, squares, background_plane = _build_scene(gs, replay)
        initial_payload = _render_two_domains(camera)
        segmentation_contract = _resolve_segmentation_motion(
            scene, initial_payload["seg"], squares, background_plane
        )
        trajectory_contract = _trajectory_parity(direction_spec)
        seg_to_square = segmentation_contract["seg_to_square"]

        def frame_provider(scene_obj, cam_obj):
            return _render_two_domains(cam_obj)

        current_x = float(direction_spec["x_start_m"])

        def move_board(x):
            nonlocal current_x
            delta = float(x) - current_x
            for entity in squares:
                pos = np.asarray(entity.get_pos(), dtype=np.float64)
                if pos.ndim > 1:
                    pos = pos[0]
                entity.set_pos(pos + np.asarray((delta, 0.0, 0.0)))
            current_x = float(x)

        def motion_provider(scene_obj, cam_obj):
            return {
                "camera_to_world": camera_to_world_cv(cam_obj),
                "object_to_world": {
                    int(seg_id): _entity_pose(square)
                    for seg_id, square in seg_to_square.items()
                },
            }

        plugin = GenesisPhysicsEventPlugin(
            preset="real_v2",
            input_space="linear",
            motion_state_provider=motion_provider,
            frame_provider=frame_provider,
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
            x = evaluate_keyframed_trajectory(
                float(t), direction_spec["keyframes"], direction_spec["easing"]
            )
            move_board(x)
            if index:
                scene.step()
            payload = frame_provider(scene, camera)
            radiance = np.asarray(payload["radiance"])
            depth = np.asarray(payload["depth"])
            seg = np.asarray(payload["seg"])
            hdr_max = max(hdr_max, float(radiance.max()))
            required_shapes = {
                "radiance": list(radiance.shape),
                "depth": list(depth.shape),
                "segmentation": list(seg.shape),
            }
            _write_png(frames_dir / f"frame_{index:06d}.png", payload["display_rgb_uint8"])
            np.savez_compressed(
                output / f"state_{index:06d}.npz",
                radiance=radiance.astype(np.float32),
                display_rgb=payload["rgb"],
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
        if hdr_max <= 1.0:
            raise RuntimeError("native radiance did not exceed display range in HDR smoke")
        metadata = {
            "status": "PASS",
            "formal_metrics_executed": False,
            "direction": args.direction,
            "duration_s": float(duration),
            "source_frame_rate_hz": source_rate,
            "frame_count": count,
            "native_radiance": True,
            "native_radiance_max": hdr_max,
            "display_rgb_archived_separately": True,
            "display_rgb_dtype": str(np.asarray(initial_payload["rgb"]).dtype),
            "display_rgb_was_clipped_from_radiance": False,
            "same_scene_state": True,
            "ours_input_domain": "scene_linear_radiance",
            "v2e_input_domain": "display_rgb",
            "required_outputs": required_shapes,
            "segmentation_expected_entity_count": segmentation_contract["expected_entity_count"],
            "segmentation_resolved_entity_count": segmentation_contract["resolved_entity_count"],
            "segmentation_unresolved_expected_entities": segmentation_contract["unresolved_expected_entities"],
            "segmentation_expected_entity_resolution_coverage": segmentation_contract["expected_entity_resolution_coverage"],
            "segmentation_ids": segmentation_contract["visible_board_ids"],
            "segmentation_all_resolved_ids": segmentation_contract["all_resolved_segmentation_ids"],
            "segmentation_background_entity_idx": segmentation_contract["background_entity_idx"],
            "segmentation_background_ids": segmentation_contract["background_segmentation_ids"],
            "segmentation_motion_map_background_overlap": segmentation_contract["motion_map_background_overlap"],
            "segmentation_visible_unknown_ids": segmentation_contract["visible_unknown_segmentation_ids"],
            "segmentation_visible_background_ids": segmentation_contract["visible_background_segmentation_ids"],
            "segmentation_motion_mapping_entity_indices": {
                str(seg_id): int(square.idx)
                for seg_id, square in segmentation_contract["seg_to_square"].items()
            },
            "segmentation_visible_board_pixels": segmentation_contract["visible_board_pixels"],
            "segmentation_mapped_board_pixels": segmentation_contract["mapped_board_pixels"],
            "duration_policy": "fixed_active_readiness_smoke_0.30s_no_event_count_adaptation",
            "trajectory_parity": trajectory_contract,
            "motion_magnitude_provenance": {
                "travel_m": abs(float(direction_spec["x_end_m"]) - float(direction_spec["x_start_m"])),
                "status": "nominal commanded replay travel, not measured real trajectory",
                "selection_rule": "preserved pre-frozen YAML value; never fit from event rate",
            },
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
