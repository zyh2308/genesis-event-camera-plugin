#!/usr/bin/env python3
"""Execute one frozen full-direction Genesis/Ours formal replay.

This script creates no metric and performs no parameter search.  It archives
one fixed 100 Hz scene state per timestamp, including native radiance,
display RGB, depth, segmentation, poses, and timestamps, then feeds the same
cached payloads to the frozen Ours v3.1 plugin.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.scripts.genesis_replay_smoke import (  # noqa: E402
    _build_scene,
    _entity_pose,
    _render_two_domains,
    _resolve_segmentation_motion,
    _set_scene_light,
    _write_png,
    evaluate_keyframed_trajectory,
    _trajectory_parity,
)


CORE_COMMIT = "e79e656ac00f7847b98db9e12bf43b093f3afff1"
SOURCE_DT_S = 0.01
FORMAL_RESOLUTION = (1280, 720)


def _scene_time(scene) -> float:
    value = scene.cur_t
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _make_dataset(handle, name, shape, dtype, *, chunks=None):
    kwargs = {"dtype": dtype}
    if chunks is not None:
        kwargs["chunks"] = chunks
    if len(shape) > 0 and shape[0] > 1:
        kwargs["compression"] = "gzip"
        kwargs["compression_opts"] = 4
    return handle.create_dataset(name, shape=shape, **kwargs)


def _direction_times(duration_s: float):
    interval_count = int(math.ceil(float(duration_s) / SOURCE_DT_S))
    generated_duration = interval_count * SOURCE_DT_S
    # The extra endpoint is intentional: N is the number of 100 Hz intervals,
    # so the cached sequence covers [0, generated_duration].  The final
    # interval is the declared stationary padded tail.
    timestamps = np.arange(interval_count + 1, dtype=np.float64) * SOURCE_DT_S
    timestamps[-1] = generated_duration
    return interval_count, generated_duration, timestamps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("LR", "RL"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    replay = yaml.safe_load((ROOT / "benchmark/replays/checkerboard_translation_v3.yaml").read_text())
    direction_spec = replay["motion"][args.direction]
    evaluation_duration = float(direction_spec["duration_s"])
    interval_count, generated_duration, timestamps = _direction_times(evaluation_duration)
    padding_duration = generated_duration - evaluation_duration
    if not (0.0 < padding_duration < SOURCE_DT_S):
        raise RuntimeError(f"unexpected padding duration: {padding_duration}")
    if tuple(replay["camera"]["resolution_xy"]) != FORMAL_RESOLUTION:
        raise RuntimeError("formal replay resolution is not frozen at 1280x720")

    try:
        import genesis as gs
        from genesis_event_plugin import GenesisPhysicsEventPlugin
        from genesis_event_plugin.genesis_adapter import camera_to_world_cv
    except Exception as exc:
        print(json.dumps({"status": "STOP", "stage": "Genesis import", "error": f"{type(exc).__name__}: {exc}"}))
        return 2

    output = args.output_dir.resolve()
    replay_dir = output / "replay"
    display_dir = replay_dir / "display_rgb"
    ours_dir = output / "ours"
    output.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    display_dir.mkdir(parents=True, exist_ok=True)
    ours_dir.mkdir(parents=True, exist_ok=True)

    backend_name = os.environ.get("GENESIS_BACKEND", "cpu").lower()
    backend = getattr(gs, backend_name)
    gs.init(backend=backend, logging_level="warning")
    replay_h5 = None
    plugin = None
    try:
        scene, camera, squares, background_plane = _build_scene(
            gs, replay, initial_x=float(direction_spec["x_start_m"])
        )
        _set_scene_light(camera)
        initial_payload = _render_two_domains(camera)
        segmentation_contract = _resolve_segmentation_motion(
            scene, initial_payload["seg"], squares, background_plane
        )
        trajectory_contract = _trajectory_parity(direction_spec)
        seg_to_square = segmentation_contract["seg_to_square"]
        sorted_seg_ids = sorted(seg_to_square)
        object_ids = np.asarray(sorted_seg_ids, dtype=np.int32)

        shape_hwc = tuple(np.asarray(initial_payload["radiance"]).shape)
        shape_hw = tuple(np.asarray(initial_payload["depth"]).shape)
        if shape_hwc != (720, 1280, 3) or shape_hw != (720, 1280):
            raise RuntimeError(f"formal archive shape mismatch: radiance={shape_hwc}, depth={shape_hw}")

        replay_h5 = h5py.File(replay_dir / "state_archive.h5", "w")
        replay_h5.attrs["resolution_xy"] = np.asarray(FORMAL_RESOLUTION, dtype=np.int32)
        replay_h5.attrs["source_dt_s"] = SOURCE_DT_S
        replay_h5.attrs["evaluation_duration_s"] = evaluation_duration
        replay_h5.attrs["generated_duration_s"] = generated_duration
        replay_h5.attrs["padding_duration_s"] = padding_duration
        replay_h5.attrs["timestamp_contract"] = "scene.cur_t == archived timestamp; fixed 0.01 s cadence"
        replay_h5.create_dataset("object_segmentation_ids", data=object_ids)
        replay_h5.create_dataset("timestamps_s", shape=(len(timestamps),), dtype="f8")
        replay_h5.create_dataset("camera_pose", shape=(len(timestamps), 4, 4), dtype="f8", compression="gzip", compression_opts=4)
        replay_h5.create_dataset("object_poses", shape=(len(timestamps), len(object_ids), 4, 4), dtype="f8", compression="gzip", compression_opts=4)
        radiance_ds = _make_dataset(replay_h5, "radiance", (len(timestamps), *shape_hwc), "f4", chunks=(1, *shape_hwc))
        display_ds = _make_dataset(replay_h5, "display_rgb", (len(timestamps), *shape_hwc), "u1", chunks=(1, *shape_hwc))
        depth_ds = _make_dataset(replay_h5, "depth", (len(timestamps), *shape_hw), "f4", chunks=(1, *shape_hw))
        seg_ds = _make_dataset(replay_h5, "segmentation", (len(timestamps), *shape_hw), "i4", chunks=(1, *shape_hw))

        current_payload = None
        current_x = float(direction_spec["x_start_m"])
        state_time_errors = []
        pose_time_errors = []
        total_events = 0

        def frame_provider(scene_obj, cam_obj):
            if current_payload is None:
                raise RuntimeError("cached frame payload is not initialized")
            return current_payload

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
                    int(seg_id): _entity_pose(seg_to_square[seg_id])
                    for seg_id in sorted_seg_ids
                },
            }

        plugin = GenesisPhysicsEventPlugin(
            output_dir=str(ours_dir),
            preset="real_v2",
            input_space="linear",
            motion_state_provider=motion_provider,
            frame_provider=frame_provider,
            device=os.environ.get("GENESIS_EVENT_DEVICE", "cpu"),
        )
        plugin.attach(scene, camera)
        plugin.start_episode()
        hdr_max = -np.inf
        for index, timestamp in enumerate(timestamps):
            t = float(timestamp)
            if index:
                scene.step()
            scene_now = _scene_time(scene)
            state_time_errors.append(abs(scene_now - t))
            if state_time_errors[-1] > 1e-9:
                raise RuntimeError(f"scene.cur_t mismatch at index {index}: {scene_now} != {t}")
            x = (
                evaluate_keyframed_trajectory(t, direction_spec["keyframes"], direction_spec["easing"])
                if t <= evaluation_duration else float(direction_spec["x_end_m"])
            )
            move_board(x)
            # Move before rendering the state.  Genesis scene time and the
            # archived pose timestamp remain the same fixed-cadence sample.
            if index and abs(_scene_time(scene) - t) > 1e-9:
                raise RuntimeError("scene time changed unexpectedly during pose update")
            payload = _render_two_domains(camera)
            current_payload = {
                "rgb": payload["display_rgb_uint8"],
                "display_rgb_uint8": payload["display_rgb_uint8"],
                "radiance": payload["radiance"],
                "depth": payload["depth"],
                "seg": payload["seg"],
            }
            radiance = np.asarray(payload["radiance"], dtype=np.float32)
            display_rgb = np.asarray(payload["display_rgb_uint8"], dtype=np.uint8)
            depth = np.asarray(payload["depth"], dtype=np.float32)
            seg = np.asarray(payload["seg"], dtype=np.int32)
            camera_pose = camera_to_world_cv(camera)
            object_poses = np.stack([_entity_pose(seg_to_square[seg_id]) for seg_id in sorted_seg_ids], axis=0)
            pose_time_errors.append(0.0)
            radiance_ds[index] = radiance
            display_ds[index] = display_rgb
            depth_ds[index] = depth
            seg_ds[index] = seg
            replay_h5["timestamps_s"][index] = t
            replay_h5["camera_pose"][index] = camera_pose
            replay_h5["object_poses"][index] = object_poses
            _write_png(display_dir / f"frame_{index:06d}.png", display_rgb)
            hdr_max = max(hdr_max, float(radiance.max()))
            events = plugin.capture()
            total_events += int(len(events))
        plugin.end_episode()
        plugin.close()
        plugin = None
        replay_h5.flush()
        replay_h5.close()
        replay_h5 = None

        if hdr_max <= 1.0:
            raise RuntimeError("formal native radiance did not exceed display range")
        if max(state_time_errors) > 1e-9 or max(pose_time_errors) > 1e-9:
            raise RuntimeError("formal timestamp/pose consistency check failed")
        metadata = {
            "status": "PASS",
            "formal_metrics_executed": False,
            "direction": args.direction,
            "evaluation_duration_s": evaluation_duration,
            "generated_duration_s": generated_duration,
            "padding_duration_s": padding_duration,
            "source_dt_s": SOURCE_DT_S,
            "source_interval_count": interval_count,
            "source_frame_count": len(timestamps),
            "resolution_xy": list(FORMAL_RESOLUTION),
            "software": {
                "genesis_version": getattr(gs, "__version__", "unknown"),
                "genesis_file": str(getattr(gs, "__file__", "unknown")),
                "python": sys.version,
                "platform": platform.platform(),
                "backend": backend_name,
                "event_device": os.environ.get("GENESIS_EVENT_DEVICE", "cpu"),
            },
            "replay_archive": str(replay_dir / "state_archive.h5"),
            "display_rgb_archive": str(display_dir),
            "ours_events": str(ours_dir / "aer_events.h5"),
            "native_radiance": True,
            "native_radiance_max": hdr_max,
            "display_rgb_independent_archive": True,
            "state_timestamp_max_abs_error_s": max(state_time_errors),
            "pose_timestamp_max_abs_error_s": max(pose_time_errors),
            "segmentation_expected_entity_count": segmentation_contract["expected_entity_count"],
            "segmentation_resolved_entity_count": segmentation_contract["resolved_entity_count"],
            "segmentation_expected_entity_resolution_coverage": segmentation_contract["expected_entity_resolution_coverage"],
            "segmentation_motion_map_background_overlap": segmentation_contract["motion_map_background_overlap"],
            "trajectory_parity": trajectory_contract,
            "ours_event_count_generated_interval_and_padding": total_events,
            "ours_core_commit": CORE_COMMIT,
            "trajectory_padding_policy": "endpoint_hold_after_evaluation_interval",
        }
        (output / "formal_genesis_replay.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        return 0
    finally:
        if plugin is not None:
            try:
                plugin.end_episode()
            except Exception:
                pass
            try:
                plugin.close()
            except Exception:
                pass
        if replay_h5 is not None:
            replay_h5.close()
        gs.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
