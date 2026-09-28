#!/usr/bin/env python3
"""Compare a fixed-camera direct sequence with the v2 physics accelerator.

The script is deliberately offline: a Genesis task adapter exports one paired
rollout to ``npz`` and this tool compares both event streams without changing
the task controller.  This makes the direct/v2 comparison reproducible before
we spend GPU time on policy training.

Expected NPZ keys:
  ``radiance`` (N,H,W or N,H,W,3), ``depth`` (N,H,W), ``seg`` (N,H,W),
  ``timestamps`` (N,), ``camera_to_world`` (N,4,4), ``K`` (3,3).

Optional articulation keys:
  ``object_ids`` (M,), ``object_to_world`` (N,M,4,4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Keep the offline comparator usable on a minimal analysis machine where the
# optional HDF5 recorder dependency is absent.  Genesis deployments import the
# normal package __init__ as usual.
_pkg = types.ModuleType("genesis_event_plugin")
_pkg.__path__ = [str(ROOT / "genesis_event_plugin")]
sys.modules.setdefault("genesis_event_plugin", _pkg)

from genesis_event_plugin.dvs_emulator import DvsEmulator  # noqa: E402
from genesis_event_plugin.noise_model import get_preset  # noqa: E402
from genesis_event_plugin.physics_interpolator import PhysicsInterpolator  # noqa: E402
from genesis_event_plugin.radiometry import prepare_linear_radiance  # noqa: E402


def _events(frames, timestamps, shape):
    config = get_preset("real_v2")
    emulator = DvsEmulator(res=shape, config=config, device="cpu")
    first = np.asarray(frames[0], dtype=np.float32)
    emulator.initialize(first, float(timestamps[0]))
    chunks = []
    for frame, timestamp in zip(frames[1:], timestamps[1:]):
        chunk = emulator.generate_events(np.asarray(frame, dtype=np.float32), float(timestamp))
        if len(chunk):
            chunks.append(chunk)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 4), dtype=np.float64)


def _required(data, name):
    if name not in data:
        raise ValueError(f"NPZ is missing required key: {name}")
    return data[name]


def compare(path: Path, keyframe_stride: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        raw = _required(data, "radiance")
        radiance = np.stack(
            [prepare_linear_radiance(frame, input_space="linear") for frame in raw], axis=0
        ).astype(np.float32)
        depth = np.asarray(_required(data, "depth"), dtype=np.float32)
        seg = np.asarray(_required(data, "seg"))
        timestamps = np.asarray(_required(data, "timestamps"), dtype=np.float64)
        camera = np.asarray(_required(data, "camera_to_world"), dtype=np.float64)
        K = np.asarray(_required(data, "K"), dtype=np.float64)
        object_ids = np.asarray(data["object_ids"], dtype=np.int64) if "object_ids" in data else np.empty(0, np.int64)
        object_pose = np.asarray(data["object_to_world"], dtype=np.float64) if "object_to_world" in data else None

    N, H, W = radiance.shape[:3]
    if depth.shape != (N, H, W) or seg.shape != (N, H, W):
        raise ValueError("radiance/depth/seg must share N,H,W")
    if timestamps.shape != (N,) or camera.shape != (N, 4, 4) or K.shape != (3, 3):
        raise ValueError("invalid timestamp/camera/K shape")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if object_pose is not None and object_pose.shape != (N, len(object_ids), 4, 4):
        raise ValueError("object_to_world must have shape (N,len(object_ids),4,4)")

    direct_frames = [radiance[i] for i in range(N)]
    direct_events = _events(direct_frames, timestamps, (H, W))

    candidate_frames = []
    candidate_times = []
    interval_stats = []
    interpolator = PhysicsInterpolator(K, disocclusion_policy="hold", composite="next_primary")
    for start in range(0, N - 1, keyframe_stride):
        end = min(start + keyframe_stride, N - 1)
        if end <= start:
            continue
        deltas = {}
        if object_pose is not None:
            for col, object_id in enumerate(object_ids):
                deltas[int(object_id)] = object_pose[end, col] @ np.linalg.inv(object_pose[start, col])
        frames, times = interpolator.interpolate(
            radiance[start], depth[start], seg[start],
            radiance[end], depth[end], seg[end],
            camera[start], camera[end], deltas,
            float(timestamps[start]), float(timestamps[end]), input_space="linear",
        )
        candidate_frames.extend(frames)
        candidate_times.extend(times)
        interval_stats.append({
            "start": int(start), "end": int(end),
            "U": int(interpolator.last_sampling.samples),
            "U_uncapped": int(interpolator.last_sampling.uncapped_samples),
            "invalid_fraction": float(interpolator.last_invalid_fraction),
        })

    candidate_events = _events(candidate_frames, candidate_times, (H, W))
    nearest_mae = []
    for frame, timestamp in zip(candidate_frames, candidate_times):
        index = int(np.argmin(np.abs(timestamps - timestamp)))
        nearest_mae.append(float(np.mean(np.abs(frame - direct_frames[index]))))
    direct_count = int(len(direct_events))
    candidate_count = int(len(candidate_events))
    return {
        "input": str(path),
        "frames": int(N), "resolution": [int(W), int(H)],
        "keyframe_stride": int(keyframe_stride),
        "direct_events": direct_count, "physics_events": candidate_count,
        "event_count_relative_error": float(abs(candidate_count - direct_count) / max(direct_count, 1)),
        "candidate_frame_mae_to_nearest_direct": float(np.mean(nearest_mae)) if nearest_mae else None,
        "candidate_timestamps_monotonic": bool(
            len(candidate_times) < 2 or np.all(np.diff(candidate_times) >= -1e-12)
        ),
        "intervals": interval_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--keyframe-stride", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.keyframe_stride < 1:
        raise SystemExit("--keyframe-stride must be >= 1")
    result = compare(args.npz, args.keyframe_stride)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
