#!/usr/bin/env python3
"""Recapture exact robot physics states at the strict-sampled time cadence.

The existing paired training cache stores HDR at 100 Hz and a state trace at
1 kHz. This CPU-only runner repeats those deterministic episodes and writes
only kinematic states at every physics step (normally 0.5 ms). It does not
render images, generate events, train, or read any future state for control.
The saved trace is later used only for offline exact-state HDR rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.robot_env import build_robot_scene
from benchmark.robot_closed_loop.run import run_oracle_trial


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"not JSON serializable: {type(value).__name__}")
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=encode) + "\n", encoding="utf-8")


def compact_state(row: dict) -> dict:
    return {
        "time_s": float(row["time_s"]),
        "object": {
            "time_s": float(row["object"]["time_s"]),
            "position_m": np.asarray(row["object"]["position_m"], dtype=np.float64).tolist(),
            "orientation_wxyz": np.asarray(row["object"]["orientation_wxyz"], dtype=np.float64).tolist(),
            "velocity_m_s": np.asarray(row["object"]["velocity_m_s"], dtype=np.float64).tolist(),
            "angular_velocity_rad_s": np.asarray(
                row["object"].get("angular_velocity_rad_s", [0, 0, 0]), dtype=np.float64,
            ).tolist(),
        },
        "robot_qpos": np.asarray(row["robot_qpos"], dtype=np.float64).tolist(),
    }


def main() -> int:
    process_started = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--scene-config", type=Path, required=True)
    p.add_argument("--control-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode-ids", default="0,1,2,3,4,5,6,7,8")
    p.add_argument("--trace-period-s", type=float, default=0.0005)
    p.add_argument("--expected-cache-manifest", type=Path)
    p.add_argument("--experiment-config", type=Path)
    args = p.parse_args()

    cache_root = args.cache_root.resolve()
    scene_path = args.scene_config.resolve()
    control_path = args.control_config.resolve()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {out}")
    episode_ids = [int(x) for x in args.episode_ids.split(",") if x.strip()]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs must be unique")

    expected_cache_hashes = {}
    if args.expected_cache_manifest:
        expected_cache_hashes = json.loads(args.expected_cache_manifest.resolve().read_text(encoding="utf-8"))
    config = json.loads(scene_path.read_text(encoding="utf-8"))
    config["shared_control"] = json.loads(control_path.read_text(encoding="utf-8"))

    genesis_init_started = time.perf_counter()
    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    genesis_import_and_init_wall_s = time.perf_counter() - genesis_init_started
    scene_build_started = time.perf_counter()
    env = build_robot_scene(config, segmentation_level="link")
    scene_build_wall_s = time.perf_counter() - scene_build_started
    dt = float(env.dt_s)
    if args.trace_period_s < dt or not np.isclose(args.trace_period_s / dt,
                                                   round(args.trace_period_s / dt), rtol=0, atol=1e-8):
        raise ValueError(f"trace period {args.trace_period_s} must be an integer multiple of physics dt {dt}")

    out.mkdir(parents=True, exist_ok=False)
    records = []
    rollout_started = time.perf_counter()
    try:
        for episode_id in episode_ids:
            episode = cache_root / f"episode_{episode_id:03d}"
            cache_path = episode / "trajectory_cache.h5"
            result_path = episode / "episode_result.json"
            old_trace_path = episode / "oracle_trace.jsonl"
            frame_manifest_path = episode / "frame_manifest.json"
            for required in (cache_path, result_path, old_trace_path, frame_manifest_path):
                if not required.is_file():
                    raise FileNotFoundError(required)
            cache_hash = sha256(cache_path)
            manifest_key = f"episode_{episode_id:03d}/trajectory_cache.h5"
            expected = expected_cache_hashes.get(manifest_key, expected_cache_hashes.get(str(cache_path)))
            if expected and expected != cache_hash:
                raise RuntimeError(f"immutable cache hash mismatch: {manifest_key}")

            source_result = json.loads(result_path.read_text(encoding="utf-8"))
            seed = int(source_result["seed"])
            speed = float(source_result.get("launch_speed_m_s", source_result.get("speed_m_s")))
            start_y = float(source_result.get("start_y_m", source_result.get("initial_y_m")))
            deceleration = float(config["controller"]["initial_deceleration_assumption_m_s2"])

            result, trace, _ = run_oracle_trial(
                env, episode_id, seed, speed, deceleration,
                initial_y_m=start_y, capture=False, trace_period_s=args.trace_period_s,
            )
            if int(result["seed"]) != seed:
                raise RuntimeError(f"rerun seed mismatch for episode {episode_id}")

            episode_out = out / f"episode_{episode_id:03d}"
            episode_out.mkdir()
            trace_out = episode_out / "state_trace_0p5ms.jsonl"
            with trace_out.open("w", encoding="utf-8", newline="\n") as stream:
                for row in trace:
                    stream.write(json.dumps(compact_state(row), separators=(",", ":"), allow_nan=False) + "\n")
            write_json(episode_out / "episode_result.json", result)

            # Compare exact state rows at each legacy 1 kHz timestamp and at
            # every 100 Hz saved HDR cache timestamp. These are consistency
            # checks, not synthetic state interpolation.
            new_times = np.asarray([float(x["time_s"]) for x in trace], dtype=np.float64)
            new_pos = np.asarray([x["object"]["position_m"] for x in trace], dtype=np.float64)
            new_vel = np.asarray([x["object"]["velocity_m_s"] for x in trace], dtype=np.float64)
            new_qpos = np.asarray([x["robot_qpos"] for x in trace], dtype=np.float64)
            old_rows = [json.loads(x) for x in old_trace_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            old_times = np.asarray([x["time_s"] for x in old_rows], dtype=np.float64)
            old_pos = np.asarray([x["object"]["position_m"] for x in old_rows], dtype=np.float64)
            old_vel = np.asarray([x["object"]["velocity_m_s"] for x in old_rows], dtype=np.float64)
            old_quat = np.asarray([x["object"]["orientation_wxyz"] for x in old_rows], dtype=np.float64)
            old_angular = np.asarray([
                x["object"].get("angular_velocity_rad_s", [0, 0, 0]) for x in old_rows
            ], dtype=np.float64)
            old_qpos = np.asarray([x["robot_qpos"] for x in old_rows], dtype=np.float64)
            index = np.searchsorted(new_times, old_times)
            valid = (index < len(new_times)) & np.isclose(new_times[np.minimum(index, len(new_times)-1)],
                                                          old_times, rtol=0, atol=1e-9)
            if not np.all(valid):
                raise RuntimeError(f"rerun misses {int((~valid).sum())} legacy trace timestamps")
            index = index[valid]
            pos_delta = np.abs(new_pos[index] - old_pos[valid])
            vel_delta = np.abs(new_vel[index] - old_vel[valid])
            new_quat = np.asarray([x["object"]["orientation_wxyz"] for x in trace], dtype=np.float64)
            new_angular = np.asarray([
                x["object"].get("angular_velocity_rad_s", [0, 0, 0]) for x in trace
            ], dtype=np.float64)
            quat_delta = np.abs(new_quat[index] - old_quat[valid])
            angular_delta = np.abs(new_angular[index] - old_angular[valid])
            qpos_delta = np.abs(new_qpos[index] - old_qpos[valid])

            with h5py.File(cache_path, "r") as h:
                cache_times = np.asarray(h["timestamps_s"], dtype=np.float64)
                cache_pos = np.asarray(h["target_position_m"], dtype=np.float64)
                cache_vel = np.asarray(h["target_velocity_m_s"], dtype=np.float64)
                cache_qpos = np.asarray(h["robot_qpos"], dtype=np.float64)
            # The old cache's forced t=0 frame precedes the first physics
            # step. Compare all subsequent exact 100 Hz timestamps.
            cache_compare = cache_times > new_times[0] + 1e-10
            cache_times_compare = cache_times[cache_compare]
            cache_pos_compare = cache_pos[cache_compare]
            cache_vel_compare = cache_vel[cache_compare]
            cache_qpos_compare = cache_qpos[cache_compare]
            cache_index = np.searchsorted(new_times, cache_times_compare)
            cache_valid = (cache_index < len(new_times)) & np.isclose(
                new_times[np.minimum(cache_index, len(new_times)-1)], cache_times_compare, rtol=0, atol=1e-9,
            )
            if not np.all(cache_valid):
                raise RuntimeError(f"rerun misses {int((~cache_valid).sum())} post-launch 100 Hz HDR cache timestamps")
            cache_index = cache_index[cache_valid]
            cache_pos_delta = np.abs(new_pos[cache_index] - cache_pos_compare[cache_valid])
            cache_vel_delta = np.abs(new_vel[cache_index] - cache_vel_compare[cache_valid])
            cache_qpos_delta = np.abs(new_qpos[cache_index] - cache_qpos_compare[cache_valid])

            episode_summary = {
                "episode_id": episode_id, "seed": seed, "speed_m_s": speed, "start_y_m": start_y,
                "legacy_cache_sha256": cache_hash,
                "legacy_frame_manifest_sha256": sha256(frame_manifest_path),
                "legacy_1khz_trace_sha256": sha256(old_trace_path),
                "new_trace_sha256": sha256(trace_out),
                "physics_dt_s": dt, "trace_period_s": args.trace_period_s,
                "new_state_rows": len(trace), "legacy_state_rows_matched": len(old_rows),
                "legacy_100hz_cache_rows_matched": int(np.count_nonzero(cache_valid)),
                "legacy_100hz_cache_rows_excluded_prelaunch": int(np.count_nonzero(~cache_compare)),
                "max_abs_delta_vs_legacy_1khz": {
                    "target_position_m": float(pos_delta.max()),
                    "target_velocity_m_s": float(vel_delta.max()),
                    "target_orientation_wxyz": float(quat_delta.max()),
                    "target_angular_velocity_rad_s": float(angular_delta.max()),
                    "robot_qpos": float(qpos_delta.max()),
                },
                "max_abs_delta_vs_legacy_100hz_cache": {
                    "target_position_m": float(cache_pos_delta.max()),
                    "target_velocity_m_s": float(cache_vel_delta.max()),
                    "robot_qpos": float(cache_qpos_delta.max()),
                },
                "result": result,
            }
            write_json(episode_out / "summary.json", episode_summary)
            records.append(episode_summary)
            print(json.dumps({"episode_id": episode_id, "rows": len(trace),
                              "max_pos_delta_legacy_1khz_m": float(pos_delta.max()),
                              "max_pos_delta_legacy_100hz_m": float(cache_pos_delta.max())}), flush=True)
    finally:
        gs.destroy()

    run = {
        "status": "PASS" if all(
            max(r["max_abs_delta_vs_legacy_1khz"]["target_position_m"],
                r["max_abs_delta_vs_legacy_1khz"]["target_velocity_m_s"],
                r["max_abs_delta_vs_legacy_1khz"]["target_orientation_wxyz"],
                r["max_abs_delta_vs_legacy_1khz"]["target_angular_velocity_rad_s"],
                r["max_abs_delta_vs_legacy_1khz"]["robot_qpos"],
                r["max_abs_delta_vs_legacy_100hz_cache"]["target_position_m"],
                r["max_abs_delta_vs_legacy_100hz_cache"]["target_velocity_m_s"],
                r["max_abs_delta_vs_legacy_100hz_cache"]["robot_qpos"]) < 1e-5 for r in records
        ) else "STATE_REPLAY_MISMATCH",
        "purpose": "CPU-only deterministic physics trace recapture; no render, events, training or GPU",
        "source_commit": os.environ.get("ROBOT_EXECUTION_CODE_COMMIT", "unknown"),
        "scene_config_sha256": sha256(scene_path), "control_config_sha256": sha256(control_path),
        "cache_root": str(cache_root), "episode_ids": episode_ids,
        "episode_split": {"train": [0, 1, 2, 3, 7, 8], "validation": [4, 5, 6]},
        "trace_period_s": args.trace_period_s, "physics_dt_s": dt,
        "runner_module": str(Path(inspect.getsourcefile(run_oracle_trial)).resolve()),
        "runner_module_sha256": sha256(Path(inspect.getsourcefile(run_oracle_trial)).resolve()),
        "robot_env_module": str(Path(inspect.getsourcefile(build_robot_scene)).resolve()),
        "robot_env_module_sha256": sha256(Path(inspect.getsourcefile(build_robot_scene)).resolve()),
        "python_executable": sys.executable,
        "genesis_module": str(Path(gs.__file__).resolve()),
        "experiment_config_sha256": sha256(args.experiment_config.resolve()) if args.experiment_config else None,
        "genesis_import_and_init_wall_s": genesis_import_and_init_wall_s,
        "scene_build_wall_s": scene_build_wall_s,
        "episode_replay_wall_s": time.perf_counter() - rollout_started,
        "elapsed_wall_s_before_artifact_hashing": time.perf_counter() - process_started,
        "episodes": records,
    }
    write_json(out / "summary.json", run)
    write_json(out / "run_manifest.json", {
        "run_id": out.name,
        "source_commit": run["source_commit"],
        "experiment_config_sha256": run["experiment_config_sha256"],
        "scene_config_sha256": run["scene_config_sha256"],
        "control_config_sha256": run["control_config_sha256"],
        "runner_module_sha256": run["runner_module_sha256"],
        "robot_env_module_sha256": run["robot_env_module_sha256"],
        "episode_inputs": [
            {"episode_id": row["episode_id"], "legacy_cache_sha256": row["legacy_cache_sha256"],
             "legacy_trace_sha256": row["legacy_1khz_trace_sha256"], "new_trace_sha256": row["new_trace_sha256"]}
            for row in records
        ],
    })
    (out / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.relative_to(out).as_posix()}\n"
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"
    ), encoding="utf-8")
    print(json.dumps({"status": run["status"], "output": str(out),
                      "elapsed_wall_s_before_artifact_hashing": run["elapsed_wall_s_before_artifact_hashing"]},
                     indent=2), flush=True)
    return 0 if run["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
