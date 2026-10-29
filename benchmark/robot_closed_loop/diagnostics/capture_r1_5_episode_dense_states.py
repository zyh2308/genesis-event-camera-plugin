#!/usr/bin/env python3
"""Run the original reference runner with a read-only per-step state tap.

The tap records actual robot and target generalized positions only within a
short requested interval. It does not write controls, advance extra physics,
modify the renderer, or change the event source. The ordinary runner remains
the authority for the rollout and writes its normal artifacts unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

import benchmark.robot_closed_loop.run as runner


def main() -> int:
    window = os.environ.get("R15_DENSE_STATE_CAPTURE_WINDOW_S", "")
    output_text = os.environ.get("R15_DENSE_STATE_CAPTURE_PATH", "")
    if not window or not output_text:
        raise RuntimeError("set R15_DENSE_STATE_CAPTURE_WINDOW_S and R15_DENSE_STATE_CAPTURE_PATH")
    start_s, end_s = (float(value) for value in window.split(","))
    if not 0 <= start_s < end_s:
        raise ValueError(f"invalid capture interval {window!r}")
    output = Path(output_text).resolve()
    if output.exists():
        raise FileExistsError(output)

    original_trial = runner.run_oracle_trial

    def wrapped_trial(env: Any, *args: Any, **kwargs: Any) -> Any:
        launch_time_s: list[float | None] = [None]
        captured_times: list[float] = []
        captured_robot_qpos: list[np.ndarray] = []
        captured_target_qpos: list[np.ndarray] = []
        original_step = env.scene.step
        original_set_velocity = env.set_initial_target_velocity

        def set_initial_velocity_and_mark_launch(velocity: Any) -> Any:
            result = original_set_velocity(velocity)
            launch_time_s[0] = runner.get_sim_time(env)
            return result

        def step_and_capture(*step_args: Any, **step_kwargs: Any) -> Any:
            result = original_step(*step_args, **step_kwargs)
            if launch_time_s[0] is None:
                return result
            time_s = runner.get_sim_time(env) - float(launch_time_s[0])
            if start_s - 1e-10 <= time_s <= end_s + 1e-10:
                obj = runner.object_state(env.target, time_s)
                captured_times.append(float(time_s))
                captured_robot_qpos.append(
                    np.asarray(runner.as_numpy(env.robot.get_dofs_position())).reshape(-1).copy()
                )
                captured_target_qpos.append(np.asarray(
                    list(obj["position_m"]) + list(obj["orientation_wxyz"]), dtype=np.float32
                ))
            return result

        env.scene.step = step_and_capture
        env.set_initial_target_velocity = set_initial_velocity_and_mark_launch
        try:
            result = original_trial(env, *args, **kwargs)
        finally:
            env.scene.step = original_step
            env.set_initial_target_velocity = original_set_velocity

        if not captured_times:
            raise RuntimeError("state tap captured no physical samples")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            times_s=np.asarray(captured_times, dtype=np.float64),
            robot_qpos=np.asarray(captured_robot_qpos),
            target_qpos=np.asarray(captured_target_qpos, dtype=np.float32),
        )
        metadata = {
            "capture_window_s": [start_s, end_s],
            "capture_hook": "immediately after each original scene.step return, using read-only qpos/pose getters",
            "physics_steps_added": 0,
            "controls_written_by_hook": 0,
            "sample_count": len(captured_times),
            "times_s": [float(captured_times[0]), float(captured_times[-1])],
            "interval_s_min": float(np.min(np.diff(captured_times))) if len(captured_times) > 1 else None,
            "interval_s_max": float(np.max(np.diff(captured_times))) if len(captured_times) > 1 else None,
            "capture_file": str(output),
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"dense_state_capture": metadata}, separators=(",", ":")), flush=True)
        return result

    runner.run_oracle_trial = wrapped_trial
    runner.main()
    with np.load(output, allow_pickle=False) as cache:
        times = np.asarray(cache["times_s"], dtype=np.float64)
    expected = np.arange(start_s, end_s + 0.00025, 0.0005, dtype=np.float64)
    if len(times) != 81 or len(expected) != 81 or not np.allclose(times, expected, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"expected 81 actual 0.5 ms physics samples, got {len(times)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
