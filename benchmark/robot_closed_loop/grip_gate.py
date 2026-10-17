"""Proprioceptive grasp-completion gate shared by oracle and learned control.

The gate deliberately consumes only Panda finger joint positions/velocities and
the close-command timestamp. It has no target pose, contact, segmentation, or
depth input. Contact and strict task success remain evaluator-only signals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


class ProprioceptiveGripGate:
    """Debounced two-finger stall within the calibrated object aperture band."""

    def __init__(self, config: Dict[str, Any], dt_s: float):
        self.min_close_command_elapsed_s = float(config["min_close_command_elapsed_s"])
        self.joint_position_range_m = tuple(float(v) for v in config["finger_joint_position_range_m"])
        self.max_finger_asymmetry_m = float(config["max_finger_joint_asymmetry_m"])
        self.max_abs_joint_velocity_m_s = float(config["max_abs_finger_joint_velocity_m_s"])
        self.stable_dwell_s = float(config["stable_dwell_s"])
        self.dt_s = float(dt_s)
        self.stable_for_s = 0.0
        self.last_time_s: Optional[float] = None

    def reset(self) -> None:
        self.stable_for_s = 0.0
        self.last_time_s = None

    def update(
        self,
        now_s: float,
        close_command_started_s: Optional[float],
        finger_joint_position_m: Sequence[float],
        finger_joint_velocity_m_s: Sequence[float],
    ) -> Dict[str, Any]:
        """Return a causal gate diagnostic, updating the continuous dwell timer."""
        now = float(now_s)
        q = np.asarray(finger_joint_position_m, dtype=np.float64).reshape(2)
        qd = np.asarray(finger_joint_velocity_m_s, dtype=np.float64).reshape(2)
        elapsed = None if close_command_started_s is None else max(0.0, now - float(close_command_started_s))

        command_active = elapsed is not None
        timing_ok = bool(command_active and elapsed >= self.min_close_command_elapsed_s)
        aperture_ok = bool(np.all((q >= self.joint_position_range_m[0]) & (q <= self.joint_position_range_m[1])))
        asymmetry = float(abs(q[0] - q[1]))
        symmetry_ok = asymmetry <= self.max_finger_asymmetry_m
        max_speed = float(np.max(np.abs(qd)))
        stopped_ok = max_speed <= self.max_abs_joint_velocity_m_s
        instantaneous = bool(timing_ok and aperture_ok and symmetry_ok and stopped_ok)

        if self.last_time_s is None:
            step_dt = 0.0
        else:
            step_dt = min(max(now - self.last_time_s, 0.0), 2.0 * self.dt_s)
        self.last_time_s = now
        self.stable_for_s = self.stable_for_s + step_dt if instantaneous else 0.0

        if not command_active:
            reason = "close_not_commanded"
        elif not timing_ok:
            reason = "close_command_too_recent"
        elif not aperture_ok:
            reason = "aperture_outside_object_band"
        elif not symmetry_ok:
            reason = "finger_positions_asymmetric"
        elif not stopped_ok:
            reason = "fingers_still_moving"
        elif self.stable_for_s + 1e-12 < self.stable_dwell_s:
            reason = "stable_dwell_incomplete"
        else:
            reason = "proprioceptive_grasp_candidate"
        return {
            "eligible": bool(instantaneous and self.stable_for_s + 1e-12 >= self.stable_dwell_s),
            "instantaneous_candidate": instantaneous,
            "reason": reason,
            "close_command_elapsed_s": elapsed,
            "minimum_close_command_elapsed_s": self.min_close_command_elapsed_s,
            "finger_joint_position_m": q.tolist(),
            "finger_joint_velocity_m_s": qd.tolist(),
            "finger_joint_asymmetry_m": asymmetry,
            "max_abs_finger_joint_velocity_m_s": max_speed,
            "stable_for_s": self.stable_for_s,
            "required_stable_dwell_s": self.stable_dwell_s,
            "input_boundary": "finger joint position/velocity and close-command time only",
        }
