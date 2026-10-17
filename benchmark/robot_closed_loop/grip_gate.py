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


class ProprioceptiveLiftStabilityMonitor:
    """Confirm a grasp during a small causal lift probe.

    The monitor uses only Panda finger-joint positions and the probe command
    timestamp. A small probe is necessary because identical closed apertures
    can arise from a real grasp or from an edge push. It never reads target
    pose, contacts, segmentation, depth, or evaluator score.
    """

    def __init__(self, config: Dict[str, Any], dt_s: float):
        self.baseline_start_s = float(config["baseline_start_s"])
        self.baseline_duration_s = float(config["baseline_duration_s"])
        self.confirmation_duration_s = float(config["confirmation_duration_s"])
        self.max_joint_drift_m = float(config["max_joint_drift_m"])
        self.violation_dwell_s = float(config["violation_dwell_s"])
        self.dt_s = float(dt_s)
        self.reset()

    def reset(self, start_s: Optional[float] = None) -> None:
        self.start_s = None if start_s is None else float(start_s)
        self.last_time_s: Optional[float] = None
        self.baseline_samples = []
        self.baseline_joint_position_m: Optional[np.ndarray] = None
        self.violation_for_s = 0.0
        self.maximum_joint_drift_m = 0.0
        self.state = "not_started"

    def update(
        self,
        now_s: float,
        probe_started_s: Optional[float],
        finger_joint_position_m: Sequence[float],
        finger_joint_velocity_m_s: Sequence[float],
    ) -> Dict[str, Any]:
        now = float(now_s)
        q = np.asarray(finger_joint_position_m, dtype=np.float64).reshape(2)
        qd = np.asarray(finger_joint_velocity_m_s, dtype=np.float64).reshape(2)
        if probe_started_s is None:
            return self._result("not_started", None, q, qd, None)
        if self.start_s is None or abs(self.start_s - float(probe_started_s)) > 1e-9:
            self.reset(float(probe_started_s))

        elapsed = max(0.0, now - self.start_s)
        if self.last_time_s is None:
            step_dt = 0.0
        else:
            step_dt = min(max(now - self.last_time_s, 0.0), 2.0 * self.dt_s)
        self.last_time_s = now

        baseline_end_s = self.baseline_start_s + self.baseline_duration_s
        if self.baseline_joint_position_m is None:
            if self.baseline_start_s <= elapsed <= baseline_end_s:
                self.baseline_samples.append(q.copy())
            if elapsed < baseline_end_s or not self.baseline_samples:
                self.state = "baseline_collection"
                return self._result(self.state, elapsed, q, qd, None)
            self.baseline_joint_position_m = np.mean(self.baseline_samples, axis=0)

        drift = float(np.max(np.abs(q - self.baseline_joint_position_m)))
        self.maximum_joint_drift_m = max(self.maximum_joint_drift_m, drift)
        if elapsed >= baseline_end_s and drift >= self.max_joint_drift_m:
            self.violation_for_s += step_dt
        else:
            self.violation_for_s = 0.0

        if self.violation_for_s + 1e-12 >= self.violation_dwell_s:
            self.state = "unstable_grasp"
        elif elapsed + 1e-12 >= self.confirmation_duration_s:
            self.state = "confirmed"
        else:
            self.state = "probing"
        return self._result(self.state, elapsed, q, qd, drift)

    def _result(
        self,
        state: str,
        elapsed: Optional[float],
        q: np.ndarray,
        qd: np.ndarray,
        drift: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "state": state,
            "confirmed": state == "confirmed",
            "unstable": state == "unstable_grasp",
            "probe_elapsed_s": elapsed,
            "baseline_joint_position_m": None if self.baseline_joint_position_m is None else self.baseline_joint_position_m.tolist(),
            "finger_joint_position_m": q.tolist(),
            "finger_joint_velocity_m_s": qd.tolist(),
            "joint_drift_from_probe_baseline_m": drift,
            "maximum_joint_drift_m": self.maximum_joint_drift_m,
            "joint_drift_threshold_m": self.max_joint_drift_m,
            "violation_dwell_s": self.violation_for_s,
            "required_violation_dwell_s": self.violation_dwell_s,
            "required_confirmation_duration_s": self.confirmation_duration_s,
            "input_boundary": "finger joint position/velocity and probe-command time only",
        }
