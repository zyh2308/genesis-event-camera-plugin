"""Versioned task scoring for the robot benchmark (kept outside frozen core)."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from benchmark.robot_closed_loop.robot_env import quaternion_matrix_wxyz


LEGACY_SCORE_VERSION = "r0_v1_legacy_center_clearance_historical_contact"
STRICT_SCORE_VERSION = "r0_5_strict_continuous_contact_clearance_slip"


def cube_bottom_clearance_m(
    position_m: Sequence[float],
    orientation_wxyz: Sequence[float],
    size_m: Sequence[float],
    support_z_m: float = 0.0,
) -> float:
    """Return the lowest transformed cube vertex above the support plane."""
    position = np.asarray(position_m, dtype=np.float64).reshape(3)
    size = np.asarray(size_m, dtype=np.float64).reshape(3)
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    local_vertices = signs * (size / 2.0)
    rotation = quaternion_matrix_wxyz(orientation_wxyz)
    world_vertices = local_vertices @ rotation.T + position
    return float(np.min(world_vertices[:, 2]) - float(support_z_m))


class StrictHoldScorer:
    """Stepwise R0.5 hold evaluator using measured pose/contact state."""

    def __init__(self, clearance_threshold_m: float, max_relative_slip_m: float, hold_s: float):
        self.clearance_threshold_m = float(clearance_threshold_m)
        self.max_relative_slip_m = float(max_relative_slip_m)
        self.hold_s = float(hold_s)
        self.grasp_relative_anchor_m = None
        self.hold_start_s = None
        self.hold_steps = 0
        self.longest_hold_steps = 0
        self.min_clearance_all_m = float("inf")
        self.min_clearance_hold_m = float("inf")
        self.min_clearance_longest_hold_m = float("inf")
        self.max_slip_m = 0.0
        self.bilateral_samples = 0
        self.support_contact_samples = 0
        self.support_contact_when_clearance_met_samples = 0
        self.clearance_samples = 0
        self.bilateral_and_clearance_samples = 0
        self.qualified_samples = 0
        self.last = {
            "bottom_clearance_m": None,
            "relative_slip_m": None,
            "bilateral_contact": False,
            "support_contact": False,
            "qualified": False,
        }

    def update(
        self,
        time_s: float,
        relative_to_tcp_m: Sequence[float],
        bottom_clearance_m: float,
        bilateral_contact: bool,
        support_contact: bool,
        dt_s: float,
    ) -> Dict[str, Any]:
        relative = np.asarray(relative_to_tcp_m, dtype=np.float64).reshape(3)
        clearance = float(bottom_clearance_m)
        self.min_clearance_all_m = min(self.min_clearance_all_m, clearance)
        self.bilateral_samples += int(bool(bilateral_contact))
        self.support_contact_samples += int(bool(support_contact))
        self.clearance_samples += int(clearance >= self.clearance_threshold_m)
        self.bilateral_and_clearance_samples += int(bool(bilateral_contact) and clearance >= self.clearance_threshold_m)
        if clearance >= self.clearance_threshold_m and support_contact:
            self.support_contact_when_clearance_met_samples += 1
        if bilateral_contact and self.grasp_relative_anchor_m is None:
            self.grasp_relative_anchor_m = relative.copy()
        slip = None
        if self.grasp_relative_anchor_m is not None:
            slip = float(np.linalg.norm(relative - self.grasp_relative_anchor_m))
            self.max_slip_m = max(self.max_slip_m, slip)
        qualified = bool(
            self.grasp_relative_anchor_m is not None
            and bilateral_contact
            and not support_contact
            and clearance >= self.clearance_threshold_m
            and slip is not None
            and slip <= self.max_relative_slip_m
        )
        if qualified:
            if self.hold_start_s is None:
                self.hold_start_s = float(time_s)
                self.min_clearance_hold_m = clearance
                self.hold_steps = 0
            self.hold_steps += 1
            self.qualified_samples += 1
            self.min_clearance_hold_m = min(self.min_clearance_hold_m, clearance)
            if self.hold_steps > self.longest_hold_steps:
                self.longest_hold_steps = self.hold_steps
                self.min_clearance_longest_hold_m = self.min_clearance_hold_m
        else:
            self.hold_start_s = None
            self.hold_steps = 0
            self.min_clearance_hold_m = float("inf")
        hold_s = float(self.hold_steps * float(dt_s))
        self.last = {
            "time_s": float(time_s),
            "bottom_clearance_m": clearance,
            "relative_slip_m": slip,
            "bilateral_contact": bool(bilateral_contact),
            "support_contact": bool(support_contact),
            "qualified": qualified,
            "continuous_hold_s": hold_s,
            "minimum_hold_clearance_m": None if not np.isfinite(self.min_clearance_hold_m) else self.min_clearance_hold_m,
        }
        return dict(self.last)

    def success_at(self, dt_s: float) -> bool:
        return self.longest_hold_steps * float(dt_s) + 1e-12 >= self.hold_s

    def summary(self, dt_s: float, total_steps: int) -> Dict[str, Any]:
        return {
            "score_version": STRICT_SCORE_VERSION,
            "success": self.success_at(dt_s),
            "continuous_qualified_hold_s": float(self.longest_hold_steps * float(dt_s)),
            "current_qualified_hold_s": float(self.hold_steps * float(dt_s)),
            "required_continuous_hold_s": self.hold_s,
            "minimum_true_bottom_clearance_m": None if not np.isfinite(self.min_clearance_all_m) else self.min_clearance_all_m,
            "minimum_true_bottom_clearance_during_longest_qualified_hold_m": None if not np.isfinite(self.min_clearance_longest_hold_m) else self.min_clearance_longest_hold_m,
            "maximum_relative_slip_since_bilateral_grasp_m": self.max_slip_m,
            "maximum_relative_slip_m": self.max_relative_slip_m,
            "clearance_threshold_samples": self.clearance_samples,
            "bilateral_and_clearance_samples": self.bilateral_and_clearance_samples,
            "qualified_hold_samples": self.qualified_samples,
            "bilateral_contact_samples": self.bilateral_samples,
            "support_contact_samples": self.support_contact_samples,
            "support_contact_while_clearance_met_samples": self.support_contact_when_clearance_met_samples,
            "contact_sample_count": int(total_steps),
            "qualified_hold_sample_fraction": float(self.qualified_samples / total_steps) if total_steps else None,
            "bilateral_contact_fraction": float(self.bilateral_samples / total_steps) if total_steps else None,
            "support_contact_fraction": float(self.support_contact_samples / total_steps) if total_steps else None,
            "grasp_relative_anchor_m": None if self.grasp_relative_anchor_m is None else self.grasp_relative_anchor_m.tolist(),
        }
