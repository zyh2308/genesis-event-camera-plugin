"""Fixed causal sliding-target predictor and bounded lateral command profile."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MotionState:
    time_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray


@dataclass(frozen=True)
class InterceptPrediction:
    reachable: bool
    ttc_s: float = math.inf
    position_m: Optional[np.ndarray] = None
    propagated_state: Optional[MotionState] = None


class FixedDecelerationPredictor:
    """Propagate a delayed observation using one public, shared deceleration."""

    def __init__(self, intercept_x_m: float, deceleration_m_s2: float):
        if deceleration_m_s2 < 0:
            raise ValueError("deceleration must be non-negative")
        self.intercept_x_m = float(intercept_x_m)
        self.deceleration_m_s2 = float(deceleration_m_s2)

    def propagate(self, state: MotionState, dt_s: float) -> MotionState:
        p = np.asarray(state.position_m, dtype=np.float64).copy()
        v = np.asarray(state.velocity_m_s, dtype=np.float64).copy()
        speed = float(np.linalg.norm(v[:2]))
        if speed > 1e-9 and self.deceleration_m_s2 > 0:
            u = v / speed
            stop_s = speed / self.deceleration_m_s2
            tau = min(max(0.0, float(dt_s)), stop_s)
            p += v * tau - 0.5 * self.deceleration_m_s2 * u * tau * tau
            v -= self.deceleration_m_s2 * u * tau
            if tau >= stop_s:
                v[:] = 0.0
        else:
            p += v * max(0.0, float(dt_s))
        return MotionState(float(state.time_s + dt_s), p, v)

    def predict(self, delayed_state: MotionState, latency_s: float) -> InterceptPrediction:
        state = self.propagate(delayed_state, latency_s)
        p0 = state.position_m
        v0 = state.velocity_m_s
        speed = float(np.linalg.norm(v0[:2]))
        if speed < 1e-8 or abs(float(v0[0])) < 1e-8:
            return InterceptPrediction(False, propagated_state=state)

        # Friction decelerates opposite the current planar velocity. Check that
        # the target reaches the intercept plane before it stops, then bisect
        # the monotonic x trajectory; no future Genesis state is consulted.
        a = self.deceleration_m_s2
        stop_s = speed / a if a > 0 else math.inf
        if a > 0:
            stop_pos = p0 + v0 * stop_s - 0.5 * a * (v0 / speed) * stop_s**2
            if (self.intercept_x_m - p0[0]) * (self.intercept_x_m - stop_pos[0]) > 0:
                return InterceptPrediction(False, propagated_state=state)
            upper = stop_s
        else:
            direction = np.sign(v0[0])
            if (self.intercept_x_m - p0[0]) * direction <= 0:
                return InterceptPrediction(False, propagated_state=state)
            upper = abs((self.intercept_x_m - p0[0]) / v0[0]) * 1.5

        def position_at(t: float) -> np.ndarray:
            if a <= 0:
                return p0 + v0 * t
            u = v0 / speed
            return p0 + v0 * t - 0.5 * a * u * t * t

        x0 = float(p0[0] - self.intercept_x_m)
        x1 = float(position_at(upper)[0] - self.intercept_x_m)
        if x0 == 0:
            t_hit = 0.0
        elif x0 * x1 > 0:
            return InterceptPrediction(False, propagated_state=state)
        else:
            lo, hi = 0.0, upper
            increasing = x1 > x0
            for _ in range(52):
                mid = 0.5 * (lo + hi)
                xm = float(position_at(mid)[0] - self.intercept_x_m)
                if (xm < 0) == increasing:
                    lo = mid
                else:
                    hi = mid
            t_hit = 0.5 * (lo + hi)
        return InterceptPrediction(True, t_hit, position_at(t_hit), state)


class BoundedLateralCommand:
    """Apply public Cartesian speed and acceleration limits to y commands."""

    def __init__(self, initial_y_m: float, speed_limit_m_s: float, accel_limit_m_s2: float):
        self.position_m = float(initial_y_m)
        self.velocity_m_s = 0.0
        self.speed_limit_m_s = float(speed_limit_m_s)
        self.accel_limit_m_s2 = float(accel_limit_m_s2)

    def step(self, target_y_m: float, dt_s: float) -> Tuple[float, float]:
        dt = float(dt_s)
        error = float(target_y_m) - self.position_m
        if abs(error) < 1e-9:
            self.velocity_m_s = 0.0
            return self.position_m, self.velocity_m_s
        desired_v = float(np.clip(error / max(dt, 1e-9), -self.speed_limit_m_s, self.speed_limit_m_s))
        delta_v = float(np.clip(
            desired_v - self.velocity_m_s,
            -self.accel_limit_m_s2 * dt,
            self.accel_limit_m_s2 * dt,
        ))
        self.velocity_m_s += delta_v
        step = self.velocity_m_s * dt
        if abs(step) >= abs(error):
            self.position_m = float(target_y_m)
            self.velocity_m_s = 0.0
        else:
            self.position_m += step
        return self.position_m, self.velocity_m_s


def fit_common_deceleration(trajectories: Sequence[Sequence[MotionState]]) -> float:
    """Fit one robust scalar deceleration from no-hand slide trajectories."""
    estimates = []
    for states in trajectories:
        if len(states) < 4:
            continue
        speed = np.asarray([np.linalg.norm(s.velocity_m_s[:2]) for s in states])
        times = np.asarray([s.time_s for s in states])
        moving = speed > 0.04
        if moving.sum() < 4:
            continue
        slope = float(np.polyfit(times[moving] - times[moving][0], speed[moving], 1)[0])
        if np.isfinite(slope) and slope < 0:
            estimates.append(-slope)
    if not estimates:
        raise ValueError("no usable deceleration estimates in slide calibration set")
    return float(np.median(estimates))
