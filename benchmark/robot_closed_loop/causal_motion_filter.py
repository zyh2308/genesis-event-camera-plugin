"""Fit a planar state from past measurements and the public slide calibration."""
import numpy as np


def causal_planar_motion_fit(times_s, positions_xy_m, cutoff_s, deceleration_m_s2):
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions_xy_m, dtype=np.float64)
    if positions.shape != (len(times), 2) or len(times) < 2:
        raise ValueError('at least two timestamped XY measurements are required')
    if not np.isfinite(times).all() or not np.isfinite(positions).all():
        raise ValueError('motion history must be finite')
    if np.any(np.diff(times) <= 0) or times[-1] > float(cutoff_s) + 1e-10:
        raise ValueError('history must be strictly ordered and no later than cutoff')
    a = float(deceleration_m_s2)
    if not np.isfinite(a) or a < 0:
        raise ValueError('public deceleration must be finite and nonnegative')
    dt = times - float(cutoff_s)
    design = np.column_stack([np.ones(len(dt)), dt])
    initial = np.linalg.lstsq(design, positions, rcond=None)[0]
    speed = float(np.linalg.norm(initial[1]))
    acceleration = np.zeros(2) if speed < 1e-9 else -a * initial[1] / speed
    corrected = positions - .5 * dt[:, None] ** 2 * acceleration
    fit = np.linalg.lstsq(design, corrected, rcond=None)[0]
    return fit[0], fit[1]
