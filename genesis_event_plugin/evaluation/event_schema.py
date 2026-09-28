"""Canonical event-array conversion and validation.

The plugin emits ``[t_s, x, y, p]`` while PECS evaluates ``[x, y, p, t]``.
Keeping that conversion explicit prevents plausible-looking metrics computed on
the wrong columns.  The canonical evaluation representation is float64
``[x, y, p, t_s]`` with polarity in ``{-1, +1}``.
"""

from __future__ import annotations

import numpy as np


_LAYOUTS = {
    "xypt": (0, 1, 2, 3),
    "txyp": (1, 2, 3, 0),
}

_TIME_SCALES = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


def canonicalize_events(
    events: np.ndarray,
    *,
    layout: str,
    timestamp_unit: str = "s",
    sort: bool = False,
) -> np.ndarray:
    """Return events as ``[x, y, p, t_s]`` with ``p`` in ``{-1,+1}``.

    Args:
        events: ``(N,4)`` numeric array.
        layout: ``"txyp"`` for plugin output or ``"xypt"`` for PECS order.
        timestamp_unit: unit used by the input timestamp column.
        sort: sort by timestamp after conversion.
    """
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"events must have shape (N,4), got {arr.shape}")
    if layout not in _LAYOUTS:
        raise ValueError(f"unknown event layout {layout!r}; expected {tuple(_LAYOUTS)}")
    if timestamp_unit not in _TIME_SCALES:
        raise ValueError(
            f"unknown timestamp unit {timestamp_unit!r}; expected {tuple(_TIME_SCALES)}"
        )

    out = np.asarray(arr[:, _LAYOUTS[layout]], dtype=np.float64).copy()
    if out.size == 0:
        return out
    if not np.isfinite(out).all():
        raise ValueError("events contain NaN or infinite values")

    out[:, 3] *= _TIME_SCALES[timestamp_unit]
    if (out[:, :2] < 0).any():
        raise ValueError("event coordinates must be non-negative")

    polarities = np.unique(out[:, 2])
    if np.isin(polarities, (0.0, 1.0)).all():
        out[:, 2] = np.where(out[:, 2] > 0, 1.0, -1.0)
    elif not np.isin(polarities, (-1.0, 1.0)).all():
        raise ValueError(
            "polarity must use {-1,+1} or {0,1}; "
            f"found {polarities[:8].tolist()}"
        )

    if sort:
        out = out[np.argsort(out[:, 3], kind="stable")]
    return out

