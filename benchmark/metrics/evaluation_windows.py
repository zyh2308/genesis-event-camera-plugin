"""Deterministic equal-duration evaluation windows.

The window policy is protocol data, not a tunable argument.  A window is
centered at a fixed normalized fraction of a declared sequence duration and
clipped only at the sequence boundaries.  All methods for a pair must use the
same declared duration, window length, and fractions.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def fixed_equal_duration_windows(
    events: np.ndarray,
    *,
    sequence_duration_s: float,
    window_duration_s: float,
    normalized_fractions: Iterable[float] = (0.25, 0.50, 0.75),
) -> list[dict[str, object]]:
    """Extract deterministic windows from canonical ``[t_s,x,y,p]`` events.

    ``sequence_duration_s`` is supplied by the frozen pair manifest.  It is
    never inferred from candidate event timestamps, which prevents a method
    with a different event onset/offset from receiving a different window.
    """
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("events must have shape (N, 4) in canonical layout")
    if sequence_duration_s <= 0 or window_duration_s <= 0:
        raise ValueError("sequence and window durations must be positive")
    if window_duration_s > sequence_duration_s:
        raise ValueError("window duration cannot exceed sequence duration")

    fractions = tuple(float(value) for value in normalized_fractions)
    if not fractions or any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("normalized fractions must lie in [0, 1]")

    max_start = sequence_duration_s - window_duration_s
    result: list[dict[str, object]] = []
    for fraction in fractions:
        center = fraction * sequence_duration_s
        start = min(max(center - window_duration_s / 2.0, 0.0), max_start)
        end = start + window_duration_s
        mask = (arr[:, 0] >= start) & (arr[:, 0] < end)
        result.append(
            {
                "fraction": fraction,
                "start_s": start,
                "end_s": end,
                "duration_s": window_duration_s,
                "events": arr[mask],
            }
        )
    return result
