"""Pre-registered distributional metrics for event-stream smoke tests.

This module intentionally contains only metrics that remain meaningful without
pixel-level correspondence.  Inputs are converted to the canonical internal
layout ``[t_s, x, y, p_01]``.  It does not perform alignment, registration, or
parameter fitting; those operations belong to a future protocol-approved
runner and must be recorded explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def canonicalize_events(
    events: np.ndarray,
    *,
    layout: str = "txyp",
    timestamp_unit: str = "us",
) -> np.ndarray:
    """Return finite events as float64 ``[t_s, x, y, p_01]``.

    Supported layouts are ``txyp`` (the plugin/V2E HDF5 convention) and
    ``xypt`` (the PECS convention).  Polarity may be ``0/1`` or ``-1/+1``;
    the output is always ``0/1``.
    """
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"events must have shape (N, 4), got {arr.shape}")
    if layout == "txyp":
        out = np.asarray(arr[:, [0, 1, 2, 3]], dtype=np.float64).copy()
    elif layout == "xypt":
        out = np.asarray(arr[:, [3, 0, 1, 2]], dtype=np.float64).copy()
    else:
        raise ValueError("layout must be 'txyp' or 'xypt'")
    if timestamp_unit == "s":
        scale = 1.0
    elif timestamp_unit == "ms":
        scale = 1e-3
    elif timestamp_unit == "us":
        scale = 1e-6
    elif timestamp_unit == "ns":
        scale = 1e-9
    else:
        raise ValueError("timestamp_unit must be one of s/ms/us/ns")
    if out.size == 0:
        return out
    if not np.isfinite(out).all():
        raise ValueError("events contain NaN or infinite values")
    out[:, 0] *= scale
    if (out[:, 1:3] < 0).any():
        raise ValueError("event coordinates must be non-negative")
    polarity = out[:, 3]
    unique = np.unique(polarity)
    if np.isin(unique, (0.0, 1.0)).all():
        out[:, 3] = polarity > 0
    elif np.isin(unique, (-1.0, 1.0)).all():
        out[:, 3] = polarity > 0
    else:
        raise ValueError(f"polarity must use 0/1 or -1/+1, found {unique[:8]}")
    return out


def _structured_to_array(events: np.ndarray) -> np.ndarray:
    names = set(events.dtype.names or ())
    required = {"x", "y", "p", "t"}
    if not required.issubset(names):
        raise ValueError(f"structured event array lacks {required - names}")
    return np.column_stack([events["t"], events["x"], events["y"], events["p"]])


def load_aer_events(
    path: str | Path,
    *,
    dataset: str = "events",
    layout: str = "txyp",
    timestamp_unit: str = "us",
    max_events: Optional[int] = None,
) -> np.ndarray:
    """Load a standard plugin/V2E HDF5 stream or Metavision EVT3 HDF5.

    ``max_events`` is only a diagnostic guard.  It takes a deterministic
    prefix, never a result-dependent sample.
    """
    path = Path(path)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"only HDF5 is supported by this loader: {path}")
    import h5py

    with h5py.File(path, "r") as handle:
        if dataset in handle:
            raw = handle[dataset]
            if max_events is not None:
                raw = raw[:max_events]
            arr = raw[...]
            return canonicalize_events(arr, layout=layout, timestamp_unit=timestamp_unit)
        if "CD/events" in handle:
            raw = handle["CD/events"]
            if max_events is not None:
                raw = raw[:max_events]
            arr = _structured_to_array(raw[...])
            return canonicalize_events(arr, layout="txyp", timestamp_unit="us")
    raise KeyError(f"no {dataset!r} or 'CD/events' dataset in {path}")


def _duration(events: np.ndarray, duration_s: Optional[float]) -> float:
    if duration_s is not None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        return float(duration_s)
    if len(events) < 2:
        return 0.0
    value = float(events[-1, 0] - events[0, 0])
    return value if value > 0 else 0.0


def _spatial_histogram(
    events: np.ndarray,
    *,
    width: int,
    height: int,
    grid: tuple[int, int],
) -> np.ndarray:
    gx, gy = grid
    if gx <= 0 or gy <= 0:
        raise ValueError("spatial grid dimensions must be positive")
    hist = np.zeros((gy, gx), dtype=np.float64)
    if len(events) == 0:
        return hist
    x = np.clip((events[:, 1].astype(np.int64) * gx) // width, 0, gx - 1)
    y = np.clip((events[:, 2].astype(np.int64) * gy) // height, 0, gy - 1)
    np.add.at(hist, (y, x), 1.0)
    total = hist.sum()
    return hist / total if total else hist


def _temporal_rate(
    events: np.ndarray,
    *,
    duration_s: float,
    bin_s: float,
) -> np.ndarray:
    if bin_s <= 0:
        raise ValueError("temporal bin must be positive")
    n_bins = max(1, int(np.ceil(duration_s / bin_s)))
    result = np.zeros(n_bins, dtype=np.float64)
    if len(events) == 0:
        return result
    origin = float(events[0, 0])
    index = np.floor((events[:, 0] - origin) / bin_s).astype(np.int64)
    index = np.clip(index, 0, n_bins - 1)
    np.add.at(result, index, 1.0)
    return result / bin_s


def summarize_events(
    events: np.ndarray,
    *,
    resolution: tuple[int, int] = (1280, 720),
    duration_s: Optional[float] = None,
    spatial_grid: tuple[int, int] = (8, 8),
    temporal_bin_s: float = 0.005,
) -> dict[str, Any]:
    """Compute the frozen distributional summary for one fixed window."""
    arr = canonicalize_events(events, layout="txyp", timestamp_unit="s")
    width, height = map(int, resolution)
    if width <= 0 or height <= 0:
        raise ValueError("resolution must be positive")
    if len(arr) and ((arr[:, 1] >= width).any() or (arr[:, 2] >= height).any()):
        raise ValueError("event coordinate exceeds declared resolution")
    dur = _duration(arr, duration_s)
    on_fraction = float(arr[:, 3].mean()) if len(arr) else 0.0
    active = np.zeros((height, width), dtype=bool)
    if len(arr):
        active[arr[:, 2].astype(np.int64), arr[:, 1].astype(np.int64)] = True
    summary = {
        "event_count": int(len(arr)),
        "duration_s": float(dur),
        "event_rate_hz": float(len(arr) / dur) if dur > 0 else None,
        "event_rate_mean_hz": float(len(arr) / dur) if dur > 0 else None,
        "on_fraction": on_fraction,
        "off_fraction": float(1.0 - on_fraction),
        "active_pixel_ratio": float(active.mean()),
        "resolution": [width, height],
        "spatial_grid": list(spatial_grid),
        "spatial_distribution": _spatial_histogram(
            arr, width=width, height=height, grid=spatial_grid
        ).tolist(),
        "temporal_bin_s": float(temporal_bin_s),
        "temporal_rate_hz": _temporal_rate(
            arr, duration_s=max(dur, temporal_bin_s), bin_s=temporal_bin_s
        ).tolist(),
    }
    temporal = np.asarray(summary["temporal_rate_hz"], dtype=np.float64)
    summary["event_rate_std_hz"] = float(temporal.std())
    return summary


def compare_summaries(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    """Return pre-registered errors; no aggregate/ranking score is produced."""
    ref_duration = float(reference["duration_s"])
    cand_duration = float(candidate["duration_s"])
    if ref_duration <= 0 or cand_duration <= 0:
        raise ValueError("both summaries need positive durations")
    if list(reference["resolution"]) != list(candidate["resolution"]):
        raise ValueError("resolution mismatch")
    if list(reference["spatial_grid"]) != list(candidate["spatial_grid"]):
        raise ValueError("spatial grid mismatch")
    ref_spatial = np.asarray(reference["spatial_distribution"], dtype=np.float64)
    cand_spatial = np.asarray(candidate["spatial_distribution"], dtype=np.float64)
    return {
        "event_rate_relative_error": float(
            abs(candidate["event_rate_hz"] - reference["event_rate_hz"])
            / max(reference["event_rate_hz"], 1e-12)
        ),
        "polarity_ratio_error": float(abs(candidate["on_fraction"] - reference["on_fraction"])),
        "active_pixel_ratio_error": float(
            abs(candidate["active_pixel_ratio"] - reference["active_pixel_ratio"])
        ),
        "spatial_distribution_l1": float(np.abs(cand_spatial - ref_spatial).sum()),
        "duration_ratio": float(cand_duration / ref_duration),
    }
