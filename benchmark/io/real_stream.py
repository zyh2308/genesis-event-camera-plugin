"""Fail-closed loader for the declared real EVK4 HDF5 source.

This module is deliberately not a metric implementation.  It proves that the
declared source can be opened and that the complete CD event dataset can be
decoded with the vendor filter.  It never aligns a method to the first event.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RealSequence:
    direction: str
    path: Path
    events_txyp_us: np.ndarray
    sequence_start_us: int
    duration_s_declared: float
    metadata: Mapping[str, object]

    @property
    def duration_s_event_span(self) -> float:
        if len(self.events_txyp_us) == 0:
            return 0.0
        return float(self.events_txyp_us[-1, 0] - self.events_txyp_us[0, 0]) / 1e6


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_real_hdf5(
    path: str | Path,
    *,
    direction: str,
    sequence_start_us: int,
    duration_s_declared: float,
    expected_resolution=(1280, 720),
) -> RealSequence:
    """Read all ``CD/events`` and convert to ``[t_rel,x,y,p]``.

    The caller supplies the declared segment origin from the protocol.  The
    function intentionally does not infer it from event timestamps.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for the declared real source") from exc

    try:
        with h5py.File(path, "r") as handle:
            if "CD/events" not in handle:
                raise RuntimeError(f"{path}: missing complete CD/events dataset")
            dataset = handle["CD/events"]
            if dataset.dtype.names is None or not {"x", "y", "p", "t"}.issubset(dataset.dtype.names):
                raise RuntimeError(f"{path}: CD/events schema is not EVT3 x,y,p,t")
            attrs = {str(k): _as_text(v) for k, v in handle.attrs.items()}
            # This is intentionally a full read.  A shape-only or first-chunk
            # check would not prove that the vendor ECF filter can decode all
            # events required by the formal pair.
            raw = dataset[:]
    except Exception as exc:
        raise RuntimeError(
            f"{path}: complete vendor HDF5 read failed; restore the official "
            f"ECF filter before Phase 5 (original error: {type(exc).__name__}: {exc})"
        ) from exc

    if raw.ndim != 1 or len(raw) == 0:
        raise RuntimeError(f"{path}: complete CD/events dataset is empty")
    width, height = (int(x) for x in str(attrs.get("geometry", "")).split("x", 1))
    if (width, height) != tuple(expected_resolution):
        raise RuntimeError(f"{path}: geometry {(width, height)} != {tuple(expected_resolution)}")
    t = np.asarray(raw["t"], dtype=np.int64) - int(sequence_start_us)
    if np.any(np.diff(t) < 0):
        raise RuntimeError(f"{path}: event timestamps are not monotonic")
    if int(t.min()) < 0:
        raise RuntimeError(
            f"{path}: an event precedes declared sequence origin {sequence_start_us}; "
            "do not replace the origin with first-event alignment"
        )
    x = np.asarray(raw["x"], dtype=np.int64)
    y = np.asarray(raw["y"], dtype=np.int64)
    p = (np.asarray(raw["p"], dtype=np.int64) > 0).astype(np.int8)
    if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
        raise RuntimeError(f"{path}: event coordinates exceed declared geometry")
    events = np.column_stack((t, x, y, p))
    return RealSequence(
        direction=str(direction),
        path=path,
        events_txyp_us=events,
        sequence_start_us=int(sequence_start_us),
        duration_s_declared=float(duration_s_declared),
        metadata=attrs,
    )
