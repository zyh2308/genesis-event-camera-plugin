"""Fail-closed loader for the declared real EVK4 HDF5 source.

This module is deliberately not a metric implementation.  It proves that the
declared source can be opened and that the complete CD event dataset can be
decoded with the vendor filter.  It never aligns a method to the first event.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
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
    ecf_plugin_path: str | None = None

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


def discover_ecf_plugin() -> str | None:
    """Find the installed vendor ECF HDF5 filter before importing h5py.

    HDF5 loads filter plugins from ``HDF5_PLUGIN_PATH``.  The previous
    implementation implicitly depended on one workstation-specific path,
    which made a complete read fail even though the official Debian package
    was installed.  Search the current environment first, then the standard
    Metavision/HDF5 plugin locations, and return the directory containing the
    actual ``libH5Zecf`` library.
    """

    def _plugin_dir(value: str | Path) -> str | None:
        path = Path(value)
        if path.is_file() and "H5Zecf" in path.name:
            return str(path.parent)
        if path.is_dir() and any(path.glob("libH5Zecf.so*")):
            return str(path)
        return None

    configured = os.environ.get("HDF5_PLUGIN_PATH", "")
    for item in configured.split(os.pathsep):
        if item:
            found = _plugin_dir(item)
            if found:
                os.environ["HDF5_PLUGIN_PATH"] = found
                return found

    candidates = (
        "/usr/local/lib/plugin",
        "/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins",
        "/usr/lib/hdf5/serial/plugins",
        "/usr/lib/hdf5/plugins",
        "/opt/metavision/lib/plugin",
    )
    for item in candidates:
        found = _plugin_dir(item)
        if found:
            os.environ["HDF5_PLUGIN_PATH"] = found
            return found
    return None


def load_real_hdf5(
    path: str | Path,
    *,
    direction: str,
    sequence_start_us: int,
    duration_s_declared: float,
    expected_resolution=(1280, 720),
    duration_tolerance_s: float = 0.010,
) -> RealSequence:
    """Read all ``CD/events`` and convert to ``[t_rel,x,y,p]``.

    The caller supplies the declared segment origin from the protocol.  The
    function intentionally does not infer it from event timestamps.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ecf_plugin_path = discover_ecf_plugin()
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
            f"ECF filter before Phase 5 (plugin path: {ecf_plugin_path or 'not found'}; "
            f"original error: {type(exc).__name__}: {exc})"
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
    event_span_s = float(t[-1] - t[0]) / 1e6
    duration_error_s = event_span_s - float(duration_s_declared)
    if abs(duration_error_s) > max(float(duration_tolerance_s), 0.01 * float(duration_s_declared)):
        raise RuntimeError(
            f"{path}: event span {event_span_s:.9f}s differs from declared "
            f"duration {float(duration_s_declared):.9f}s by {duration_error_s:+.9f}s; "
            "do not repair this by shifting the first event"
        )
    attrs = dict(attrs)
    attrs["ecf_plugin_path"] = ecf_plugin_path or ""
    attrs["event_span_s"] = event_span_s
    attrs["duration_error_s"] = duration_error_s
    return RealSequence(
        direction=str(direction),
        path=path,
        events_txyp_us=events,
        sequence_start_us=int(sequence_start_us),
        duration_s_declared=float(duration_s_declared),
        metadata=attrs,
        ecf_plugin_path=ecf_plugin_path,
    )
