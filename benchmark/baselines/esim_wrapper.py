#!/usr/bin/env python3
"""Fail-closed adapter for the official ESIM output contract.

This module does not implement event generation. It only normalizes an event
stream produced by an official ESIM executable/binding into the benchmark
schema ``[timestamp_us, x, y, polarity_01]``. If no official executable is
provided, the wrapper refuses to invent one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
from typing import Optional

import numpy as np


def canonicalize_esim_events(
    events: np.ndarray,
    *,
    timestamp_unit: str = "s",
    polarity_encoding: str = "pm1",
) -> np.ndarray:
    """Convert official ESIM-style ``[t,x,y,p]`` to ``uint64 [t_us,x,y,p]``."""
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected (N,4) [t,x,y,p], got {arr.shape}")
    out = np.asarray(arr, dtype=np.float64).copy()
    scale = {"s": 1e6, "ms": 1e3, "us": 1.0, "ns": 1e-3}.get(timestamp_unit)
    if scale is None:
        raise ValueError("timestamp_unit must be s/ms/us/ns")
    if len(out) == 0:
        return np.empty((0, 4), dtype=np.uint64)
    if not np.isfinite(out).all():
        raise ValueError("ESIM output contains NaN or infinity")
    if (out[:, 0] < 0).any() or (out[:, 1:3] < 0).any():
        raise ValueError("timestamps and coordinates must be non-negative")
    if polarity_encoding == "pm1":
        if not np.isin(np.unique(out[:, 3]), (-1.0, 1.0)).all():
            raise ValueError("pm1 polarity encoding requires {-1,+1}")
        out[:, 3] = out[:, 3] > 0
    elif polarity_encoding == "01":
        if not np.isin(np.unique(out[:, 3]), (0.0, 1.0)).all():
            raise ValueError("01 polarity encoding requires {0,1}")
    else:
        raise ValueError("polarity_encoding must be pm1 or 01")
    timestamps = out[:, 0] * scale
    if np.any(np.abs(timestamps - np.round(timestamps)) > 1e-6):
        raise ValueError("timestamps cannot be represented as integer microseconds")
    out[:, 0] = np.round(timestamps)
    out[:, 1:3] = np.round(out[:, 1:3])
    if (out[:, 1:3] > np.iinfo(np.uint32).max).any():
        raise ValueError("coordinates exceed uint32 range")
    if np.any(np.diff(out[:, 0]) < 0):
        raise ValueError("ESIM timestamps are not monotonic")
    return out.astype(np.uint64)


def run_official_esim(command: str, *, cwd: Optional[Path] = None, timeout_s: int = 3600) -> None:
    """Run a caller-supplied official ESIM command without altering it."""
    if not command.strip():
        raise ValueError("no official ESIM command supplied; refusing to emulate ESIM")
    subprocess.run(shlex.split(command), cwd=str(cwd) if cwd else None, check=True, timeout=timeout_s)


def write_aer_h5(events: np.ndarray, output: Path, *, resolution: tuple[int, int]) -> None:
    """Write the unified benchmark HDF5 schema."""
    import h5py

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        ds = handle.create_dataset("events", data=events, dtype=np.uint64)
        ds.attrs["columns"] = "timestamp_us,x,y,polarity"
        ds.attrs["polarity_encoding"] = "0=OFF, 1=ON"
        ds.attrs["source"] = "official ESIM output converted by benchmark wrapper"
        handle.attrs["resolution"] = resolution
        handle.attrs["event_count"] = int(len(events))


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert official ESIM events; never implements ESIM.")
    parser.add_argument("--events-npy", type=Path, help="official ESIM [t,x,y,p] .npy output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--timestamp-unit", choices=["s", "ms", "us", "ns"], default="s")
    parser.add_argument("--polarity-encoding", choices=["pm1", "01"], default="pm1")
    args = parser.parse_args()
    if args.events_npy is None:
        raise SystemExit("No official ESIM output was supplied; wrapper refuses to generate a substitute.")
    raw = np.load(args.events_npy)
    events = canonicalize_esim_events(
        raw, timestamp_unit=args.timestamp_unit, polarity_encoding=args.polarity_encoding
    )
    if len(events) and ((events[:, 1] >= args.width).any() or (events[:, 2] >= args.height).any()):
        raise SystemExit("ESIM event coordinate exceeds declared resolution")
    write_aer_h5(events, args.output, resolution=(args.width, args.height))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

