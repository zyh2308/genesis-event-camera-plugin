"""Common v4 QualityAdapter contract.

Adapters are intentionally limited to raw-artifact loading, explicit format
conversion, and provenance validation.  They do not calculate metrics, align
streams, fit parameters, or mutate simulator outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CANONICAL_SCHEMA_VERSION = "genesis_event_quality_v4_canonical-1.0"
CANONICAL_LAYOUT = "xypt"
CANONICAL_TIMESTAMP_UNIT = "s"
CANONICAL_POLARITY_ENCODING = "-1=OFF,1=ON"

_TIME_TO_SECONDS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}

REQUIRED_ARTIFACT_METADATA = (
    "sequence_id",
    "generator_name",
    "resolution_xy",
    "event_layout",
    "polarity_encoding",
    "timestamp_unit",
    "declared_start",
    "declared_end",
    "observed_first_timestamp",
    "observed_last_timestamp",
    "event_count",
    "config_sha256",
    "protocol_sha256",
)


@dataclass(frozen=True)
class AdaptedEventStream:
    """Canonical events and the validated provenance attached to them."""

    events: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_canonical_events(
            self.events,
            resolution_xy=tuple(self.metadata["resolution_xy"]),
            declared_start_s=float(self.metadata["canonical_declared_start_s"]),
            declared_end_s=float(self.metadata["canonical_declared_end_s"]),
        )


def _time_scale(unit: str) -> float:
    try:
        return _TIME_TO_SECONDS[str(unit)]
    except KeyError as exc:
        raise ValueError(f"unsupported timestamp_unit {unit!r}") from exc


def validate_canonical_events(
    events: np.ndarray,
    *,
    resolution_xy: tuple[int, int] | None = None,
    declared_start_s: float | None = None,
    declared_end_s: float | None = None,
) -> None:
    """Validate the v4 canonical ``[x,y,p,t_s]`` array without changing it."""

    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"canonical events must have shape (N,4), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("canonical events must be numeric")
    if arr.size == 0:
        return
    if not np.isfinite(arr).all():
        raise ValueError("canonical events contain NaN or infinity")
    if (arr[:, :2] < 0).any():
        raise ValueError("canonical event coordinates must be non-negative")
    if not np.allclose(arr[:, :2], np.rint(arr[:, :2])):
        raise ValueError("canonical event coordinates must be integer-valued")
    if not np.isin(arr[:, 2], (-1.0, 1.0)).all():
        raise ValueError("canonical polarity must use {-1,+1}")
    if np.any(np.diff(arr[:, 3]) < 0):
        raise ValueError("canonical timestamps must be nondecreasing")
    if resolution_xy is not None:
        width, height = (int(value) for value in resolution_xy)
        if width <= 0 or height <= 0:
            raise ValueError("resolution_xy must contain positive dimensions")
        if (arr[:, 0] >= width).any() or (arr[:, 1] >= height).any():
            raise ValueError("canonical event coordinate exceeds resolution_xy")
    if declared_start_s is not None and float(arr[:, 3].min()) < declared_start_s:
        raise ValueError("canonical event precedes declared_start_s")
    if declared_end_s is not None and float(arr[:, 3].max()) > declared_end_s:
        raise ValueError("canonical event exceeds declared_end_s")


def _require_artifact_metadata(metadata: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_ARTIFACT_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"artifact metadata missing required fields: {missing}")
    resolution = metadata["resolution_xy"]
    if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
        raise ValueError("resolution_xy must be [width,height]")
    if any(int(value) <= 0 for value in resolution):
        raise ValueError("resolution_xy dimensions must be positive")
    unit = str(metadata["timestamp_unit"])
    _time_scale(unit)
    start = float(metadata["declared_start"])
    end = float(metadata["declared_end"])
    if end < start:
        raise ValueError("declared_end must be >= declared_start")
    event_count = metadata["event_count"]
    if isinstance(event_count, bool) or int(event_count) != event_count or int(event_count) < 0:
        raise ValueError("event_count must be a non-negative integer")
    event_count = int(event_count)
    first = metadata["observed_first_timestamp"]
    last = metadata["observed_last_timestamp"]
    if event_count == 0:
        if first is not None or last is not None:
            raise ValueError("empty artifact must use null observed timestamps")
    else:
        if first is None or last is None:
            raise ValueError("non-empty artifact requires observed timestamps")
        first = float(first)
        last = float(last)
        if not np.isfinite([first, last]).all():
            raise ValueError("observed timestamps must be finite")
        if first < start or last > end or last < first:
            raise ValueError("observed timestamps must lie within declared interval")
    for key in ("sequence_id", "generator_name", "event_layout", "polarity_encoding"):
        if not str(metadata[key]).strip():
            raise ValueError(f"artifact metadata field {key} must be non-empty")
    for key in ("config_sha256", "protocol_sha256"):
        if not str(metadata[key]).strip():
            raise ValueError(f"artifact metadata field {key} must be non-empty")


def _raw_time_from_seconds(value_s: float, unit: str) -> float:
    return float(value_s) / _time_scale(unit)


def build_output_metadata(
    metadata: Mapping[str, Any],
    events: np.ndarray,
    *,
    adapter_name: str,
    artifact_path: str | Path,
) -> dict[str, Any]:
    """Validate sidecar inputs and append canonical observed-range metadata."""

    _require_artifact_metadata(metadata)
    arr = np.asarray(events, dtype=np.float64)
    resolution = tuple(int(value) for value in metadata["resolution_xy"])
    start_s = float(metadata["declared_start"]) * _time_scale(str(metadata["timestamp_unit"]))
    end_s = float(metadata["declared_end"]) * _time_scale(str(metadata["timestamp_unit"]))
    validate_canonical_events(
        arr,
        resolution_xy=resolution,
        declared_start_s=start_s,
        declared_end_s=end_s,
    )

    result = dict(metadata)
    observed_first = (
        _raw_time_from_seconds(float(arr[0, 3]), str(metadata["timestamp_unit"]))
        if len(arr)
        else None
    )
    observed_last = (
        _raw_time_from_seconds(float(arr[-1, 3]), str(metadata["timestamp_unit"]))
        if len(arr)
        else None
    )
    expected_count = int(metadata["event_count"])
    if expected_count != len(arr):
        raise ValueError(
            f"sidecar event_count={expected_count} does not match artifact row count={len(arr)}"
        )
    for field, actual in (
        ("observed_first_timestamp", observed_first),
        ("observed_last_timestamp", observed_last),
    ):
        expected = metadata[field]
        if expected is None or actual is None:
            if expected is not actual:
                raise ValueError(f"sidecar {field} does not match artifact observation")
        elif not np.isclose(float(expected), float(actual), rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"sidecar {field}={expected} does not match artifact value={actual}"
            )
    result.update(
        {
            "adapter_name": adapter_name,
            "artifact_path": str(Path(artifact_path)),
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "canonical_event_layout": CANONICAL_LAYOUT,
            "canonical_timestamp_unit": CANONICAL_TIMESTAMP_UNIT,
            "canonical_polarity_encoding": CANONICAL_POLARITY_ENCODING,
            "canonical_declared_start_s": start_s,
            "canonical_declared_end_s": end_s,
            "event_count": int(len(arr)),
            "observed_first_timestamp": observed_first,
            "observed_last_timestamp": observed_last,
        }
    )
    return result


class QualityAdapter(ABC):
    """Base contract for raw artifact → v4 canonical event conversion."""

    source_name: str

    @abstractmethod
    def adapt(
        self,
        artifact: str | Path,
        *,
        metadata: Mapping[str, Any],
    ) -> AdaptedEventStream:
        """Load one raw artifact and return events plus validated metadata."""
