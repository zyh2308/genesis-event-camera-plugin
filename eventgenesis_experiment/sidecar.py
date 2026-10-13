"""JSON sidecar contract for timestamp-aligned physical state and labels."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Mapping, Optional

import numpy as np


SIDECAR_SCHEMA_VERSION = "eventgenesis-sidecar-v1"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


class SidecarWriter:
    """Write immutable run metadata and one synchronized JSONL row per capture."""

    def __init__(self, metadata_dir: str | Path):
        self.root = Path(metadata_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "metadata.json"
        self.state_path = self.root / "state.jsonl"
        self._state_handle = None

    def write_metadata(self, metadata: Mapping[str, Any]) -> Path:
        payload = {"schema_version": SIDECAR_SCHEMA_VERSION, **json_safe(dict(metadata))}
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        return self.metadata_path

    def open(self) -> None:
        if self._state_handle is None:
            self._state_handle = self.state_path.open("w", encoding="utf-8")

    def append(
        self,
        *,
        episode_index: int = 0,
        frame_index: int,
        timestamp_s: float,
        event_index_start: int,
        event_index_end: int,
        events_timestamp_start_s: Optional[float],
        events_timestamp_end_s: Optional[float],
        state: Mapping[str, Any],
        labels: Mapping[str, Any],
        action: Any = None,
    ) -> None:
        self.open()
        row = {
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "timestamp_s": float(timestamp_s),
            "event_index": {"start": int(event_index_start), "end": int(event_index_end)},
            "event_timestamp_s": {
                "start": None if events_timestamp_start_s is None else float(events_timestamp_start_s),
                "end": None if events_timestamp_end_s is None else float(events_timestamp_end_s),
            },
            "state": dict(state),
            "labels": dict(labels),
            "action": action,
            "action_timestamp_s": float(timestamp_s) if action is not None else None,
        }
        self._state_handle.write(json.dumps(json_safe(row), sort_keys=True, ensure_ascii=False) + "\n")
        self._state_handle.flush()

    def close(self) -> None:
        if self._state_handle is not None:
            self._state_handle.close()
            self._state_handle = None
