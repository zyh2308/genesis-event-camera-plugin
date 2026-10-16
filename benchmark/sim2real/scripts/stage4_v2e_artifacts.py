"""Resolve immutable outputs emitted by the registered Stage-4 V2E shards."""
from __future__ import annotations

from pathlib import Path


GPU_SHARDS = (1, 2, 4, 5)


def locate_shard_artifacts(root: Path, rollout: str) -> tuple[Path, Path]:
    """Return the unique HDF5/status pair for a rollout, failing closed."""
    candidates = [root / f"gpu{gpu}" / f"{rollout}.h5" for gpu in GPU_SHARDS]
    candidates.append(root / f"{rollout}.h5")
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one V2E HDF5 for {rollout}, found {len(matches)}: "
            f"{[str(path) for path in matches]}"
        )
    event_path = matches[0]
    status_path = event_path.with_name(f"{rollout}.status.json")
    if not status_path.is_file():
        raise RuntimeError(f"missing V2E status record for {rollout}: {status_path}")
    return event_path, status_path
