"""Dataset layout and provenance manifest for all three paper lines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import EventGenesisConfig, config_sha256, config_to_dict
from .sidecar import SidecarWriter


DATASET_SCHEMA_VERSION = "eventgenesis-dataset-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DatasetExporter:
    """Create the stable package layout without duplicating simulator logic."""

    def __init__(self, root: str | Path, config: EventGenesisConfig, *,
                 config_path: Optional[str | Path] = None,
                 experiment_kind: str = "quality", run_id: str = "run-0000"):
        if experiment_kind not in {"quality", "sim2real", "robot"}:
            raise ValueError("experiment_kind must be quality, sim2real, or robot")
        self.root = Path(root)
        self.events_dir = self.root / "events"
        self.labels_dir = self.root / "labels"
        self.metadata_dir = self.root / "metadata"
        for directory in (self.events_dir, self.labels_dir, self.metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.config.validate()
        self.config_path = Path(config_path) if config_path is not None else None
        self.experiment_kind = experiment_kind
        self.run_id = run_id
        self.sidecar = SidecarWriter(self.metadata_dir)

    @property
    def plugin_output_dir(self) -> str:
        return str(self.events_dir)

    def write_metadata(self, *, scene: Optional[Mapping[str, Any]] = None,
                       extra: Optional[Mapping[str, Any]] = None) -> Path:
        payload: Dict[str, Any] = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment_kind": self.experiment_kind,
            "core_commit": self.config.core_commit,
            "genesis_version": self.config.genesis_version,
            "config": config_to_dict(self.config),
            "scene": dict(scene or {}),
            "files": {
                "events": "events/aer_events.h5",
                "state": "metadata/state.jsonl",
                "metadata": "metadata/metadata.json",
                "labels": "labels/",
            },
        }
        if self.config_path is not None:
            payload["config_path"] = str(self.config_path)
            payload["config_sha256"] = config_sha256(self.config_path)
        if extra:
            payload["extra"] = dict(extra)
        return self.sidecar.write_metadata(payload)

    def finalize(self) -> Path:
        """Write a manifest after plugin/sidecar handles have been closed."""
        self.sidecar.close()
        files = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files[str(path.relative_to(self.root))] = _sha256(path)
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "run_id": self.run_id,
            "experiment_kind": self.experiment_kind,
            "core_commit": self.config.core_commit,
            "config_sha256": config_sha256(self.config_path) if self.config_path else None,
            "files_sha256": files,
        }
        path = self.root / "manifest.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path
