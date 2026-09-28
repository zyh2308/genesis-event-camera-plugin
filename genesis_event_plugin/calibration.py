"""Versioned sensor-calibration bundle loader.

The bundle layout is intentionally data-first so a measured camera can be
added without changing simulator code::

    calibration/<camera_id>/v3/
      metadata.json
      intrinsics.json
      radiometry.json
      contrast_threshold.json
      background_noise.npz          # optional
      temporal_response.json        # optional
      readout.json                  # optional

No values in this module are EVK4 defaults.  Missing measured artifacts remain
missing and calibrated transfer requests fail closed.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .radiometry import RadiometricResponse


_SOURCE_STATUSES = {"source", "measured", "nominal", "fitted", "assumed"}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"calibration file {path} must contain a JSON object")
    return payload


@dataclass(frozen=True)
class CalibrationBundle:
    """A validated v3 calibration directory and its optional artifacts."""

    root: Path
    camera_id: str
    metadata: Dict[str, Any]
    intrinsics: Optional[Dict[str, Any]]
    radiometry: Optional[Dict[str, Any]]
    contrast_threshold: Optional[Dict[str, Any]]
    temporal_response: Optional[Dict[str, Any]]
    readout: Optional[Dict[str, Any]]
    background_noise_path: Optional[Path]

    @classmethod
    def from_directory(cls, root: str) -> "CalibrationBundle":
        directory = Path(root).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"calibration directory not found: {directory}")
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"v3 calibration bundle requires {metadata_path}")
        metadata = _read_json(metadata_path)
        if metadata.get("schema_version") != "genesis-event-calibration-v3":
            raise ValueError("calibration metadata must declare genesis-event-calibration-v3")
        camera_id = str(metadata.get("camera_id", ""))
        if not camera_id:
            raise ValueError("calibration metadata requires camera_id")
        if directory.name != "v3" or directory.parent.name != camera_id:
            raise ValueError("bundle path must be calibration/<camera_id>/v3")
        source = metadata.get("source_status", "")
        if source not in _SOURCE_STATUSES:
            raise ValueError("metadata.source_status must identify measured/nominal/fitted/etc.")

        def optional_json(name: str) -> Optional[Dict[str, Any]]:
            path = directory / name
            return _read_json(path) if path.is_file() else None

        noise_path = directory / "background_noise.npz"
        return cls(
            root=directory,
            camera_id=camera_id,
            metadata=metadata,
            intrinsics=optional_json("intrinsics.json"),
            radiometry=optional_json("radiometry.json"),
            contrast_threshold=optional_json("contrast_threshold.json"),
            temporal_response=optional_json("temporal_response.json"),
            readout=optional_json("readout.json"),
            background_noise_path=noise_path if noise_path.is_file() else None,
        )

    def radiometric_response(self) -> RadiometricResponse:
        """Build the explicit response object, if radiometry.json exists."""
        if self.radiometry is None:
            raise ValueError("bundle has no radiometry.json")
        fields = dict(self.radiometry)
        transfer = fields.get("transfer_path")
        if transfer:
            fields["transfer_path"] = str((self.root / transfer).resolve())
        response = RadiometricResponse(**{
            key: fields[key] for key in (
                "mode", "gain", "offset", "floor", "calibration_status",
                "calibration_id", "transfer_path",
            ) if key in fields
        })
        response.validate()
        return response

    def summary(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "schema_version": self.metadata["schema_version"],
            "source_status": self.metadata["source_status"],
            "files": {
                "intrinsics": self.intrinsics is not None,
                "radiometry": self.radiometry is not None,
                "contrast_threshold": self.contrast_threshold is not None,
                "background_noise": self.background_noise_path is not None,
                "temporal_response": self.temporal_response is not None,
                "readout": self.readout is not None,
            },
        }
