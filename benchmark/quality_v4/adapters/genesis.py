"""QualityAdapter for EventGenesis recorder HDF5 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from genesis_event_plugin.evaluation.event_reader import read_genesis_hdf5

from .base import AdaptedEventStream, QualityAdapter, build_output_metadata


class GenesisAdapter(QualityAdapter):
    """Convert Genesis recorder output to canonical ``[x,y,p,t_s]``."""

    source_name = "eventgenesis"

    def adapt(
        self,
        artifact: str | Path,
        *,
        metadata: Mapping[str, Any],
    ) -> AdaptedEventStream:
        events = read_genesis_hdf5(str(artifact))
        output_metadata = build_output_metadata(
            metadata,
            events,
            adapter_name=self.source_name,
            artifact_path=artifact,
        )
        return AdaptedEventStream(events=events, metadata=output_metadata)
