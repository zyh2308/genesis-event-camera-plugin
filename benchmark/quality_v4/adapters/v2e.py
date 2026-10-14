"""QualityAdapter for official V2E HDF5 event artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from benchmark.metrics.event_statistics import load_aer_events

from .base import AdaptedEventStream, QualityAdapter, build_output_metadata


class V2EAdapter(QualityAdapter):
    """Convert official V2E ``events`` HDF5 output to canonical events."""

    source_name = "v2e_official_core"

    def adapt(
        self,
        artifact: str | Path,
        *,
        metadata: Mapping[str, Any],
    ) -> AdaptedEventStream:
        raw = load_aer_events(
            artifact,
            dataset="events",
            layout="txyp",
            timestamp_unit=str(metadata["timestamp_unit"]),
        )
        # Existing distributional loader returns [t_s,x,y,p_01].
        events = np.asarray(raw[:, [1, 2, 3, 0]], dtype=np.float64)
        events[:, 2] = np.where(events[:, 2] > 0, 1.0, -1.0)
        output_metadata = build_output_metadata(
            metadata,
            events,
            adapter_name=self.source_name,
            artifact_path=artifact,
        )
        return AdaptedEventStream(events=events, metadata=output_metadata)
