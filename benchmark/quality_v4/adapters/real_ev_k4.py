"""QualityAdapter for the declared Real EVK4 HDF5 artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from benchmark.io.real_stream import load_real_hdf5

from .base import AdaptedEventStream, QualityAdapter, build_output_metadata


class RealEVK4Adapter(QualityAdapter):
    """Convert vendor EVT3 ``CD/events`` into canonical ``[x,y,p,t_s]``."""

    source_name = "real_evk4"

    def adapt(
        self,
        artifact: str | Path,
        *,
        metadata: Mapping[str, Any],
    ) -> AdaptedEventStream:
        """Use the existing fail-closed loader; do not infer a first-event origin.

        Adapter-specific loader options are explicit metadata fields so the
        declared acquisition origin and duration remain auditable.
        """

        if "sequence_start_us" not in metadata:
            raise ValueError("RealEVK4 metadata requires explicit sequence_start_us")
        if "duration_s_declared" not in metadata:
            raise ValueError("RealEVK4 metadata requires explicit duration_s_declared")
        width, height = (int(value) for value in metadata["resolution_xy"])
        sequence = load_real_hdf5(
            artifact,
            direction=str(metadata.get("direction", metadata["sequence_id"])),
            sequence_start_us=int(metadata["sequence_start_us"]),
            duration_s_declared=float(metadata["duration_s_declared"]),
            expected_resolution=(width, height),
        )
        raw = np.asarray(sequence.events_txyp_us, dtype=np.float64)
        # Existing loader returns [t_rel_us, x, y, p_01].
        events = raw[:, [1, 2, 3, 0]].copy()
        events[:, 2] = np.where(events[:, 2] > 0, 1.0, -1.0)
        events[:, 3] *= 1e-6
        output_metadata = build_output_metadata(
            metadata,
            events,
            adapter_name=self.source_name,
            artifact_path=artifact,
        )
        return AdaptedEventStream(events=events, metadata=output_metadata)
