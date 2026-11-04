"""Pluggable event readout interface.

No EVK4 arbiter, dead-time, saturation or latency model is assumed here.  The
default is an ideal pass-through until measured readout data is available.
"""

from typing import Any


class ReadoutModel:
    """Interface for post-pixel event readout models."""

    name = "base"

    def process(self, events: Any, t_previous: float, t_current: float) -> Any:
        raise NotImplementedError


class IdealReadout(ReadoutModel):
    """Return pixel events unchanged."""

    name = "ideal_readout"

    def process(self, events: Any, t_previous: float, t_current: float) -> Any:
        return events


def build_readout_model(name: str) -> ReadoutModel:
    if name in {"ideal_readout", "none"}:
        return IdealReadout()
    raise ValueError(
        f"unknown readout model {name!r}; measured models are not bundled yet"
    )
