"""Stage 5 Q2 quality-interface design.

This package defines the v4 boundary between raw event artifacts and future
metrics.  It deliberately does not execute a benchmark, perform alignment, or
change the frozen simulator.
"""

from .adapters import (
    AdaptedEventStream,
    GenesisAdapter,
    QualityAdapter,
    RealEVK4Adapter,
    V2EAdapter,
)

__all__ = [
    "AdaptedEventStream",
    "GenesisAdapter",
    "QualityAdapter",
    "RealEVK4Adapter",
    "V2EAdapter",
]
