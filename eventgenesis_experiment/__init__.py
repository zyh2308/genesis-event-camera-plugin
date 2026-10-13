"""Experiment infrastructure for the frozen EventGenesis simulator core.

This package deliberately lives outside :mod:`genesis_event_plugin`.  It owns
experiment adapters, provenance/configuration, and dataset packaging; the
physics/radiance event-generation path remains the single implementation in
the core package.
"""

from .config import EventGenesisConfig, load_config
from .adapter import (
    ExperimentAdapter,
    QualityAdapter,
    Sim2RealAdapter,
    RobotAdapter,
)
from .runner import ExperimentRunner

__all__ = [
    "EventGenesisConfig",
    "load_config",
    "ExperimentAdapter",
    "QualityAdapter",
    "Sim2RealAdapter",
    "RobotAdapter",
    "ExperimentRunner",
]
