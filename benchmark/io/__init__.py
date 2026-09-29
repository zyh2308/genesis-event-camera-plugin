"""Readiness-only data loaders used by the pre-formal benchmark gate."""

from .real_stream import RealSequence, load_real_hdf5

__all__ = ["RealSequence", "load_real_hdf5"]
