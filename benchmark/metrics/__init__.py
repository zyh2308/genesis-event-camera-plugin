"""Dependency-light, pre-registered benchmark metrics."""

from .event_statistics import (
    compare_summaries,
    load_aer_events,
    summarize_events,
)

__all__ = ["compare_summaries", "load_aer_events", "summarize_events"]

