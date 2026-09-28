"""
PECS-based evaluation metrics for event camera simulators.

Modules:
  pecs_metrics    — CD/GD computation with KD-tree acceleration
  evaluator       — Unified evaluation (6 internal + PECS external metrics)
  event_reader    — HDF5/RAW/numpy event data readers
  plan_a_runner   — Plan A: 6-scene Genesis vs real camera benchmark
"""

from .pecs_metrics import (
    chamfer_distance,
    gaussian_distance,
    compute_both,
    normalize_events,
    compare_to_baselines,
    PECS_BASELINES,
    V2E_BASELINES,
    ESIM_BASELINES,
)

from .evaluator import (
    EventSimEvaluator,
    InternalMetrics,
    ExternalMetrics,
    EvaluationReport,
    batch_evaluate,
)

from .event_reader import (
    read_events,
    read_genesis_hdf5,
    read_genesis_event_frames,
    read_prophesee_raw,
)

from .event_schema import canonicalize_events

from .plan_a_runner import (
    run_plan_a,
    generate_genesis_events,
    PLAN_A_SCENES,
)
