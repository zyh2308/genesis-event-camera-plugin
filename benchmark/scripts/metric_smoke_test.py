#!/usr/bin/env python3
"""Verify metric implementations and schema conversions without a formal run."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmark.metrics.event_statistics import (  # noqa: E402
    canonicalize_events,
    compare_summaries,
    summarize_events,
)
try:  # SciPy is an optional environment dependency for this Phase 1-4 check.
    from genesis_event_plugin.evaluation.pecs_metrics import (  # noqa: E402
        chamfer_distance,
        gaussian_distance,
    )
    PECS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # keep the non-CD/GD checks runnable
    chamfer_distance = None
    gaussian_distance = None
    PECS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def main() -> int:
    raw = np.array(
        [
            [0, 1, 1, 1],
            [1000, 2, 1, 0],
            [2000, 2, 2, 1],
            [3000, 3, 2, 0],
        ],
        dtype=np.uint64,
    )
    canonical = canonicalize_events(raw, layout="txyp", timestamp_unit="us")
    assert canonical.shape == (4, 4)
    assert np.isclose(canonical[-1, 0], 0.003)
    assert set(np.unique(canonical[:, 3])) == {0.0, 1.0}

    reference = summarize_events(
        canonical,
        resolution=(4, 4),
        duration_s=0.004,
        spatial_grid=(2, 2),
        temporal_bin_s=0.001,
    )
    candidate = summarize_events(
        canonical,
        resolution=(4, 4),
        duration_s=0.004,
        spatial_grid=(2, 2),
        temporal_bin_s=0.001,
    )
    errors = compare_summaries(reference, candidate)
    assert all(np.isclose(value, 0.0) for key, value in errors.items() if key != "duration_ratio")
    assert np.isclose(errors["duration_ratio"], 1.0)

    # The existing PECS implementation is verified on identical point clouds
    # when SciPy is installed. This remains a mathematical smoke test only,
    # not a claim that CD/GD are scientifically enabled for current data.
    pecs = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, -1.0, 0.001],
            [2.0, 2.0, 1.0, 0.002],
        ],
        dtype=np.float32,
    )
    if PECS_IMPORT_ERROR is None:
        cd = chamfer_distance(pecs, pecs, normalize=True)
        gd = gaussian_distance(pecs, pecs, normalize=True)
        assert np.isclose(cd, 0.0)
        assert np.isclose(gd, 0.0)
        pecs_result = {
            "status": "verified",
            "identical_cloud_cd": float(cd),
            "identical_cloud_gd": float(gd),
        }
    else:
        pecs_result = {
            "status": "blocked_missing_dependency",
            "error": PECS_IMPORT_ERROR,
            "required_for_math_smoke": "scipy",
        }

    result = {
        "status": "metric_smoke_pass_with_optional_gate" if PECS_IMPORT_ERROR else "metric_smoke_pass",
        "formal_benchmark_run": False,
        "event_statistics": {
            "canonical_layout": "[t_s,x,y,p_01]",
            "spatial_grid": [2, 2],
            "temporal_bin_s": 0.001,
            "zero_error_self_comparison": errors,
        },
        "pecs_metric_math": {**pecs_result, "scientific_gate": "disabled_pending_matched_rig"},
        "no_composite_score": True,
    }
    out = REPO / "benchmark/results/raw/metric_smoke_test.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
