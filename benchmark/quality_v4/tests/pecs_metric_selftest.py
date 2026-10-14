#!/usr/bin/env python3
"""Small deterministic self-test for the existing PECS-style metric.

This file intentionally imports the legacy metric implementation without
changing it.  It tests only basic behavior needed before a future v4 protocol
can freeze a formal metric contract.  It does not load project data, align
streams, fit parameters, or run a benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genesis_event_plugin.evaluation.event_schema import canonicalize_events  # noqa: E402
from genesis_event_plugin.evaluation.pecs_metrics import (  # noqa: E402
    chamfer_distance,
    compute_both,
    gaussian_distance,
)


def _canonical_cloud() -> np.ndarray:
    # Canonical legacy-metric layout: [x,y,p,t_s].
    return np.asarray(
        [
            [2.0, 3.0, -1.0, 0.0000],
            [4.0, 3.0, 1.0, 0.0010],
            [4.0, 5.0, -1.0, 0.0020],
            [7.0, 8.0, 1.0, 0.0040],
        ],
        dtype=np.float64,
    )


def test_identity() -> dict[str, float]:
    cloud = _canonical_cloud()
    cd = chamfer_distance(cloud, cloud, normalize=True)
    gd = gaussian_distance(cloud, cloud, normalize=True)
    assert np.isclose(cd, 0.0, atol=1e-7), cd
    assert np.isclose(gd, 0.0, atol=1e-7), gd
    return {"cd": float(cd), "gd": float(gd)}


def test_empty_input_behavior() -> str:
    cloud = _canonical_cloud()
    empty = np.empty((0, 4), dtype=np.float64)
    for left, right in ((empty, cloud), (cloud, empty), (empty, empty)):
        try:
            compute_both(left, right)
        except ValueError:
            continue
        raise AssertionError("empty event input did not raise ValueError")
    return "ValueError for all empty/non-empty combinations"


def test_timestamp_unit_consistency() -> dict[str, float]:
    raw_us = np.asarray(
        [[0, 2, 3, 0], [1000, 4, 3, 1], [2000, 4, 5, 0], [4000, 7, 8, 1]],
        dtype=np.float64,
    )
    raw_s = raw_us.copy()
    raw_s[:, 0] *= 1e-6
    from_us = canonicalize_events(raw_us, layout="txyp", timestamp_unit="us", sort=True)
    from_s = canonicalize_events(raw_s, layout="txyp", timestamp_unit="s", sort=True)
    assert np.allclose(from_us, from_s), (from_us, from_s)
    cd_us = chamfer_distance(from_us, from_us, normalize=True)
    cd_s = chamfer_distance(from_s, from_s, normalize=True)
    assert np.isclose(cd_us, cd_s, atol=1e-7)
    return {"canonical_max_abs_diff": float(np.max(np.abs(from_us - from_s))), "cd": float(cd_us)}


def test_polarity_conversion_consistency() -> dict[str, float]:
    raw_01 = np.asarray(
        [[2, 3, 0, 0], [4, 3, 1, 1000], [4, 5, 0, 2000], [7, 8, 1, 4000]],
        dtype=np.float64,
    )
    raw_pm1 = raw_01.copy()
    raw_pm1[:, 2] = np.where(raw_pm1[:, 2] > 0, 1.0, -1.0)
    from_01 = canonicalize_events(raw_01, layout="xypt", timestamp_unit="us", sort=True)
    from_pm1 = canonicalize_events(raw_pm1, layout="xypt", timestamp_unit="us", sort=True)
    assert np.allclose(from_01, from_pm1), (from_01, from_pm1)
    return {"canonical_max_abs_diff": float(np.max(np.abs(from_01 - from_pm1)))}


def test_deterministic_result() -> dict[str, float]:
    left = np.vstack([_canonical_cloud(), _canonical_cloud() + np.asarray([1, 0, 0, 0])])
    right = _canonical_cloud()
    first = compute_both(left, right, normalize=True, subsample=3)
    second = compute_both(left, right, normalize=True, subsample=3)
    assert first == second, (first, second)
    return {"cd": float(first["cd"]), "gd": float(first["gd"]), "subsample": 3.0}


def run_selftests() -> dict[str, object]:
    results = {
        "identity": test_identity(),
        "empty_input_behavior": test_empty_input_behavior(),
        "timestamp_unit_consistency": test_timestamp_unit_consistency(),
        "polarity_conversion_consistency": test_polarity_conversion_consistency(),
        "deterministic_result": test_deterministic_result(),
    }
    return {
        "status": "PASS",
        "formal_benchmark_executed": False,
        "formal_metrics_executed": False,
        "tests": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_selftests(), indent=2, ensure_ascii=False))
