#!/usr/bin/env python3
"""Smoke-test the frozen equal-duration window extractor only."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmark.metrics.evaluation_windows import fixed_equal_duration_windows  # noqa: E402


def main() -> int:
    events = np.array(
        [[t, int(t * 100), int(t * 50), int(i % 2)] for i, t in enumerate(np.linspace(0, 1.7, 18))],
        dtype=np.float64,
    )
    windows = fixed_equal_duration_windows(
        events,
        sequence_duration_s=1.756258,
        window_duration_s=0.005,
        normalized_fractions=(0.25, 0.50, 0.75),
    )
    assert len(windows) == 3
    assert all(np.isclose(float(item["duration_s"]), 0.005) for item in windows)
    assert all(float(item["end_s"]) > float(item["start_s"]) for item in windows)
    assert all(float(item["end_s"]) <= 1.756258 + 1e-12 for item in windows)
    result = {
        "status": "pass",
        "formal_benchmark_run": False,
        "window_duration_s": 0.005,
        "fractions": [0.25, 0.50, 0.75],
        "window_bounds_s": [
            [float(item["start_s"]), float(item["end_s"])] for item in windows
        ],
        "event_counts": [int(len(item["events"])) for item in windows],
    }
    out = REPO / "benchmark/results/raw/window_smoke_test.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
