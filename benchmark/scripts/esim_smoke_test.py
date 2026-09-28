#!/usr/bin/env python3
"""Smoke-test the ESIM provenance and output adapter without fabricating ESIM."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parents[1]
sys.path.insert(0, str(REPO))

from benchmark.baselines.esim_wrapper import canonicalize_esim_events  # noqa: E402


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    esim = PROJECT / "code/rpg_esim"
    # This is an adapter-only synthetic contract check. It is intentionally not
    # labeled as an ESIM-generated stream.
    synthetic_official_contract = np.array(
        [[0.001, 10, 2, 1], [0.002, 11, 2, -1], [0.003, 12, 2, 1]], dtype=np.float64
    )
    converted = canonicalize_esim_events(synthetic_official_contract, timestamp_unit="s")
    assert converted.tolist() == [[1000, 10, 2, 1], [2000, 11, 2, 0], [3000, 12, 2, 1]]
    empty = canonicalize_esim_events(np.empty((0, 4)), timestamp_unit="s")
    assert empty.shape == (0, 4)
    assert np.all(np.diff(converted[:, 0]) >= 0)
    assert np.all((converted[:, 1] < 1280) & (converted[:, 2] < 720))
    assert set(np.unique(converted[:, 3])) == {0, 1}

    result = {
        "status": "supported_but_not_reproducibly_built",
        "formal_benchmark_run": False,
        "official_algorithm_execution": "not_run",
        "source": str(esim),
        "source_commit": git_head(esim),
        "build_record": str(REPO / "benchmark/baselines/ESIM_BUILD.md"),
        "adapter_contract_smoke": {
            "status": "pass",
            "input_contract": "official ESIM [t,x,y,p] with t in seconds and p in {-1,+1}",
            "output_contract": "uint64 [timestamp_us,x,y,polarity_01]",
            "empty_input": "pass",
            "monotonic_time": "pass",
            "coordinate_range": "pass",
            "polarity_conversion": "pass",
            "xy_order_preserved": "pass",
        },
        "reason_not_executable": "catkin_make is unavailable; see ESIM_BUILD.md",
    }
    out = REPO / "benchmark/results/raw/esim_smoke_test.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

