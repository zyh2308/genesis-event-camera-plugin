#!/usr/bin/env python3
"""Validate the protocol-approved V2E IIR cadence without a formal replay.

This is deliberately a numerical-contract smoke test.  It reads the official
V2E source's ``check_lowpass`` guard and reproduces only its documented
``eps = dt/tau`` check.  It does not generate event metrics or select a
cadence based on benchmark output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

import yaml


REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "benchmark/protocol_v2.yaml"
V2E_UTILS = Path("/home/科研/Eventbased_WAM/code/v2e_source/v2ecore/v2e_utils.py")


def main() -> int:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    spec = protocol["methods"]["v2e"]["formal_input_cadence"]
    source = V2E_UTILS.read_text(encoding="utf-8")
    maxeps_match = re.search(r"def check_lowpass[\s\S]*?maxeps\s*=\s*([0-9.]+)", source)
    if maxeps_match is None:
        raise SystemExit("STOP: official V2E check_lowpass maxeps could not be located")

    maxeps = float(maxeps_match.group(1))
    cutoff_hz = float(spec["cutoff_hz"])
    cadence_hz = float(spec["effective_cadence_hz"])
    dt = 1.0 / cadence_hz
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    eps = dt / tau
    previous_eps = (1.0 / 100.0) / tau
    assert math.isclose(maxeps, 0.3)
    assert eps < maxeps
    assert previous_eps > maxeps
    assert math.isclose(float(spec["timestamp_resolution_s"]), dt)

    result = {
        "status": "pass",
        "formal_benchmark_run": False,
        "official_source": str(V2E_UTILS),
        "official_guard": "check_lowpass",
        "maxeps": maxeps,
        "cutoff_hz": cutoff_hz,
        "effective_cadence_hz": cadence_hz,
        "timestamp_resolution_s": dt,
        "eps_at_protocol_cadence": eps,
        "eps_at_previous_100Hz": previous_eps,
        "previous_100Hz_warning_expected": previous_eps > maxeps,
        "protocol_cadence_warning_expected": eps > maxeps,
        "official_executable_replay": "not_run_in_phase_4_6",
    }
    out = REPO / "benchmark/results/raw/v2e_cadence_smoke.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
