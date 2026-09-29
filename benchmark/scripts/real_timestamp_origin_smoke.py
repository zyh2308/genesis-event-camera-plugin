#!/usr/bin/env python3
"""Readiness smoke for complete real LR/RL streams and timestamp origin.

This is a loader/provenance check only.  It does not compute any formal metric.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from benchmark.io.real_stream import load_real_hdf5


def main() -> int:
    protocol = yaml.safe_load((ROOT / "benchmark/protocol_v3.yaml").read_text())
    output = {"formal_metrics_executed": False, "streams": {}}
    try:
        for direction in ("LR", "RL"):
            spec = protocol["real_sequences"][direction]
            sequence = load_real_hdf5(
                spec["file"], direction=direction,
                sequence_start_us=int(spec["declared_sequence_start_us"])
                if "declared_sequence_start_us" in spec else int(protocol["time_origin"]["real_sequence_origin"]["declared_start_us"]),
                duration_s_declared=float(spec["duration_s"]),
            )
            events = sequence.events_txyp_us
            output["streams"][direction] = {
                "status": "PASS",
                "path": str(sequence.path),
                "source": "vendor_hdf5_ecf",
                "ecf_plugin_path": sequence.ecf_plugin_path,
                "complete_event_count": int(len(events)),
                "sequence_start_us": sequence.sequence_start_us,
                "first_t_rel_us": int(events[0, 0]),
                "last_t_rel_us": int(events[-1, 0]),
                "event_span_s": sequence.duration_s_event_span,
                "declared_duration_s": sequence.duration_s_declared,
                "duration_error_s": sequence.duration_s_event_span - sequence.duration_s_declared,
                "first_event_was_not_used_as_origin": True,
            }
    except Exception as exc:
        output["status"] = "STOP"
        output["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 2
    output["status"] = "PASS"
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
