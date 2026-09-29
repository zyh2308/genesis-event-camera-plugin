#!/usr/bin/env python3
"""Audit baseline availability and validate existing artifacts.

This script deliberately does not launch Genesis, V2E, ROS/ESIM, or ICNS.  It
checks existing artifacts and dependency readiness only; therefore its output
cannot be mistaken for formal benchmark results.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parents[1]
sys.path.insert(0, str(REPO))

from benchmark.metrics.event_statistics import load_aer_events  # noqa: E402


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def core_diff_is_empty(repo: Path, frozen_commit: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "diff", "--exit-code", frozen_commit, "--", "genesis_event_plugin/"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0


def inspect_standard_h5(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    with h5py.File(path, "r") as handle:
        if "events" not in handle:
            record.update({"valid": False, "reason": "missing events dataset"})
            return record
        ds = handle["events"]
        record.update(
            {
                "valid": ds.ndim == 2 and ds.shape[1] == 4,
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
                "root_attrs": {str(k): str(v) for k, v in handle.attrs.items()},
                "dataset_attrs": {str(k): str(v) for k, v in ds.attrs.items()},
            }
        )
        if ds.shape[0]:
            first = np.asarray(ds[0], dtype=np.uint64).tolist()
            last = np.asarray(ds[-1], dtype=np.uint64).tolist()
            record["first_event"] = first
            record["last_event"] = last
            record["timestamps_monotonic_sampled"] = int(last[0]) >= int(first[0])
    return record


def inspect_metavision_h5(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    with h5py.File(path, "r") as handle:
        ds = handle.get("CD/events")
        if ds is None:
            record.update({"valid": False, "reason": "missing CD/events"})
            return record
        names = list(ds.dtype.names or ())
        record.update(
            {
                "valid": {"x", "y", "p", "t"}.issubset(names),
                "shape": list(ds.shape),
                "fields": names,
                "root_attrs": {str(k): str(v) for k, v in handle.attrs.items()},
            }
        )
    return record


def main() -> int:
    real_v2 = PROJECT / "validation/simulator-quality/real_v2/results/aer_events_real_v2.h5"
    v2e = PROJECT / "validation/simulator-quality/current_data_precheck/external_baselines/v2e_paper_defaults/aer_events.h5"
    v2e_meta = v2e.parent / "run_metadata.json"
    real_evt3 = PROJECT / "data/real/evk4_checkerboard_translation_20260914/derived/one_round_h5/segment_01_LR.hdf5"
    esim = PROJECT / "code/rpg_esim"
    iebcs = PROJECT / "code/IEBCS"  # intentionally absent in this checkout

    try:
        converted = load_aer_events(real_evt3, max_events=64)
        real_loader = {
            "status": "pass",
            "sample_shape": list(converted.shape),
            "canonical_layout": "[t_s,x,y,p_01]",
            "first_timestamp_s": float(converted[0, 0]) if len(converted) else None,
        }
    except Exception as exc:  # surface data/schema problems without hiding them
        real_loader = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}

    result = {
        "status": "smoke_only_not_formal",
        "formal_benchmark_run": False,
        "benchmark_head": git_head(REPO),
        "frozen_core_commit": "e79e656ac00f7847b98db9e12bf43b093f3afff1",
        "frozen_core_diff": "empty" if core_diff_is_empty(REPO, "e79e656ac00f7847b98db9e12bf43b093f3afff1") else "nonempty",
        "ours_v3_1": {
            "artifact": inspect_standard_h5(real_v2),
            "config_name": "real_v2",
            "parameter_tuning": "frozen project calibration; no smoke-test tuning",
        },
        "real_evt3_loader": {
            "artifact": inspect_metavision_h5(real_evt3),
            "conversion_smoke": real_loader,
        },
        "v2e": {
            "artifact": inspect_standard_h5(v2e),
            "run_metadata": json.loads(v2e_meta.read_text(encoding="utf-8")) if v2e_meta.exists() else None,
            "source_path": str(PROJECT / "code/v2e_source"),
            "source_version": "1.5.1 (setup.py; source snapshot has no git metadata)",
            "executable_smoke": False,
            "note": "existing official-core artifact checked; no new formal replay launched",
        },
        "esim": {
            "source_exists": esim.exists(),
            "source_commit": git_head(esim),
            "build_or_devel_exists": any((esim / name).exists() for name in ("build", "devel", "install")),
            "executable_smoke": False,
            "reason": "ROS/catkin build and executable are not available in this environment",
        },
        "icns_iebcs": {
            "local_checkout_exists": iebcs.exists(),
            "executable_smoke": False,
            "reason": "official source is not present locally; do not label a substitute as ICNS/IEBCS",
        },
        "pecs": {
            "executable_smoke": False,
            "reason": "reference-only for protocol/metrics; not required for Phase 1-4",
        },
    }
    # Keep this explicit assertion close to the output: a failed frozen-version
    # check is a stop condition, not a reason to silently benchmark another tree.
    if result["frozen_core_diff"] != "empty":
        result["status"] = "blocked_modified_frozen_core"
    out = REPO / "benchmark/results/raw/baseline_smoke_test.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "smoke_only_not_formal" else 2


if __name__ == "__main__":
    raise SystemExit(main())
