#!/usr/bin/env python3
"""Replay cached source frames to inspect Ours event timing within each keyframe interval.

This is an offline event-generator replay: no Genesis rendering or physics is
advanced. It records the adaptive sampler cadence and event-time fractions,
and verifies that added diagnostics leave the original event arrays unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.diagnose_r1_native_hdr_replay import (  # noqa: E402
    _sha256,
    _write_hash_manifest,
    generate_ours,
    verify_source_hashes,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True,
                        help="completed native_hdr_cutoff059 replay with per-episode source caches")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", default="004,005")
    args = parser.parse_args()
    experiment_root = args.experiment_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output must be a new experiment path: {output}")
    episode_ids = [int(value) for value in args.episodes.split(",") if value]
    if not episode_ids:
        raise ValueError("at least one episode id is required")
    parent_summary = json.loads((experiment_root / "native_hdr_replay_summary.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    episode_results: dict[str, Any] = {}

    for episode_id in episode_ids:
        episode_name = f"episode_{episode_id:03d}"
        episode_root = experiment_root / episode_name
        cache_path = episode_root / "same_state_source_cache.h5"
        expected_native_path = episode_root / "ours_native_hdr" / "raw_events.npz"
        expected_legacy_path = episode_root / "ours_legacy_display_gray" / "raw_events.npz"
        verified = verify_source_hashes(
            experiment_root,
            [cache_path, expected_native_path, expected_legacy_path],
        )
        episode_output = output / episode_name
        episode_output.mkdir()
        expected = {
            "ours_native_hdr": np.asarray(np.load(expected_native_path)["events"], dtype=np.float64),
            "ours_legacy_display_gray": np.asarray(np.load(expected_legacy_path)["events"], dtype=np.float64),
        }
        routes = {}
        for name, response_mode, display_gray in (
            ("ours_native_hdr", "physical_log", False),
            ("ours_legacy_display_gray", "video_linlog", True),
        ):
            events, info = generate_ours(
                cache_path, episode_output / name, response_mode, display_gray=display_gray,
            )
            exact = events.shape == expected[name].shape and np.array_equal(events, expected[name])
            routes[name] = {
                "event_count": int(len(events)),
                "matches_original_raw_event_array_exactly": bool(exact),
                "original_raw_event_sha256": _sha256(
                    expected_native_path if name == "ours_native_hdr" else expected_legacy_path
                ),
                "replay_raw_event_sha256": _sha256(episode_output / name / "raw_events.npz"),
                "generation": info,
            }
        if not all(route["matches_original_raw_event_array_exactly"] for route in routes.values()):
            raise RuntimeError(f"cached event replay was not bitwise stable for {episode_name}")
        episode_results[episode_name] = {
            "input_cache_sha256": verified[str(cache_path.relative_to(experiment_root))],
            "verified_parent_raw_event_hashes": {
                name: verified[str(path.relative_to(experiment_root))]
                for name, path in (("ours_native_hdr", expected_native_path),
                                   ("ours_legacy_display_gray", expected_legacy_path))
            },
            "routes": routes,
        }

    result = {
        "status": "PASS",
        "purpose": "causal diagnosis of event density around source-frame boundaries; no model/control evaluation",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "parent_run": str(experiment_root),
        "parent_code_commit": parent_summary["code_commit"],
        "parent_config_sha256": parent_summary["config_sha256"],
        "event_replay_wall_s": time.perf_counter() - started,
        "physics_or_camera_rerendered": False,
        "frozen_core_modified": False,
        "episodes": episode_results,
    }
    (output / "event_timing_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    _write_hash_manifest(output)
    print(json.dumps({
        "output": str(output),
        "status": result["status"],
        "wall_s": result["event_replay_wall_s"],
        "bitwise_stable": all(
            route["matches_original_raw_event_array_exactly"]
            for episode in episode_results.values() for route in episode["routes"].values()
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
