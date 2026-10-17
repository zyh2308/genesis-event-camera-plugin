#!/usr/bin/env python3
"""Capture supplementary shared robot/physics rollouts for R1 training.

The script uses the existing R0.8 scene, oracle runner and trajectory archive
contract. It captures only the seven missing speed/lateral combinations; Ours
and V2E generation/training are a later cache-only step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (  # noqa: E402
    RolloutArchive,
    _resolve_motion_bodies,
    _sha256,
)
from benchmark.robot_closed_loop.robot_env import build_robot_scene  # noqa: E402
from benchmark.robot_closed_loop.run import jsonable, run_oracle_trial, save_gif, write_trace  # noqa: E402


DEFAULT_PLAN = ROOT / "benchmark/robot_closed_loop/configs/r1_shared_ideal_small9_v1.json"


def _plan_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [row for row in config["rollout_plan"] if row["cache_source"] == "new shared rollout"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-config", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--scene-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json")
    parser.add_argument("--control-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/shared_control_r0_8_candidate.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.plan_config.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scene_path = args.scene_config.resolve()
    control_path = args.control_config.resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    scene["shared_control"] = control
    rows = _plan_rows(plan)
    if len(rows) != 7 or len({int(row["episode_id"]) for row in rows}) != 7:
        raise ValueError("small9 rollout plan must declare seven unique missing episodes")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    import genesis as gs
    gs.init(backend=gs.cpu, logging_level="warning")
    env = None
    started = time.perf_counter()
    records = []
    try:
        env = build_robot_scene(scene, segmentation_level="link")
        env.camera.render(rgb=True, depth=True, segmentation=True, radiance=True)
        pose_bodies = _resolve_motion_bodies(env)
        deceleration = float(scene["controller"]["initial_deceleration_assumption_m_s2"])
        for row in rows:
            episode_id = int(row["episode_id"])
            episode_seed = int(row["seed"])
            episode_dir = out / f"episode_{episode_id:03d}"
            episode_dir.mkdir(parents=True, exist_ok=False)
            archive = RolloutArchive(env, episode_dir, pose_bodies)
            episode_started = time.perf_counter()
            result, trace, frames = run_oracle_trial(
                env, episode_id, episode_seed, float(row["speed_m_s"]), deceleration,
                initial_y_m=float(row["launch_y_m"]), trajectory_cache=archive,
            )
            cache_path = archive.write(scene)
            if frames:
                save_gif(episode_dir / "oracle_rollout.gif", frames)
            write_trace(episode_dir / "oracle_trace.jsonl", trace)
            (episode_dir / "episode_result.json").write_text(
                json.dumps(jsonable(result), indent=2), encoding="utf-8"
            )
            record = {
                "episode_id": episode_id, "seed": episode_seed,
                "speed_m_s": float(row["speed_m_s"]), "launch_y_m": float(row["launch_y_m"]),
                "cache_sha256": _sha256(cache_path),
                "frame_manifest_sha256": _sha256(episode_dir / "frame_manifest.json"),
                "trace_sha256": _sha256(episode_dir / "oracle_trace.jsonl"),
                "episode_result_sha256": _sha256(episode_dir / "episode_result.json"),
                "strict_success": bool(result.get("success", False)),
                "episode_wall_s": time.perf_counter() - episode_started,
            }
            records.append(record)
            (episode_dir / "rollout_provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    finally:
        if env is not None:
            gs.destroy()
        else:
            gs.destroy()

    summary = {
        "status": "PASS",
        "purpose": "seven additional shared physical/robot caches for development-only R1; no strict success claim",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "plan_config_sha256": _sha256(plan_path),
        "scene_config_sha256": _sha256(scene_path),
        "control_config_sha256": _sha256(control_path),
        "physics_dt_s": float(scene["physics"]["dt_s"]),
        "source_cadence_hz": 100,
        "rollouts": records,
        "wall_s": time.perf_counter() - started,
        "source_scope": "new rollout caches are shared by both generators; no generator controls physics; labels are stored only for supervised development training/evaluator scoring",
    }
    (out / "supplementary_rollouts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    files = {
        str(path.relative_to(out)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(out.rglob("*")) if path.is_file()
    }
    (out / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {relative}\n" for relative, item in files.items()), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(out), "episode_count": len(records),
                      "wall_s": summary["wall_s"], "manifest_sha256": _sha256(out / "SHA256SUMS")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
