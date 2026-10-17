#!/usr/bin/env python3
"""Regenerate paired training events and fit Ours/V2E from cached rollouts.

This path performs no Genesis rollout. It reuses immutable RGB/HDR/depth/seg,
pose, timestamp, and trace caches, applies one explicit shared pixel-response
overlay, and makes two identically configured development fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.robot_closed_loop.diagnostics.run_paired_generator_smoke import (  # noqa: E402
    _load_v2e_events,
    _make_ours_events,
    _sha256,
    _write_hash_manifest,
    _write_source_episode,
)


DEFAULT_EXPERIMENT = ROOT / "benchmark/robot_closed_loop/configs/r1_shared_ideal_response_v1.json"


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="existing trajectory_cache directory containing episode_NNN caches")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--v2e-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/r1_shared_ideal_response_v1_v2e.yaml")
    parser.add_argument("--scene-config", type=Path,
                        default=ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json")
    parser.add_argument("--v2e-source", type=Path, required=True)
    parser.add_argument("--slomo-model", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable,
                        help="robot/plugin and estimator-training interpreter")
    parser.add_argument("--v2e-python", default=None,
                        help="official V2E interpreter; defaults to --python")
    parser.add_argument("--episode-ids", default="0,1")
    parser.add_argument("--train-episode-ids", default=None)
    parser.add_argument("--validation-episode-ids", default=None)
    args = parser.parse_args()

    experiment_path = args.experiment_config.resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    import yaml
    v2e_config_path = args.v2e_config.resolve()
    v2e_parameters = yaml.safe_load(v2e_config_path.read_text(encoding="utf-8"))["parameters"]
    response = experiment["pixel_response"]
    parameter_map = {
        "pos_thres": "pos_thres", "neg_thres": "neg_thres", "sigma_thres": "sigma_thres",
        "cutoff_hz": "cutoff_hz", "leak_rate_hz": "leak_rate_hz",
        "leak_jitter_fraction": "leak_jitter_fraction",
        "noise_rate_cov_decades": "noise_rate_cov_decades",
        "shot_noise_rate_hz": "shot_noise_rate_hz",
        "refractory_period_s": "refractory_period_s", "seed": "seed",
    }
    mismatched = {
        v2e_key: {"ours": response[ours_key], "v2e": v2e_parameters.get(v2e_key)}
        for ours_key, v2e_key in parameter_map.items()
        if float(response[ours_key]) != float(v2e_parameters.get(v2e_key, float("nan")))
    }
    if mismatched:
        raise ValueError(f"shared-response configuration mismatch: {mismatched}")
    training = experiment["evaluation"]
    default_split = training["episode_split"]
    episode_ids = [int(value) for value in args.episode_ids.split(",") if value.strip()]
    train_ids = [int(value) for value in (args.train_episode_ids or ",".join(map(str, default_split["train_episode_ids"]))).split(",") if value.strip()]
    validation_ids = [int(value) for value in (args.validation_episode_ids or ",".join(map(str, default_split["validation_episode_ids"]))).split(",") if value.strip()]
    if not episode_ids or set(train_ids) & set(validation_ids):
        raise ValueError("episode list must be nonempty and train/validation IDs disjoint")
    if set(episode_ids) != set(train_ids) | set(validation_ids):
        raise ValueError("every selected episode must appear in exactly one declared split")

    cache_root = args.cache_root.resolve()
    out = args.output.resolve()
    _clean_dir(out)
    replay_root = out / "generator_replays"
    _clean_dir(replay_root)
    ours_root, v2e_root = out / "ours_training", out / "v2e_training"
    _clean_dir(ours_root)
    _clean_dir(v2e_root)
    split_contract = {
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
        "split_unit": "episode",
        "training_seed": int(training["training_seed"]),
        "architecture": training["network"],
        "fixed_budget": {
            "max_epochs": int(training["max_epochs"]),
            "patience": int(training["patience"]),
            "batch_size": int(training["batch_size"]),
            "threads": int(training["threads"]),
        },
        "purpose": "R1 minimum-cache common-response development; not held-out formal test",
    }
    split_hash = hashlib.sha256(json.dumps(split_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for source_root in (ours_root, v2e_root):
        (source_root / "episode_split_contract.json").write_text(json.dumps(split_contract, indent=2), encoding="utf-8")

    runner_python = Path(args.python).resolve()
    v2e_python = Path(args.v2e_python or args.python).resolve()
    v2e_source = args.v2e_source.resolve()
    slomo_model = args.slomo_model.resolve()
    v2e_config = v2e_config_path
    generation_started = time.perf_counter()
    episode_records: List[Dict[str, Any]] = []
    for episode_id in episode_ids:
        cache_dir = cache_root / f"episode_{episode_id:03d}"
        cache_path = cache_dir / "trajectory_cache.h5"
        frames_dir = cache_dir / "frames"
        result_path = cache_dir / "episode_result.json"
        trace_path = cache_dir / "oracle_trace.jsonl"
        for path in (cache_path, result_path, trace_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if not frames_dir.is_dir() or not list(frames_dir.glob("*.png")):
            raise FileNotFoundError(f"cached official V2E PNG frames missing: {frames_dir}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        trace = _jsonl(trace_path)
        replay_dir = replay_root / f"episode_{episode_id:03d}"
        _clean_dir(replay_dir)
        ours_out = replay_dir / "ours"
        v2e_out = replay_dir / "v2e"
        if ours_out.exists() or v2e_out.exists():
            raise FileExistsError(f"refusing to overwrite prior per-cache replay: {ours_out} or {v2e_out}")

        ours_events = _make_ours_events(cache_path, ours_out, frames_dir=frames_dir, response_config=experiment)
        command = [
            str(runner_python), str(ROOT / "benchmark/scripts/v2e_official_pipeline_smoke.py"),
            "--frames", str(frames_dir), "--output-dir", str(v2e_out),
            "--source", str(v2e_source), "--python", str(v2e_python),
            "--slomo-model", str(slomo_model), "--config", str(v2e_config),
        ]
        process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, check=False)
        (replay_dir / "v2e_driver.log").write_text(process.stdout, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"official V2E failed for episode {episode_id}: {process.stdout[-2500:]}")
        v2e_events = _load_v2e_events(v2e_out / "events.h5")
        _write_source_episode(ours_root, episode_id, result, trace, ours_events)
        _write_source_episode(v2e_root, episode_id, result, trace, v2e_events)

        frame_manifest = cache_dir / "frame_manifest.json"
        if not frame_manifest.is_file():
            raise FileNotFoundError(frame_manifest)
        pair = {
            "episode_id": episode_id,
            "shared_cache_sha256": _sha256(cache_path),
            "frame_manifest_sha256": _sha256(frame_manifest),
            "episode_result_sha256": _sha256(result_path),
            "trace_sha256": _sha256(trace_path),
            "replay_artifact_root": str(replay_dir),
            "split_contract_sha256": split_hash,
            "split": "train" if episode_id in train_ids else "validation",
            "ours_event_count": int(len(ours_events)),
            "v2e_event_count": int(len(v2e_events)),
            "same_cached_png_endpoint_manifest": True,
            "same_GT_labels": True,
        }
        for root in (ours_root, v2e_root):
            episode_dir = root / "episodes" / f"episode_{episode_id:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            (episode_dir / "pair_provenance.json").write_text(json.dumps(pair, indent=2), encoding="utf-8")
        episode_records.append(pair)

    generation_wall_s = time.perf_counter() - generation_started
    ours_files = _write_hash_manifest(ours_root)
    v2e_files = _write_hash_manifest(v2e_root)

    fit_started = time.perf_counter()
    fit_records = {}
    for source_name, source_root in (("ours", ours_root), ("v2e", v2e_root)):
        fit_out = out / f"{source_name}_fit"
        command = [
            str(runner_python), "-m", "benchmark.robot_closed_loop.diagnostics.train_event_state_development",
            "--config", str(args.scene_config.resolve()), "--source-results", str(source_root),
            "--out", str(fit_out), "--train-episodes", ",".join(map(str, train_ids)),
            "--validation-episodes", ",".join(map(str, validation_ids)),
            "--epochs", str(int(training["max_epochs"])), "--patience", str(int(training["patience"])),
            "--batch-size", str(int(training["batch_size"])), "--seed", str(int(training["training_seed"])),
            "--threads", str(int(training["threads"])), "--architecture", str(training["network"]),
            "--source-label", f"R1 shared-response {source_name} training events",
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": str(training["threads"]),
            "MKL_NUM_THREADS": str(training["threads"]),
            "OPENBLAS_NUM_THREADS": str(training["threads"]),
            "PYTHONPATH": str(ROOT),
        }
        process = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (out / f"{source_name}_fit.log").write_text(process.stdout, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"{source_name} event-state fit failed: {process.stdout[-2500:]}")
        summary_path = fit_out / "development_summary.json"
        checkpoint_path = fit_out / "spatial_event_state.pt"
        fit_records[source_name] = {
            "training_summary_sha256": _sha256(summary_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "source_manifest_sha256": _sha256(source_root / "SHA256SUMS"),
            "validation_metrics": json.loads(summary_path.read_text(encoding="utf-8"))["validation_metrics"],
        }
    fit_wall_s = time.perf_counter() - fit_started

    result = {
        "status": "PASS",
        "experiment_id": experiment["experiment_id"],
        "purpose": "paired common-response training and minimum-cache R1 endpoint-to-endpoint development validation; no ranking or formal test claim",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scene_config_sha256": _sha256(args.scene_config.resolve()),
        "experiment_config_sha256": _sha256(experiment_path),
        "v2e_overlay_sha256": _sha256(v2e_config),
        "official_v2e_default_config_sha256": _sha256(ROOT / "benchmark/configs/v2e.yaml"),
        "v2e_source": str(v2e_source),
        "v2e_entry_sha256": _sha256(v2e_source / "v2e.py"),
        "slomo_checkpoint_sha256": _sha256(slomo_model),
        "episode_split": {**split_contract, "sha256": split_hash},
        "episodes": episode_records,
        "paired_training_event_roots": {
            "ours": {"path": str(ours_root), "manifest_sha256": _sha256(ours_root / "SHA256SUMS"), "files": ours_files},
            "v2e": {"path": str(v2e_root), "manifest_sha256": _sha256(v2e_root / "SHA256SUMS"), "files": v2e_files},
        },
        "fits": fit_records,
        "fairness_boundary": {
            "same_cache_timestamps_labels_split_and_training_seed": True,
            "same_pixel_endpoint": "same archived PNG byte content read with OpenCV grayscale; endpoint manifest hash is shared",
            "shared_pixel_response_parameters": experiment["pixel_response"],
            "shared_transfer": "V2E-compatible lin_log threshold 20 DN",
            "temporal_reconstruction": experiment["temporal_paths"],
            "frozen_core_modified": False,
            "official_v2e_default_yaml_modified": False,
            "formal_test_set_reserved": "all nine existing Direct-HDR evaluation episodes are prior development material",
        },
        "cost_s": {
            "shared_rollout_capture": 0.0,
            "paired_event_generation_from_existing_cache": generation_wall_s,
            "two_equal_budget_model_fits": fit_wall_s,
            "wall_scope": "end-to-end replay and training process wall time; excludes original cached rollouts and common-reference evaluation",
        },
    }
    result_path = out / "r1_cached_rollout_development.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    files = _write_hash_manifest(out)
    (out / "cache_input_hashes.json").write_text(json.dumps({
        str(cache_root / f"episode_{episode_id:03d}" / "trajectory_cache.h5"):
        _sha256(cache_root / f"episode_{episode_id:03d}" / "trajectory_cache.h5")
        for episode_id in episode_ids
    }, indent=2), encoding="utf-8")
    _write_hash_manifest(out)
    print(json.dumps({
        "status": "PASS", "output": str(out), "episodes": len(episode_records),
        "checkpoint_sha256": {name: record["checkpoint_sha256"] for name, record in fit_records.items()},
        "cost_s": result["cost_s"], "output_file_count": len(files),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
