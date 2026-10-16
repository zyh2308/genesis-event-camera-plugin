#!/usr/bin/env python3
"""Train a compact spatial event-state model with episode-level development split.

GT is used only to build training labels and score predictions after inference.
Inputs are causal Direct-HDR event tensors. No RGB, segmentation, depth, or
future state is passed to the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark.robot_closed_loop.diagnostics.replay_event_state import (
    _phase,
    _read_trace,
    _sha256,
    _truth_at,
)
from benchmark.robot_closed_loop.event_state_model import (
    INPUT_CONTRACT,
    SpatialEventStateNet,
    five_bin_tensor,
)
from benchmark.robot_closed_loop.predictor import FixedDecelerationPredictor, MotionState


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"
DEFAULT_TRAIN_EPISODES = "0,2,3,4,7,8"
DEFAULT_VALIDATION_EPISODES = "1,5,6"


def _build_dataset(source_root: Path, episode_ids: Sequence[int], config: Dict[str, Any]):
    tensors: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    input_hashes: Dict[str, str] = {}
    width, height = (int(v) for v in config["camera"]["resolution_wh"])
    cadence_s = 1.0 / float(config["controller"]["high_level_hz"])
    latency_s = float(config["controller"]["observation_latency_s"])
    burn_in_s = float(config["controller"]["event_burn_in_s"])

    for episode_id in episode_ids:
        episode_dir = source_root / "episodes" / f"episode_{episode_id:03d}"
        episode_csv = episode_dir / "episode.csv"
        trace_path = episode_dir / "state_control_trace.jsonl"
        event_path = episode_dir / "direct_hdr_events_1khz_2khz.npz"
        for path in (episode_csv, trace_path, event_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            input_hashes[str(path)] = _sha256(path)
        with episode_csv.open(newline="") as stream:
            episode = next(csv.DictReader(stream))
        trace = _read_trace(trace_path)
        with np.load(event_path) as archive:
            events = np.asarray(archive["events_2000hz"], dtype=np.float64)
        events = events[np.argsort(events[:, 0], kind="stable")]

        contact_s = float(episode["first_contact_s"])
        lift_s = None if not episode["lift_start_s"] else float(episode["lift_start_s"])
        duration_s = float(episode["sim_duration_s"])
        first_poll = np.ceil(burn_in_s / cadence_s - 1e-10) * cadence_s
        poll_times = np.arange(first_poll, duration_s + 1e-9, cadence_s)
        for now_s in poll_times:
            cutoff_s = float(now_s - latency_s)
            if cutoff_s <= 0.025:
                continue
            voxel = five_bin_tensor(events, cutoff_s, width, height)
            position, velocity = _truth_at(trace, cutoff_s)
            tensors.append(np.log1p(voxel).reshape(10, 64, 64))
            labels.append(np.concatenate((position, velocity)).astype(np.float32))
            bin_counts = voxel.sum(axis=(1, 2, 3))
            metadata.append({
                "episode_id": int(episode_id),
                "speed_m_s": float(episode["speed_m_s"]),
                "launch_y_m": float(episode["launch_y_m"]),
                "evaluation_time_s": float(now_s),
                "observation_cutoff_s": cutoff_s,
                "phase": _phase(cutoff_s, contact_s, lift_s),
                "input_event_count": int(voxel.sum()),
                "input_time_bin_count": int(np.count_nonzero(bin_counts)),
                "input_state_age_ms": latency_s * 1000.0,
            })

    if not tensors:
        raise RuntimeError("no causal event windows were constructed")
    return np.asarray(tensors, dtype=np.float32), np.asarray(labels, dtype=np.float32), metadata, input_hashes


def _phase_sample_weights(metadata: Sequence[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    groups = [(int(row["episode_id"]), str(row["phase"])) for row in metadata]
    counts = {group: groups.count(group) for group in set(groups)}
    weights = np.asarray([1.0 / (len(counts) * counts[group]) for group in groups], dtype=np.float32)
    return torch.as_tensor(weights, device=device)


def _weighted_loss(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_sample = torch.mean((prediction - target) ** 2, dim=1)
    return torch.sum(per_sample * weights) / torch.clamp(weights.sum(), min=1e-9)


def _predict(model, x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (model(x) * scale + mean).cpu().numpy()


def _percentiles(values: Iterable[float]) -> Dict[str, Any]:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": int(len(values)), "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
    }


def _score_rows(
    predictions: np.ndarray,
    labels: np.ndarray,
    metadata: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    error = predictions.astype(np.float64) - labels.astype(np.float64)
    pos_xy = np.linalg.norm(error[:, :2], axis=1)
    vel_xy = np.linalg.norm(error[:, 3:5], axis=1)
    counts = np.asarray([row["input_event_count"] for row in metadata])
    bins = np.asarray([row["input_time_bin_count"] for row in metadata])
    valid = (counts >= 12) & (bins >= 3)
    by_phase: Dict[str, Any] = {}
    for phase in ("pre_contact", "contact_to_lift", "post_lift"):
        indices = np.asarray([row["phase"] == phase for row in metadata], dtype=bool)
        if not np.any(indices):
            by_phase[phase] = {"n": 0}
            continue
        by_phase[phase] = {
            "n": int(indices.sum()),
            "position_xy_error_m": _percentiles(pos_xy[indices]),
            "velocity_xy_error_m_s": _percentiles(vel_xy[indices]),
            "event_state_valid_fraction": float(np.mean(valid[indices])),
            "event_state_age_ms_p95": float(np.percentile(
                [metadata[i]["input_state_age_ms"] for i in np.where(indices & valid)[0]], 95,
            )) if np.any(indices & valid) else None,
        }

    controller_cfg = config["controller"]
    predictor = FixedDecelerationPredictor(
        float(controller_cfg["intercept_x_m"]),
        float(controller_cfg["initial_deceleration_assumption_m_s2"]),
    )
    arrival_errors, intercept_y_errors = [], []
    reachable_pairs = 0
    for index, row in enumerate(metadata):
        if row["phase"] != "pre_contact" or not valid[index]:
            continue
        age_s = float(controller_cfg["observation_latency_s"])
        cutoff_s = float(row["observation_cutoff_s"])
        pred_state = MotionState(cutoff_s, predictions[index, :3], predictions[index, 3:])
        true_state = MotionState(cutoff_s, labels[index, :3], labels[index, 3:])
        predicted_arrival = predictor.predict(pred_state, age_s)
        reference_arrival = predictor.predict(true_state, age_s)
        if predicted_arrival.reachable and reference_arrival.reachable:
            reachable_pairs += 1
            arrival_errors.append(abs(predicted_arrival.ttc_s - reference_arrival.ttc_s))
            intercept_y_errors.append(abs(float(predicted_arrival.position_m[1] - reference_arrival.position_m[1])))
    return {
        "n_windows": int(len(metadata)),
        "position_xy_error_m": _percentiles(pos_xy),
        "velocity_xy_error_m_s": _percentiles(vel_xy),
        "event_state_valid_fraction": float(np.mean(valid)),
        "event_state_valid_windows": int(valid.sum()),
        "event_state_age_ms_when_valid": _percentiles(
            [float(row["input_state_age_ms"]) for row, ok in zip(metadata, valid) if ok],
        ),
        "phase": by_phase,
        "pre_contact_same_fixed_predictor_reference": {
            "reachable_prediction_pairs": reachable_pairs,
            "absolute_intercept_arrival_error_s": _percentiles(arrival_errors),
            "absolute_intercept_y_error_m": _percentiles(intercept_y_errors),
            "reference_definition": "both estimated and GT state at the causal cutoff are passed through the same frozen FixedDecelerationPredictor with the configured observation latency; future GT arrival is not an input",
        },
    }


def _write_predictions(path: Path, metadata, labels, predictions) -> None:
    meta_fields = list(metadata[0])
    label_fields = [
        "gt_x_m", "gt_y_m", "gt_z_m", "gt_vx_m_s", "gt_vy_m_s", "gt_vz_m_s",
        "pred_x_m", "pred_y_m", "pred_z_m", "pred_vx_m_s", "pred_vy_m_s", "pred_vz_m_s",
        "position_xy_error_m", "velocity_xy_error_m_s", "event_state_status",
    ]
    fields = meta_fields + label_fields
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for meta, truth, pred in zip(metadata, labels, predictions):
            row = dict(meta)
            row.update(dict(zip(label_fields[:6], truth.tolist())))
            row.update(dict(zip(label_fields[6:12], pred.tolist())))
            row["position_xy_error_m"] = float(np.linalg.norm(pred[:2] - truth[:2]))
            row["velocity_xy_error_m_s"] = float(np.linalg.norm(pred[3:5] - truth[3:5]))
            row["event_state_status"] = "fresh" if row["input_event_count"] >= 12 and row["input_time_bin_count"] >= 3 else (
                "no_event" if row["input_event_count"] == 0 else "tracking_lost"
            )
            writer.writerow(row)


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-episodes", default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--validation-episodes", default=DEFAULT_VALIDATION_EPISODES)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    train_ids = [int(value) for value in args.train_episodes.split(",")]
    validation_ids = [int(value) for value in args.validation_episodes.split(",")]
    if set(train_ids) & set(validation_ids):
        raise ValueError("episode-level train and validation sets must be disjoint")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text())
    source_root = args.source_results.resolve()
    x_train_np, y_train_np, train_meta, train_hashes = _build_dataset(source_root, train_ids, config)
    x_val_np, y_val_np, val_meta, val_hashes = _build_dataset(source_root, validation_ids, config)
    device = torch.device("cpu")
    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    x_val = torch.from_numpy(x_val_np)
    y_val = torch.from_numpy(y_val_np)
    label_mean = y_train.mean(dim=0)
    label_scale = y_train.std(dim=0, unbiased=False).clamp_min(0.01)
    y_train_n = (y_train - label_mean) / label_scale
    y_val_n = (y_val - label_mean) / label_scale
    train_weights = _phase_sample_weights(train_meta, device)
    val_weights = _phase_sample_weights(val_meta, device)
    # Keep phase weights aligned with samples while DataLoader shuffles.
    dataset = TensorDataset(x_train, y_train_n, train_weights)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = SpatialEventStateNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)

    started = time.perf_counter()
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y, batch_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _weighted_loss(model(batch_x), batch_y, batch_weights)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(_weighted_loss(model(x_train), y_train_n, train_weights).item())
            validation_loss = float(_weighted_loss(model(x_val), y_val_n, val_weights).item())
        history.append({"epoch": epoch + 1, "train_phase_balanced_normalized_mse": train_loss,
                        "validation_phase_balanced_normalized_mse": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    training_wall_s = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    train_predictions = _predict(model, x_train, label_mean, label_scale)
    val_predictions = _predict(model, x_val, label_mean, label_scale)
    train_metrics = _score_rows(train_predictions, y_train_np, train_meta, config)
    val_metrics = _score_rows(val_predictions, y_val_np, val_meta, config)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_path = out / "spatial_event_state.pt"
    torch.save({
        "state_dict": best_state,
        "label_mean": label_mean.cpu(), "label_scale": label_scale.cpu(),
        "architecture": SpatialEventStateNet.architecture,
        "input_contract": INPUT_CONTRACT,
        "training_episodes": train_ids,
        "validation_episodes": validation_ids,
        "inference_gt_inputs": False,
        "causal_timestamp": "observation cutoff, exclusive upper bound for event window",
        "validity_policy": {"minimum_events": 12, "minimum_nonempty_5ms_bins": 3,
                            "maximum_state_age_s": float(config["controller"].get("maximum_event_state_age_s", 0.03))},
        "best_epoch": best_epoch,
        "config_sha256": _sha256(args.config),
        "training_code_commit": _git_value("rev-parse", "HEAD"),
    }, checkpoint_path)
    _write_predictions(out / "train_predictions.csv", train_meta, y_train_np, train_predictions)
    _write_predictions(out / "validation_predictions.csv", val_meta, y_val_np, val_predictions)
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    input_hashes = {**train_hashes, **val_hashes}
    summary = {
        "phase": "small Direct-HDR event-state development fit; not a generator comparison or model ranking",
        "code_commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "config_sha256": _sha256(args.config),
        "source_results_root": str(source_root),
        "source_manifest_sha256": _sha256(source_root / "SHA256SUMS"),
        "source_input_sha256": input_hashes,
        "train_episodes": train_ids,
        "validation_episodes": validation_ids,
        "episode_level_split_only": True,
        "split_rationale": "both splits contain every launch speed and every lateral position; distinct speed-position combinations are held out",
        "all_nine_episodes_are_development_material": True,
        "model_architecture": SpatialEventStateNet.architecture,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_contract": INPUT_CONTRACT,
        "training_labels": "simulator object position and velocity at each causal observation cutoff",
        "model_inputs": "only Direct-HDR raw events binned into causal five-bin tensors; no RGB, segmentation, depth, or GT at inference",
        "sampling": {"bin_ms": 5, "bins": 5, "window_ms": 25, "cadence_ms": 10,
                     "observation_latency_ms": float(config["controller"]["observation_latency_s"]) * 1000.0,
                     "poll_timestamp": "labels and model state use observation_cutoff_s, not wall/poll time"},
        "training": {"device": str(device), "threads": args.threads, "seed": args.seed,
                     "max_epochs": args.epochs, "best_epoch": best_epoch, "patience": args.patience,
                     "batch_size": args.batch_size, "training_wall_s": training_wall_s,
                     "best_validation_phase_balanced_normalized_mse": best_loss,
                     "phase_reweighting": "equal aggregate weight per episode-phase group within each split"},
        "validity_policy": {"minimum_events": 12, "minimum_nonempty_5ms_bins": 3,
                            "maximum_state_age_s": float(config["controller"].get("maximum_event_state_age_s", 0.03)),
                            "missing_or_stale_state_action": "hold lateral command and do not start closure; existing controller behavior"},
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "metric_boundary": "position/velocity and same-fixed-predictor arrival/intercept metrics use GT only after inference as evaluator; arrival is compared at the causal cutoff and does not use future GT arrival",
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    (out / "development_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name not in {"run_manifest.json", "SHA256SUMS"}:
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    (out / "SHA256SUMS").write_text("".join(f"{v['sha256']}  {k}\n" for k, v in outputs.items()), encoding="utf-8")
    manifest = {
        "phase": summary["phase"], "code_commit": summary["code_commit"],
        "config_sha256": summary["config_sha256"], "source_manifest_sha256": summary["source_manifest_sha256"],
        "checkpoint_sha256": summary["checkpoint_sha256"], "outputs": outputs,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "train": train_metrics, "validation": val_metrics,
                      "best_epoch": best_epoch, "training_wall_s": training_wall_s}, indent=2))


if __name__ == "__main__":
    main()
