#!/usr/bin/env python3
"""Small supervised overfit check for causal event-to-state perception.

The network receives only causal five-bin event tensors. Simulator states are
used as training labels and for offline scoring, never as network inputs. All
selected episodes are used for fitting, so this script tests the data/model
plumbing and capacity only; it does not estimate generalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark.robot_closed_loop.diagnostics.replay_event_state import (
    _five_bin_tensor,
    _phase,
    _read_trace,
    _sha256,
    _truth_at,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "benchmark/robot_closed_loop/configs/scene_pilot_r0_5_candidate.json"


class TinyEventStateNet(nn.Module):
    """Compact 2D CNN over ten time-polarity channels."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 6))

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(events))


def _build_dataset(source_root: Path, episode_ids: List[int], config: Dict[str, Any]):
    tensors, labels, metadata = [], [], []
    width, height = (int(v) for v in config["camera"]["resolution_wh"])
    cadence_s = 1.0 / float(config["controller"]["high_level_hz"])
    latency_s = float(config["controller"]["observation_latency_s"])
    burn_in_s = float(config["controller"]["event_burn_in_s"])

    for episode_id in episode_ids:
        episode_dir = source_root / "episodes" / f"episode_{episode_id:03d}"
        with (episode_dir / "episode.csv").open(newline="") as stream:
            episode = next(csv.DictReader(stream))
        trace_path = episode_dir / "state_control_trace.jsonl"
        trace = _read_trace(trace_path)
        with np.load(episode_dir / "direct_hdr_events_1khz_2khz.npz") as archive:
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
            voxel = _five_bin_tensor(events, cutoff_s, width, height)
            position, velocity = _truth_at(trace, cutoff_s)
            tensors.append(np.log1p(voxel).reshape(10, 64, 64))
            labels.append(np.concatenate((position, velocity)).astype(np.float32))
            metadata.append({
                "episode_id": episode_id,
                "speed_m_s": float(episode["launch_speed_m_s"]),
                "launch_y_m": float(episode["start_y_m"]),
                "evaluation_time_s": float(now_s),
                "observation_cutoff_s": cutoff_s,
                "phase": _phase(cutoff_s, contact_s, lift_s),
                "input_event_count": int(voxel.sum()),
            })

    if not tensors:
        raise RuntimeError("no causal event windows were constructed")
    return (
        np.asarray(tensors, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        metadata,
    )


def _score(model, x: torch.Tensor, y: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor):
    model.eval()
    with torch.no_grad():
        prediction = model(x) * scale + mean
    truth = y * scale + mean
    error = prediction - truth
    position_error = torch.linalg.vector_norm(error[:, :3], dim=1).cpu().numpy()
    planar_position_error = torch.linalg.vector_norm(error[:, :2], dim=1).cpu().numpy()
    planar_velocity_error = torch.linalg.vector_norm(error[:, 3:5], dim=1).cpu().numpy()
    full_velocity_error = torch.linalg.vector_norm(error[:, 3:], dim=1).cpu().numpy()
    return prediction, {
        "position_3d_error_m": {
            "rmse": float(torch.sqrt(torch.mean(error[:, :3] ** 2)).item() * np.sqrt(3.0)),
            "p95": float(np.percentile(position_error, 95)),
        },
        "position_xy_error_m": {
            "rmse": float(np.sqrt(np.mean(planar_position_error ** 2))),
            "p95": float(np.percentile(planar_position_error, 95)),
        },
        "velocity_xyz_error_m_s": {
            "rmse": float(np.sqrt(np.mean(full_velocity_error ** 2))),
            "p95": float(np.percentile(full_velocity_error, 95)),
        },
        "velocity_xy_error_m_s": {
            "rmse": float(np.sqrt(np.mean(planar_velocity_error ** 2))),
            "p95": float(np.percentile(planar_velocity_error, 95)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", default="0,4,8")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    config = json.loads(args.config.read_text())
    episode_ids = [int(value) for value in args.episodes.split(",")]
    x_np, y_np, metadata = _build_dataset(args.source_results, episode_ids, config)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    mean = y.mean(dim=0)
    scale = y.std(dim=0, unbiased=False).clamp_min(0.01)
    y_normalized = (y - mean) / scale
    dataset = TensorDataset(x, y_normalized)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = TinyEventStateNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    started = time.perf_counter()
    losses = []
    for _epoch in range(args.epochs):
        model.train()
        total_loss, count = 0.0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(batch_x) - batch_y) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            count += len(batch_y)
        losses.append(total_loss / max(count, 1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_wall = time.perf_counter() - started

    _, overall_metrics = _score(model, x, y_normalized, mean, scale)
    summary_by_phase = {}
    with torch.no_grad():
        prediction = model(x) * scale + mean
    error = prediction - y
    for phase in ("pre_contact", "contact_to_lift", "post_lift"):
        idx = [i for i, row in enumerate(metadata) if row["phase"] == phase]
        if not idx:
            summary_by_phase[phase] = {"n": 0}
            continue
        e = error[idx]
        pos_xy = torch.linalg.vector_norm(e[:, :2], dim=1).cpu().numpy()
        vel_xy = torch.linalg.vector_norm(e[:, 3:5], dim=1).cpu().numpy()
        summary_by_phase[phase] = {
            "n": len(idx),
            "position_xy_rmse_m": float(np.sqrt(np.mean(pos_xy ** 2))),
            "position_xy_p95_m": float(np.percentile(pos_xy, 95)),
            "velocity_xy_rmse_m_s": float(np.sqrt(np.mean(vel_xy ** 2))),
            "velocity_xy_p95_m_s": float(np.percentile(vel_xy, 95)),
        }

    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.out / "tiny_event_state_overfit.pt"
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "label_mean": mean.cpu(), "label_scale": scale.cpu(),
        "architecture": "Conv2D(10x64x64)->16->32->32->global-mean->MLP(32,32,6)",
        "input": "causal five 5ms bins, two polarities, 64x64, log1p counts, 10ms latency",
        "output": "world position xyz and velocity xyz at observation cutoff",
        "training_episodes": episode_ids,
        "inference_gt_inputs": False,
    }, checkpoint_path)

    prediction = prediction.detach().cpu().numpy()
    labels_cpu = y.detach().cpu().numpy()
    with (args.out / "per_window_predictions.csv").open("w", newline="") as stream:
        fields = list(metadata[0]) + [
            "gt_x_m", "gt_y_m", "gt_z_m", "gt_vx_m_s", "gt_vy_m_s", "gt_vz_m_s",
            "pred_x_m", "pred_y_m", "pred_z_m", "pred_vx_m_s", "pred_vy_m_s", "pred_vz_m_s",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for meta, truth, pred in zip(metadata, labels_cpu, prediction):
            row = dict(meta)
            row.update(dict(zip(fields[7:13], truth.tolist())))
            row.update(dict(zip(fields[13:19], pred.tolist())))
            writer.writerow(row)

    # Model-only latency excludes event binning, DVS generation, rendering,
    # physics, and all output serialization; report it separately.
    model.eval()
    latency_ms = []
    with torch.no_grad():
        for _ in range(200):
            tick = time.perf_counter()
            model(x[:1])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_ms.append((time.perf_counter() - tick) * 1000.0)

    source_manifest = args.source_results / "SHA256SUMS"
    expected = {}
    if source_manifest.exists():
        for line in source_manifest.read_text().splitlines():
            pieces = line.split(None, 1)
            if len(pieces) == 2:
                expected[pieces[1].lstrip("* ")] = pieces[0]
    input_hashes = {}
    for episode_id in episode_ids:
        rel = f"episodes/episode_{episode_id:03d}"
        for name in ("direct_hdr_events_1khz_2khz.npz", "state_control_trace.jsonl", "episode.csv"):
            path = args.source_results / rel / name
            digest = _sha256(path)
            input_hashes[f"{rel}/{name}"] = digest
            expected_digest = expected.get(f"{rel}/{name}")
            if expected_digest is not None and expected_digest != digest:
                raise RuntimeError(f"source checksum mismatch: {rel}/{name}")

    summary = {
        "phase": "R0.6 tiny supervised overfit diagnosis; not a generalization or ranking result",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": _sha256(args.config),
        "source_sha256sums_sha256": None if not source_manifest.exists() else _sha256(source_manifest),
        "raw_event_inputs_verified_against_source_manifest": all(
            f"episodes/episode_{episode_id:03d}/direct_hdr_events_1khz_2khz.npz" in expected
            for episode_id in episode_ids
        ),
        "input_hashes": input_hashes,
        "episodes_fit_and_scored": episode_ids,
        "split": "none; all listed episodes are used for fitting and in-sample scoring",
        "training_labels": "simulator object pose/velocity at each causal observation cutoff",
        "network_inputs": "only five-bin Direct-HDR event counts; no GT, segmentation, depth, RGB, or future state is passed to the model",
        "input_shape_per_sample": [10, 64, 64],
        "sample_count": len(metadata),
        "sample_count_by_phase": {phase: sum(row["phase"] == phase for row in metadata) for phase in ("pre_contact", "contact_to_lift", "post_lift")},
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "training_wall_s": train_wall,
        "final_normalized_mse": losses[-1],
        "initial_normalized_mse": losses[0],
        "in_sample_errors": overall_metrics,
        "in_sample_errors_by_phase": summary_by_phase,
        "model_only_latency_ms": {
            "mean": float(np.mean(latency_ms)),
            "p50": float(np.percentile(latency_ms, 50)),
            "p95": float(np.percentile(latency_ms, 95)),
        },
        "latency_boundary": "batch-1 network only; excludes event binning, Direct-HDR render, DVS, physics, and I/O",
    }
    manifest_path = args.out / "run_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n")
    outputs = [checkpoint_path, args.out / "per_window_predictions.csv", manifest_path]
    (args.out / "SHA256SUMS").write_text("".join(
        f"{_sha256(path)}  {path.name}\n" for path in outputs
    ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
