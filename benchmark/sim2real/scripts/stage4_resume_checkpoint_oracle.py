#!/usr/bin/env python3
"""Validate a real Stage-4 process-exit resume against its exact source checkpoint.

CUDA's STE-FlowNet feature-warp backward uses grid_sample, which PyTorch reports
as nondeterministic on CUDA. A full replay from seed can therefore drift before
the resume boundary. This check verifies the actual resumed process from the
saved boundary checkpoint, its independently derived next batch/augmentation,
and a separate forward-only loss oracle from that same checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_next_batch(data_root: Path, generator: str, seed: int,
                           epoch: int, position: int) -> tuple[list[dict], list[list[int]]]:
    """Build the next batch from the sorted NPZ inventory and registered seed."""
    samples = []
    for path in sorted((data_root / generator / "train").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            ids = np.asarray(data["sample_ids"])
            if ids.shape != (66,):
                raise RuntimeError(f"{path}: expected 66 sample IDs, got {ids.shape}")
            samples.extend({"rollout": path.stem, "sample_index": index,
                            "sample_id": str(ids[index])} for index in range(66))
    if len(samples) != 1584 or position % 8:
        raise RuntimeError(f"unexpected training inventory/position: {len(samples)}/{position}")
    order = np.random.default_rng(seed + 1000003 * epoch).permutation(len(samples))
    batch_index = position // 8
    selected = order[position:position + 8]
    if len(selected) != 8:
        raise RuntimeError("source checkpoint does not point to a complete next batch")
    batch = [samples[int(index)] for index in selected]
    augmentation = [[seed, epoch, batch_index, micro] for micro in range(8)]
    return batch, augmentation


def optimizer_step_range(state: dict) -> tuple[int, int, int]:
    steps = []
    for entry in state["optimizer"]["state"].values():
        value = entry.get("step")
        if value is not None:
            steps.append(int(value.item() if torch.is_tensor(value) else value))
    if not steps:
        raise RuntimeError("optimizer checkpoint has no Adam step counters")
    return min(steps), max(steps), len(steps)


def forward_oracle(repo: Path, model_root: Path, data_root: Path,
                   checkpoint_path: Path, generator: str, seed: int,
                   batch: list[dict], augmentation: list[list[int]]) -> dict:
    scripts = repo / "benchmark/sim2real/scripts"
    compat = repo / "benchmark/sim2real/compat"
    sys.path[:0] = [str(scripts), str(compat), str(model_root)]

    from run_steflow_stage2_pilot import (
        objective_dt1, pure_torch_spatial_correlation, set_seed, state_sha256,
    )
    from steflow_align_corners import patch_ste
    from train_steflow_stage4 import SIZE, apply_joint_transform

    corr_module = importlib.import_module("models.corr")
    corr_module.spatial_correlation_sample = pure_torch_spatial_correlation
    import models
    import multiscaleloss
    patch_ste(models, multiscaleloss, model_root)

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("resume oracle requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

    model = models.steflow().to(device)
    model.train()
    optimizer = torch.optim.Adam([
        {"params": model.bias_parameters(), "weight_decay": 0.0},
        {"params": model.weight_parameters(), "weight_decay": .0004},
    ], lr=.0004, betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=.7)
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if source["generator"] != generator or source["seed"] != seed:
        raise RuntimeError("source checkpoint generator/seed mismatch")
    model.load_state_dict(source["state_dict"])
    optimizer.load_state_dict(source["optimizer"])
    scheduler.load_state_dict(source["scheduler"])
    random.setstate(source["python_rng"])
    np.random.set_state(source["numpy_rng"])
    torch.set_rng_state(source["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in source["cuda_rng"]])
    if state_sha256(model) != source["current_state_sha256"]:
        raise RuntimeError("oracle did not rehydrate the source model state exactly")

    arrays = {}
    event_rows, gray_rows = [], []
    for micro, item in enumerate(batch):
        rollout = item["rollout"]
        if rollout not in arrays:
            with np.load(data_root / generator / "train" / f"{rollout}.npz", allow_pickle=False) as d:
                arrays[rollout] = {key: np.asarray(d[key]) for key in (
                    "events", "gray_prev", "gray_next", "flow_gt", "flow_valid")}
        ix = item["sample_index"]
        sample = {key: arrays[rollout][key][ix:ix + 1] for key in (
            "events", "gray_prev", "gray_next", "flow_gt", "flow_valid")}
        event, gray, _, _ = apply_joint_transform(
            sample, np.random.default_rng(np.random.SeedSequence(augmentation[micro])), device)
        event_rows.append(event)
        gray_rows.append(gray)
    event_batch = torch.cat(event_rows, dim=0)
    gray_batch = torch.cat(gray_rows, dim=0)
    with torch.no_grad():
        output = model(event_batch, SIZE, use_dt4=False, flow_seq=False)
        loss, photo, smooth = objective_dt1(
            output, gray_batch[:, 0:1], gray_batch[:, 1:2], event_batch, multiscaleloss)
    torch.cuda.synchronize(device)
    values = {"loss": float(loss.cpu()), "photometric_loss": float(photo.cpu()),
              "smoothness_loss": float(smooth.cpu())}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"non-finite independent forward oracle values: {values}")
    return {**values, "source_model_state_sha256": source["current_state_sha256"],
            "batch_sample_ids": [item["sample_id"] for item in batch],
            "augmentation_seed_sequences": augmentation}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("repo-root", "config", "data-root", "model-root", "source-checkpoint",
                 "resumed-run-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-data-used", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.real_data_used:
        raise SystemExit("resume oracle is synthetic-only")
    for name in ("repo_root", "config", "data_root", "model_root", "source_checkpoint",
                 "resumed_run_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite resume oracle output {args.output}")

    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    resumed_root = args.resumed_run_root / args.generator
    resumed_checkpoint_path = resumed_root / "last.pt"
    resumed = torch.load(resumed_checkpoint_path, map_location="cpu", weights_only=False)
    live = json.loads((resumed_root / "live_status.json").read_text())
    metadata = json.loads((resumed_root / "run_metadata.json").read_text())
    metric_rows = [json.loads(line) for line in (resumed_root / "training_metrics.jsonl").read_text().splitlines()
                   if line.strip()]
    metric_rows = [row for row in metric_rows if row.get("optimizer_update") == 199]
    if len(metric_rows) != 1:
        raise RuntimeError(f"expected one real resumed update-199 metric row, got {len(metric_rows)}")
    metric = metric_rows[0]

    if (source.get("optimizer_update"), source.get("next_epoch"), source.get("next_position_samples")) != (198, 1, 0):
        raise RuntimeError("source checkpoint is not the validated update-198 epoch boundary")
    if source.get("resume_allowed") is not True or source.get("resume_phase") != "train":
        raise RuntimeError("source checkpoint is not resumable")
    if (resumed.get("optimizer_update"), resumed.get("next_epoch"), resumed.get("next_position_samples")) != (199, 1, 8):
        raise RuntimeError("resumed checkpoint did not advance exactly one update from the source boundary")
    if live.get("completed_updates") != 199 or live.get("status") != "paused_at_update_budget":
        raise RuntimeError(f"resumed process did not exit cleanly at update 199: {live}")
    if metadata.get("generator") != args.generator or metadata.get("seed") != args.seed:
        raise RuntimeError("resumed process generator/seed mismatch")
    manifest_path = args.data_root / "training_bank_manifest.json"
    if sha256(args.config) != source.get("config_sha256"):
        raise RuntimeError("source checkpoint config hash is stale")
    if sha256(manifest_path) != source.get("data_manifest_sha256"):
        raise RuntimeError("source checkpoint training data manifest hash is stale")
    for key in ("initial_state_sha256", "config_sha256", "data_manifest_sha256"):
        if resumed.get(key) != source.get(key):
            raise RuntimeError(f"resume lineage mismatch in {key}")
    if metadata.get("real_data_used") is not False or metadata.get("flow_gt_used_in_optimizer") is not False:
        raise RuntimeError("resumed process violated the synthetic-only training contract")

    batch, augmentation = independent_next_batch(
        args.data_root, args.generator, args.seed, source["next_epoch"], source["next_position_samples"])
    expected_ids = [item["sample_id"] for item in batch]
    if metric.get("rollout_sample_ids") != expected_ids:
        raise RuntimeError("resumed update sample IDs do not match independent sampler reconstruction")
    if metric.get("augmentation_seed_sequences") != augmentation:
        raise RuntimeError("resumed update augmentation seeds do not match the source boundary schedule")
    if metric.get("optimizer_update") != source["optimizer_update"] + 1:
        raise RuntimeError("optimizer update did not advance from the saved boundary")
    if not math.isfinite(float(metric["loss"])) or not math.isfinite(float(metric["gradient_l2_norm"])) or metric["gradient_l2_norm"] <= 0:
        raise RuntimeError("resumed update did not produce a finite positive gradient")
    if metric.get("finite_gradient_tensors", 0) <= 0:
        raise RuntimeError("resumed update has no finite gradient tensors")

    expected = forward_oracle(args.repo_root, args.model_root, args.data_root,
                              args.source_checkpoint, args.generator, args.seed,
                              batch, augmentation)
    tolerance = {key: max(1e-5, abs(expected[key]) * 1e-6)
                 for key in ("loss", "photometric_loss", "smoothness_loss")}
    errors = {key: abs(float(metric[key]) - expected[key])
              for key in tolerance}
    if any(errors[key] > tolerance[key] for key in errors):
        raise RuntimeError(f"actual resumed forward metrics disagree with source-checkpoint oracle: {errors}")

    before_steps = optimizer_step_range(source)
    after_steps = optimizer_step_range(resumed)
    if before_steps[:2] != (198, 198) or after_steps[:2] != (199, 199):
        raise RuntimeError(f"Adam step counters did not restore/advance exactly once: {before_steps} -> {after_steps}")
    if resumed.get("current_state_sha256") == source.get("current_state_sha256"):
        raise RuntimeError("resumed optimizer update did not change model state")
    if resumed["scheduler"]["last_epoch"] != source["scheduler"]["last_epoch"]:
        raise RuntimeError("scheduler unexpectedly advanced inside the new epoch")
    rng_fields = ("python_rng", "numpy_rng", "torch_rng", "cuda_rng")
    if any(field not in source for field in rng_fields) or not source["cuda_rng"]:
        raise RuntimeError("source checkpoint is missing one or more CPU/CUDA RNG states")

    result = {
        "schema": "stage4_checkpoint_resume_oracle_v1", "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "generator": args.generator,
        "seed": args.seed, "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_bytes": args.source_checkpoint.stat().st_size,
        "source_checkpoint_sha256": sha256(args.source_checkpoint),
        "resumed_checkpoint": str(resumed_checkpoint_path),
        "resumed_checkpoint_bytes": resumed_checkpoint_path.stat().st_size,
        "resumed_checkpoint_sha256": sha256(resumed_checkpoint_path),
        "resumed_process_pid": metadata.get("pid"), "resumed_process_started_utc": metadata.get("started_utc"),
        "resume_boundary": {"source_update": source["optimizer_update"], "source_next_epoch": source["next_epoch"],
                            "source_next_position_samples": source["next_position_samples"],
                            "resumed_update": resumed["optimizer_update"],
                            "resumed_next_epoch": resumed["next_epoch"],
                            "resumed_next_position_samples": resumed["next_position_samples"]},
        "independent_batch_reconstruction": {"sample_ids": expected_ids,
                                             "augmentation_seed_sequences": augmentation},
        "actual_resumed_metrics": {key: float(metric[key]) for key in tolerance},
        "independent_forward_oracle": expected,
        "forward_abs_errors": errors, "forward_tolerances": tolerance,
        "optimizer_step_counters_before_after": {"min": [before_steps[0], after_steps[0]],
                                                 "max": [before_steps[1], after_steps[1]],
                                                 "parameter_states": [before_steps[2], after_steps[2]]},
        "scheduler_last_epoch_before_after": [source["scheduler"]["last_epoch"], resumed["scheduler"]["last_epoch"]],
        "model_state_changed_after_update": True,
        "source_checkpoint_rng_fields_present": list(rng_fields),
        "training_data": "synthetic only", "real_data_used": False,
        "full_seed_replay_note": "A separate full replay from seed did not match bitwise; PyTorch deterministic-algorithm guard identifies grid_sampler_2d_backward_cuda as nondeterministic. The gate checks actual boundary-state restoration and next-step inputs/forward loss instead of requiring cumulative replay weights to be identical.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
