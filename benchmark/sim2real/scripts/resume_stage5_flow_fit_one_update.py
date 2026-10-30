#!/usr/bin/env python3
"""Resume a Stage 5 STE-FlowNet fit checkpoint for exactly one supervised update.

This is a bounded integrity probe. It never edits the source checkpoint and
does not claim bitwise equivalence with the original CUDA training process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmark/sim2real/scripts"))
from stage5_flow_supervised_pilot import (  # noqa: E402
    FLOW_UNIT,
    load_sample,
    model_source,
    multiscale_supervised_epe,
    pad_sample,
    sha256,
    state_sha256,
)


def optimizer_step_values(optimizer: torch.optim.Optimizer) -> list[int]:
    return sorted({int(state["step"].item()) for state in optimizer.state.values()
                   if "step" in state})


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def repository_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output_dir}")
    output_dir.mkdir(parents=True)

    config = json.loads(args.config.read_text())
    sample_cfg = config["sample"]
    source_checkpoint = args.checkpoint.resolve()
    source_sha_before = sha256(source_checkpoint)
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("source checkpoint must be a mapping")
    if int(checkpoint.get("updates", -1)) != 200:
        raise RuntimeError(f"expected source update 200, found {checkpoint.get('updates')!r}")
    if checkpoint.get("sample_id") != sample_cfg["sample_id"]:
        raise RuntimeError("source checkpoint sample ID differs from frozen config")
    if checkpoint.get("flow_unit") != FLOW_UNIT:
        raise RuntimeError("source checkpoint flow unit mismatch")
    if checkpoint.get("real_data_used") is not False:
        raise RuntimeError("source checkpoint is not certified synthetic-only")

    manifest_path = args.data_root.resolve() / "training_bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    if checkpoint.get("data_manifest_sha256") != manifest_sha:
        raise RuntimeError("checkpoint and paired data manifest hashes differ")
    if manifest.get("flow_unit") != FLOW_UNIT:
        raise RuntimeError("paired bank flow unit mismatch")

    rows = {(row["generator"], row["rollout_id"]): row for row in manifest["entries"]}
    rollout = sample_cfg["rollout_id"]
    arms = list(config["pairing"]["arms"])
    if set(arms) != {"eventgenesis", "v2e"}:
        raise RuntimeError(f"unexpected paired arms in config: {arms}")
    paired_samples = {}
    for arm in arms:
        row = rows[(arm, rollout)]
        if row["split"] != sample_cfg["split"]:
            raise RuntimeError(f"{arm}/{rollout} split differs from frozen sample config")
        path = Path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"registered artifact missing or hash-mismatched: {path}")
        paired_samples[arm] = load_sample(path, int(sample_cfg["window_index"]))
        if str(paired_samples[arm]["sample_ids"]) != sample_cfg["sample_id"]:
            raise RuntimeError(f"{arm} sample identity differs from frozen config")
    label_fields = ("flow_gt", "flow_valid", "sample_ids", "label_sha256")
    for field in label_fields:
        if not np.array_equal(paired_samples["eventgenesis"][field], paired_samples["v2e"][field]):
            raise RuntimeError(f"paired physical label field differs: {field}")

    generator = checkpoint.get("generator")
    if generator not in paired_samples:
        raise RuntimeError(f"unknown checkpoint generator: {generator!r}")
    sample = paired_samples[generator]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the one-update resume probe requires CUDA")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    model, source_patch = model_source(args.model_root.resolve(), device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    state_before = state_sha256(model)
    if state_before != checkpoint.get("state_sha256"):
        raise RuntimeError("restored model state hash differs from source checkpoint metadata")
    optimizer = torch.optim.Adam([
        {"params": model.bias_parameters(), "weight_decay": 0.0},
        {"params": model.weight_parameters(), "weight_decay": 0.0004},
    ], lr=0.0004, betas=(0.9, 0.999))
    optimizer.load_state_dict(checkpoint["optimizer"])
    steps_before = optimizer_step_values(optimizer)
    if steps_before != [200]:
        raise RuntimeError(f"expected all restored Adam counters at 200, got {steps_before}")
    if len(optimizer.param_groups) != 2:
        raise RuntimeError("restored Adam parameter group count mismatch")
    group_sizes = [len(group["params"]) for group in optimizer.param_groups]

    events, gt, valid = pad_sample(sample, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(events, 352, use_dt4=False, flow_seq=False)
    loss, scale_losses = multiscale_supervised_epe(outputs, gt, valid)
    if not torch.isfinite(loss):
        raise RuntimeError("resumed update loss is not finite")
    loss_before_step = float(loss.detach().cpu())
    scale_losses = [float(value) for value in scale_losses]
    if not np.isfinite(scale_losses).all():
        raise RuntimeError("resumed multi-scale losses are not all finite")
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads or any(not torch.isfinite(grad).all() for grad in grads):
        raise RuntimeError("resumed update has missing or non-finite gradients")
    gradient_l2 = float(torch.sqrt(sum(grad.detach().float().square().sum() for grad in grads)).cpu())
    if not np.isfinite(gradient_l2) or gradient_l2 <= 0:
        raise RuntimeError(f"invalid resumed gradient norm: {gradient_l2}")

    optimizer.step()
    steps_after = optimizer_step_values(optimizer)
    if steps_after != [201]:
        raise RuntimeError(f"Adam did not advance exactly 200 -> 201: {steps_after}")
    state_after = state_sha256(model)
    if state_after == state_before:
        raise RuntimeError("model state did not change after the optimizer update")

    output_checkpoint = output_dir / "resume_update_201.pt"
    payload = {
        key: value for key, value in checkpoint.items()
        if key not in {"state_dict", "optimizer", "updates", "state_sha256"}
    }
    payload.update({
        "state_dict": cpu_tree(model.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "updates": 201,
        "state_sha256": state_after,
        "resume_probe_parent_sha256": source_sha_before,
        "resume_probe_created_utc": datetime.now(timezone.utc).isoformat(),
        "resume_probe_loss": loss_before_step,
        "resume_probe_gradient_l2": gradient_l2,
    })
    torch.save(payload, output_checkpoint)
    source_sha_after = sha256(source_checkpoint)
    if source_sha_after != source_sha_before:
        raise RuntimeError("source checkpoint changed during the probe")

    result = {
        "schema": "stage7_flow_checkpoint_resume_one_update_v1",
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": repository_commit(),
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "generator": generator,
        "seed": checkpoint.get("seed"),
        "sample_id": sample_cfg["sample_id"],
        "rollout_id": sample_cfg["rollout_id"],
        "window_index": int(sample_cfg["window_index"]),
        "config_sha256": sha256(args.config.resolve()),
        "data_manifest_sha256": manifest_sha,
        "paired_labels_identical": True,
        "flow_unit": FLOW_UNIT,
        "real_data_used": False,
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "size_bytes": source_checkpoint.stat().st_size,
            "sha256_before": source_sha_before,
            "sha256_after": source_sha_after,
            "updates": 200,
            "model_state_sha256": state_before,
            "optimizer_step_values": steps_before,
            "scheduler_present": "scheduler" in checkpoint,
        },
        "update": {
            "from": 200,
            "to": 201,
            "loss_before_step": loss_before_step,
            "scale_losses_before_step": scale_losses,
            "gradient_l2_norm": gradient_l2,
            "gradient_tensor_count": len(grads),
            "optimizer_step_values_after": steps_after,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "parameter_groups": group_sizes,
            "scheduler_present": "scheduler" in checkpoint,
        },
        "new_checkpoint": {
            "path": str(output_checkpoint.resolve()),
            "size_bytes": output_checkpoint.stat().st_size,
            "sha256": sha256(output_checkpoint),
            "updates": 201,
        },
        "source_patch": source_patch,
        "strict_full_run_bitwise_equivalence_claimed": False,
        "real_scoring_enabled": False,
    }
    result_path = output_dir / "resume_probe.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
