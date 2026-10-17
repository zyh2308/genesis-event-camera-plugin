#!/usr/bin/env python3
"""Compare uninterrupted and process-exit/resume STE training across epoch 1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare_value(left, right, path="root", errors=None):
    errors = [] if errors is None else errors
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            errors.append({"path": path, "error": "numpy array/type mismatch"})
        elif left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            errors.append({"path": path, "error": "numpy array state mismatch"})
        return errors
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            errors.append({"path": path, "error": "tensor/type mismatch"}); return errors
        a, b = left.detach().cpu(), right.detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            errors.append({"path": path, "error": f"shape/dtype mismatch {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"})
        elif a.is_floating_point():
            max_abs = float((a - b).abs().max()) if a.numel() else 0.
            if not np.isfinite(max_abs) or max_abs > 1e-6:
                errors.append({"path": path, "max_abs": max_abs})
        elif not torch.equal(a, b):
            errors.append({"path": path, "error": "integer tensor mismatch"})
        return errors
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            errors.append({"path": path, "error": "dictionary keys differ"}); return errors
        for key in left:
            compare_value(left[key], right[key], f"{path}.{key}", errors)
        return errors
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            errors.append({"path": path, "error": "sequence lengths differ"}); return errors
        for index, (a, b) in enumerate(zip(left, right)):
            compare_value(a, b, f"{path}[{index}]", errors)
        return errors
    if left != right:
        errors.append({"path": path, "left": repr(left), "right": repr(right)})
    return errors


def max_tensor_abs(left, right) -> float:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.numel() == 0:
            return 0.
        return float((left.detach().cpu().float() - right.detach().cpu().float()).abs().max())
    if isinstance(left, dict) and isinstance(right, dict):
        return max((max_tensor_abs(left[key], right[key]) for key in left if key in right), default=0.)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return max((max_tensor_abs(a, b) for a, b in zip(left, right)), default=0.)
    return 0.


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("repo-root", "config", "bank-root", "v2e-root", "data-root", "model-root", "genesis-root",
                 "receipt", "pilot-run-root", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    for name in ("repo_root", "config", "bank_root", "v2e_root", "data_root", "model_root", "genesis_root",
                 "receipt", "pilot_run_root", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    split_checkpoint = args.pilot_run_root / "eventgenesis/last.pt"
    if not split_checkpoint.is_file():
        raise SystemExit(f"resume comparison blocked: missing epoch-boundary pilot checkpoint {split_checkpoint}")
    checkpoint = torch.load(split_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["optimizer_update"] != 198 or checkpoint["next_epoch"] != 1 or checkpoint["next_position_samples"] != 0:
        raise SystemExit("resume comparison requires the fully validated post-epoch-1 checkpoint at update 198")
    if checkpoint.get("resume_allowed") is not True or checkpoint.get("resume_phase") != "train":
        raise SystemExit("pilot checkpoint is not a resumable training checkpoint")
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite resume evidence directory {args.output_root}")
    args.output_root.mkdir(parents=True)
    gate = args.repo_root / "benchmark/sim2real/scripts/stage4_launch_gate.py"
    run_root = args.output_root / "runs"
    env_base = os.environ.copy()
    env_base.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "STAGE4_PHYSICAL_GPU": str(args.gpu),
                     "STAGE4_REPO_COMMIT": subprocess.check_output(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True).strip(),
                     "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"})

    common = [sys.executable, str(gate), "--phase", "resume-check", "--receipt", str(args.receipt),
              "--repo-root", str(args.repo_root), "--config", str(args.config),
              "--bank-root", str(args.bank_root), "--v2e-root", str(args.v2e_root),
              "--data-root", str(args.data_root), "--model-root", str(args.model_root),
              "--genesis-root", str(args.genesis_root), "--generator", "eventgenesis", "--max-updates", "199"]

    def execute(label: str, root: Path, resume: Path | None):
        cmd = common + ["--run-root", str(root)]
        if resume is not None:
            cmd += ["--resume", str(resume)]
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        with (args.output_root / f"{label}.stdout.log").open("w") as out, (args.output_root / f"{label}.stderr.log").open("w") as err:
            proc = subprocess.run(cmd, env=env_base, stdout=out, stderr=err)
        result = {"label": label, "command": cmd, "started_utc": started,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_s": round(time.monotonic() - t0, 3), "exit_code": proc.returncode,
                  "stdout": str(args.output_root / f"{label}.stdout.log"),
                  "stderr": str(args.output_root / f"{label}.stderr.log")}
        if proc.returncode != 0:
            raise RuntimeError(f"{label} process failed; inspect {result['stderr']}")
        return result

    continuous_root = run_root / "continuous_from_seed0"
    resumed_root = run_root / "process_exit_resume_from_update198"
    continuous_record = execute("continuous", continuous_root, None)
    resumed_record = execute("resumed", resumed_root, split_checkpoint)
    continuous_ckpt = torch.load(continuous_root / "eventgenesis/last.pt", map_location="cpu", weights_only=False)
    resumed_ckpt = torch.load(resumed_root / "eventgenesis/last.pt", map_location="cpu", weights_only=False)
    if continuous_ckpt["optimizer_update"] != 199 or resumed_ckpt["optimizer_update"] != 199:
        raise RuntimeError("comparison checkpoints did not stop at update 199")
    continuous_rows = [json.loads(line) for line in (continuous_root / "eventgenesis/training_metrics.jsonl").read_text().splitlines()
                      if line.strip() and json.loads(line).get("optimizer_update") == 199]
    resumed_rows = [json.loads(line) for line in (resumed_root / "eventgenesis/training_metrics.jsonl").read_text().splitlines()
                    if line.strip() and json.loads(line).get("optimizer_update") == 199]
    if len(continuous_rows) != 1 or len(resumed_rows) != 1:
        raise RuntimeError("missing unique update-199 training metrics in one path")
    metric_keys = ("rollout_sample_ids", "augmentation_seed_sequences", "learning_rate")
    for key in metric_keys:
        if continuous_rows[0][key] != resumed_rows[0][key]:
            raise RuntimeError(f"resume diverged in next-batch {key}")
    loss_errors = {key: abs(float(continuous_rows[0][key]) - float(resumed_rows[0][key]))
                   for key in ("loss", "photometric_loss", "smoothness_loss", "gradient_l2_norm")}
    if max(loss_errors.values()) > 1e-6:
        raise RuntimeError(f"resume update-199 metrics exceeded tolerance: {loss_errors}")
    model_errors = compare_value(continuous_ckpt["state_dict"], resumed_ckpt["state_dict"], "model")
    optimizer_errors = compare_value(continuous_ckpt["optimizer"], resumed_ckpt["optimizer"], "optimizer")
    scheduler_errors = compare_value(continuous_ckpt["scheduler"], resumed_ckpt["scheduler"], "scheduler")
    rng_errors = compare_value({"python": continuous_ckpt["python_rng"], "numpy": continuous_ckpt["numpy_rng"],
                                "torch": continuous_ckpt["torch_rng"], "cuda": continuous_ckpt["cuda_rng"]},
                               {"python": resumed_ckpt["python_rng"], "numpy": resumed_ckpt["numpy_rng"],
                                "torch": resumed_ckpt["torch_rng"], "cuda": resumed_ckpt["cuda_rng"]}, "rng")
    if model_errors or optimizer_errors or scheduler_errors or rng_errors:
        raise RuntimeError(json.dumps({"model": model_errors[:5], "optimizer": optimizer_errors[:5],
                                       "scheduler": scheduler_errors[:5], "rng": rng_errors[:5]}, indent=2))
    result = {
        "schema": "stage4_resume_equivalence_v1", "status": "PASS", "split_checkpoint": str(split_checkpoint),
        "split_checkpoint_sha256": digest(split_checkpoint), "split_boundary": {
            "optimizer_update": checkpoint["optimizer_update"], "next_epoch": checkpoint["next_epoch"],
            "next_position_samples": checkpoint["next_position_samples"],
            "scheduler_last_epoch": checkpoint["scheduler"]["last_epoch"]},
        "continuous_process": continuous_record, "resumed_process": resumed_record,
        "compared_next_update": 199, "compared_sample_ids": continuous_rows[0]["rollout_sample_ids"],
        "compared_augmentation_seed_sequences": continuous_rows[0]["augmentation_seed_sequences"],
        "metric_abs_errors": loss_errors,
        "max_model_parameter_abs_error": max_tensor_abs(continuous_ckpt["state_dict"], resumed_ckpt["state_dict"]),
        "optimizer_scheduler_rng": "exact state match within floating tensor tolerance 1e-6",
        "continuous_checkpoint_sha256": digest(continuous_root / "eventgenesis/last.pt"),
        "resumed_checkpoint_sha256": digest(resumed_root / "eventgenesis/last.pt"),
        "all_training_data_synthetic": True, "real_data_used": False,
    }
    out = args.output_root / "resume_equivalence.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
