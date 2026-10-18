#!/usr/bin/env python3
"""Synthetic-only Stage-4 E2VID+ training with real process-boundary resume.

The pinned E2VID model/loader/losses remain unchanged. Spatial transforms are
seeded jointly across event tensors, target images, displacement and validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
STOP_REQUESTED = False


def _request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_count(value) -> int:
    if torch.is_tensor(value):
        return 1
    if isinstance(value, dict):
        return sum(tensor_count(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_count(v) for v in value)
    return 0


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value: dict) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    # Checkpoints are loaded with map_location=device; explicitly move RNG
    # ByteTensors back to CPU because torch.set_rng_state requires CPU state.
    torch.set_rng_state(value["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in value["torch_cuda"]])


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sample_digest(sequence: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in sequence:
        for key in ("events", "frame", "flow", "flow_valid"):
            value = item[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().contiguous().numpy()
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value).view(np.uint8))
        digest.update(str(item["sample_id"]).encode())
    return digest.hexdigest()


def scalar_losses(losses: dict) -> dict[str, float]:
    return {key: float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            for key, v in losses.items()}


def optimizer_step_counts(optimizer) -> list[int]:
    return sorted(int(state["step"].item()) for state in optimizer.state.values() if "step" in state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--generator", choices=("eventgenesis", "v2e"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=0,
                        help="stop on an optimizer boundary; zero means the configured full budget")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.max_updates < 0:
        raise ValueError("max-updates must be >= 0")
    model_root, data_root = args.model_root.resolve(), args.data_root.resolve()
    config_path, run_root = args.config.resolve(), args.output_root.resolve()
    if not model_root.is_dir() or not data_root.is_dir() or not config_path.is_file():
        raise FileNotFoundError("model, data, or config path missing")
    run_root.mkdir(parents=True, exist_ok=True)
    trainer_script_sha = sha256(Path(__file__).resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered E2VID+ path")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    sys.path.insert(0, str(model_root))
    sys.path.insert(0, str(model_root / "PerceptualSimilarity"))
    import skimage.measure
    from skimage.metrics import structural_similarity
    if not hasattr(skimage.measure, "compare_ssim"):
        skimage.measure.compare_ssim = structural_similarity
    try:
        import IPython  # noqa: F401
    except ImportError:
        stub = types.ModuleType("IPython")
        stub.embed = lambda *unused_args, **unused_kwargs: None
        sys.modules["IPython"] = stub

    from data_loader import dataset as dataset_module
    import data_loader.data_loaders as data_loaders_module
    from data_loader.data_loaders import HDF5DataLoader
    from data_loader.dataset import DynamicH5Dataset, SequenceDataset
    from model.loss import perceptual_loss, temporal_consistency_loss
    from model.model import E2VIDRecurrent
    from trainer.trainer import Trainer
    from utils.data import data_sources

    class Stage4SequenceDataset(SequenceDataset):
        """Permit one full-rollout validation sequence at an exact-length boundary.

        The pinned upstream SequenceDataset returns length zero when
        ``sequence_length >= dataset.length``. For this fixed validation case,
        one sequence of 200 samples is well-defined: sample indices 0..199
        cover all 200 labelled 10 ms intervals without using the terminal
        duplicate image at index 200. Keep the upstream train behavior intact.
        """

        def __init__(self, *sequence_args, **sequence_kwargs):
            super().__init__(*sequence_args, **sequence_kwargs)
            self.stage4_full_rollout_override = False
            if self.L == 200:
                if self.dataset.length != 200 or self.step_size != 200 or self.length != 0:
                    raise RuntimeError(
                        "unexpected upstream full-rollout validation boundary: "
                        f"dataset_length={self.dataset.length}, step={self.step_size}, sequences={self.length}"
                    )
                self.length = 1
                self.stage4_full_rollout_override = True

    data_loaders_module.SequenceDataset = Stage4SequenceDataset

    config = json.loads(config_path.read_text())
    cfg = config["reconstruction"]
    if cfg["model"] != "pinned_E2VIDRecurrent":
        raise RuntimeError("unexpected reconstruction model")
    if (int(cfg["sequence_length"]), *map(int, cfg["train_crop_hw"]),
            int(cfg["effective_batch_clips"]), int(cfg["epochs"])) != (40, 112, 112, 2, 200):
        raise RuntimeError("reconstruction configuration differs from the registered recipe")
    total_budget = int(cfg["expected_optimizer_updates"])
    updates_per_epoch = int(cfg["expected_updates_per_epoch"])
    if (total_budget, updates_per_epoch) != (12000, 60):
        raise RuntimeError("unexpected Stage-4 E2VID optimizer budget")
    data_manifest_path = data_root / "adapter_manifest.json"
    if not data_manifest_path.is_file():
        raise FileNotFoundError(data_manifest_path)
    data_manifest = json.loads(data_manifest_path.read_text())
    data_manifest_sha = sha256(data_manifest_path)
    config_sha = sha256(config_path)
    if data_manifest.get("real_data_used") is not False or data_manifest.get("real_scoring_enabled") is not False:
        raise RuntimeError("real-data training/scoring guard is not closed")
    arm_rows = [r for r in data_manifest["rollout_rows"] if r.get("generator") == args.generator]
    if len(arm_rows) != 32:
        raise RuntimeError(f"expected 32 {args.generator} rollouts in adapter manifest, got {len(arm_rows)}")
    train_list = data_root / f"{args.generator}_train_files.txt"
    val_list = data_root / f"{args.generator}_validation_files.txt"
    if not train_list.is_file() or not val_list.is_file():
        raise FileNotFoundError("registered train/validation file lists missing")

    class Stage4E2VIDDataset(DynamicH5Dataset):
        def load_data(self, data_path):
            self.h5_file = h5py.File(data_path, "r")
            self.sensor_resolution = (self.h5_file.attrs["sensor_resolution"][0:2]
                                      if self.sensor_resolution is None else self.sensor_resolution[0:2])
            self.has_flow = "flow" in self.h5_file and len(self.h5_file["flow"]) > 0
            event_ts = np.asarray(self.h5_file["events/ts"], dtype=np.float64)
            self.num_events = int(self.h5_file.attrs["num_events"])
            self.t0 = float(event_ts[0]) if len(event_ts) else 0.0
            self.tk = float(event_ts[-1]) if len(event_ts) else 0.0
            self.num_frames = int(self.h5_file.attrs["num_imgs"])
            names = sorted(self.h5_file["images"].keys())
            self.frame_ts = [float(self.h5_file[f"images/{name}"].attrs["timestamp"]) for name in names]
            try:
                self.data_source_idx = data_sources.index(self.h5_file.attrs.get("source", "unknown"))
            except ValueError:
                self.data_source_idx = -1
            self.rollout_id = str(self.h5_file.attrs["sim2real_episode_id"])

        def compute_frame_indices(self):
            timestamps = np.asarray(self.h5_file["events/ts"], dtype=np.float64)
            result, start = [], 0
            for timestamp in self.frame_ts:
                end = int(np.searchsorted(timestamps, timestamp, side="right"))
                result.append([start, end])
                start = end
            return result

        def get_flow_valid(self, index):
            return np.asarray(self.h5_file[f"flow_valid/flow_valid{index:09d}"], dtype=np.uint8)

        def __getitem__(self, index, seed=None):
            item = super().__getitem__(index, seed=seed)
            seed = random.randint(0, 2**32) if seed is None else seed
            item["flow"] = self.transform_flow(np.asarray(self.get_flow(index), dtype=np.float32), seed)
            valid = torch.from_numpy(self.get_flow_valid(index).astype(np.float32)[None])
            if self.transform:
                random.seed(seed)
                valid = self.transform(valid, is_flow=False)
            item["flow_valid"] = valid[0] > 0.5
            item["sample_id"] = f"{self.rollout_id}:{index:03d}"
            item["rollout_id"] = self.rollout_id
            return item

    dataset_module.Stage4E2VIDDataset = Stage4E2VIDDataset
    from utils.data_augmentation import CenterCrop, RandomCrop, RandomFlip  # noqa: F401

    common = {
        "dataset_type": "Stage4E2VIDDataset", "proba_pause_when_running": 0.0,
        "proba_pause_when_paused": 0.0, "normalize_image": False,
        "noise_kwargs": {}, "hot_pixel_kwargs": {},
        "dataset_kwargs": {"num_bins": 5, "combined_voxel_channels": True, "filter_hot_events": False},
    }
    train_kwargs = {**common, "sequence_length": 40, "step_size": 40}
    train_kwargs["dataset_kwargs"] = {
        **common["dataset_kwargs"],
        "transforms": {"RandomCrop": {"size": [112, 112]},
                       "RandomFlip": {"p_hflip": 0.5, "p_vflip": 0.5}},
    }
    val_kwargs = {**common, "sequence_length": 200, "step_size": 200}
    val_kwargs["dataset_kwargs"] = {
        **common["dataset_kwargs"], "transforms": {"CenterCrop": {"size": [160, 160]}},
    }
    train_loader = HDF5DataLoader(str(train_list), batch_size=1, shuffle=False, num_workers=0,
                                  pin_memory=True, sequence_kwargs=train_kwargs)
    val_loader = HDF5DataLoader(str(val_list), batch_size=1, shuffle=False, num_workers=0,
                                pin_memory=True, sequence_kwargs=val_kwargs)
    if len(train_loader.dataset) != 120 or len(val_loader.dataset) != 8:
        raise RuntimeError(f"expected 120 train clips / 8 validation rollouts, got {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    validation_sequence = val_loader.dataset[0]
    validation_ids = [item["sample_id"] for item in validation_sequence]
    if (len(validation_sequence) != 200 or validation_ids[0].rsplit(":", 1)[-1] != "000"
            or validation_ids[-1].rsplit(":", 1)[-1] != "199"
            or len({item["rollout_id"] for item in validation_sequence}) != 1):
        raise RuntimeError(
            "full-rollout validation probe must cover one rollout's 200 labelled intervals 000..199"
        )
    atomic_json(run_root / "loader_preflight.json", {
        "status": "pass", "train_clips": len(train_loader.dataset),
        "validation_rollouts": len(val_loader.dataset), "validation_sequence_length": len(validation_sequence),
        "validation_first_sample_id": validation_ids[0], "validation_last_sample_id": validation_ids[-1],
        "upstream_exact_length_boundary_overridden": True,
        "rule": "for L=200 and dataset.length=200 only, expose one sequence of indices 0..199; leave terminal duplicate frame index 200 unused",
        "real_data_used": False,
    })

    model_args = dict(cfg["model_args"])
    if int(model_args["num_bins"]) != 5:
        raise RuntimeError("E2VID network must consume five bins")
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = E2VIDRecurrent(model_args).to(device)
    initial_hash = state_sha256(model)
    optim_cfg = cfg["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(optim_cfg["lr"]), betas=tuple(optim_cfg["betas"]),
        weight_decay=float(optim_cfg["weight_decay"]), amsgrad=bool(optim_cfg["amsgrad"]),
    )
    perceptual = perceptual_loss(weight=float(cfg["loss"]["LPIPS_VGG_weight"]), net="vgg", use_gpu=True)
    for parameter in perceptual.model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in perceptual.model.parameters()):
        raise RuntimeError("LPIPS-VGG features must remain frozen")
    temporal = temporal_consistency_loss(weight=float(cfg["loss"]["temporal_weight"]),
                                         L0=int(cfg["loss"]["temporal_L0"]))
    temporal.calls, temporal.values = 0, []
    original_temporal_call = temporal_consistency_loss.__call__

    def recording_temporal_call(self, i, image, processed, flow, output_images=False):
        value = original_temporal_call(self, i, image, processed, flow, output_images)
        if value is not None and not output_images:
            self.calls += 1
            self.values.append(float(value.detach().cpu()))
        return value
    temporal_consistency_loss.__call__ = recording_temporal_call
    losses = [perceptual, temporal]
    shim = SimpleNamespace(model=model, loss_ftns=losses, device=device)
    shim.to_device = lambda item: Trainer.to_device(shim, item)
    shim.forward_sequence = lambda sequence: Trainer.forward_sequence(shim, sequence)

    rollout_files = sorted((data_root / args.generator / "train").glob("*.h5"))
    val_files = sorted((data_root / args.generator / "validation").glob("*.h5"))
    train_ids, val_ids = [p.stem for p in rollout_files], [p.stem for p in val_files]
    if len(train_ids) != 24 or len(val_ids) != 8 or set(train_ids) & set(val_ids):
        raise RuntimeError("expected disjoint whole-rollout split 24/8")
    clip_ids = [(path.stem, clip) for path in rollout_files for clip in range(5)]
    if len(clip_ids) != 120:
        raise RuntimeError("train sampler is not 24 rollouts × 5 non-overlapping clips")

    next_epoch, next_batch, update = 0, 0, 0
    best_lpips, best_epoch = float("inf"), None
    resume_path = args.resume.resolve() if args.resume else None
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt.get("schema") != "e2vid_stage4_resume_v1":
            raise RuntimeError("unrecognized resume checkpoint")
        if (ckpt.get("config_sha256"), ckpt.get("data_manifest_sha256")) != (config_sha, data_manifest_sha):
            raise RuntimeError("resume code/config/data fingerprints do not match")
        if (ckpt.get("generator"), ckpt.get("seed")) != (args.generator, args.seed):
            raise RuntimeError("resume generator/seed does not match")
        saved_script_sha = ckpt.get("trainer_script_sha256")
        if saved_script_sha and saved_script_sha != trainer_script_sha:
            raise RuntimeError(
                f"resume trainer fingerprint differs: {saved_script_sha} != {trainer_script_sha}"
            )
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        if state_sha256(model) != ckpt["model_sha256"]:
            raise RuntimeError("restored E2VID model-state hash mismatch")
        update, next_epoch, next_batch = int(ckpt["optimizer_update"]), int(ckpt["next_epoch"]), int(ckpt["next_batch_in_epoch"])
        best_lpips, best_epoch = float(ckpt["best_synthetic_validation_lpips_vgg"]), ckpt.get("best_epoch")
        restore_rng(ckpt["rng_state"])

    def get_batch(index):
        return default_collate([train_loader.dataset[index]])

    resume_oracle = None
    if resume_path:
        saved_rng = rng_state()
        order = np.random.default_rng(args.seed + next_epoch).permutation(120).tolist()
        indices = order[next_batch * 2:(next_batch + 1) * 2]
        if len(indices) != 2:
            raise RuntimeError("resume checkpoint has no complete next effective batch")
        probe_hashes, probe_losses = [], []
        model.train()
        with torch.no_grad():
            for index in indices:
                seq = get_batch(index)
                probe_hashes.append(sample_digest(seq))
                probe_losses.append(scalar_losses(shim.forward_sequence(seq))["loss"])
        restore_rng(saved_rng)
        resume_oracle = {
            "status": "pending_first_resumed_update", "checkpoint_path": str(resume_path),
            "checkpoint_sha256": sha256(resume_path), "source_update": update,
            "source_checkpoint_trainer_script_sha256": ckpt.get("trainer_script_sha256"),
            "resumed_trainer_script_sha256": trainer_script_sha,
            "next_epoch_zero_based": next_epoch, "next_batch_in_epoch": next_batch,
            "next_clip_indices": indices, "probe_sample_sha256": probe_hashes,
            "probe_loss_components_total": probe_losses, "rng_restored_before_training": True,
            "restored_model_sha256": state_sha256(model),
            "restored_optimizer_step_counts": optimizer_step_counts(optimizer),
            "real_data_used": False,
        }
        atomic_json(run_root / "resume_oracle.json", resume_oracle)

    def recurrent_state_fixture():
        sequence = get_batch(0)
        a, b = sequence[0]["events"].to(device), sequence[1]["events"].to(device)
        model.train(); model.reset_states()
        with torch.no_grad():
            y0 = model(a)["image"].detach().clone()
            n0 = tensor_count(model.states)
            y1 = model(b)["image"].detach().clone()
            n1 = tensor_count(model.states)
            model.reset_states()
            y0_replay = model(a)["image"].detach().clone()
        record = {"state_tensors_after_first_step": n0, "state_tensors_after_second_step": n1,
                  "continues_within_clip": n0 > 0 and n1 == n0,
                  "reset_first_step_max_abs_error": float((y0 - y0_replay).abs().max().cpu()),
                  "distinct_step_output_mean_abs_delta": float((y0 - y1).abs().mean().cpu())}
        if not record["continues_within_clip"] or record["reset_first_step_max_abs_error"] > 1e-7:
            raise RuntimeError(f"recurrent-state continuation/reset preflight failed: {record}")
        return record

    if update == 0:
        atomic_json(run_root / "recurrent_state_preflight.json", {"status": "pass", **recurrent_state_fixture()})

    metrics_path = run_root / "training_metrics.jsonl"
    metrics = metrics_path.open("a" if resume_path and metrics_path.exists() else "w")
    signal.signal(signal.SIGTERM, _request_stop); signal.signal(signal.SIGINT, _request_stop)
    run_started_utc = datetime.now(timezone.utc).isoformat()
    run_started = time.monotonic()

    def save_checkpoint(epoch_cursor: int, batch_cursor: int, event: str) -> dict:
        data = {
            "schema": "e2vid_stage4_resume_v1", "generator": args.generator, "seed": args.seed,
            "model": model.state_dict(), "model_sha256": state_sha256(model),
            "optimizer": optimizer.state_dict(), "optimizer_step_counts": optimizer_step_counts(optimizer),
            "optimizer_update": update, "next_epoch": epoch_cursor, "next_batch_in_epoch": batch_cursor,
            "best_synthetic_validation_lpips_vgg": best_lpips, "best_epoch": best_epoch,
            "rng_state": rng_state(), "config_sha256": config_sha, "data_manifest_sha256": data_manifest_sha,
            "trainer_script_sha256": trainer_script_sha,
            "train_rollouts": train_ids, "validation_rollouts": val_ids,
            "event": event, "saved_utc": datetime.now(timezone.utc).isoformat(),
        }
        path = run_root / "last.pt"
        temporary = run_root / "last.pt.partial"
        torch.save(data, temporary); os.replace(temporary, path)
        if update == 20 and event in ("first_20_update_checkpoint", "training_boundary_checkpoint"):
            import shutil
            shutil.copy2(path, run_root / "update_000020.pt")
        return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path),
                "model_sha256": data["model_sha256"]}

    atomic_json(run_root / "live_status.json", {
        "status": "running", "generator": args.generator, "seed": args.seed,
        "optimizer_update": update, "optimizer_budget": total_budget,
        "epoch": next_epoch + 1, "batch_in_epoch": next_batch,
        "started_utc": run_started_utc, "device": str(device),
        "resume_from": str(resume_path) if resume_path else None,
        "config_sha256": config_sha, "data_manifest_sha256": data_manifest_sha,
        "trainer_script_sha256": trainer_script_sha,
        "real_scoring_enabled": False,
    })

    try:
        for epoch in range(next_epoch, int(cfg["epochs"])):
            order = np.random.default_rng(args.seed + epoch).permutation(120).tolist()
            first_batch = next_batch if epoch == next_epoch else 0
            for batch_idx in range(first_batch, updates_per_epoch):
                indices = order[batch_idx * 2:(batch_idx + 1) * 2]
                if len(indices) != 2:
                    raise RuntimeError("effective batch size is not two clips")
                optimizer.zero_grad(set_to_none=True)
                rows, sample_hashes, valid_fractions = [], [], []
                calls_before = temporal.calls
                tc_grad_norm = None
                for micro_idx, clip_index in enumerate(indices):
                    sequence = get_batch(clip_index)
                    if len(sequence) != 40 or tuple(sequence[0]["events"].shape) != (1, 5, 112, 112):
                        raise RuntimeError("wrong E2VID temporal or spatial input shape")
                    sample_hashes.append(sample_digest(sequence))
                    valid_fractions.append(float(torch.stack([x["flow_valid"].float().mean() for x in sequence]).mean()))
                    values = shim.forward_sequence(sequence)
                    if not torch.isfinite(values["loss"]):
                        raise RuntimeError(f"non-finite loss at update {update + 1}")
                    tc_keys = [k for k in values if k.startswith("temporal_consistency_loss/")]
                    if not tc_keys:
                        raise RuntimeError("TC did not execute in a 40-frame training clip")
                    if update == 0 and micro_idx == 0:
                        grads = torch.autograd.grad(values[tc_keys[0]], model.parameters(), retain_graph=True, allow_unused=True)
                        tc_grad_norm = float(sum(float((g.detach().float() ** 2).sum().cpu()) for g in grads if g is not None) ** 0.5)
                        if not np.isfinite(tc_grad_norm) or tc_grad_norm <= 0:
                            raise RuntimeError("TC objective has no finite, nonzero gradient to E2VID")
                    (values["loss"] / 2.0).backward()
                    rows.append(scalar_losses(values))
                grad_sq, grad_tensors = 0.0, 0
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("non-finite E2VID gradient")
                    grad_sq += float((parameter.grad.detach().float() ** 2).sum().cpu())
                    grad_tensors += 1
                grad_norm = grad_sq ** 0.5
                if grad_tensors == 0 or not np.isfinite(grad_norm) or grad_norm <= 0:
                    raise RuntimeError("optimizer update has no finite nonzero gradient")
                optimizer.step(); update += 1
                if resume_oracle is not None and update == resume_oracle["source_update"] + 1:
                    if sample_hashes != resume_oracle["probe_sample_sha256"]:
                        raise RuntimeError("resumed next batch differs from saved-cursor oracle")
                    actual_losses = [r["loss"] for r in rows]
                    if not np.allclose(actual_losses, resume_oracle["probe_loss_components_total"], rtol=1e-5, atol=1e-6):
                        raise RuntimeError(f"resumed forward losses mismatch: {actual_losses}")
                    resume_oracle.update({"status": "pass", "observed_next_update": update,
                                          "observed_sample_sha256": sample_hashes,
                                          "observed_losses": actual_losses,
                                          "optimizer_step_counts_after_next_update": optimizer_step_counts(optimizer)})
                    atomic_json(run_root / "resume_oracle.json", resume_oracle)
                means = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
                epoch_cursor, batch_cursor = epoch, batch_idx + 1
                if batch_cursor == updates_per_epoch:
                    epoch_cursor, batch_cursor = epoch + 1, 0
                record = {
                    "event": "optimizer_update", "generator": args.generator, "seed": args.seed,
                    "optimizer_update": update, "epoch": epoch + 1, "batch_in_epoch": batch_idx + 1,
                    "clip_indices": indices, "clip_ids": [f"{clip_ids[i][0]}:{clip_ids[i][1]:03d}" for i in indices],
                    "sample_sha256": sample_hashes, "loss_components": means,
                    "temporal_consistency_valid_calls": temporal.calls - calls_before,
                    "temporal_consistency_first_update_gradient_l2": tc_grad_norm,
                    "flow_valid_fraction_mean": float(np.mean(valid_fractions)),
                    "gradient_l2_norm": grad_norm, "finite_gradient_tensors": grad_tensors,
                    "recurrent_state_tensor_count": tensor_count(model.states),
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                }
                metrics.write(json.dumps(record, sort_keys=True) + "\n"); metrics.flush()
                checkpoint_info = None
                if update == 20 or update % 10 == 0:
                    if update == 20 or update % 300 == 0:
                        label = "first_20_update_checkpoint" if update == 20 else "periodic_training_checkpoint"
                        checkpoint_info = save_checkpoint(epoch_cursor, batch_cursor, label)
                    atomic_json(run_root / "live_status.json", {
                        "status": "running", "generator": args.generator, "seed": args.seed,
                        "optimizer_update": update, "optimizer_budget": total_budget,
                        "epoch": epoch + 1, "batch_in_epoch": batch_idx + 1,
                        "latest_loss_components": means,
                        "tc_valid_calls_last_update": record["temporal_consistency_valid_calls"],
                        "gradient_l2_norm": grad_norm, "finite_gradient_tensors": grad_tensors,
                        "last_checkpoint": checkpoint_info, "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "real_scoring_enabled": False, "trainer_script_sha256": trainer_script_sha,
                    })
                    print(json.dumps(record, sort_keys=True), flush=True)
                if (args.max_updates and update >= args.max_updates) or update >= total_budget or STOP_REQUESTED:
                    save_checkpoint(epoch_cursor, batch_cursor, "training_boundary_checkpoint")
                    next_epoch, next_batch = epoch_cursor, batch_cursor
                    break
            epoch_complete = next_batch == 0 and next_epoch == epoch + 1
            if not epoch_complete and ((args.max_updates and update >= args.max_updates) or update >= total_budget or STOP_REQUESTED):
                break
            next_epoch, next_batch = epoch + 1, 0
            if next_epoch % int(cfg["validate_every_epochs"]) == 0 or next_epoch == int(cfg["epochs"]):
                model.eval()
                val_rows = []
                with torch.no_grad():
                    for val_index in range(len(val_loader.dataset)):
                        sequence = default_collate([val_loader.dataset[val_index]])
                        values = scalar_losses(shim.forward_sequence(sequence))
                        if not np.isfinite(values["loss"]):
                            raise RuntimeError("non-finite synthetic validation loss")
                        val_rows.append({"rollout_id": sequence[-1]["rollout_id"][0], "components": values})
                lpips_key = next((k for k in val_rows[0]["components"] if k.startswith("perceptual_loss/")), None)
                tc_key = next((k for k in val_rows[0]["components"] if k.startswith("temporal_consistency_loss/")), None)
                if lpips_key is None or tc_key is None or len(val_rows) != 8:
                    raise RuntimeError("validation rollout/loss contract mismatch")
                lpips = float(np.mean([row["components"][lpips_key] for row in val_rows]))
                tc_val = float(np.mean([row["components"][tc_key] for row in val_rows]))
                if lpips < best_lpips:
                    best_lpips, best_epoch = lpips, next_epoch
                    save_checkpoint(next_epoch, 0, "synthetic_validation_best")
                    import shutil
                    shutil.copy2(run_root / "last.pt", run_root / "synthetic_best.pt")
                    atomic_json(run_root / "synthetic_best.json", {
                        "epoch": best_epoch, "optimizer_update": update,
                        "synthetic_validation_lpips_vgg": lpips,
                        "synthetic_validation_temporal_consistency": tc_val,
                        "checkpoint": str(run_root / "synthetic_best.pt"),
                        "size_bytes": (run_root / "synthetic_best.pt").stat().st_size,
                        "sha256": sha256(run_root / "synthetic_best.pt"),
                        "selection_source": "synthetic validation only",
                    })
                metrics.write(json.dumps({
                    "event": "synthetic_validation", "generator": args.generator, "seed": args.seed,
                    "epoch": next_epoch, "optimizer_update": update,
                    "lpips_vgg_equal_rollout_mean": lpips,
                    "temporal_consistency_equal_rollout_mean": tc_val,
                    "rollouts": val_rows, "best_epoch": best_epoch, "best_lpips_vgg": best_lpips,
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                }, sort_keys=True) + "\n"); metrics.flush()
                model.train()
                save_checkpoint(next_epoch, 0, "epoch_boundary")
                atomic_json(run_root / "live_status.json", {
                    "status": "running", "generator": args.generator, "seed": args.seed,
                    "optimizer_update": update, "optimizer_budget": total_budget, "epoch": next_epoch,
                    "best_epoch": best_epoch, "best_synthetic_validation_lpips_vgg": best_lpips,
                    "latest_validation_lpips_vgg": lpips, "latest_validation_temporal_consistency": tc_val,
                    "updated_utc": datetime.now(timezone.utc).isoformat(), "real_scoring_enabled": False,
                })
            if STOP_REQUESTED or (args.max_updates and update >= args.max_updates) or update >= total_budget:
                break
        complete = update >= total_budget
        summary = {
            "status": "complete" if complete else ("interrupted_at_safe_boundary" if STOP_REQUESTED else "stopped_at_requested_boundary"),
            "generator": args.generator, "seed": args.seed, "optimizer_update": update,
            "optimizer_budget": total_budget, "completed_epochs": update // updates_per_epoch,
            "best_epoch": best_epoch,
            "best_synthetic_validation_lpips_vgg": best_lpips if math.isfinite(best_lpips) else None,
            "final_checkpoint": str(run_root / "last.pt"), "final_checkpoint_sha256": sha256(run_root / "last.pt"),
            "started_utc": run_started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.monotonic() - run_started, "real_scoring_enabled": False,
            "initial_model_sha256": initial_hash, "config_sha256": config_sha,
            "trainer_script_sha256": trainer_script_sha,
            "data_manifest_sha256": data_manifest_sha, "resume_from": str(resume_path) if resume_path else None,
            "resume_oracle": resume_oracle,
        }
        atomic_json(run_root / "live_status.json", summary)
        atomic_json(run_root / "run_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        metrics.close()


if __name__ == "__main__":
    main()
