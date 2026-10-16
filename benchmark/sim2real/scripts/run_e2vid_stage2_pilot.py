#!/usr/bin/env python3
"""Multi-rollout, synthetic-only E2VID+ pilot with audited temporal loss."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch


MODEL_ARGS = {
    "num_bins": 5,
    "skip_type": "sum",
    "recurrent_block_type": "convlstm",
    "num_encoders": 3,
    "base_num_channels": 32,
    "num_residual_blocks": 2,
    "use_upsample_conv": True,
    "final_activation": "",
    "norm": "none",
}
SEQUENCE_LENGTH = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_state_count(value) -> int:
    if torch.is_tensor(value):
        return 1
    if isinstance(value, (list, tuple)):
        return sum(tensor_state_count(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_state_count(item) for item in value.values())
    return 0


def run_boundary_fixtures(dataset_class, output_root: Path) -> dict:
    expected_flow = np.zeros((2, 8, 8), dtype=np.float32)
    expected_flow[0] = 2.0
    expected_flow[1] = -1.0
    cases = {
        "empty": np.empty((0, 4), dtype=np.float64),
        "sparse_two": np.asarray([[0.002, 1, 2, 1], [0.008, 3, 4, 0]], dtype=np.float64),
        "tied_three": np.asarray([[0.005, 1, 1, 1], [0.005, 2, 2, 1], [0.005, 3, 3, 0]], dtype=np.float64),
    }
    observations = {}
    flow_hashes = set()
    with tempfile.TemporaryDirectory(prefix="eventgenesis-e2vid-boundaries-") as temp_root:
        for case, events in cases.items():
            path = Path(temp_root) / f"{case}.h5"
            with h5py.File(path, "w") as out:
                out.attrs["sensor_resolution"] = np.asarray([8, 8], dtype=np.int32)
                out.attrs["num_events"] = np.int64(len(events))
                out.attrs["num_imgs"] = np.int64(2)
                out.attrs["source"] = "unknown"
                out.attrs["sim2real_episode_id"] = case
                event_group = out.create_group("events")
                event_group.create_dataset("xs", data=events[:, 1].astype(np.uint16))
                event_group.create_dataset("ys", data=events[:, 2].astype(np.uint16))
                event_group.create_dataset("ts", data=events[:, 0].astype(np.float64))
                event_group.create_dataset("ps", data=events[:, 3].astype(np.uint8))
                images = out.create_group("images")
                for index in range(2):
                    image = images.create_dataset(f"image{index:09d}", data=np.zeros((8, 8), dtype=np.uint8))
                    image.attrs["timestamp"] = np.float64(0.01)
                    image.attrs["event_idx"] = np.int64(len(events))
                flow_group = out.create_group("flow")
                flow_group.create_dataset("flow000000000", data=expected_flow)
                valid_group = out.create_group("flow_valid")
                valid_group.create_dataset("flow_valid000000000", data=np.ones((8, 8), dtype=np.uint8))
            dataset = dataset_class(
                str(path), transforms={}, num_bins=5,
                combined_voxel_channels=True, filter_hot_events=False,
            )
            sample = dataset[0]
            flow = sample["flow"].detach().cpu().numpy()
            voxel = sample["events"].detach().cpu().numpy()
            if not np.array_equal(flow, expected_flow):
                raise RuntimeError(f"event span changed shared physical flow label for {case}")
            if not np.isfinite(voxel).all():
                raise RuntimeError(f"non-finite event representation for {case}")
            if case != "tied_three" and np.count_nonzero(voxel) != 0:
                raise RuntimeError(f"expected pinned empty voxel behavior for {case}")
            if case == "tied_three" and not np.isclose(np.abs(voxel).sum(), 3.0):
                raise RuntimeError("tied-time center-bin fallback did not preserve event polarities")
            flow_hash = hashlib.sha256(np.ascontiguousarray(flow).view(np.uint8)).hexdigest()
            flow_hashes.add(flow_hash)
            observations[case] = {
                "raw_event_count": int(len(events)),
                "observed_event_span_s": float(events[:, 0].max() - events[:, 0].min()) if len(events) else 0.0,
                "voxel_finite": bool(np.isfinite(voxel).all()),
                "voxel_nonzero_values": int(np.count_nonzero(voxel)),
                "voxel_abs_sum": float(np.abs(voxel).sum()),
                "flow_label_sha256": flow_hash,
                "flow_valid_pixels": int(sample["flow_valid"].sum()),
            }
            dataset.h5_file.close()
    if len(flow_hashes) != 1:
        raise RuntimeError("boundary fixture flow labels changed with event count/span")
    result = {
        "status": "pass",
        "cases": observations,
        "identical_shared_flow_hash_across_empty_sparse_tied_events": True,
        "timestamp_tie_policy": "searchsorted(side=right) assigns all events at a boundary to that right endpoint; constant-time events use center-bin accumulation",
    }
    (output_root / "e2vid_boundary_fixtures.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def loss_scalars(losses: dict) -> dict[str, float]:
    result = {}
    for key, value in losses.items():
        if torch.is_tensor(value):
            result[key] = float(value.detach().cpu())
        else:
            result[key] = float(value)
    return result


def run_flow_direction_fixture(temporal_loss_class) -> dict:
    """Verify forward (+x) labels align a right-moving image under the pinned TC warp."""
    previous = torch.zeros((1, 1, 5, 7), dtype=torch.float32)
    current = torch.zeros_like(previous)
    previous[0, 0, 2, 2] = 1.0
    current[0, 0, 2, 3] = 1.0
    forward_flow = torch.zeros((1, 2, 5, 7), dtype=torch.float32)
    forward_flow[:, 0] = 1.0

    def measure(flow: torch.Tensor) -> dict[str, float]:
        objective = temporal_loss_class(weight=1.0, L0=1)
        objective(0, previous, previous, flow)
        value, maps = objective(1, current, current, flow, output_images=True)
        warp_error = float(torch.abs(maps["processed0_warped_to1"] - current).mean())
        return {"temporal_loss": float(value.detach()), "unmasked_warp_l1": warp_error}

    forward = measure(forward_flow)
    reversed_flow = measure(-forward_flow)
    if not forward["unmasked_warp_l1"] < reversed_flow["unmasked_warp_l1"]:
        raise RuntimeError(f"E2VID temporal warp direction check failed: {forward} vs {reversed_flow}")
    return {
        "status": "pass",
        "scene_motion": "one-pixel rightward translation",
        "label_contract": "forward source-to-target displacement (+1 px in x)",
        "forward_label": forward,
        "reversed_label": reversed_flow,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    model_root = args.model_root.resolve()
    sys.path.insert(0, str(model_root))
    sys.path.insert(0, str(model_root / "PerceptualSimilarity"))
    import skimage.measure
    from skimage.metrics import structural_similarity
    if not hasattr(skimage.measure, "compare_ssim"):
        skimage.measure.compare_ssim = structural_similarity
    try:
        import IPython  # noqa: F401
    except ImportError:
        ipython_stub = types.ModuleType("IPython")
        ipython_stub.embed = lambda *unused_args, **unused_kwargs: None
        sys.modules["IPython"] = ipython_stub

    from data_loader import dataset as dataset_module
    from data_loader.data_loaders import HDF5DataLoader
    from data_loader.dataset import DynamicH5Dataset
    from model.loss import perceptual_loss, temporal_consistency_loss
    from model.model import E2VIDRecurrent
    from trainer.trainer import Trainer
    from utils.data import data_sources

    original_temporal_call = temporal_consistency_loss.__call__

    def recording_temporal_call(self, i, image, processed, flow, output_images=False):
        value = original_temporal_call(self, i, image, processed, flow, output_images)
        if value is not None and not output_images and hasattr(self, "calls"):
            self.calls += 1
            self.values.append(float(value.detach().cpu()))
        return value

    # Trainer dispatches by the pinned loss class name, so instrument its
    # method in-memory while retaining the exact official forward/recipe.
    temporal_consistency_loss.__call__ = recording_temporal_call

    class Stage2RightEndpointH5Dataset(DynamicH5Dataset):
        """Right-tie indexing, empty-event support, and fixed physical flow labels."""

        def load_data(self, data_path):
            self.h5_file = h5py.File(data_path, "r")
            if self.sensor_resolution is None:
                self.sensor_resolution = self.h5_file.attrs["sensor_resolution"][0:2]
            else:
                self.sensor_resolution = self.sensor_resolution[0:2]
            self.has_flow = "flow" in self.h5_file and len(self.h5_file["flow"]) > 0
            event_ts = np.asarray(self.h5_file["events/ts"], dtype=np.float64)
            self.num_events = int(self.h5_file.attrs["num_events"])
            self.t0 = float(event_ts[0]) if len(event_ts) else 0.0
            self.tk = float(event_ts[-1]) if len(event_ts) else 0.0
            self.num_frames = int(self.h5_file.attrs["num_imgs"])
            self.frame_ts = [
                float(self.h5_file[f"images/{name}"].attrs["timestamp"])
                for name in sorted(self.h5_file["images"].keys())
            ]
            source = self.h5_file.attrs.get("source", "unknown")
            try:
                self.data_source_idx = data_sources.index(source)
            except ValueError:
                self.data_source_idx = -1
            self.rollout_id = str(self.h5_file.attrs.get("sim2real_episode_id", "unknown"))

        def compute_frame_indices(self):
            timestamps = np.asarray(self.h5_file["events/ts"], dtype=np.float64)
            indices = []
            start = 0
            for timestamp in self.frame_ts:
                end = int(np.searchsorted(timestamps, timestamp, side="right"))
                indices.append([start, end])
                start = end
            return indices

        def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
            # The pinned voxelizer normalizes by timestamp span. The tied-time
            # limit is represented by depositing all events into the center bin.
            if len(ts) > 1 and float((ts.max() - ts.min()).item()) == 0.0:
                if not combined_voxel_channels:
                    return super().get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels)
                voxel = torch.zeros((self.num_bins, *self.sensor_resolution), dtype=torch.float32)
                channel = self.num_bins // 2
                voxel[channel].index_put_(
                    (ys.to(torch.int64), xs.to(torch.int64)), ps.to(torch.float32), accumulate=True
                )
                return voxel
            return super().get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels)

        def get_flow_valid(self, index):
            return np.asarray(self.h5_file[f"flow_valid/flow_valid{index:09d}"], dtype=np.uint8)

        def __getitem__(self, index, seed=None):
            # Parent conversion multiplies flow by the observed event span.
            # Restore the shared per-interval physical displacement directly.
            item = super().__getitem__(index, seed=seed)
            seed = random.randint(0, 2**32) if seed is None else seed
            displacement = np.asarray(self.get_flow(index), dtype=np.float32)
            item["flow"] = self.transform_flow(displacement, seed)
            item["flow_valid"] = torch.from_numpy(self.get_flow_valid(index).astype(bool))
            item["rollout_id"] = self.rollout_id
            return item

    dataset_module.Stage2RightEndpointH5Dataset = Stage2RightEndpointH5Dataset
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback is used for this pilot")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    boundary_evidence = run_boundary_fixtures(Stage2RightEndpointH5Dataset, args.output_root)
    flow_direction_evidence = run_flow_direction_fixture(temporal_consistency_loss)
    (args.output_root / "e2vid_flow_direction_fixture.json").write_text(
        json.dumps(flow_direction_evidence, indent=2) + "\n"
    )

    results = []
    initial_hashes = {}
    training_rows = []
    validation_rows = []
    for generator in ("eventgenesis", "v2e"):
        generator_root = args.data_root / generator
        train_paths = sorted((generator_root / "train").glob("*.h5"))
        val_paths = sorted((generator_root / "validation").glob("*.h5"))
        if len(train_paths) < 2 or len(val_paths) < 2:
            raise RuntimeError(f"expected multiple train and validation rollouts for {generator}")
        train_filelist = args.output_root / f"{generator}_train_files.csv"
        val_filelist = args.output_root / f"{generator}_validation_files.csv"
        train_filelist.write_text("".join(f"{path}\n" for path in train_paths))
        val_filelist.write_text("".join(f"{path}\n" for path in val_paths))
        sequence_kwargs = {
            "sequence_length": SEQUENCE_LENGTH,
            "step_size": 1,
            "dataset_type": "Stage2RightEndpointH5Dataset",
            "proba_pause_when_running": 0.0,
            "proba_pause_when_paused": 0.0,
            "normalize_image": False,
            "noise_kwargs": {},
            "hot_pixel_kwargs": {},
            "dataset_kwargs": {
                "transforms": {},
                "num_bins": 5,
                "combined_voxel_channels": True,
                "filter_hot_events": False,
            },
        }
        train_loader = HDF5DataLoader(
            data_file=str(train_filelist), batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, sequence_kwargs=sequence_kwargs,
        )
        val_loader = HDF5DataLoader(
            data_file=str(val_filelist), batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True, sequence_kwargs=sequence_kwargs,
        )
        if len(train_loader) < 4 or len(val_loader) < 2:
            raise RuntimeError(f"insufficient multi-rollout sequences: train={len(train_loader)}, val={len(val_loader)}")
        with h5py.File(train_paths[0], "r") as handle:
            height, width = map(int, handle.attrs["sensor_resolution"])
        expected_event_shape = (1, 5, height, width)

        set_seed(args.seed)
        losses = [perceptual_loss(weight=1.0, net="vgg", use_gpu=True)]
        temporal_train = temporal_consistency_loss(weight=2.0, L0=2)
        temporal_train.calls = 0
        temporal_train.values = []
        losses.append(temporal_train)
        model = E2VIDRecurrent(MODEL_ARGS).to(device)
        model.train()
        initial_hash = model_sha256(model)
        initial_hashes[generator] = initial_hash
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)
        shim = SimpleNamespace(model=model, loss_ftns=losses, device=device)
        shim.to_device = lambda item: Trainer.to_device(shim, item)
        shim.forward_sequence = lambda sequence: Trainer.forward_sequence(shim, sequence)

        train_start = time.perf_counter()
        update_records = []
        for epoch in range(args.epochs):
            model.train()
            for batch_index, sequence in enumerate(train_loader):
                if len(sequence) != SEQUENCE_LENGTH:
                    raise RuntimeError(f"{generator}: got sequence length {len(sequence)}")
                for step in sequence:
                    if tuple(step["events"].shape) != expected_event_shape:
                        raise RuntimeError(f"wrong event tensor shape {step['events'].shape}")
                    if tuple(step["flow"].shape) != (1, 2, height, width):
                        raise RuntimeError(f"wrong shared flow shape {step['flow'].shape}")
                optimizer.zero_grad(set_to_none=True)
                before_calls = temporal_train.calls
                train_losses = shim.forward_sequence(sequence)
                total = train_losses["loss"]
                if not torch.isfinite(total):
                    raise RuntimeError(f"non-finite E2VID loss {generator} epoch {epoch} batch {batch_index}")
                total.backward()
                grad_sq = 0.0
                finite_gradient_tensors = 0
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("non-finite E2VID gradient")
                        grad_sq += float(torch.sum(parameter.grad.detach().float() ** 2).cpu())
                        finite_gradient_tensors += 1
                grad_norm = float(grad_sq**0.5)
                if not np.isfinite(grad_norm) or grad_norm <= 0 or finite_gradient_tensors == 0:
                    raise RuntimeError("E2VID backward produced no finite nonzero gradient")
                optimizer.step()
                scalars = loss_scalars(train_losses)
                rollout_id = sequence[-1]["rollout_id"][0]
                update_records.append({
                    "epoch": epoch + 1,
                    "batch_index": batch_index,
                    "rollout_id": str(rollout_id),
                    "loss_components": scalars,
                    "temporal_loss_calls": temporal_train.calls - before_calls,
                    "gradient_l2_norm": grad_norm,
                    "finite_gradient_tensors": finite_gradient_tensors,
                })
                training_rows.append({"generator": generator, **update_records[-1]})
        train_seconds = time.perf_counter() - train_start
        if temporal_train.calls <= 0:
            raise RuntimeError("temporal-consistency objective never executed")

        temporal_val = temporal_consistency_loss(weight=2.0, L0=2)
        temporal_val.calls = 0
        temporal_val.values = []
        val_losses = [perceptual_loss(weight=1.0, net="vgg", use_gpu=True), temporal_val]
        val_shim = SimpleNamespace(model=model, loss_ftns=val_losses, device=device)
        val_shim.to_device = lambda item: Trainer.to_device(val_shim, item)
        val_shim.forward_sequence = lambda sequence: Trainer.forward_sequence(val_shim, sequence)
        model.eval()
        val_records = []
        with torch.no_grad():
            for sequence_index, sequence in enumerate(val_loader):
                before_calls = temporal_val.calls
                vals = val_shim.forward_sequence(sequence)
                scalars = loss_scalars(vals)
                if not np.isfinite(scalars["loss"]):
                    raise RuntimeError(f"non-finite validation loss for {generator}")
                val_records.append({
                    "sequence_index": sequence_index,
                    "rollout_id": str(sequence[-1]["rollout_id"][0]),
                    "loss_components": scalars,
                    "temporal_loss_calls": temporal_val.calls - before_calls,
                })
                validation_rows.append({"generator": generator, **val_records[-1]})
        if temporal_val.calls <= 0:
            raise RuntimeError("temporal-consistency validation never executed")

        saved_hash = model_sha256(model)
        checkpoint = args.output_root / f"{generator}_final.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator,
            "seed": args.seed,
            "epochs": args.epochs,
            "updates": len(update_records),
            "model_args": MODEL_ARGS,
            "sequence_length": SEQUENCE_LENGTH,
            "model_sha256": saved_hash,
        }, checkpoint)

        # Restore into a fresh model/optimizer and compare deterministic
        # synthetic validation output/loss on one held-out rollout window.
        restored = E2VIDRecurrent(MODEL_ARGS).to(device)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
        restored.load_state_dict(checkpoint_data["model"])
        restored_optimizer.load_state_dict(checkpoint_data["optimizer"])
        if model_sha256(restored) != saved_hash:
            raise RuntimeError("E2VID checkpoint restore model hash mismatch")
        restored.eval()
        probe_sequence = next(iter(val_loader))
        probe_loss = temporal_consistency_loss(weight=2.0, L0=2)
        probe_loss.calls = 0
        probe_loss.values = []
        probe_losses = [perceptual_loss(weight=1.0, net="vgg", use_gpu=True), probe_loss]
        probe_shim = SimpleNamespace(model=restored, loss_ftns=probe_losses, device=device)
        probe_shim.to_device = lambda item: Trainer.to_device(probe_shim, item)
        probe_shim.forward_sequence = lambda sequence: Trainer.forward_sequence(probe_shim, sequence)
        with torch.no_grad():
            restored_val = loss_scalars(probe_shim.forward_sequence(probe_sequence))
        if not np.allclose(restored_val["loss"], val_records[0]["loss_components"]["loss"], rtol=1e-5, atol=1e-6):
            # First val_loader sequence is the same probe used above only if
            # iteration is deterministic; otherwise compare model/checkpoint
            # integrity and retain the restored probe value without asserting.
            first_id = val_records[0]["rollout_id"]
            probe_id = str(probe_sequence[-1]["rollout_id"][0])
            if first_id == probe_id:
                raise RuntimeError("E2VID restored validation loss differs on identical synthetic sequence")

        temporal_count_train = temporal_train.calls
        temporal_count_val = temporal_val.calls
        results.append({
            "generator": generator,
            "seed": args.seed,
            "train_rollouts": [path.stem for path in train_paths],
            "validation_rollouts": [path.stem for path in val_paths],
            "train_sequences_per_epoch": len(train_loader),
            "validation_sequences": len(val_loader),
            "sequence_length": SEQUENCE_LENGTH,
            "epochs": args.epochs,
            "updates": len(update_records),
            "initial_state_sha256": initial_hash,
            "final_model_state_sha256": saved_hash,
            "temporal_consistency_loss": {
                "weight": 2.0,
                "L0": 2,
                "train_valid_calls": temporal_count_train,
                "validation_valid_calls": temporal_count_val,
                "train_value_mean": float(np.mean(temporal_train.values)),
                "validation_value_mean": float(np.mean(temporal_val.values)),
                "train_value_count": len(temporal_train.values),
                "validation_value_count": len(temporal_val.values),
            },
            "recursive_state_tensor_count_after_sequence": tensor_state_count(model.states),
            "train_wall_seconds": train_seconds,
            "mean_train_components": {
                key: float(np.mean([row["loss_components"][key] for row in update_records]))
                for key in update_records[0]["loss_components"]
            },
            "mean_validation_components": {
                key: float(np.mean([row["loss_components"][key] for row in val_records]))
                for key in val_records[0]["loss_components"]
            },
            "backprop": {
                "all_steps_finite_nonzero_gradient": True,
                "gradient_l2_norm_min": min(row["gradient_l2_norm"] for row in update_records),
                "gradient_l2_norm_max": max(row["gradient_l2_norm"] for row in update_records),
                "finite_gradient_tensors_min": min(row["finite_gradient_tensors"] for row in update_records),
            },
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_restore_model_hash_match": model_sha256(restored) == saved_hash,
            "checkpoint_restore_synthetic_validation_probe": restored_val,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        })
        del model, optimizer, restored, restored_optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError(f"paired initialization hashes differ: {initial_hashes}")
    result = {
        "status": "pass",
        "scope": "multi-physical-rollout synthetic-only short training and independent synthetic validation per generator",
        "real_data_accessed": False,
        "initial_state_hashes_match": True,
        "paired_initial_state_sha256": next(iter(initial_hashes.values())),
        "model": "pinned E2VIDRecurrent from event_cnn_minimal",
        "model_args": MODEL_ARGS,
        "optimizer": {"type": "Adam", "lr": 1e-4, "weight_decay": 0, "amsgrad": True},
        "losses": ["LPIPS-VGG weight=1.0", "official temporal consistency weight=2.0 L0=2"],
        "flow_adapter": "shared geometry flow displacement and validity are independent of raw event count/span; event data only form the event voxel",
        "event_boundaries": "empty streams supported; <3 events use pinned zero voxel; tied timestamps use deterministic center-bin accumulation; right endpoints include all ties",
        "event_boundary_fixture_evidence": boundary_evidence,
        "flow_direction_fixture_evidence": flow_direction_evidence,
        "paired_results": results,
        "train_updates": training_rows,
        "validation_sequences": validation_rows,
    }
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
