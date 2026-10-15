#!/usr/bin/env python3
"""Run one matched synthetic-only E2VID+ update per generator on g1."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


MODEL_ARGS = {
    "unet_kwargs": {
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
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    model_root = args.model_root.resolve()
    sys.path.insert(0, str(model_root))
    sys.path.insert(0, str(model_root / "PerceptualSimilarity"))

    # Pinned PerceptualSimilarity imports this removed scikit-image alias even
    # though the training loss does not call it. Keep the compatibility shim
    # local to this runner instead of editing the third-party checkout.
    import skimage.measure
    from skimage.metrics import structural_similarity

    if not hasattr(skimage.measure, "compare_ssim"):
        skimage.measure.compare_ssim = structural_similarity

    from data_loader import dataset as dataset_module
    from data_loader.data_loaders import HDF5DataLoader
    from data_loader.dataset import DynamicH5Dataset
    from model.loss import perceptual_loss, temporal_consistency_loss
    from model.model import E2VIDRecurrent
    from trainer.trainer import Trainer

    class Stage0RightEndpointH5Dataset(DynamicH5Dataset):
        """Use a deterministic inclusive right endpoint for repeated ts values."""

        def compute_frame_indices(self):
            event_ts = np.asarray(self.h5_file["events/ts"], dtype=np.float64)
            indices = []
            start = 0
            for timestamp in self.frame_ts:
                end = int(np.searchsorted(event_ts, timestamp, side="right"))
                indices.append([start, end])
                start = end
            return indices

    dataset_module.Stage0RightEndpointH5Dataset = Stage0RightEndpointH5Dataset
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback is used for the experiment")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    results = []
    initial_hashes = {}
    for generator in ("eventgenesis", "v2e"):
        generator_root = args.data_root / generator
        train_filelist = args.output_root / f"{generator}_train_files.csv"
        val_filelist = args.output_root / f"{generator}_validation_files.csv"
        train_filelist.write_text(str(generator_root / "train.h5") + "\n")
        val_filelist.write_text(str(generator_root / "validation.h5") + "\n")
        sequence_kwargs = {
            "sequence_length": 2,
            "dataset_type": "Stage0RightEndpointH5Dataset",
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
        if len(train_loader) < 1 or len(val_loader) < 1:
            raise RuntimeError(f"empty synthetic loader for {generator}: {len(train_loader)}, {len(val_loader)}")

        train_sequence = next(iter(train_loader))
        val_sequence = next(iter(val_loader))
        expected_shape = (1, 5, 128, 160)
        for label, sequence in (("train", train_sequence), ("validation", val_sequence)):
            if len(sequence) != 2:
                raise RuntimeError(f"{generator}/{label}: expected 2 time steps, got {len(sequence)}")
            if tuple(sequence[0]["events"].shape) != expected_shape:
                raise RuntimeError(f"{generator}/{label}: wrong event shape {sequence[0]['events'].shape}")
            if tuple(sequence[0]["frame"].shape) != (1, 1, 128, 160):
                raise RuntimeError(f"{generator}/{label}: wrong frame shape {sequence[0]['frame'].shape}")
            if tuple(sequence[0]["flow"].shape) != (1, 2, 128, 160):
                raise RuntimeError(f"{generator}/{label}: wrong flow shape {sequence[0]['flow'].shape}")

        # Instantiate the official pinned E2VID+ model/losses. Reset the same
        # seed immediately before model initialization for paired parity.
        loss_ftns = [
            perceptual_loss(weight=1.0, net="vgg", use_gpu=True),
            temporal_consistency_loss(weight=2.0, L0=2),
        ]
        set_seed(args.seed)
        model = E2VIDRecurrent(MODEL_ARGS).to(device)
        model.train()
        initial_hash = state_sha256(model)
        initial_hashes[generator] = initial_hash
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)

        shim = SimpleNamespace(model=model, loss_ftns=loss_ftns, device=device)
        shim.to_device = lambda item: Trainer.to_device(shim, item)
        shim.forward_sequence = lambda sequence: Trainer.forward_sequence(shim, sequence)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        train_losses = shim.forward_sequence(train_sequence)
        train_loss = train_losses["loss"]
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"non-finite training loss for {generator}: {train_loss.item()}")
        train_loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - start

        model.eval()
        set_seed(args.seed)
        with torch.no_grad():
            val_losses = shim.forward_sequence(val_sequence)
        val_loss = val_losses["loss"]
        if not torch.isfinite(val_loss):
            raise RuntimeError(f"non-finite synthetic validation loss for {generator}: {val_loss.item()}")

        args.output_root.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output_root / f"{generator}_step000001.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": generator,
            "seed": args.seed,
            "updates": 1,
            "source_model": "E2VIDRecurrent from event_cnn_minimal",
            "train_loss": float(train_loss.detach().cpu()),
            "validation_loss": float(val_loss.detach().cpu()),
        }, checkpoint)
        results.append({
            "generator": generator,
            "seed": args.seed,
            "updates": 1,
            "train_loader_sequences": len(train_loader),
            "validation_loader_sequences": len(val_loader),
            "sequence_length": 2,
            "channels": 5,
            "initial_state_sha256": initial_hash,
            "train_loss": float(train_loss.detach().cpu()),
            "validation_loss": float(val_loss.detach().cpu()),
            "train_step_wall_seconds": train_seconds,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        })
        del model, optimizer, loss_ftns, train_loader, val_loader
        torch.cuda.empty_cache()

    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError(f"paired model initializations differ: {initial_hashes}")
    result = {
        "status": "pass",
        "scope": "one synthetic-only E2VID+ update and one synthetic validation sequence per generator",
        "real_data_accessed": False,
        "initial_state_hashes_match": True,
        "model_args": MODEL_ARGS,
        "optimizer": {"type": "Adam", "lr": 1e-4, "weight_decay": 0, "amsgrad": True},
        "losses": ["LPIPS-VGG weight=1.0", "temporal_consistency weight=2.0 L0=2"],
        "adapter": "right-endpoint HDF5 loader; duplicate timestamp ties use numpy.searchsorted(side=right); fixed seed 71",
        "results": results,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
