#!/usr/bin/env python3
"""Verify pinned E2VID crop/flip transforms agree across event/image/flow/mask."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--trainer-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.model_root.resolve()
    trainer_source = args.trainer_script.resolve().read_text()
    upstream_sequence_source = (source / "data_loader/dataset.py").read_text()
    adapter_contracts = {
        "upstream sequence sampler passes one shared seed": (
            "seed = random.randint(0, 2**32)" in upstream_sequence_source
            and "item = self.dataset.__getitem__(j, seed)" in upstream_sequence_source
        ),
        "stage4 adapter forwards same seed to events/images": (
            "item = super().__getitem__(index, seed=seed)" in trainer_source
        ),
        "stage4 adapter reuses same seed for flow": (
            "item[\"flow\"] = self.transform_flow(np.asarray(self.get_flow(index), dtype=np.float32), seed)"
            in trainer_source
        ),
        "stage4 adapter reseeds validity transform identically": (
            "random.seed(seed)\n                valid = self.transform(valid, is_flow=False)" in trainer_source
        ),
    }
    if not all(adapter_contracts.values()):
        raise RuntimeError(f"adapter does not share crop/flip seed across all modalities: {adapter_contracts}")
    sys.path.insert(0, str(source))
    from utils.data_augmentation import Compose, RandomCrop, RandomFlip

    height, width = 260, 346
    transform = Compose([
        RandomCrop(size=[112, 112]),
        RandomFlip(p_hflip=0.5, p_vflip=0.5),
    ])
    plane = torch.arange(1, height * width + 1, dtype=torch.float32).reshape(1, height, width)
    events = plane.repeat(5, 1, 1)
    image = plane.clone()
    flow = plane.repeat(2, 1, 1)
    valid = plane.clone()
    rows = []
    for seed in (0, 1, 271828, 20261019):
        def apply(value: torch.Tensor, *, is_flow: bool) -> torch.Tensor:
            random.seed(seed)
            return transform(value.clone(), is_flow=is_flow)

        events_out = apply(events, is_flow=False)
        image_out = apply(image, is_flow=False)
        flow_out = apply(flow, is_flow=True)
        valid_out = apply(valid, is_flow=False)
        event_marker = events_out[0]
        checks = {
            "events_image": torch.equal(event_marker, image_out[0]),
            "events_validity": torch.equal(event_marker, valid_out[0]),
            "events_flow_x_spatial": torch.equal(event_marker, flow_out[0].abs()),
            "events_flow_y_spatial": torch.equal(event_marker, flow_out[1].abs()),
            "all_expected_shapes": (
                tuple(events_out.shape) == (5, 112, 112)
                and tuple(image_out.shape) == (1, 112, 112)
                and tuple(flow_out.shape) == (2, 112, 112)
                and tuple(valid_out.shape) == (1, 112, 112)
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"joint augmentation failed for seed {seed}: {checks}")
        rows.append({"seed": seed, "checks": checks, "event_marker_sha256": hashlib.sha256(
            event_marker.contiguous().numpy().tobytes()).hexdigest()})

    result = {
        "status": "pass",
        "source": str(source),
        "source_module": "utils.data_augmentation.Compose(RandomCrop(112x112), RandomFlip(0.5,0.5))",
        "input_hw": [height, width],
        "seed_count": len(rows),
        "results": rows,
        "trainer_script_sha256": hashlib.sha256(args.trainer_script.resolve().read_bytes()).hexdigest(),
        "upstream_dataset_script_sha256": hashlib.sha256((source / "data_loader/dataset.py").read_bytes()).hexdigest(),
        "adapter_seed_contract": adapter_contracts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
