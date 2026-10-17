#!/usr/bin/env python3
"""Reproduce the Stage-4 static-ground preflight indexing diagnosis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    with h5py.File(args.bank_h5, "r") as bank:
        for rollout in sorted(bank["episodes"]):
            if not rollout.startswith("s1_"):
                continue
            group = bank[f"episodes/{rollout}"]
            physical = np.asarray(group["segmentation_link_idx"], dtype=np.int32)
            flow = np.asarray(group["flow_forward_30ms_px"], dtype=np.float32)
            valid = np.asarray(group["flow_valid_30ms"], dtype=bool)
            ground_id = int(group.attrs["ground_segmentation_link_id"])
            for index in (0, 32, 65):
                mask = valid[index] & (physical[index * 3] == ground_id)
                if not np.any(mask):
                    raise RuntimeError(f"empty registered ground support: {rollout}/{index}")
                # This reproduces the old expression exactly. Mixed advanced
                # indexing moves selected pixels before the channel dimension.
                legacy_selected = flow[index, :, mask]
                legacy_value = float(np.max(np.linalg.norm(legacy_selected, axis=0)))
                # CHW slicing retains the explicit vector-first shape.
                selected = flow[index][:, mask]
                if selected.shape != (2, int(mask.sum())):
                    raise RuntimeError(f"expected (2,N) selected flow, got {selected.shape}")
                per_pixel = np.linalg.norm(selected.astype(np.float64), axis=0)
                cases.append({
                    "rollout": rollout,
                    "interval_index": index,
                    "valid_ground_pixels": int(mask.sum()),
                    "legacy_selected_shape": list(legacy_selected.shape),
                    "legacy_axis0_value": legacy_value,
                    "correct_selected_shape": list(selected.shape),
                    "correct_max_per_pixel_epe_px_per_30ms": float(per_pixel.max()),
                })
    old_max = max(row["legacy_axis0_value"] for row in cases)
    corrected_max = max(row["correct_max_per_pixel_epe_px_per_30ms"] for row in cases)
    result = {
        "schema": "stage4_static_ground_flow_indexing_diagnosis_v1",
        "status": "PASS" if len(cases) == 24 and corrected_max <= 1e-5 and old_max > 1e-5 else "FAIL",
        "source_bank_h5": str(args.bank_h5.resolve()),
        "checked_cases": len(cases),
        "old_wrong_axis_max": old_max,
        "correct_per_pixel_max": corrected_max,
        "threshold_px_per_30ms": 1e-5,
        "cause": "NumPy mixed boolean indexing returns (N,2), so norm(axis=0) aggregated across pixels rather than computing one 2D vector norm per pixel.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
