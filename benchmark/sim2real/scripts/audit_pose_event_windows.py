#!/usr/bin/env python3
"""Independent endpoint/count fixture for the pose event-window adapter."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch


SCRIPT = Path(__file__).with_name("run_pose_stage2_pilot.py")
SPEC = importlib.util.spec_from_file_location("run_pose_stage2_pilot", SCRIPT)
POSE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(POSE)


def write_fixture(shared_path: Path, v2e_root: Path) -> dict[str, np.ndarray]:
    v2e_root.mkdir(parents=True)
    # The two intervals are (0.00, 0.10] and (0.10, 0.20]. Events at
    # t=0.00, 0.10, and 0.20 exercise both exclusion and inclusion boundaries;
    # 0.100001 and 0.200001 are just after their respective endpoints.
    raw_events = {
        "eventgenesis": np.asarray([
            [0.000000, 16, 12, 1],
            [0.025000, 16, 12, -1],
            [0.100000, 16, 12, 1],
            [0.100000, 16, 12, -1],
            [0.100001, 16, 12, 1],
            [0.125000, 16, 12, -1],
            [0.200000, 16, 12, 1],
            [0.200001, 16, 12, -1],
        ], dtype=np.float64),
        "v2e": np.asarray([
            [0.000000, 16, 12, 1],
            [0.025000, 16, 12, 0],
            [0.050000, 16, 12, 1],
            [0.100000, 16, 12, 1],
            [0.100000, 16, 12, 0],
            [0.100001, 16, 12, 1],
            [0.125000, 16, 12, 0],
            [0.200000, 16, 12, 1],
            [0.200001, 16, 12, 0],
        ], dtype=np.float64),
    }
    with h5py.File(shared_path, "w") as shared:
        episodes = shared.create_group("episodes")
        episode = episodes.create_group("rollout_fixture")
        episode.attrs["split"] = "train"
        episode.create_dataset("frame_timestamps_s", data=np.asarray([0.0, 0.1, 0.2]))
        episode.create_dataset(
            "camera_intrinsics_K_px",
            data=np.asarray([[20.0, 0.0, 16.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]]),
        )
        episode.create_dataset("camera_to_world_cv", data=np.repeat(np.eye(4)[None], 3, axis=0))
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 2, 3] = 1.0
        episode.create_dataset("target_pose", data=poses)
        episode.create_dataset("v2e_display_luma_u8", data=np.zeros((3, 24, 32), np.uint8))
        episode.create_dataset(
            "eventgenesis_events_txyp", data=raw_events["eventgenesis"],
        )
    v2e_us = raw_events["v2e"].copy()
    v2e_us[:, 0] *= 1_000_000
    with h5py.File(v2e_root / "rollout_fixture.h5", "w") as handle:
        handle.create_dataset("events", data=v2e_us.astype(np.uint32))
    return raw_events


def independently_recount(raw_events: np.ndarray, start_s: float, end_s: float) -> int:
    # Deliberately separate from the adapter's selector and encoder.
    timestamps_s = raw_events[:, 0]
    return int(sum(start_s < float(timestamp) <= end_s for timestamp in timestamps_s))


def audit_event_windows() -> list[dict]:
    records = []
    with tempfile.TemporaryDirectory(prefix="pose-window-audit-") as temp:
        root = Path(temp)
        shared_path = root / "shared.h5"
        v2e_root = root / "v2e"
        raw_events = write_fixture(shared_path, v2e_root)
        expected_by_generator = {
            "eventgenesis": [3, 3],
            "v2e": [4, 3],
        }
        for generator in ("eventgenesis", "v2e"):
            rows = POSE.collect_samples(
                generator, shared_path, v2e_root, [0.1, 0.1, 0.1],
                "train", windows_per_rollout=2, count_scale=8,
            )
            assert [row["interval_index"] for row in rows] == [0, 1]
            for index, row in enumerate(rows):
                start_s, end_s = ((0.0, 0.1), (0.1, 0.2))[index]
                assert row["sample_id"] == f"rollout_fixture/interval_{index:03d}"
                assert row["rollout_id"] == "rollout_fixture"
                assert row["event_start_s"] == start_s
                assert row["event_end_s"] == end_s
                assert row["label_time_s"] == end_s
                assert row["event_end_s"] == row["label_time_s"]
                recounted = independently_recount(raw_events[generator], start_s, end_s)
                assert recounted == expected_by_generator[generator][index]
                assert row["event_count"] == recounted
                selected = (raw_events[generator][:, 0] > start_s) & (raw_events[generator][:, 0] <= end_s)
                selected_times = raw_events[generator][selected, 0]
                assert len(selected_times) == recounted
                assert len(selected_times) == 0 or float(selected_times.max()) <= row["label_time_s"]
                records.append({
                    "generator": generator,
                    "sample_id": row["sample_id"],
                    "interval_index": index,
                    "event_start_s": start_s,
                    "event_end_s": end_s,
                    "label_time_s": end_s,
                    "event_count_from_adapter": row["event_count"],
                    "event_count_independent_raw_recount": recounted,
                    "max_selected_event_time_s": float(selected_times.max()),
                })
    return records


def audit_confidence_decode() -> dict:
    # Construct outputs where the GT centroid cell stores exact keypoints but
    # a different cell has the confidence maximum. This isolates decoder logic.
    h = w = 4
    points = np.asarray([
        [0.625, 0.625],
        [0.500, 0.500], [0.500, 0.750], [0.750, 0.500], [0.750, 0.750],
        [0.500, 0.500], [0.500, 0.750], [0.750, 0.500], [0.750, 0.750],
    ], dtype=np.float32)
    output = torch.zeros((1, 2 * POSE.NUM_KEYPOINTS + 2, h, w), dtype=torch.float32)
    output[0, 2 * POSE.NUM_KEYPOINTS, 0, 0] = 8.0  # wrong high-confidence cell
    output[0, 2 * POSE.NUM_KEYPOINTS, 2, 2] = -8.0  # GT cell is not selected
    for index, (x_norm, y_norm) in enumerate(points):
        local_x = float(x_norm * w - 2)
        local_y = float(y_norm * h - 2)
        if index == 0:
            # Centroid offsets use sigmoid in the official decoder.
            output[0, 2 * index, 2, 2] = 0.0
            output[0, 2 * index + 1, 2, 2] = 0.0
        else:
            output[0, 2 * index, 2, 2] = local_x
            output[0, 2 * index + 1, 2, 2] = local_y
    argmax_xy, argmax_conf, argmax_cell = POSE.decode_normalized_keypoints(output)
    gt_xy, gt_conf, gt_cell = POSE.decode_normalized_keypoints(output, (2, 2))
    argmax_epe = float(np.linalg.norm(argmax_xy[1:] - points[1:], axis=1).mean() * 416)
    gt_cell_epe = float(np.linalg.norm(gt_xy[1:] - points[1:], axis=1).mean() * 416)
    assert argmax_cell == (0, 0) and gt_cell == (2, 2)
    assert gt_cell_epe < 1e-5 and argmax_epe > 100
    assert np.isfinite(points).all() and np.all((points >= 0) & (points < 1))
    return {
        "target_coordinate_shape": list(points.shape),
        "target_coordinate_min_max": [float(points.min()), float(points.max())],
        "target_coordinates_finite_and_in_unit_square": True,
        "confidence_argmax_cell_yx": list(argmax_cell),
        "confidence_argmax_cell_confidence": argmax_conf,
        "confidence_argmax_keypoint_mean_epe_px": argmax_epe,
        "gt_centroid_cell_yx": list(gt_cell),
        "gt_centroid_cell_confidence": gt_conf,
        "gt_centroid_cell_keypoint_mean_epe_px": gt_cell_epe,
        "interpretation": "deterministic decoder fixture only; does not assert the Stage-2 checkpoint selected these cells",
    }


def main() -> None:
    report = {
        "status": "pass",
        "scope": "independent raw timestamp recount and deterministic decoder fixture; no model training or real data",
        "pose_event_windows": audit_event_windows(),
        "confidence_decode_fixture": audit_confidence_decode(),
        "stage2_confidence_supervision_source_check": {
            "configured_epochs": 2,
            "pinned_pretrain_num_epochs": 15,
            "confidence_loss_included_condition": "epoch > pretrain_num_epochs",
            "confidence_loss_included_in_stage2_epochs": False,
            "implication": "confidence argmax cell was not directly supervised in the Stage-2 pilot",
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
