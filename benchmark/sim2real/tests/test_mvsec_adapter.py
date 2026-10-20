#!/usr/bin/env python3
"""Controlled checks for the Stage 3 MVSEC adapter."""
from __future__ import annotations

import sys
import tempfile
import unittest
import os
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import run_mvsec_candidate_inference as adapter

REPOSITORY_PARENT = Path(__file__).resolve().parents[5]
SOURCE_CANDIDATES = [
    Path(os.environ["STE_FLOWNET_ROOT"]).expanduser() if os.environ.get("STE_FLOWNET_ROOT") else None,
    REPOSITORY_PARENT / "external_sources" / "STE-FlowNet",
    REPOSITORY_PARENT / "external" / "sources" / "downstream" / "STE-FlowNet",
]
STE_SOURCE = next((path.resolve() for path in SOURCE_CANDIDATES if path is not None and path.is_dir()), None)
if STE_SOURCE is None:
    raise FileNotFoundError(
        "pinned STE-FlowNet source is required for the independent GT-helper parity test; "
        "set STE_FLOWNET_ROOT to the verified checkout/archive path"
    )
sys.path.insert(0, str(STE_SOURCE))
from multiscaleloss import estimate_corresponding_gt_flow


class MVSECAdapterTests(unittest.TestCase):
    def test_two_component_epe_examples(self):
        gt = np.zeros((2, 2, 3), dtype=np.float32)
        gt[0] = 3.0
        gt[1] = 4.0
        mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8)

        perfect, count = adapter.strict_compute_epe(
            gt, gt, mask,
            prediction_unit=adapter.FLOW_UNIT,
            ground_truth_unit=adapter.FLOW_UNIT,
        )
        self.assertEqual(perfect, 0.0)
        self.assertEqual(count, 4)

        pred = gt.copy()
        pred[1] += 2.0
        vertical_only, _ = adapter.strict_compute_epe(
            pred, gt, mask,
            prediction_unit=adapter.FLOW_UNIT,
            ground_truth_unit=adapter.FLOW_UNIT,
        )
        self.assertEqual(vertical_only, 2.0)

        pred[0] += 3.0
        both_channels = np.sqrt(3.0**2 + 2.0**2)
        actual, _ = adapter.strict_compute_epe(
            pred, gt, mask,
            prediction_unit=adapter.FLOW_UNIT,
            ground_truth_unit=adapter.FLOW_UNIT,
        )
        self.assertAlmostEqual(actual, both_channels, places=6)

    def test_empty_mask_and_invalid_inputs_fail_closed(self):
        field = np.zeros((2, 2, 2), dtype=np.float32)
        empty, count = adapter.strict_compute_epe(
            field, field, np.zeros((2, 2), dtype=bool),
            prediction_unit=adapter.FLOW_UNIT,
            ground_truth_unit=adapter.FLOW_UNIT,
        )
        self.assertIsNone(empty)
        self.assertEqual(count, 0)
        with self.assertRaisesRegex(ValueError, "must share"):
            adapter.strict_compute_epe(
                field[:1], field, np.ones((2, 2), dtype=bool),
                prediction_unit=adapter.FLOW_UNIT,
                ground_truth_unit=adapter.FLOW_UNIT,
            )
        with self.assertRaisesRegex(ValueError, "flow units"):
            adapter.strict_compute_epe(
                field, field, np.ones((2, 2), dtype=bool),
                prediction_unit="px_per_second", ground_truth_unit=adapter.FLOW_UNIT,
            )
        bad_mask = np.ones((2, 2), dtype=np.float32)
        bad_mask[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "mask must contain"):
            adapter.strict_compute_epe(
                field, field, bad_mask,
                prediction_unit=adapter.FLOW_UNIT,
                ground_truth_unit=adapter.FLOW_UNIT,
            )

    def test_endpoint_windows_exclude_start_include_end_and_do_not_look_ahead(self):
        # Two disjoint 100 ms windows. Exact start ties are excluded, exact
        # right-end ties included, and events after each right endpoint remain
        # outside the corresponding sample.
        timestamps = np.array([0.00, 0.10, 0.100, 0.15, 0.20, 0.200, 0.201, 0.30])
        rows = np.column_stack((
            np.arange(len(timestamps)), np.zeros(len(timestamps)), timestamps,
            np.ones(len(timestamps)),
        ))
        first, info1 = adapter.collect_interval_events(rows, timestamps, 0.0, 0.1)
        second, info2 = adapter.collect_interval_events(rows, timestamps, 0.1, 0.2)
        self.assertEqual(info1["event_count"], 2)
        self.assertEqual(first[:, 2].tolist(), [0.1, 0.1])
        self.assertEqual(info2["event_count"], 3)
        self.assertEqual(second[:, 2].tolist(), [0.15, 0.2, 0.2])
        self.assertFalse(np.any(first[:, 2] > 0.1))
        self.assertFalse(np.any(second[:, 2] > 0.2))

    def test_native_padding_and_crop_preserve_u_v_and_sensor_pixels(self):
        target = adapter.round_model_size(260, 346)
        self.assertEqual(target, 352)
        volume = np.zeros((2, 260, 346, 20), dtype=np.uint8)
        volume[0, 0, 0, 0] = 5
        volume[1, 259, 345, 19] = 9
        padded, offset = adapter.center_pad_event_volume(volume, target)
        self.assertEqual(padded.shape, (2, 352, 352, 20))
        self.assertEqual(offset, (46, 3))
        self.assertTrue(np.array_equal(padded[:, 46:306, 3:349], volume))

        flow = np.zeros((2, target, target), dtype=np.float32)
        flow[0] = 12.5
        flow[1] = -3.25
        cropped = adapter.crop_flow_to_sensor(flow, (260, 346), offset)
        self.assertEqual(cropped.shape, (2, 260, 346))
        self.assertTrue(np.all(cropped[0] == 12.5))
        self.assertTrue(np.all(cropped[1] == -3.25))

    def test_model_output_extraction_keeps_both_flow_components(self):
        output = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
        output[:, 0] = 6.0
        output[:, 1] = -2.0
        flow = adapter.strict_model_output_to_hw(output)
        self.assertEqual(flow.shape, (2, 3, 4))
        self.assertTrue(np.all(flow[0] == 6.0))
        self.assertTrue(np.all(flow[1] == -2.0))
        with self.assertRaisesRegex(ValueError, r"must be \[1,2,H,W\]"):
            adapter.strict_model_output_to_hw(torch.zeros((1, 1, 3, 4)))

    def test_event_encoding_handles_empty_sparse_and_tied_time_intervals(self):
        empty = adapter.gaussian_count_representation(np.zeros((0, 4)), 260, 346)
        self.assertEqual(int(empty.sum()), 0)
        sparse = np.array([[12, 34, 1.0, 1]], dtype=np.float64)
        encoded_sparse = adapter.gaussian_count_representation(sparse, 260, 346)
        self.assertGreater(int(encoded_sparse.sum()), 0)
        tied = np.array([[12, 34, 1.0, 1], [12, 34, 1.0, -1]], dtype=np.float64)
        encoded_tied = adapter.gaussian_count_representation(tied, 260, 346)
        self.assertTrue(np.isfinite(encoded_tied).all())
        self.assertGreater(int(encoded_tied.sum()), 0)

    def test_dt1_and_dt4_use_explicit_aps_endpoint_indices(self):
        aps = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64)
        self.assertAlmostEqual(aps[3] - aps[2], 0.1, places=7)
        self.assertAlmostEqual(aps[4] - aps[0], 0.4, places=7)

    def test_gt_timestamp_propagation_matches_pinned_official_helper(self):
        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        flow = np.zeros((4, 2, 260, 346), dtype=np.float32)
        flow[:, 0, :, :] = 1.0
        flow[:, 1, :, :] = -0.5
        with tempfile.NamedTemporaryFile(suffix=".h5") as stream:
            with h5py.File(stream.name, "w") as handle:
                dataset = handle.create_dataset("flow_dist", data=flow)
                # Include a non-aligned interval, an exact internal GT endpoint,
                # and the last timestamp covered by the available GT maps.
                for start, end, expected_indices, expected_center, expected_noop in (
                    (0.05, 0.25, [0, 2], [2.0, -1.0], False),
                    (0.05, 0.20, [0, 1], [1.5, -0.75], False),
                    (0.10, 0.20, [1, 1], [1.0, -0.5], True),
                ):
                    actual, association = adapter.official_flow_association_from_h5(
                        dataset, timestamps, start, end
                    )
                    expected_u, expected_v = estimate_corresponding_gt_flow(
                        flow[:, 0], flow[:, 1], timestamps,
                        np.array(start), np.array(end),
                    )
                    expected = np.stack((expected_u, expected_v), axis=0).astype(np.float32)
                    self.assertEqual(actual.shape, (2, 260, 346))
                    self.assertTrue(np.allclose(actual, expected, rtol=0, atol=1e-6))
                    self.assertEqual(association["gt_map_indices_inclusive"], expected_indices)
                    self.assertTrue(np.allclose(actual[:, 100, 100], expected_center, atol=1e-6))
                    self.assertEqual(association["right_endpoint_zero_duration_map_read"], expected_noop)

                # The pinned helper indexes one GT timestamp beyond the end
                # when the window reaches the final available timestamp. The
                # adapter must handle that physical boundary without reading
                # beyond source data; its interior result remains exact.
                final_endpoint, final_meta = adapter.official_flow_association_from_h5(
                    dataset, timestamps, 0.30, 0.40
                )
                self.assertTrue(np.isfinite(final_endpoint).all())
                self.assertTrue(np.allclose(final_endpoint[:, 100, 100], [1.0, -0.5], atol=1e-6))
                self.assertFalse(final_meta["right_endpoint_zero_duration_map_read"])


if __name__ == "__main__":
    unittest.main()
