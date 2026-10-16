#!/usr/bin/env python3
"""Independent hand-calculated checks for strict 2-D STE flow EPE."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from run_steflow_stage2_pilot import FLOW_UNIT, compute_epe, model_output_to_flow_hw  # noqa: E402


class FlowEpeTests(unittest.TestCase):
    def setUp(self):
        # u=3 and v=4 makes the vector magnitude exactly 5 px/interval.
        self.gt = np.stack((np.full((2, 2), 3.0), np.full((2, 2), 4.0)))
        self.mask = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)

    def epe(self, prediction, gt=None, mask=None, prediction_unit=FLOW_UNIT, gt_unit=FLOW_UNIT):
        return compute_epe(
            prediction,
            self.gt if gt is None else gt,
            self.mask if mask is None else mask,
            prediction_unit=prediction_unit,
            gt_unit=gt_unit,
        )

    def test_perfect_two_component_prediction_is_zero(self):
        self.assertEqual(self.epe(self.gt.copy()), (0.0, 2))

    def test_u_and_v_both_contribute_to_vector_magnitude(self):
        prediction = np.zeros_like(self.gt)
        self.assertEqual(self.epe(prediction), (5.0, 2))

    def test_v_only_error_is_not_dropped(self):
        prediction = self.gt.copy()
        prediction[1, self.mask.astype(bool)] += 2.0
        self.assertEqual(self.epe(prediction), (2.0, 2))

    def test_validity_mask_excludes_invalid_region(self):
        prediction = self.gt.copy()
        prediction[0, 0, 1] += 3.0  # invalid pixel: ignored
        prediction[1, 1, 0] += 4.0  # valid pixel: EPE=4
        self.assertEqual(self.epe(prediction), (2.0, 2))

    def test_empty_validity_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "no valid flow pixels"):
            self.epe(self.gt.copy(), mask=np.zeros((2, 2), dtype=np.uint8))

    def test_missing_vector_component_rejects_broadcast(self):
        with self.assertRaisesRegex(ValueError, r"same \[2,H,W\]"):
            self.epe(self.gt[0])

    def test_bad_mask_shape_rejects_broadcast(self):
        with self.assertRaisesRegex(ValueError, r"mask must have shape"):
            self.epe(self.gt.copy(), mask=self.mask[None, ...])

    def test_unit_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires both prediction and GT"):
            self.epe(self.gt.copy(), prediction_unit="px_per_second")

    def test_nonfinite_flow_is_rejected(self):
        prediction = self.gt.copy()
        prediction[1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self.epe(prediction)

    def test_nonbinary_mask_is_rejected(self):
        invalid = self.mask.astype(np.float32)
        invalid[0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "only finite 0/1"):
            self.epe(self.gt.copy(), mask=invalid)

    def test_model_output_extraction_keeps_u_and_v(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment-specific
            self.skipTest(str(exc))
        output = torch.as_tensor(self.gt[None, ...])
        extracted = model_output_to_flow_hw(output)
        self.assertEqual(extracted.shape, (2, 2, 2))
        np.testing.assert_array_equal(extracted, self.gt)
        with self.assertRaisesRegex(ValueError, r"\[1,2,H,W\]"):
            model_output_to_flow_hw(output[:, :1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
