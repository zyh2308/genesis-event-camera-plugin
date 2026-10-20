#!/usr/bin/env python3
"""Regression checks for the pinned Genesis display transfer adapter."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from generate_stage5_flow_coverage_bank import genesis_display_rgb_to_linear_proxy


class GenesisDisplayTransferTests(unittest.TestCase):
    def test_inverse_gamma22_and_clipping_record(self):
        rgb = np.asarray([[[0, 128, 255]]], dtype=np.uint8)
        linear, metadata = genesis_display_rgb_to_linear_proxy(rgb)
        expected = (rgb.astype(np.float32) / 255.0) ** 2.2
        np.testing.assert_allclose(linear, expected, rtol=0, atol=1e-7)
        self.assertFalse(metadata["native_hdr_radiance"])
        self.assertAlmostEqual(metadata["saturated_channel_fraction"], 1 / 3)
        self.assertEqual(float(linear.min()), 0.0)
        self.assertEqual(float(linear.max()), 1.0)

    def test_rejects_unexpected_renderer_contract(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            genesis_display_rgb_to_linear_proxy(np.zeros((4, 4), dtype=np.uint8))
        with self.assertRaisesRegex(TypeError, "uint8"):
            genesis_display_rgb_to_linear_proxy(np.zeros((4, 4, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
