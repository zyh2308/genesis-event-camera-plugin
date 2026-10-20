import math
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from benchmark.sim2real.scripts.evaluate_synthetic_reconstruction_fixture import (
    classical_metrics,
    lpips_input,
    validate_pair,
)


class ReconstructionMetricContractTests(unittest.TestCase):
    def test_hqf_protocol_model_config_fingerprint_matches_path(self):
        root = Path(__file__).resolve().parents[3]
        protocol_path = root / "benchmark/sim2real/protocol/hqf_14_reconstruction_test_candidate_20261021_v3.json"
        protocol = json.loads(protocol_path.read_text())
        config_path = root / "benchmark/sim2real" / protocol["model"]["model_args_source"].split("::", 1)[0]
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        self.assertEqual(protocol["model"]["model_args_source_sha256"], digest)

    @unittest.skipIf(importlib.util.find_spec("skimage") is None, "scikit-image is available in the pinned g1 evaluation environment")
    def test_perfect_pair_has_zero_error_and_unit_ssim(self):
        target = np.full((16, 20), 64, dtype=np.uint8)
        prediction = np.full((16, 20), 64.0 / 255.0, dtype=np.float32)
        target01 = validate_pair(prediction, target, (16, 20))
        values = classical_metrics(prediction, target01)
        self.assertEqual(values["mse"], 0.0)
        self.assertTrue(math.isinf(values["psnr_db"]))
        self.assertAlmostEqual(values["ssim"], 1.0, places=6)

    @unittest.skipIf(importlib.util.find_spec("skimage") is None, "scikit-image is available in the pinned g1 evaluation environment")
    def test_known_constant_error_is_hand_computable(self):
        target = np.zeros((16, 20), dtype=np.float32)
        prediction = np.full((16, 20), 0.25, dtype=np.float32)
        values = classical_metrics(prediction, target)
        self.assertAlmostEqual(values["mse"], 1.0 / 16.0, places=8)
        self.assertAlmostEqual(values["psnr_db"], 10.0 * math.log10(16.0), places=6)
        self.assertGreaterEqual(values["ssim"], 0.0)
        self.assertLessEqual(values["ssim"], 1.0)

    def test_lpips_grayscale_mapping_is_fixed_rgb_replication(self):
        image = np.asarray([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
        tensor = lpips_input(image, torch.device("cpu"))
        self.assertEqual(tuple(tensor.shape), (1, 3, 2, 2))
        self.assertEqual(float(tensor.min()), -1.0)
        self.assertEqual(float(tensor.max()), 1.0)
        self.assertTrue(torch.equal(tensor[:, 0], tensor[:, 1]))
        self.assertTrue(torch.equal(tensor[:, 1], tensor[:, 2]))

    def test_rejects_shape_dtype_nonfinite_and_out_of_range(self):
        valid_target = np.zeros((16, 20), dtype=np.uint8)
        valid_prediction = np.zeros((16, 20), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_pair(valid_prediction[:15], valid_target, (16, 20))
        with self.assertRaises(ValueError):
            validate_pair(valid_prediction.astype(np.float64), valid_target, (16, 20))
        nan_prediction = valid_prediction.copy(); nan_prediction[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_pair(nan_prediction, valid_target, (16, 20))
        high_prediction = valid_prediction.copy(); high_prediction[0, 0] = 1.01
        with self.assertRaises(ValueError):
            validate_pair(high_prediction, valid_target, (16, 20))


if __name__ == "__main__":
    unittest.main()
