"""Small controls for scientific units, geometry, and missing-output scoring."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np

p = Path(__file__).resolve().parents[1] / 'pixel_metrics.py'
s = importlib.util.spec_from_file_location('pixel_metrics', p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


class PixelMetricsTests(unittest.TestCase):
    def test_original_pixel_units(self):
        np.testing.assert_allclose(m.project_world_to_pixel([1., -1., 2.], np.eye(4), [100., 200., 160., 120.]), [210., 20.])

    def test_rotated_translated_camera(self):
        pose = np.eye(4); pose[:3, :3] = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
        pose[:3, 3] = [10., 20., 30.]
        world = (pose @ [1., -1., 2., 1.])[:3]
        np.testing.assert_allclose(m.project_world_to_pixel(world, pose, [100., 200., 160., 120.]), [210., 20.])

    def test_invalid_projection(self):
        for point in ([0., 0., -1.], [0., 0., 0.], [np.nan, 0., 1.]):
            with self.assertRaises(ValueError): m.project_world_to_pixel(point, np.eye(4), [100., 100., 160., 120.])

    def test_missing_outputs_are_failures(self):
        x = m.summarize_pixel_failures([{'pixel_euclidean_error_px': x} for x in [0., 10., 11., None, float('nan')]])
        self.assertEqual(x['missing_or_invalid'], 2)
        self.assertEqual(x['thresholds_px']['10.0']['failed_or_missing'], 3)
        self.assertEqual(x['thresholds_px']['10.0']['fraction_all_polls'], .6)

    def test_empty_and_perfect(self):
        self.assertIsNone(m.summarize_pixel_failures([])['thresholds_px']['10.0']['fraction_all_polls'])
        self.assertEqual(m.summarize_pixel_failures([{'pixel_euclidean_error_px': 0.}])['thresholds_px']['2.0']['failed_or_missing'], 0)


if __name__ == '__main__': unittest.main()
