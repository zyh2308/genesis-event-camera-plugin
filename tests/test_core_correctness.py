"""Regression tests for scientifically consequential simulator behavior."""

import tempfile
import unittest

import numpy as np

from genesis_event_plugin import (
    DvsEmulator,
    GenesisEventPlugin,
    GenesisEventRecorder,
    custom_config,
    get_camera_profile,
    get_preset,
)
from genesis_event_plugin.evaluation.event_schema import canonicalize_events
from genesis_event_plugin.evaluation.pecs_metrics import gaussian_distance


class TestCoreCorrectness(unittest.TestCase):
    def test_evk4_nominal_optics_and_intrinsics(self):
        profile = get_camera_profile('evk4')
        self.assertEqual(profile.native_resolution_wh, (1280, 720))
        self.assertEqual(profile.lens_model, 'SFA 0820-5M')
        self.assertAlmostEqual(profile.sensor_size_mm[0], 6.2208, places=6)
        K = profile.nominal_intrinsics()
        expected_f = 360.0 / np.tan(np.deg2rad(23.6 / 2.0))
        self.assertAlmostEqual(K[0, 0], expected_f, places=8)
        self.assertAlmostEqual(K[1, 1], expected_f, places=8)
        self.assertEqual(
            profile.genesis_camera_kwargs(),
            {'res': (1280, 720), 'fov': 23.6, 'model': 'pinhole'},
        )
        field_w, field_h = profile.field_size_at_distance(0.5)
        self.assertAlmostEqual(field_w, 0.5 * 2 * np.tan(np.deg2rad(20.7)))
        self.assertAlmostEqual(field_h, 0.5 * 2 * np.tan(np.deg2rad(11.8)))
        self.assertGreater(
            profile.minimum_distance_for_extent(0.30, 0.20), 0.5
        )

    def test_evk4_camera_validation_is_fail_closed_in_strict_mode(self):
        class FakeCamera:
            res = (1280, 720)
            fov = 40.0
            model = 'pinhole'

        with self.assertRaises(ValueError):
            get_camera_profile('evk4').validate_genesis_camera(
                FakeCamera(), strict=True
            )

    def test_evk4_non_strict_only_relaxes_native_resolution(self):
        class DownsampledCamera:
            res = (320, 180)
            fov = 23.6
            model = 'pinhole'

        messages = get_camera_profile('evk4').validate_genesis_camera(
            DownsampledCamera(), strict=False
        )
        self.assertTrue(any('resized RL representation' in m for m in messages))

        DownsampledCamera.res = (64, 64)
        with self.assertRaises(ValueError):
            get_camera_profile('evk4').validate_genesis_camera(
                DownsampledCamera(), strict=False
            )

    def test_set_preset_updates_recorded_provenance(self):
        plugin = GenesisEventPlugin(preset='clean', device='cpu')
        plugin.set_preset('moderate')
        self.assertEqual(plugin.preset_name, 'moderate')

    def test_evk4_nominal_sensor_preset_uses_published_1klux_values(self):
        config = get_preset('evk4_nominal_1klux')
        self.assertEqual(config.pos_thres, 0.25)
        self.assertEqual(config.neg_thres, 0.25)
        self.assertEqual(config.sigma_thres, 0.015)
        self.assertEqual(config.shot_noise_rate_hz, 0.1)
        self.assertEqual(config.cutoff_hz, 0.0)

    def test_presets_are_not_shared_mutable_objects(self):
        first = get_preset('clean')
        second = get_preset('clean')
        first.pos_thres = 9.0
        self.assertEqual(second.pos_thres, 0.2)

    def test_float_input_ranges_are_equivalent(self):
        cfg = custom_config(pos_thres=0.2, neg_thres=0.2, sigma_thres=0, seed=1)
        a = DvsEmulator(res=(2, 2), config=cfg, device='cpu')
        b = DvsEmulator(res=(2, 2), config=cfg, device='cpu')
        a.initialize(np.full((2, 2), 0.1, np.float32), 0.0)
        b.initialize(np.full((2, 2), 25.5, np.float32), 0.0)
        ea = a.generate_events(np.full((2, 2), 0.8, np.float32), 0.01)
        eb = b.generate_events(np.full((2, 2), 204.0, np.float32), 0.01)
        np.testing.assert_array_equal(ea, eb)

    def test_threshold_crossings_have_per_pixel_times(self):
        cfg = custom_config(pos_thres=0.2, neg_thres=0.2, sigma_thres=0, seed=3)
        emulator = DvsEmulator(res=(2, 3), config=cfg, device='cpu')
        emulator.initialize(np.full((2, 3), 20, np.uint8), 0.0)
        events = emulator.generate_events(
            np.array([[30, 60, 120], [180, 220, 250]], np.uint8), 0.01
        )
        self.assertTrue(np.all(np.diff(events[:, 0]) >= 0))
        self.assertGreater(len(np.unique(events[:, 0])), 10)

    def test_pecs_gaussian_distance_squares_euclidean_distance(self):
        r = np.array([[0.0, 0.0, -1.0, 0.0]], dtype=np.float32)
        q = np.array([[2.0, 0.0, -1.0, 0.0]], dtype=np.float32)
        actual = gaussian_distance(r, q, normalize=False, sigma=0.4)
        expected = 2.0 * (1.0 - np.exp(-(2.0 ** 2) / 0.4))
        self.assertAlmostEqual(actual, expected, places=7)

    def test_schema_converts_units_layout_and_polarity(self):
        legacy = np.array([[1_000_000, 4, 5, 0], [2_000_000, 6, 7, 1]])
        canonical = canonicalize_events(
            legacy, layout='txyp', timestamp_unit='us', sort=True
        )
        np.testing.assert_allclose(
            canonical,
            np.array([[4, 5, -1, 1.0], [6, 7, 1, 2.0]], dtype=np.float64),
        )

    def test_recorder_writes_schema_and_episode_boundaries(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            recorder = GenesisEventRecorder(
                directory,
                resolution=(8, 10),
                sensor_config={'pos_thres': 0.2},
                camera_config={'target': 'evk4'},
            )
            recorder.start_episode()
            recorder.record_aer_events(
                np.array([[0.01, 2, 3, -1], [0.02, 4, 5, 1]], dtype=np.float32)
            )
            recorder.end_episode()
            recorder.start_episode()
            recorder.end_episode()
            recorder.close()

            with h5py.File(f'{directory}/aer_events.h5', 'r') as handle:
                self.assertEqual(handle.attrs['schema_version'], 'genesis-event-v2')
                self.assertEqual(handle['events'].attrs['timestamp_unit'], 'us')
                self.assertIn('evk4', handle.attrs['camera_config_json'])
                np.testing.assert_array_equal(
                    handle['episode_event_ranges'][:],
                    np.array([[0, 0, 2], [1, 2, 2]], dtype=np.uint64),
                )

    def test_event_frame_log_scaling_preserves_rate_and_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = GenesisEventRecorder(
                directory,
                output_event_frames=True,
                event_frame_window='capture_interval',
                event_frame_normalization='log1p',
            )
            recorder.start_episode()
            two = np.array(
                [[0.001, 1, 1, 1], [0.002, 1, 1, 1]], dtype=np.float64
            )
            recorder.record_aer_events(two)
            frame_two = recorder.build_event_frame_from_buffer((3, 3), 0.002)
            four = np.array(
                [[0.003 + i * 0.001, 1, 1, 1] for i in range(4)],
                dtype=np.float64,
            )
            recorder.record_aer_events(four)
            frame_four = recorder.build_event_frame_from_buffer((3, 3), 0.006)
            silent = recorder.build_event_frame_from_buffer((3, 3), 0.007)
            self.assertGreater(frame_four[1, 1, 0], frame_two[1, 1, 0])
            self.assertEqual(float(silent.sum()), 0.0)
            recorder.end_episode()
            recorder.close()


if __name__ == '__main__':
    unittest.main()
