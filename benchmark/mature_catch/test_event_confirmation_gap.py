"""The shared confirmation candidate changes time support, not motion evidence."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from event_sphere_tracker_rgbd_base import EventSphereTracker


def tracker(gap=None):
    values = {} if gap is None else {'CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS': str(gap)}
    with patch.dict(os.environ, values, clear=False):
        if gap is None:
            os.environ.pop('CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS', None)
        obj = EventSphereTracker(np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]), (640, 480), .04)
    obj.initialize_rgb(dict(valid=True, position=[0., 2., 1.], camera_origin=[0., 0., 1.5],
                            radius_px=12., range_std_m=.01, initialization_mode='single_frame_RGBD'))
    return obj


def add_measurement(obj, timestamp, sign=1.):
    obj.samples.append(dict(t=timestamp, position=[sign*.055, 2.-sign*.15, 1.+sign*.05],
                            camera_origin=[0., 0., 1.5], radius_px=12., available_s=timestamp+.01,
                            maximum_event_s=timestamp, quality=10., uv=[320., 240.]))
    return obj._predict(timestamp)


class ConfirmationGapTests(unittest.TestCase):
    def test_default_is_exactly_original_fifty_ms(self):
        a, b = tracker(), tracker(50)
        self.assertEqual(a.confirmation_max_gap_s, .05)
        for t in [.10, .14, .18]:
            sa, sb = add_measurement(a, t), add_measurement(b, t)
            for key in ['position', 'velocity', 'latest_s']:
                np.testing.assert_array_equal(sa[key], sb[key])
            self.assertEqual(a.motion_onset_s, b.motion_onset_s)

    def test_eighty_ms_confirmation_requires_registered_hundred(self):
        a, b = tracker(50), tracker(100)
        for obj in (a, b):
            add_measurement(obj, .10)
            add_measurement(obj, .18)
        self.assertIsNone(a.motion_onset_s)
        self.assertEqual(b.motion_onset_s, .18)

    def test_association_uses_the_same_time_support(self):
        a, b = tracker(50), tracker(100)
        for obj in (a, b):
            obj.motion_suspicion = [dict(t=.10, innovation=np.array([.055, -.15, .05]))]
        np.testing.assert_array_equal(a._association_prediction(.18)['position'], a.static_position)
        expected = b.static_position + b.motion_suspicion[-1]['innovation'] + b.initial_velocity_prior*.08 + .5*b.gravity*.08**2
        np.testing.assert_allclose(b._association_prediction(.18)['position'], expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(b._association_prediction(.21)['position'], b.static_position)

    def test_opposite_motion_does_not_confirm(self):
        obj = tracker(100)
        add_measurement(obj, .10, 1.)
        add_measurement(obj, .18, -1.)
        self.assertIsNone(obj.motion_onset_s)

    def test_minimum_time_span_is_unchanged(self):
        obj = tracker(100)
        add_measurement(obj, .10)
        add_measurement(obj, .102)
        self.assertIsNone(obj.motion_onset_s)
        add_measurement(obj, .106)
        self.assertEqual(obj.motion_onset_s, .106)

    def test_unregistered_values_are_rejected(self):
        for value in [0, 20, 99, 101, 'nan', 'inf']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                tracker(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
