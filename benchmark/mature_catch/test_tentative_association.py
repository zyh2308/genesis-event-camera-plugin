"""Association may follow one tentative motion sample; public state may not."""
import unittest
from unittest.mock import patch
import numpy as np

from event_sphere_tracker_rgbd_base import EventSphereTracker


class TentativeAssociationTests(unittest.TestCase):
    def make_tracker(self, position=(0., 2.4, 1.)):
        K = np.array([[433., 0., 320.], [0., 433., 240.], [0., 0., 1.]])
        tracker = EventSphereTracker(K, (640, 480), .04)
        tracker.initialize_rgb(dict(valid=True, position=list(position),
            camera_origin=[0., 0., 1.4], radius_px=7.2, range_std_m=.01,
            initialization_mode='single_frame_RGBD'))
        return tracker

    def test_no_prediction_and_no_suspicion(self):
        K = np.eye(3)
        empty = EventSphereTracker(K, (640, 480), .04)
        self.assertIsNone(empty._association_prediction(.7))
        tracker = self.make_tracker()
        result = tracker._association_prediction(.7)
        np.testing.assert_array_equal(result['position'], tracker.static_position)
        self.assertIsNone(tracker.motion_onset_s)

    def test_association_only_public_stays_static(self):
        tracker = self.make_tracker()
        innovation = np.array([.01, -.20, .08])
        tracker.motion_suspicion = [dict(t=.7, innovation=innovation)]
        before = tracker._predict(.74)
        result = tracker._association_prediction(.74)
        dt = .74 - .7
        expected = (tracker.static_position + innovation +
                    tracker.initial_velocity_prior * dt +
                    .5 * tracker.gravity * dt * dt)
        np.testing.assert_allclose(result['position'], expected, atol=1e-12)
        np.testing.assert_array_equal(before['position'], tracker.static_position)
        np.testing.assert_array_equal(tracker._predict(.74)['position'], tracker.static_position)
        self.assertIsNone(tracker.motion_onset_s)
        self.assertIsNone(tracker.filter_mean)
        self.assertEqual(len(tracker.motion_suspicion), 1)

    def test_original_prediction_dictionary_not_mutated(self):
        tracker = self.make_tracker()
        tracker.motion_suspicion = [dict(t=.7, innovation=np.array([0., -.2, .1]))]
        original = dict(position=tracker.static_position.copy(), velocity=np.zeros(3))
        expected = original['position'].copy()
        with patch.object(tracker, '_predict', return_value=original):
            result = tracker._association_prediction(.74)
        self.assertIsNot(result, original)
        np.testing.assert_array_equal(original['position'], expected)

    def test_confirmed_stale_future_and_age_boundary(self):
        tracker = self.make_tracker()
        original = dict(position=tracker.static_position.copy(), velocity=np.zeros(3))
        for age in [-.001, .050001]:
            tracker.motion_suspicion = [dict(t=0., innovation=np.array([0., -.2, .1]))]
            with patch.object(tracker, '_predict', return_value=original):
                self.assertIs(tracker._association_prediction(age), original)
        for age in [0., .05]:
            with patch.object(tracker, '_predict', return_value=original):
                self.assertIsNot(tracker._association_prediction(age), original)
        tracker.motion_onset_s = .6
        with patch.object(tracker, '_predict', return_value=original):
            self.assertIs(tracker._association_prediction(.02), original)

    def circle_packet(self, time_s=.74):
        angle = np.linspace(0., 2*np.pi, 128, endpoint=False)
        events = np.column_stack((np.full(128, time_s-.006),
            320.+8.66*np.cos(angle), 326.6+8.66*np.sin(angle), np.ones(128)))
        T = np.eye(4)
        T[:3, :3] = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        T[:3, 3] = [0., 0., 1.4]
        return dict(events=events, available_s=time_s+.01, end_s=time_s, T=T)

    def test_detector_uses_hook_without_relaxing_distance_gate(self):
        tracker = self.make_tracker()
        packet = self.circle_packet()
        # The synthetic sphere is around world y=2.0, outside the original
        # 0.30m gate centered at the initialized y=2.4 position.
        self.assertIsNone(tracker._detect(packet))
        tracker.motion_suspicion = [dict(t=.7, innovation=np.array([0., -.28, 0.]))]
        self.assertIsNotNone(tracker._detect(packet))
        np.testing.assert_array_equal(tracker._predict(.74)['position'], tracker.static_position)
        self.assertIsNone(tracker.motion_onset_s)
        # The same hook still rejects a candidate over0.30m from its association
        # prediction. A stale tentative sample falls back to the original gate.
        tracker.motion_suspicion[0]['t'] = .6
        self.assertIsNone(tracker._detect(packet))


if __name__ == '__main__':
    unittest.main()
