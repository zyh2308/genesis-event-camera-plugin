"""Small state-estimator checks; this is not a physical catch experiment."""
import json
import numpy as np

from event_sphere_tracker import EventSphereTracker

K = np.array([[433., 0., 320.], [0., 433., 240.], [0., 0., 1.]])
tracker = EventSphereTracker(K, (640, 480), .04)
p0 = np.array([.1, 2.4, 1.5])
origin = np.array([0., -.1366, 1.4364])
tracker.initialize_rgb(dict(valid=True, position=p0.tolist(),
                            camera_origin=origin.tolist(), radius_px=7.2),
                       time_s=0., available_s=0.)
hand = np.eye(4)
hand[:3, 3] = [0., .5, .8]
state = tracker.observe(.04, hand, np.zeros(12))
assert np.allclose(state['position'], p0)
assert np.array_equal(state['velocity'], np.zeros(3))
assert np.array_equal(state['acceleration'], np.zeros(3))
assert state['maximum_event_s'] is None
assert state['fresh_event'] is False

# Inject calibrated measurement samples, not a simulator release flag.
for t in np.arange(.50, .70, .01):
    dt = max(0., float(t) - .60)
    F, b, _ = tracker._transition(dt)
    p = (F @ np.r_[p0, [0., -1.8, 2.5]] + b)[:3]
    tracker.samples.append(dict(t=float(t), position=p.tolist(),
                                 camera_origin=origin.tolist(), radius_px=7.2,
                                 maximum_event_s=float(t), available_s=float(t)+.01))
    tracker.last_event_time = float(t)
    state = tracker.observe(float(t)+.01, hand, np.zeros(12))
assert tracker.motion_onset_s is not None
assert state['motion_phase'] == 'event_tracked_flight'
assert state['velocity'][1] < 0.
assert state['maximum_event_s'] <= state['time_s'] + 1e-9

# Once an estimated grasp is latched, actor position/velocity depend only on
# the hand transform and latched offset. Additional detections cannot change
# these actor inputs, which justifies stopping unused post-grasp acquisition.
tracker.attached_offset = np.array([.02, .03, .04])
first = tracker.observe(.8, hand, np.zeros(12))
tracker.samples.append(dict(t=.75, position=[.2, 1.8, 1.2],
                             camera_origin=origin.tolist(), radius_px=10.,
                             maximum_event_s=.75, available_s=.76))
tracker.last_event_time = .75
second = tracker.observe(.8, hand, np.zeros(12))
assert np.array_equal(first['position'], second['position'])
assert np.array_equal(first['velocity'], second['velocity'])
print(json.dumps(dict(status='PASS', scope='state-estimator unit checks, no physics',
                      static_velocity_zero=True, motion_detected_from_measurements=True,
                      motion_onset_s=tracker.motion_onset_s,
                      post_grasp_actor_inputs_independent_of_event_updates=True)))
