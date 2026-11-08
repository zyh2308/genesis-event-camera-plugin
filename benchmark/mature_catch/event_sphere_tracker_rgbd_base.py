"""One image establishes a static target; events detect and track its motion.

The image is never refreshed. Motion onset is inferred from compensated event
measurements, not from the simulator's release flag or target state.
"""
import numpy as np

from event_sphere_tracker_kalman import EventSphereTracker as MotionFilter


class EventSphereTracker(MotionFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb_initialization = None
        self.static_position = None
        self.static_covariance = None
        self.static_latest_s = 0.
        self.static_measurements = 0
        self.motion_suspicion = []
        self.motion_onset_s = None
        self.minimum_filter_measurements = 1
        self.minimum_filter_span_s = 0.

    def initialize_rgb(self, detection, time_s=0., available_s=0.):
        assert detection['valid']
        self.rgb_initialization = dict(detection, time_s=float(time_s),
                                       available_s=float(available_s))
        self.static_position = np.asarray(detection['position'], dtype=float)
        sample = dict(position=detection['position'],
                      camera_origin=detection['camera_origin'],
                      radius_px=detection['radius_px'])
        R,n=self._measurement_cov(sample)
        ratio=float(detection.get('radius_uncertainty_px',.65))/.65
        radial=float(n@R@n)
        if 'range_std_m' in detection:
            self.static_covariance=R+(float(detection['range_std_m'])**2-radial)*n[:,None]*n[None,:]
        else:self.static_covariance = R+(ratio*ratio-1)*radial*n[:,None]*n[None,:]
        self.initialization_label=('one RGB-D frame at time 0' if detection.get('initialization_mode')=='single_frame_RGBD' else 'one RGB frame at time 0')
        self.static_latest_s = float(time_s)

    def _association_prediction(self, time_s):
        prediction = super()._association_prediction(time_s)
        if (prediction is None or self.motion_onset_s is not None or
                not self.motion_suspicion):
            return prediction
        last = self.motion_suspicion[-1]
        dt = float(time_s - last['t'])
        if not 0. <= dt <= .05:
            return prediction
        # A first motion observation can guide candidate association before
        # confirmation, while the public state remains the initialized static
        # target. Keep the original distance gate and two-observation rule.
        center = (self.static_position + last['innovation'] +
                  self.initial_velocity_prior * dt + .5 * self.gravity * dt * dt)
        return dict(prediction, position=center)

    def _start_motion(self, sample):
        t = sample['t']
        dt = t - self.motion_suspicion[0]['t']
        predicted = self.static_position + self.initial_velocity_prior * dt
        origin = np.asarray(sample['camera_origin'])
        ray = np.asarray(sample['position']) - origin
        ray /= np.linalg.norm(ray)
        # Preserve the observed bearing and the range established before motion.
        position = origin + ray * float((predicted - origin) @ ray)
        self.filter_mean = np.r_[position, self.initial_velocity_prior]
        self.filter_cov = np.zeros((6, 6))
        self.filter_cov[:3, :3] = self.static_covariance + np.eye(3) * .02**2
        self.filter_cov[3:, 3:] = np.diag([.75**2, .45**2, 3.**2])
        self.filter_time = t
        self.filter_first_time = t
        self.filter_accepted = 1
        self.motion_onset_s = t

    def _predict(self, time_s):
        if self.rgb_initialization is None:
            return super()._predict(time_s)
        if time_s < self.rgb_initialization['available_s'] - 1e-9:
            return None
        if self.motion_onset_s is None:
            eligible = [s for s in self.samples if s['t'] <= time_s + 1e-8]
            for sample in eligible[self.filter_processed:]:
                self.filter_processed += 1
                R, ray = self._measurement_cov(sample)
                innovation = np.asarray(sample['position']) - self.static_position
                S = self.static_covariance + R
                mahal = float(innovation @ np.linalg.solve(S, innovation))
                transverse = float(np.linalg.norm(np.cross(innovation, ray)))
                suspicious = mahal > 25 and transverse > .010
                if suspicious:
                    if (self.motion_suspicion and
                            (sample['t'] - self.motion_suspicion[-1]['t'] > .05 or
                             float(innovation @ self.motion_suspicion[-1]['innovation']) < 0)):
                        self.motion_suspicion = []
                    self.motion_suspicion.append(dict(t=sample['t'], innovation=innovation))
                    if (len(self.motion_suspicion) >= 2 and
                            sample['t'] - self.motion_suspicion[0]['t'] >= .005):
                        self._start_motion(sample)
                        break
                    continue
                self.motion_suspicion = []
                if mahal <= 25:
                    # Keep the one-frame world position while the target is held.
                    # Event contour radii have correlated quantization error;
                    # repeated static detections validate visibility and onset,
                    # rather than overpowering the image's initial range.
                    self.static_latest_s = sample['t']
                    self.static_measurements += 1
            if self.motion_onset_s is None:
                return dict(time_s=float(time_s), position=self.static_position.copy(),
                            velocity=np.zeros(3), acceleration=np.zeros(3), latest_s=float(self.static_latest_s),
                            n=self.static_measurements, fit_error_m=0.,
                            motion_phase='static_prediction',
                            initialization=self.initialization_label)
        result = super()._predict(time_s)
        if result is not None:
            result.update(motion_phase='event_tracked_flight',
                          motion_onset_s=float(self.motion_onset_s),
                          initialization=self.initialization_label)
        return result

    def observe(self, now_s, hand_pose, hand_joint_positions):
        while self.pending and self.pending[0]['available_s'] <= now_s + 1e-9:
            packet = self.pending.popleft()
            sample = self._detect(packet)
            if sample is not None:
                self.samples.append(sample)
                self.last_event_time = max(self.last_event_time, sample['maximum_event_s'])
        state = self._predict(now_s)
        if state is None:
            return None
        if self.motion_onset_s is None:
            # A prediction from an old frame is explicitly not a fresh event observation.
            if now_s - self.static_latest_s > 1.0:
                return None
            latest_event = float(self.last_event_time) if np.isfinite(self.last_event_time) else None
            state.update(source=('single_RGBD_initialized_static_target_prediction' if self.rgb_initialization.get('initialization_mode')=='single_frame_RGBD' else 'single_RGB_initialized_static_target_prediction'),
                         maximum_event_s=latest_event,
                         fresh_event=self.static_measurements > 0 and now_s-self.static_latest_s <= .04,
                         RGB_frame_time_s=self.rgb_initialization['time_s'],
                         RGB_available_s=self.rgb_initialization['available_s'])
            self.diagnostics.append({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in state.items()})
            return state
        result = super().observe(now_s, hand_pose, hand_joint_positions)
        if result is not None:
            result.update(RGB_frame_time_s=self.rgb_initialization['time_s'],
                          RGB_available_s=self.rgb_initialization['available_s'])
        return result
