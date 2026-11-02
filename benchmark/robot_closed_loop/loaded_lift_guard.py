"""Proprioceptive guard: coarse integrity in motion, micro drift at rest."""
import numpy as np
from benchmark.robot_closed_loop.grip_gate import ProprioceptiveGripGate, ProprioceptiveLiftStabilityMonitor


class LoadedLiftGuard:
    def __init__(self, micro_config, grip_config, dt_s, settled_speed_m_s=.01):
        self.dt_s = float(dt_s)
        self.settled_speed = float(settled_speed_m_s)
        self.coarse = ProprioceptiveGripGate(grip_config, dt_s)
        # A baseline collected once the arm has settled already describes the
        # final load; it need not wait the probe's initial movement interval.
        micro = dict(micro_config, baseline_start_s=0.)
        self.micro = ProprioceptiveLiftStabilityMonitor(micro, dt_s)
        self.reset()

    def reset(self, start_s=None):
        self.start_s = start_s
        self.coarse.reset()
        self.micro.reset()
        self.settled_for_s = 0.
        self.coarse_bad_for_s = 0.
        self.micro_started_s = None

    def update(self, now_s, close_started_s, tcp_speed_m_s, finger_q_m, finger_qd_m_s):
        speed = float(tcp_speed_m_s)
        if not np.isfinite(speed):
            return dict(state='invalid_robot_state', unstable=True, confirmed=False)
        coarse = self.coarse.update(now_s, close_started_s, finger_q_m, finger_qd_m_s)
        self.coarse_bad_for_s = (
            0. if coarse['instantaneous_candidate'] else self.coarse_bad_for_s + self.dt_s
        )
        gross_failure = self.coarse_bad_for_s + 1e-12 >= self.coarse.stable_dwell_s
        if speed <= self.settled_speed:
            self.settled_for_s += self.dt_s
        else:
            self.settled_for_s = 0.
            self.micro_started_s = None
            self.micro.reset()
        if (self.micro_started_s is None
                and self.settled_for_s + 1e-12 >= self.coarse.stable_dwell_s):
            self.micro_started_s = float(now_s)
            self.micro.reset(self.micro_started_s)
        if self.micro_started_s is None:
            result = dict(state='full_lift_in_motion', confirmed=False, unstable=False,
                          maximum_joint_drift_m=None)
        else:
            result = self.micro.update(now_s, self.micro_started_s, finger_q_m, finger_qd_m_s)
        result = dict(result)
        result.update(unstable=bool(result['unstable'] or gross_failure),
                      monitor_stage='loaded_lift', robot_TCP_speed_m_s=speed,
                      robot_settled_for_s=self.settled_for_s,
                      micro_baseline_started_s=self.micro_started_s,
                      coarse_joint_guard=coarse,
                      coarse_violation_for_s=self.coarse_bad_for_s,
                      joint_drift_threshold_m=self.micro.max_joint_drift_m,
                      input_boundary='robot TCP speed, finger q/qd and own command times only')
        if gross_failure:
            result['state'] = 'gross_grip_integrity_failure'
        return result
