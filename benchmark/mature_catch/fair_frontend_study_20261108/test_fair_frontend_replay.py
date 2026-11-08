"""Small contract tests, never reported as real robot/cache results."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import copy
import os
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import numpy as np
from fair_frontend_replay import (CANDIDATES, CANONICAL, Candidate, PublicCache,
    aggregate_and_select, evaluate, grade, infer, native_visibility, write)

HERE = Path(__file__).resolve().parent


def trace_rows():
    return [dict(t_before=t, scoring_only_GT_position_at_control=[1., 2., 3.],
        qvel=[1., 0., 0., 0., 0., 0.], hand_contact=False) for t in [0., .1, .2]]


def visible_state(t):
    return dict(time_s=t, available=True, moving=True, position=[1., 2., 3.], velocity=[1., 0., 0.])


class ToyTracker:
    """Checks clocks/data routing only; it is not a perception implementation."""
    def __init__(self, K, resolution, radius):
        self.observation_window_ms = float(os.environ['CATCHIT_EVENT_OBSERVATION_WINDOW_MS'])
        self.use_flight_fit = os.environ['CATCHIT_EVENT_FLIGHT_FIT'] == '1'
        self.confirmation_max_gap_s = float(os.environ['CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS'])/1000.
        self.pending = deque(); self.samples = []; self.motion_onset_s = None
        self.static_latest_s = 0.; self.last_event_time = -np.inf

    def initialize_rgb(self, initial, time_s, available_s):
        self.initial = initial

    def push(self, events, available_s, packet_time_s, camera_to_world, parent_packet_id):
        self.pending.append(dict(events=events, available_s=available_s, end_s=packet_time_s))

    def _detect(self, packet):
        if not len(packet['events']): return None
        t = float(packet['events'][-1, 0]); self.motion_onset_s = t
        return dict(t=t, position=[1., 2., 3.], velocity=[1., 0., 0.],
                    available_s=packet['available_s'], maximum_event_s=t)

    def _predict(self, now):
        return dict(position=[1., 2., 3.], velocity=[1., 0., 0.], fit_error_m=0.,
                    latest_s=self.samples[-1]['t'] if self.samples else 0.)


class FairScreenTests(unittest.TestCase):
    def test_missing_queries_cannot_disappear_from_error_score(self):
        states = [visible_state(0.), visible_state(.1), dict(time_s=.2, available=False, moving=False)]
        metric = grade(states, trace_rows(), .05, [1, 2])
        self.assertEqual(metric['queries'], 2)
        self.assertEqual(metric['moving_coverage'], .5)
        self.assertEqual(metric['position_error_m_conditional']['mean'], 0.)
        self.assertEqual(metric['penalized_position_error_m']['mean'], .25)
        self.assertEqual(metric['penalized_velocity_error_m_s']['mean'], 1.)
        self.assertEqual(metric['normalized_equal_weight_loss'], .5)

    def test_large_errors_are_capped_but_uncapped_diagnostics_remain(self):
        states = [visible_state(0.), visible_state(.1), visible_state(.2)]
        states[1]['position'] = [11., 2., 3.]; states[1]['velocity'] = [11., 0., 0.]
        metric = grade(states, trace_rows(), .05, [1])
        self.assertEqual(metric['normalized_equal_weight_loss'], 1.)
        self.assertEqual(metric['position_error_m_conditional']['mean'], 10.)
        self.assertEqual(metric['velocity_error_m_s_conditional']['mean'], 10.)

    def test_false_published_motion_has_a_separate_fraction_penalty(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='score_contract_') as folder:
            run = Path(folder); rows = trace_rows()
            write(run/'episode_000/trace.json', rows)
            write(run/'episode_000/result.json', dict(initial=dict(release_time_s=.05), success=False))
            inference = dict(states=[visible_state(t) for t in [0., .1, .2]])
            metric = evaluate(run, inference, dict(exact_public_state_prefix_queries=3))
            self.assertEqual(metric['false_moving_pre_release_fraction'], 1.)
            self.assertEqual(metric['full_precontact']['selection_loss'], 1.)

    def test_visibility_keeps_native_eighty_ms_predictions(self):
        state = dict(position=np.ones(3), velocity=np.zeros(3), latest_s=.10, fit_error_m=.01)
        tracker = SimpleNamespace(motion_onset_s=.10, static_latest_s=0.)
        self.assertIs(native_visibility(state, tracker, .185), state)
        self.assertIsNone(native_visibility(state, tracker, .501))
        self.assertIsNone(native_visibility(dict(state, fit_error_m=.121), tracker, .12))

    def test_raw_parent_availability_and_query_schedule_are_preserved(self):
        public = PublicCache(dict(valid=True, public_world_gravity_m_s2=[0., 0., -9.81]),
            np.array([0., .04, .08]), np.array([0., .04]), np.repeat(np.eye(4)[None], 2, axis=0),
            np.eye(3), (640, 480),
            [dict(time_s=0., available_s=.01, events=0), dict(time_s=.04, available_s=.05, events=1)],
            np.array([[.035, 3., 4., 1.]]), np.array([0, 0, 1]), .04, {}, {})
        a = infer(public, ToyTracker, Candidate(0, False, 50))
        b = infer(public, ToyTracker, Candidate(0, False, 50))
        self.assertEqual(a['states'], b['states'])
        self.assertEqual([s['time_s'] for s in a['states']], [0., .04, .08])
        self.assertEqual([s['moving'] for s in a['states']], [False, False, True])
        self.assertEqual(a['consumed_packets'], 2)
        self.assertFalse(hasattr(public, 'target_GT'))

    def test_same_candidate_union_goes_to_both_sources_and_variants_do_not_add_cells(self):
        records = []
        for source in ['ours', 'v2e']:
            for seed in [1, 2]:
                values = {}
                for candidate in CANDIDATES:
                    preferred = candidate.flight_fit if source == 'ours' else not candidate.flight_fit
                    loss = .1 if preferred else .3
                    values[candidate.identifier] = dict(status='PASS', metrics=dict(full_precontact=dict(selection_loss=loss, moving_coverage=.8)))
                records.append(dict(entry=dict(method=source, seed=seed, endpoint_hz=25.,
                    capture_variant='a', cache_id=f'{source}_{seed}'), candidates=values))
        base = aggregate_and_select(records)
        repeated = []
        for record in records:
            duplicate = copy.deepcopy(record); duplicate['entry']['capture_variant'] = 'b'
            repeated.extend([record, duplicate])
        other = aggregate_and_select(repeated)
        self.assertEqual(base['same_live_validation_candidate_union'], other['same_live_validation_candidate_union'])
        self.assertEqual(base['ranking_by_source']['ours'][0]['loss'], other['ranking_by_source']['ours'][0]['loss'])
        self.assertEqual(base['ranking_by_source']['ours'][0]['independent_seed_cadence_cells'], 2)
        ids = {c['candidate_id'] for c in base['same_live_validation_candidate_union']}
        self.assertIn(CANONICAL, ids); self.assertLessEqual(len(ids), 5)

    def test_unmatched_source_screening_budget_is_rejected(self):
        with self.assertRaises(AssertionError):
            aggregate_and_select([dict(entry=dict(cache_id='unmatched', method='ours', seed=1, endpoint_hz=25., capture_variant='a'), candidates={})])


if __name__ == '__main__':
    unittest.main(verbosity=2)
