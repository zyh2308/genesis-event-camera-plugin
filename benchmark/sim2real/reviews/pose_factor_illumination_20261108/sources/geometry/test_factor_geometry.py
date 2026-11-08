"""Meaningful CPU controls for anchor geometry, alignment and macro metrics."""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import factor_geometry as f


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])


def fixture():
    # The nonzero anchor and rotated GT distinguish anchor depth from origin z.
    vertices = np.array(list(__import__('itertools').product([-.02, .02], repeat=3)))
    anchor = np.array([.017, -.008, .011])
    keys = np.vstack((anchor, vertices))
    assets = f.FactorAssets(vertices, keys, .04 * math.sqrt(3))
    K = np.array([[300., 0., 173.], [0., 301., 130.], [0., 0., 1.]])
    R = ry(.7) @ rz(.4)
    t = np.array([.03, -.02, .6])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    p = R @ anchor + t
    pixel = K @ p
    uv = pixel[:2] / pixel[2] / f.PIXELS
    record = dict(method='eventgenesis', seed=0, condition='normal', rid='r0', sample_id='w0',
                  K=K.tolist(), T_gt=T.tolist(), anchor=anchor.tolist(), uv_gt=uv.tolist(),
                  gt_log_size=math.log(.1), gt_log_shape=0., predictions={})
    exact = dict(uv=uv, z_m=p[2], R=R, size=.1)
    record['predictions'] = {a: copy.deepcopy(exact) for a in f.CANONICAL_ARMS}
    return record, assets


def flatten(record):
    result = []
    for arm, p in record['predictions'].items():
        row = {k: v for k, v in record.items() if k != 'predictions'}
        row.update(arm=arm, pred_uv=p['uv'].tolist(), z_m=float(p['z_m']), size=float(p['size']),
                   R_pred=p['R'].tolist(), t_pred_m=f.decode_translation(record['K'], record['anchor'], p['uv'], p['z_m'], p['R']).tolist())
        result.append(row)
    return result


class GeometryTests(unittest.TestCase):
    def test_anchor_decode_zero_and_all_gt_ceiling(self):
        rec, assets = fixture()
        gt = f._gt(rec)
        self.assertGreater(abs(gt['z_m'] - gt['t'][2]), .001)
        np.testing.assert_allclose(f.decode_translation(rec['K'], rec['anchor'], gt['uv'], gt['z_m'], gt['R']), gt['t'], atol=1e-15)
        score = f.score_factors(rec, rec['predictions']['C0'], assets)
        self.assertLess(score['translation_error_m'], 1e-14)
        self.assertLess(score['adds_m'], 1e-14)
        self.assertEqual(score['success'], 1)
        rec['predictions']['C0']['uv'] += [.1, -.05]
        rec['predictions']['C0']['z_m'] *= 1.2
        rec['predictions']['C0']['R'] = np.eye(3)
        oracles = f.factor_oracles(rec, 'C0', assets)
        self.assertEqual(len(oracles), 8)
        allgt = [r for r in oracles if r['variant'].endswith('u1z1r1')][0]
        self.assertLess(allgt['metrics']['translation_error_m'], 1e-14)
        self.assertLess(allgt['metrics']['adds_m'], 1e-14)
        self.assertTrue(allgt['uses_privileged_gt'])
        self.assertFalse(oracles[0]['uses_privileged_gt'])

    def test_rotation_substitution_recomputes_origin_translation(self):
        rec, assets = fixture()
        rec['predictions']['C0']['R'] = np.eye(3)
        native = f.score_factors(rec, rec['predictions']['C0'], assets)
        self.assertLess(native['anchor_position_error_m'], 1e-14)
        self.assertGreater(native['translation_error_m'], .005)
        oracle = [r for r in f.factor_oracles(rec, 'C0', assets) if r['variant'].endswith('u0z0r1')][0]
        self.assertLess(oracle['metrics']['translation_error_m'], 1e-14)

    def test_all8_prediction_crosses_identity_and_source(self):
        rec, assets = fixture()
        crosses = f.prediction_crosses(rec, assets)
        self.assertEqual(len(crosses), 8)
        self.assertEqual(len({r['variant'] for r in crosses}), 8)
        self.assertTrue(all(r['metrics']['success'] == 1 for r in crosses))
        self.assertTrue(all(not r['uses_privileged_gt'] for r in crosses))
        rec['predictions']['C1']['z_m'] += .1
        for row in f.prediction_crosses(rec, assets):
            self.assertAlmostEqual(row['metrics']['depth_mae_m'], .1 if row['sources']['z_m'] == 'C1' else 0)

    def test_rotation_angle_symmetry_caveat(self):
        rec, assets = fixture()
        rec['T_gt'] = np.eye(4).tolist()
        rec['T_gt'][2][3] = .5
        rec['anchor'] = [0., 0., 0.]
        assets.keypoints_object_m[0] = [0., 0., 0.]
        rec['uv_gt'] = [.5, .5]
        p = dict(uv=np.array([.5, .5]), z_m=.5, R=rz(math.pi), size=.1)
        score = f.score_factors(rec, p, assets)
        self.assertAlmostEqual(score['rotation_error_deg'], 180.)
        self.assertLess(score['adds_m'], 1e-14)
        self.assertEqual(score['success'], 1.)

    def test_native_frontality_all_keypoints(self):
        rec, assets = fixture()
        # Anchor stays in front; a required distant keypoint goes behind.
        assets.keypoints_object_m = np.vstack((assets.keypoints_object_m, [0., 0., -2.]))
        score = f.score_factors(rec, rec['predictions']['C0'], assets)
        self.assertEqual(score['valid_pose'], 0)
        self.assertIsNone(score['adds_m'])
        self.assertEqual(score['success'], 0.)
        self.assertEqual(score['auc'], 0.)

    def test_success_threshold_is_strict_and_metric_not_percent(self):
        assets = f.FactorAssets(np.zeros((1,3)), np.zeros((1,3)), 1.25)
        T = np.eye(4); T[2,3] = .5
        rec = dict(K=[[300,0,173],[0,300,130],[0,0,1]], anchor=[0,0,0],
                   T_gt=T, uv_gt=[.5,.5], gt_log_size=math.log(.1))
        p = dict(uv=np.array([.5,.5]),z_m=.625,R=np.eye(3),size=.1)
        score = f.score_factors(rec,p,assets)
        self.assertEqual(score['adds_m'], .125)
        self.assertEqual(score['success'], 0.)

    def test_invalid_rotation_units_and_labels_rejected(self):
        rec, assets = fixture()
        bad = copy.deepcopy(rec['predictions']['C0'])
        bad['R'] *= 1.1
        with self.assertRaisesRegex(ValueError, 'SO'):
            f.score_factors(rec, bad, assets)
        bad = copy.deepcopy(rec)
        bad['uv_gt'] = (np.array(rec['uv_gt']) * f.PIXELS).tolist()
        with self.assertRaisesRegex(ValueError, 'projection|projected'):
            f.score_factors(bad, rec['predictions']['C0'], assets)
        rows = flatten(rec)
        rows[0]['t_pred_m'] = (np.array(rows[0]['t_pred_m']) * 100).tolist()
        with self.assertRaisesRegex(ValueError, 't_pred_m'):
            f.group_flat_predictions(rows)

    def test_explicit_alignment_duplicate_missing_mismatch(self):
        rec, _ = fixture()
        rows = flatten(rec)
        self.assertEqual(len(f.group_flat_predictions(list(reversed(rows)))), 1)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            f.group_flat_predictions(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            f.group_flat_predictions(rows[:-1])
        bad = copy.deepcopy(rows)
        bad[1]['gt_log_size'] += .01
        with self.assertRaisesRegex(ValueError, 'label mismatch'):
            f.group_flat_predictions(bad)
        bad = copy.deepcopy(rows)
        bad[1]['sample_id'] = 'different_window'
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            f.group_flat_predictions(bad)

    def test_equal_rollout_not_pooled_mean_and_invalid_accounting(self):
        rec, assets = fixture()
        score = f.score_factors(rec, rec['predictions']['C0'], assets)
        rows = []
        for rid, n, center in [('a', 1, 0), ('b', 9, 10)]:
            for i in range(n):
                row = dict(method='eg', seed=0, condition='normal', variant='actual/C0', rid=rid, sample_id=str(i), metrics=dict(score))
                row['metrics']['center_px'] = center
                rows.append(row)
        result = f.aggregate_rollout_equal(rows)[0]
        self.assertEqual(result['summary']['center_px'], 5)
        self.assertEqual(result['sample_count'], 10)
        self.assertEqual(result['rollout_count'], 2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            f.aggregate_rollout_equal(rows + [rows[0]])

    def test_native_parity_true_and_detects_formula_change(self):
        rec, assets = fixture()
        aggregates = f.aggregate_rollout_equal(list(f.evaluate_records([rec], assets)))
        native = [dict(method=r['method'], seed=r['seed'], condition=r['condition'], arm=r['variant'].split('/')[1],
                       score=dict(summary={k:r['summary'][k] for k in f.NATIVE_METRICS},
                                  by_rollout={rid:{k:v[k] for k in f.NATIVE_METRICS} for rid,v in r['by_rollout'].items()}))
                  for r in aggregates if r['variant'].startswith('actual/')]
        self.assertEqual(f.native_parity_report(aggregates, native)['status'], 'passed')
        native[0]['score']['summary']['depth_mae_m'] += .01
        self.assertEqual(f.native_parity_report(aggregates, native)['status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            f.native_parity_report(aggregates, native[:-1])

    def test_cli_unique_output_and_diagnostic_artifacts(self):
        rec, assets = fixture()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d/'vertices.npy', assets.vertices_m)
            np.save(d/'keypoints.npy', assets.keypoints_object_m)
            (d/'pred.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in flatten(rec)))
            argv = [sys.executable, '-B', str(Path(f.__file__)), 'summarize', '--predictions', str(d/'pred.jsonl'),
                    '--vertices', str(d/'vertices.npy'), '--keypoints', str(d/'keypoints.npy'), '--diameter-m', str(assets.diameter_m), '--out', str(d/'out')]
            result = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            completion = json.loads((d/'out'/'completion.json').read_text())
            self.assertEqual(completion['optimizer_updates'], 0)
            self.assertFalse(completion['gt_oracles_deployable'])
            self.assertEqual(completion['scored_variants'], 35)
            self.assertEqual(completion['native_parity'], 'not_checked')
            second = subprocess.run(argv, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('fresh directory', second.stderr)


class CountGainTests(unittest.TestCase):
    def representation(self):
        counts = np.array([[[0, 1], [2, 8]], [[0, 0], [3, 8]]], dtype=float)
        return np.concatenate((np.minimum(np.log1p(counts)/math.log(9), 1),
                               np.minimum(np.log1p(counts.sum(0))[None]/math.log(9), 1))).astype(np.float32)

    def test_identity_byte_exact_and_no_mutation(self):
        image = self.representation()
        before = image.tobytes()
        result = f.controlled_count_gain(image, 1)
        self.assertEqual(result.tobytes(), before)
        self.assertEqual(image.tobytes(), before)
        self.assertFalse(np.shares_memory(image, result))

    def test_registered_gain_support_and_total_reconstruction(self):
        image = self.representation()
        for gain in (.5, 2):
            result = f.controlled_count_gain(image, gain)
            np.testing.assert_array_equal(result > 0, image > 0)
            self.assertEqual(result.dtype, image.dtype)
            self.assertGreaterEqual(float(result.min()), 0)
            self.assertLessEqual(float(result.max()), 1)
            # Neither polarity clipped at [1,0], so a simple known count oracle.
            self.assertAlmostEqual(float(result[0,0,1]), math.log1p(gain)/math.log(9), places=6)
            self.assertAlmostEqual(float(result[2,0,1]), math.log1p(gain)/math.log(9), places=6)

    def test_reject_resized_inconsistency_invalid_domain(self):
        # Independent channel interpolation is NOT invertible as raw counts.
        image = np.array([[[.5]], [[.5]], [[1.]]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'resized'):
            f.controlled_count_gain(image, 1)
        good = self.representation()
        bad = good.copy(); bad[0,0,0] = -1e-6
        with self.assertRaises(ValueError):
            f.controlled_count_gain(bad, 2)
        with self.assertRaises(ValueError):
            f.controlled_count_gain(good, 0)
        with self.assertRaises(ValueError):
            f.controlled_count_gain(good.astype(np.int32), .5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
