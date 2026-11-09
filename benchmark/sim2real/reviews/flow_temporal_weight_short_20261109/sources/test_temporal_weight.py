import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np
import torch

import run_temporal_weight as runner
import temporal_photo as loss


def load(path, name, digest):
    if runner.sha(path) != digest:
        raise ValueError(f'Wrong frozen fixture bytes for {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemporalWeightTest(unittest.TestCase):
    def test_static_and_uniform_change_are_unit_weights(self):
        first = torch.zeros(2, 1, 8, 8, requires_grad=True)
        second = first.detach().clone()
        second[1] = .125
        weight = loss.endpoint_weights(first, second)
        torch.testing.assert_close(weight, torch.ones_like(weight), rtol=0, atol=0)
        self.assertFalse(weight.requires_grad)

    def test_changed_region_weight_is_bounded_and_mass_preserving(self):
        first = torch.zeros(2, 1, 4, 4)
        second = first.clone()
        second[0, 0, 0, 0] = .5
        second[1, 0, 2, :] = .025
        weight = loss.endpoint_weights(first, second)
        torch.testing.assert_close(weight.mean(dim=(-2, -1)), torch.ones(2, 1), rtol=0, atol=1e-7)
        self.assertAlmostEqual(float(weight[0, 0, 0, 0] / weight[0, 0, 0, 1]), 2.)
        self.assertAlmostEqual(float(weight[1, 0, 2, 0] / weight[1, 0, 0, 0]), 1.5)
        self.assertGreaterEqual(float(weight.min()), .5)
        self.assertLessEqual(float(weight.max()), 2.)

    def test_actual_native_loss_and_gradients_on_fixed_weights(self):
        fixtures = Path(__file__).resolve().parent.parent / 'flow_photometric_alignment_v2_20261108'
        native = load(Path(os.getenv('STE_NATIVE_LOSS_SOURCE', fixtures / 'pinned_multiscaleloss.py')),
                      'temporal_weight_original_loss', 'ff401317e49ce37ae0f8e9bfd6b82f133549b4a0271be35095f74bdc50a439c0')
        compat = load(Path(os.getenv('STE_NATIVE_COMPAT_SOURCE', fixtures / 'pinned_compat.py')),
                      'temporal_weight_original_compat', '0f7290aa45b8ba020c03d2d26b64a0cc213ccec7536e386bce186e4d0446393e')
        native.warp = compat.loss_warp
        rng = torch.Generator().manual_seed(841)
        first = torch.rand((2, 32, 32), generator=rng)
        last = first.clone()
        last[:, 6:12, 5:13] = .95
        fields = tuple((1.4 * torch.rand((2, 2, n, n), generator=rng) - .7).requires_grad_()
                       for n in (32, 16, 8, 4))
        level_weights = (1., 1., 1., 1.)
        # The original public entry converts each flow with Tensor.cuda().
        # Adapt that operation only so its full CPU loss remains testable.
        from unittest.mock import patch
        with patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
            official = native.compute_photometric_loss(first, last, None, fields, weights=level_weights)
        levels = loss.prepare(first, last, fields)
        equivalent, weighted = loss.photometric_terms(fields, levels, native.warp, level_weights)
        torch.testing.assert_close(equivalent, official, rtol=0, atol=0)
        unit_levels = [(a, b, torch.ones_like(w)) for a, b, w in levels]
        _, unit = loss.photometric_terms(fields, unit_levels, native.warp, level_weights)
        torch.testing.assert_close(unit, official, rtol=0, atol=0)
        old_gradients = torch.autograd.grad(official, fields, retain_graph=True)
        unit_gradients = torch.autograd.grad(unit, fields, retain_graph=True)
        new_gradients = torch.autograd.grad(weighted, fields)
        for old, identity in zip(old_gradients, unit_gradients, strict=True):
            torch.testing.assert_close(old, identity, rtol=0, atol=0)
        self.assertGreater(float((new_gradients[0] - old_gradients[0]).abs().max()), 1e-4)

    def test_primary_scene_two_rollouts_and_empty_support(self):
        results = []
        for method, arm, value in [('eventgenesis', 'native_control', .6), ('eventgenesis', 'temporal_weight', .4),
                                   ('v2e', 'native_control', .8), ('v2e', 'temporal_weight', .7)]:
            rollouts = []
            for rid in ('a', 'b'):
                windows = [{runner.REGION: {'pixels': 0, 'epe_px': None}}]
                windows += [{runner.REGION: {'pixels': 1, 'epe_px': value}}] * 65
                rollouts.append(dict(rollout=rid, stratum=runner.SCENE, windows=windows,
                                     regions={runner.REGION: {'epe_px': value}}))
            results.append(dict(method=method, arm=arm, validation=dict(rollouts=rollouts)))
        cache = {'eventgenesis': {'validation': {rid: {'sample_ids': np.arange(66)} for rid in ('a','b')}}}
        rows = runner.scene_rows(results, cache)
        self.assertEqual(len(rows['per_window']), 132)
        self.assertIsNone(rows['per_window'][0]['EG_candidate_minus_EG_control'])
        self.assertAlmostEqual(rows['per_window'][1]['EG_candidate_minus_EG_control'], -.2)
        self.assertTrue(rows['EG_beats_V2E_candidate_in_both_rollouts'])


if __name__ == '__main__':
    unittest.main()
