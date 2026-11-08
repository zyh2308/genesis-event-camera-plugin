"""Meaningful CPU-only checks; no physical data, real tests, or GPU contexts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pose_priority_policy import (BoundedFitter, center_retention, center_supervision,
                                  draw_auxiliary, gradient_evidence, masked_mean, state_sha)


class TinyCenterScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 1)
        self.center = nn.Conv2d(4, 1, 1)
        self.size = nn.Linear(4, 2)
        coords = (torch.arange(16) + .5) / 16
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('grid', torch.stack((xx, yy), -1).reshape(-1, 2))

    def forward(self, images):
        features = self.features(images).tanh()
        logits = self.center(features).flatten(1)
        uv = logits.softmax(1) @ self.grid
        size, shape = self.size(features.mean((2, 3))).unbind(1)
        return logits, uv, size, shape


def affine(images, labels, rng):
    # Independent analytical implementation of the existing shared augmentation.
    values = torch.as_tensor(rng.uniform(-1, 1, (len(images), 3)), dtype=images.dtype)
    zoom, shift = 1 + .15 * values[:, 0], .06 * values[:, 1:]
    theta = torch.zeros(len(images), 2, 3)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / zoom
    theta[:, :, 2] = -2 * shift / zoom[:, None]
    result = labels.clone()
    result[:, :2] = (result[:, :2] - .5) * zoom[:, None] + .5 + shift
    result[:, 2:4] += zoom.log()[:, None]
    return F.grid_sample(images, F.affine_grid(theta, images.shape, align_corners=False), align_corners=False), result


def matched_affine(images, normal, labels, rng):
    other = np.random.default_rng(); other.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    first, target = affine(images, labels, rng)
    second, paired_target = affine(normal, labels, other)
    assert torch.equal(target, paired_target)
    return first, second, target


def write(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, allow_nan=False)


def fake_supervised(model, predicted, target):
    return (center_supervision(model, predicted, target)
            + F.smooth_l1_loss(predicted[2], target[:, 2], beta=.1)
            + F.smooth_l1_loss(predicted[3], target[:, 4], beta=.1))


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_paired_condition_schedule_is_reproducible_and_half_prioritized(self):
        a, mask = draw_auxiliary(np.random.default_rng(12), 17, 6000)
        b, other = draw_auxiliary(np.random.default_rng(12), 17, 6000)
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(mask, other)
        self.assertTrue(np.all(a[mask] < 17) and np.all(a[~mask] >= 17) and np.all(a < 34))
        self.assertLess(abs(mask.mean() - .75), .02)
        with self.assertRaises(ValueError): draw_auxiliary(np.random.default_rng(), 0, 3)

    def test_eighth_retention_is_exactly_zero_without_renormalizing_half(self):
        values = torch.tensor([2., 100.], requires_grad=True)
        loss = masked_mean(values, torch.tensor([True, False]))
        self.assertEqual(float(loss.detach()), 1.)
        loss.backward(); torch.testing.assert_close(values.grad, torch.tensor([.5, 0.]))
        self.assertEqual(float(masked_mean(values, torch.zeros(2, dtype=torch.bool)).detach()), 0.)

    def test_same_view_frozen_teacher_has_zero_initial_retention_gradient(self):
        torch.manual_seed(3); model = TinyCenterScale().double()
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        images = torch.rand(2, 3, 16, 16, dtype=torch.double)
        prediction = model(images)
        with torch.no_grad(): reference = teacher(images)
        loss = center_retention(prediction, reference).mean()
        loss.backward()
        self.assertLess(abs(float(loss.detach())), 1e-12)
        self.assertTrue(all(p.grad is None or float(p.grad.abs().max()) < 1e-12 for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_gradient_probe_identifies_direct_conflict(self):
        parameter = nn.Parameter(torch.tensor([1., 2.]))
        value = gradient_evidence(parameter.sum(), -2 * parameter.sum(), (parameter,))
        self.assertAlmostEqual(value['cosine'], -1.)
        self.assertAlmostEqual(value['consistency_gradient_l2'] / value['supervised_gradient_l2'], 2.)

    def test_cpu_smoke_all_paired_arms_rebuild_and_label_identity(self):
        torch.manual_seed(22)
        normal = torch.rand(64, 3, 16, 16)
        target = torch.zeros(64, 9); target[:, :2] = .45; target[:, 2] = -.2; target[:, 4] = .3
        aux_clean = torch.cat((normal, normal))
        relit = torch.cat((normal * .7, normal * .1))
        aux_target = torch.cat((target, target))
        state = copy.deepcopy(TinyCenterScale().state_dict()); expected = state_sha(state)
        shared = SimpleNamespace(METHODS=('eventgenesis', 'v2e'), affine=affine,
                                 supervised=fake_supervised, write=write,
                                 sha=lambda p: __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest())
        native = SimpleNamespace(shared=shared, matched_affine=matched_affine)
        with tempfile.TemporaryDirectory() as temporary:
            fitter = BoundedFitter(native, Path(temporary), True)
            with contextlib.redirect_stdout(io.StringIO()):
                for method in shared.METHODS:
                    for arm in ('clean_control', 'physical_supervised', 'paired_consistency'):
                        aux = aux_clean if arm == 'clean_control' else relit
                        for branch in ('depth', 'center'):
                            fn = fitter.fit if branch == 'depth' else fitter.fit_center
                            model, history = fn(SimpleNamespace(CenterScale=TinyCenterScale), state,
                                                normal, target, aux, aux_target, 0, True, aux_clean,
                                                2 if arm == 'paired_consistency' else 0)
                            self.assertEqual(len(history), 1)
            self.assertEqual(len(fitter.records), 12)
            self.assertEqual(len({r['labels_sha256'] for r in fitter.records}), 1)
            self.assertTrue(all(r['updates'] == 1 and r['teacher_unchanged']
                                and r['source_state_sha256'] == expected
                                and r['independent_model_optimizer_rebuild'] for r in fitter.records))
            self.assertEqual(state_sha(state), expected)
            # Probes must use the physical supervised arm, never clean duplicate inputs.
            self.assertEqual(len(fitter.probes), 4)


if __name__ == '__main__': unittest.main(verbosity=2)
