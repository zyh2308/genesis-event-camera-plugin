"""CPU checks for the coordinate algebra and actual C1 one-step integration."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from depth_coordinates import (DepthCoordinateLinear, fit_depth, install_depth_coordinates,
                               native_state, state_sha, verify_prediction_parity)
import native_paired_source as native
from pose_priority_policy import BoundedFitter
from run_depth_coordinate_short import validate_contract


class TinyCenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8, padding=1), nn.SiLU())
        self.center = nn.Conv2d(4, 1, 1)
        self.size = nn.Sequential(nn.Linear(8, 6), nn.SiLU(), nn.Linear(6, 2))
        y, x = torch.meshgrid((torch.arange(16) + .5) / 16, (torch.arange(16) + .5) / 16,
                              indexing='ij')
        self.register_buffer('grid', torch.stack((x, y), -1).reshape(-1, 2))

    def forward(self, image):
        features = self.features(image)
        logits = self.center(features).flatten(1)
        attention = logits.softmax(1)
        features = features.flatten(2)
        pooled = (features * attention[:, None]).sum(2)
        scale, shape = self.size(torch.cat((pooled, features.mean(2)), 1)).unbind(1)
        return logits, attention @ self.grid, scale - 2.3, shape - 2.3


def training_fixture():
    rng = torch.Generator().manual_seed(824)
    normal = torch.rand(33, 3, 128, 128, generator=rng)
    target = torch.zeros(33, 9)
    target[:, :2] = .5
    target[:, 2] = -2.
    target[:, 3] = -.5
    target[:, 4] = .7
    return normal, target, torch.cat((normal * .5, normal * .125)), torch.cat((target, target)), torch.cat((normal, normal))


class DepthCoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_affine_algebra_in_float64_preserves_outputs_and_parameter_count(self):
        old = nn.Linear(5, 2).double()
        new = DepthCoordinateLinear(old)
        features = torch.randn(32, 5, dtype=torch.float64)
        torch.testing.assert_close(old(features), new(features), rtol=0, atol=1e-14)
        self.assertEqual(sum(p.numel() for p in old.parameters()), sum(p.numel() for p in new.parameters()))

    def test_native_round_trip_preserves_rng_and_all_other_parameters(self):
        original = TinyCenter().eval()
        modified = copy.deepcopy(original)
        rng = torch.get_rng_state().clone()
        install_depth_coordinates(modified)
        output = native_state(modified)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in original.state_dict().items():
            if not name.startswith('size.2.'):
                self.assertTrue(torch.equal(value, output[name]))
        rebuilt = TinyCenter().eval()
        rebuilt.load_state_dict(output, strict=True)
        inputs = training_fixture()[0][:2]
        verify_prediction_parity(original, modified, inputs, torch.zeros(2))
        verify_prediction_parity(original, rebuilt, inputs, torch.zeros(2))

    def test_depth_derivative_has_no_direct_scale_row_component(self):
        old = nn.Linear(3, 2).double()
        new = DepthCoordinateLinear(old)
        features = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
        for layer in (old, new):
            scale, shape = layer(features).unbind(-1)
            (shape - scale).sum().backward()
        self.assertTrue(torch.equal(old.weight.grad[0], -features[0]))
        self.assertTrue(torch.equal(new.weight.grad[0], torch.zeros(3, dtype=torch.float64)))
        self.assertTrue(torch.equal(new.weight.grad[1], features[0]))

    def test_actual_native_and_candidate_update_match_tape_and_first_scalar(self):
        normal, target, auxiliary, auxiliary_target, clean = training_fixture()
        scale = SimpleNamespace(CenterScale=TinyCenter)
        source = TinyCenter().state_dict()
        source_sha = state_sha(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / 'control'
            candidate = root / 'candidate'
            control.mkdir()
            candidate.mkdir()
            original = BoundedFitter(native, control, True)
            original.order = [('eventgenesis', 0, 'physical_supervised', 'depth')]
            original.fit(scale, source, normal, target, auxiliary, auxiliary_target, 0, True, clean, 0)
            path = control / 'training_evidence/eventgenesis_seed0_physical_supervised_depth.jsonl'
            old_log = [json.loads(row) for row in path.read_text().splitlines()]
            model, receipt = fit_depth(native, scale, source, (normal, target, auxiliary, auxiliary_target, clean),
                                       'eventgenesis', 0, True, candidate, old_log)
            self.assertEqual(receipt['updates'], 1)
            self.assertEqual(receipt['labels_sha256'], original.records[0]['labels_sha256'])
            self.assertTrue(receipt['first_native_loss_oracle']['passed'])
            self.assertTrue(receipt['teacher_unchanged'] and receipt['independent_model_adam_readback'])
            self.assertIs(type(model.size[-1]), nn.Linear)
            self.assertNotEqual(receipt['final_coordinate_state_sha256'], receipt['initial_coordinate_state_sha256'])
            self.assertEqual(state_sha(source), source_sha)
            changed = copy.deepcopy(old_log)
            changed[0]['normal_indices'][0] = -1
            rejected = root / 'rejected'
            rejected.mkdir()
            with self.assertRaisesRegex(ValueError, 'draw differs'):
                fit_depth(native, scale, source, (normal, target, auxiliary, auxiliary_target, clean),
                          'eventgenesis', 0, True, rejected, changed)
            self.assertFalse(any(rejected.rglob('*.pt')))

    def test_registered_budget_role_and_unexpected_head_fail_closed(self):
        contract = json.loads((Path(__file__).parent / 'DEPTH_COORDINATE_CONTRACT_TEMPLATE.json').read_text())
        validate_contract(contract)
        for name, value in [('independent32_read', True), ('normal_teacher_weight', 0), ('updates_per_run', 100)]:
            wrong = dict(contract, **{name: value})
            with self.assertRaises(ValueError):
                validate_contract(wrong)
        with self.assertRaises(ValueError):
            DepthCoordinateLinear(nn.Linear(3, 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
