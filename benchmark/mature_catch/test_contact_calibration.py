"""Pure contact-profile checks; no Genesis, policy, rendering or physics."""
import copy
import unittest
from types import SimpleNamespace

from contact_calibration import ContactCalibration, profile_config


class Solver:
    is_built = True

    def __init__(self):
        self.mu = {0: 1.0, 1: .8, 2: 1.0, 3: .95, 4: 1.1}
        self.ratio = {i: 1.0 for i in self.mu}
        self.params = {i: [.004, 1., .9, .95, .001, .5, 2.] for i in self.mu}

    def get_geoms_friction(self, ids):
        return [self.mu[i] for i in ids]

    def get_geoms_friction_ratio(self, ids):
        return [[self.ratio[i] for i in ids]]

    def get_sol_params(self, *, geoms_idx):
        return [self.params[i][:] for i in geoms_idx]


class Geom:
    def __init__(self, solver, idx, name):
        self._solver = solver
        self.idx = idx
        self.link = SimpleNamespace(name=name)
        self.friction = solver.mu[idx]

    def set_friction(self, value):
        self.friction = value
        self._solver.mu[self.idx] = value


def robot():
    s = Solver()
    names = ("floor", "palm", "finger", "object", "wheel")
    return SimpleNamespace(solver=s, geoms=[Geom(s, i, n) for i, n in enumerate(names)])


class ContactProfileTests(unittest.TestCase):
    def test_configs_are_explicit_and_fresh(self):
        self.assertEqual(profile_config("combined"), dict(name="combined", substeps=2, hand_friction_scale=1.5))
        x = profile_config("baseline"); x["substeps"] = 99
        self.assertEqual(profile_config("baseline")["substeps"], 1)
        with self.assertRaises(ValueError): profile_config("guess")

    def test_baseline_does_not_modify_any_collider(self):
        r = robot(); before = copy.deepcopy(r.solver.mu)
        a = ContactCalibration(r, "baseline", [1, 2]).apply()
        self.assertEqual(before, r.solver.mu)
        self.assertEqual(a["changed_geom_ids"], [])

    def test_only_selected_hand_changes_and_never_compounds(self):
        r = robot(); c = ContactCalibration(r, "hand_friction1p5", [1, 2])
        for _ in range(3):
            a = c.apply()
            self.assertAlmostEqual(r.solver.mu[1], 1.2)
            self.assertEqual(r.solver.mu[2], 1.5)
            self.assertEqual([r.solver.mu[i] for i in [0, 3, 4]], [1., .95, 1.1])
            self.assertEqual(a["changed_geom_ids"], [1, 2])
            self.assertTrue(a["unselected_unchanged"])
        self.assertEqual(c.baseline_mu, {0: 1., 1: .8, 2: 1., 3: .95, 4: 1.1})

    def test_substeps_only_does_not_set_friction(self):
        r = robot(); c = ContactCalibration(r, "substeps2", [1, 2])
        for g in r.geoms:
            g.set_friction = lambda value: self.fail("Substeps-only profile set friction")
        a = c.apply()
        self.assertEqual(a["profile"]["substeps"], 2)

    def test_refuses_invalid_or_duplicate_ids(self):
        for ids in ([], [1, 1], [99]):
            with self.assertRaises(ValueError): ContactCalibration(robot(), "baseline", ids)

    def test_detects_unselected_or_ratio_changes(self):
        r = robot(); c = ContactCalibration(r, "combined", [1, 2])
        r.geoms[0].set_friction(2.)
        with self.assertRaises(RuntimeError): c.apply()
        r = robot(); c = ContactCalibration(r, "combined", [1, 2]); r.solver.ratio[3] = 2.
        with self.assertRaises(RuntimeError): c.apply()

    def test_reads_live_solver_and_constraints(self):
        r = robot(); c = ContactCalibration(r, "baseline", [1, 2]); r.solver.mu[1] = 2.
        with self.assertRaises(RuntimeError): c.apply()
        r = robot(); c = ContactCalibration(r, "baseline", [1, 2]); r.solver.params[3][0] = .9
        with self.assertRaises(RuntimeError): c.apply()


if __name__ == "__main__":
    unittest.main()
