"""Common benchmark contact profiles; no perception, policy or score changes.

Instantiate after scene.build(), before applying a profile. ``hand_geom_ids``
must be an explicit, audited list of Genesis collision-geometry global IDs.
The object, ground, wheels and other robot links are never selected implicitly.
"""
from __future__ import annotations

import math


_PROFILES = {
    "baseline": (1, 1.0),
    "hand_friction1p5": (1, 1.5),
    "substeps2": (2, 1.0),
    "combined": (2, 1.5),
}


def profile_config(name):
    """Return a fresh configuration for scene construction and hand friction."""
    if name not in _PROFILES:
        raise ValueError(f"Unknown contact profile {name!r}; expected {tuple(_PROFILES)}")
    substeps, scale = _PROFILES[name]
    return dict(name=name, substeps=substeps, hand_friction_scale=scale)


def _flat(value):
    """Read tensor/array/list scalars without importing Genesis or torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [x for part in value for x in _flat(part)]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite contact parameter")
    return [number]


def _same(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


class ContactCalibration:
    """Apply hand-only friction from an immutable imported baseline.

    ``robot`` must expose actual Genesis collision ``geoms`` and their solver.
    This helper is intended for one environment, matching the current adapter.
    """

    def __init__(self, robot, profile_name, hand_geom_ids):
        self.config = profile_config(profile_name)
        self.geoms = {int(g.idx): g for g in robot.geoms}
        if not self.geoms or len(self.geoms) != len(robot.geoms):
            raise ValueError("Expected distinct actual collision-geometry IDs")
        supplied = [int(i) for i in hand_geom_ids]
        self.hand_ids = sorted(set(supplied))
        if not self.hand_ids or len(self.hand_ids) != len(supplied):
            raise ValueError("A nonempty unique hand-collider whitelist is required")
        if not set(self.hand_ids) <= self.geoms.keys():
            raise ValueError("Hand whitelist contains an unknown collision geometry")
        self.ids = sorted(self.geoms)
        self.solver = getattr(robot, "solver", None)
        if self.solver is None:
            self.solver = next(iter(self.geoms.values()))._solver
        if not self.solver.is_built:
            raise RuntimeError("ContactCalibration must be created after scene.build()")
        self.baseline = self._snapshot()
        self.baseline_mu = {r["geom_id"]: r["friction"] for r in self.baseline}
        for i in self.hand_ids:
            target = self.baseline_mu[i] * self.config["hand_friction_scale"]
            if not 0.01 <= target <= 5.0:
                raise ValueError(f"Unstable requested hand friction {target} for geom {i}")

    def _snapshot(self):
        mu = _flat(self.solver.get_geoms_friction(self.ids))
        ratios = _flat(self.solver.get_geoms_friction_ratio(self.ids))
        params = _flat(self.solver.get_sol_params(geoms_idx=self.ids))
        n = len(self.ids)
        if len(mu) != n or len(ratios) != n or len(params) != 7 * n:
            raise RuntimeError("Unexpected live contact parameter shape; expected one environment")
        rows = []
        for j, i in enumerate(self.ids):
            g = self.geoms[i]
            py_mu = float(g.friction)
            if not _same(py_mu, mu[j]):
                raise RuntimeError(f"Python/live solver friction differs for geom {i}")
            rows.append(dict(
                geom_id=i, link=str(g.link.name),
                friction=mu[j], friction_ratio=ratios[j],
                effective_friction=mu[j] * ratios[j],
                live_sol_params=params[7*j:7*j+7],
                selected_hand=i in self.hand_ids,
            ))
        return rows

    def apply(self):
        """Set the exact profile after reset and return an all-collider audit."""
        before = self._snapshot()
        if self.config["hand_friction_scale"] != 1.0:
            for i in self.hand_ids:
                self.geoms[i].set_friction(
                    self.baseline_mu[i] * self.config["hand_friction_scale"])
        after = self._snapshot()
        changed = []
        for base, row in zip(self.baseline, after):
            i = row["geom_id"]
            expected = base["friction"] * (
                self.config["hand_friction_scale"] if i in self.hand_ids else 1.0)
            if not _same(row["friction"], expected):
                raise RuntimeError(f"Profile friction mismatch for geom {i}")
            if not _same(row["friction_ratio"], base["friction_ratio"]):
                raise RuntimeError(f"Friction ratio unexpectedly changed for geom {i}")
            if any(not _same(a, b) for a, b in zip(row["live_sol_params"], base["live_sol_params"])):
                raise RuntimeError(f"Constraint solver parameters unexpectedly changed for geom {i}")
            if not _same(row["friction"], base["friction"]):
                changed.append(i)
        if not set(changed) <= set(self.hand_ids):
            raise RuntimeError("A non-hand collider was modified")
        return dict(
            profile=dict(self.config), hand_geom_ids=list(self.hand_ids),
            object_friction_changed=False, unselected_unchanged=True,
            changed_geom_ids=changed, baseline=self.baseline,
            before=before, after=after,
            baseline_source="Live solver immediately after scene.build, before profile application",
            substeps_note="Requested at scene construction; outer control dt remains unchanged",
        )
