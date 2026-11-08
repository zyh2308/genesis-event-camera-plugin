"""CPU-only cached test of the common, intentional motion-range bootstrap.

No production edits, finite-difference velocity, GT estimator inputs, physics,
rendering or CUDA. Deploy beside replay_static_association.py. All eight
normal25 archived runs are required, each tested in three declared modes.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np


def public_association(self, time_s):
    return self._predict(time_s)


def install_bootstrap(tracker, native_class, serial):
    """Patch only this instance, preserving the full native outer method."""
    tracker._measurement_bootstrap_history = []

    def start(self, sample):
        native_class._start_motion(self, sample)
        old_mean = self.filter_mean.copy(); old_cov = self.filter_cov.copy()
        R, _ = self._measurement_cov(sample)
        self.filter_mean[:3] = np.asarray(sample["position"], dtype=float)
        self.filter_cov[:3, :3] = R + np.eye(3) * .02**2
        np.testing.assert_array_equal(self.filter_mean[3:], old_mean[3:])
        np.testing.assert_array_equal(self.filter_cov[3:, :], old_cov[3:, :])
        np.testing.assert_array_equal(self.filter_cov[:, 3:], old_cov[:, 3:])
        assert self.filter_time == sample["t"]
        assert self.motion_onset_s == sample["t"]
        self._measurement_bootstrap_history.append(serial(dict(
            measurement_t=float(sample["t"]), available_s=float(sample["available_s"]),
            first_suspicion_t=float(self.motion_suspicion[0]["t"]),
            suspicion_times=[float(x["t"]) for x in self.motion_suspicion],
            static_position=self.static_position, measurement_position=sample["position"],
            camera_origin=sample["camera_origin"],
            before_mean=old_mean, after_mean=self.filter_mean.copy(),
            before_covariance=old_cov, after_covariance=self.filter_cov.copy(),
            measurement_covariance=R, release_bracket=getattr(self, "release_bracket", None),
            velocity_prior_unchanged=True, GT_input_used=False)))

    tracker._start_motion = types.MethodType(start, tracker)


def metric_summary(states):
    metrics = {}
    for metric in ("scoring_only_position_error_m", "scoring_only_velocity_error_m_s",
                   "scoring_only_absolute_range_error_m", "scoring_only_transverse_position_error_m"):
        values = [s[metric] for s in states if metric in s]
        metrics[metric] = dict(n=len(values), median=float(np.median(values)) if values else None,
                              p95=float(np.percentile(values, 95)) if values else None)
    return dict(queries=len(states), available_states=sum(s["position"] is not None for s in states), metrics=metrics)


def grade_phases(payload, release_s):
    states = [s for s in payload["states"] if s["time_s"] >= release_s - 1e-9]
    moving = [s for s in states if s["position"] is not None and s["motion_onset_s"] is not None]
    first = moving[0]["time_s"] if moving else None
    early = [] if first is None else [s for s in states if first-1e-9 <= s["time_s"] <= first+.1+1e-9]
    pre_fit = [s for s in moving if "flight_fit_observations" not in s]
    return dict(first_control_moving_s=first,
        first_moving_control_plus_100ms=metric_summary(early),
        moving_before_first_flight_fit=metric_summary(pre_fit),
        all_postrelease_precontact=metric_summary(states))


def run_one(run, mode, source, output, until, common, native_class, module):
    created = []
    fit = mode != "original"

    def factory(*args, **kwargs):
        tracker = native_class(*args, **kwargs)
        if mode == "original":
            tracker._association_prediction = types.MethodType(public_association, tracker)
        elif mode == "measurement_bootstrap":
            install_bootstrap(tracker, native_class, common.serial)
        created.append(tracker)
        return tracker

    module.EventSphereTracker = factory
    try:
        result = common.replay(run, mode, source, output, until, flight_fit=fit)
    finally:
        module.EventSphereTracker = native_class
    assert len(created) == 1
    tracker = created[0]
    assert bool(tracker.use_flight_fit) == fit
    path = output / result["file"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mode == "original":
        assert payload["baseline_parity"]["status"] == "PASS"
    history = getattr(tracker, "_measurement_bootstrap_history", [])
    assert len(history) <= 1, "Unexpected multiple motion initializations"
    payload["bootstrap_history"] = history
    payload["candidate_change"] = (
        "Instance-only _start_motion: full native call retained; position=current measurement; "
        "position covariance=existing measurement covariance+0.02m squared identity. "
        "Velocity prior/covariance, gates, count, windows and fit remain unchanged. "
        "Shared intentional range-prior candidate; not a simulator-core bug claim."
        if mode == "measurement_bootstrap" else
        "Archived public association, fit disabled; exact archive parity required" if mode == "original" else
        "Native current association and causal flight fit, no patches")
    # Object truth is read only after the helper has returned all decisions.
    result_gt = json.loads((run / "episode_000/result.json").read_text())
    phases = grade_phases(payload, float(result_gt["initial"]["release_time_s"]))
    payload["phase_metrics_scoring_only"] = phases
    path.write_text(json.dumps(common.serial(payload), indent=2), encoding="utf-8")
    result.update(bootstrap_history=history, phase_metrics_scoring_only=phases)
    return result, payload


def unchanged_prefix(native, candidate):
    a = native["states"]; b = candidate["states"]
    assert len(a) == len(b)
    onset_a = native["motion_onset_s"]; onset_b = candidate["motion_onset_s"]
    assert (onset_a is None) == (onset_b is None), "Motion confirmation changed"
    if onset_a is not None: assert abs(onset_a-onset_b) < 1e-10
    count = 0; maximum = 0.
    for x, y in zip(a, b):
        assert x["time_s"] == y["time_s"]
        if x["motion_onset_s"] is not None: break
        assert y["motion_onset_s"] is None
        assert (x["position"] is None) == (y["position"] is None)
        if x["position"] is not None:
            for key in ("position", "velocity"):
                delta = float(np.max(np.abs(np.asarray(x[key])-y[key])))
                maximum = max(maximum, delta); assert delta < 1e-10
        count += 1
    # Every detection available no later than the initial confirmation is also
    # identical: bootstrap cannot change the evidence that triggered itself.
    old = [s for s in native["samples"] if onset_a is None or s["t"] <= onset_a+1e-10]
    new = [s for s in candidate["samples"] if onset_b is None or s["t"] <= onset_b+1e-10]
    assert len(old) == len(new)
    for x, y in zip(old, new):
        for key in ("t", "position", "uv", "radius_px", "quality", "available_s", "maximum_event_s", "camera_origin"):
            np.testing.assert_allclose(x[key], y[key], rtol=0, atol=1e-10)
    return dict(static_control_queries=count, maximum_static_state_difference=maximum,
                initial_confirmation_time_unchanged=True, onset_s=onset_a,
                detection_prefix_samples=len(old))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--helper", type=Path, default=Path(__file__).with_name("replay_static_association.py"))
    p.add_argument("--until", type=float, default=.96)
    a = p.parse_args()
    assert len(a.run) == 8 and len({str(x.resolve()) for x in a.run}) == 8, "All eight distinct registered caches required"
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source))
    os.environ.update(CATCHIT_REJECT_PARTIAL_CONTOURS="0", CATCHIT_EVENT_FLIGHT_FIT="0", CATCHIT_REVOCABLE_GRASP="0")
    spec = importlib.util.spec_from_file_location("shared_bootstrap_cpu_replay", a.helper)
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    import event_sphere_tracker as module
    native_class = module.EventSphereTracker
    results = []; checks = []
    for run in a.run:
        these = {}
        for mode in ("original", "native_flight_fit", "measurement_bootstrap"):
            result, payload = run_one(run, mode, a.source, a.output, a.until, common, native_class, module)
            results.append(result); these[mode] = payload
        check = unchanged_prefix(these["native_flight_fit"], these["measurement_bootstrap"])
        check["run"] = str(run); checks.append(check)
    assert len(results) == 24
    assert "genesis" not in sys.modules and "torch" not in sys.modules
    receipt = dict(status="PASS", physics_steps=0, renderer_calls=0, CUDA_calls=0,
        scope="All eight cached trajectories; shared intentional range-prior bootstrap candidate, not a simulator-core defect claim; no catch-success claim",
        archived_parity_required=True, naive_finite_difference_velocity=False,
        modes=["original fit0/public association", "native fit1", "measurement bootstrap fit1"],
        results=results, static_prefix_and_confirmation_checks=checks)
    (a.output/"SUMMARY.json").write_text(json.dumps(common.serial(receipt), indent=2), encoding="utf-8")
    files = [Path(__file__), a.helper] + list(a.source.glob("event_sphere_tracker*.py")) + [a.source/"event_bearing_flight.py"]
    (a.output/"SOURCE_HASHES.json").write_text(json.dumps({str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="PASS", results=len(results), output=str(a.output))), flush=True)


if __name__ == "__main__":
    main()
