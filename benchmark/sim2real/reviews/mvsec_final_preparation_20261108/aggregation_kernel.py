"""Frozen MVSEC mathematical aggregation; no parser, network or model loading.

Each metric map has shape:
    {method: {seed: {sequence: finite_sequence_metric}}}
Call aggregate_matrices(aee, outlier_percent, zero_flow_aee, roi='full_native').
All five methods, three seeds and four sequences are mandatory. Run
`python -B aggregation_kernel.py --self-test` for synthetic CPU fixtures only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from numbers import Real

import numpy as np

METHODS = ("eventgenesis", "v2e", "vid2e", "film_esim", "rife_esim")
SEEDS = (0, 1, 2)
SEQUENCES = ("indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1")
METRICS = ("aee_px_per_interval", "outlier_3px_and_5pct_percent", "zero_flow_aee_px_per_interval")
B = 10000
RNG_SEED = 20261108


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matrix(values, metric):
    require(isinstance(values, dict) and set(values) == set(METHODS), metric + ": full five-method matrix required")
    arrays = {}
    for method in METHODS:
        runs = values[method]
        require(isinstance(runs, dict), method + ": seed mapping required")
        normalized = {}
        for key, value in runs.items():
            require(type(key) is int or (isinstance(key, str) and key in ("0", "1", "2")), "Invalid seed key")
            seed = int(key)
            require(seed in SEEDS and seed not in normalized, "Duplicate/unregistered seed")
            normalized[seed] = value
        require(set(normalized) == set(SEEDS), method + ": all three seeds required")
        rows = []
        for seed in SEEDS:
            row = normalized[seed]
            require(isinstance(row, dict) and set(row) == set(SEQUENCES), method + ": all four sequences required")
            ordered = []
            for sequence in SEQUENCES:
                value = row[sequence]
                require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                        and math.isfinite(float(value)), metric + ": nonfinite/nonnumeric sequence value")
                value = float(value)
                require(value >= 0, metric + ": negative error/fraction")
                if metric == METRICS[1]:
                    require(value <= 100, "Outlier percentage outside[0,100]")
                ordered.append(value)
            rows.append(ordered)
        arrays[method] = np.asarray(rows, dtype=np.float64)
    return arrays


def summarize(arrays, draws):
    summaries = {}
    for method in METHODS:
        values = arrays[method]
        seed_values = values.mean(axis=1)  # equal four-sequence weight
        summaries[method] = dict(per_seed={str(seed): float(value) for seed, value in zip(SEEDS, seed_values)},
            mean=float(seed_values.mean()), sample_sd=float(seed_values.std(ddof=1)),
            per_sequence_seed_mean={sequence: float(value) for sequence, value in zip(SEQUENCES, values.mean(axis=0))})
    paired = {}
    for method in METHODS[1:]:
        differences = arrays["eventgenesis"] - arrays[method]
        clusters = differences.mean(axis=0)  # retain all three paired seeds inside every sequence cluster
        boot = clusters[draws].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975], method="linear")
        paired[method] = dict(difference_sign="eventgenesis minus comparator; negative favors eventgenesis",
            paired_seed_by_sequence_differences=differences.tolist(),
            sequence_mean_paired_difference={s: float(v) for s, v in zip(SEQUENCES, clusters)},
            mean_difference=float(clusters.mean()), percentile95_ci=[float(low), float(high)],
            bootstrap_replicates=B, rng_seed=RNG_SEED, cluster_count=4,
            conditional_on_observed_three_seeds=True, significance_stars=False)
    return dict(methods=summaries, paired_comparisons=paired)


def aggregate_matrices(aee, outlier_percent, zero_flow_aee, *, roi="full_native"):
    require(roi in ("full_native", "legacy_roi"), "Unregistered ROI")
    # Validate all input matrices before returning any table.
    errors = matrix(aee, METRICS[0])
    outliers = matrix(outlier_percent, METRICS[1])
    zero = matrix(zero_flow_aee, METRICS[2])
    reference = zero["eventgenesis"][0]
    for method in METHODS:
        require(np.array_equal(zero[method], np.broadcast_to(reference, (3, 4))),
                "Zero-flow reference differs across methods/seeds; refuse full table")
    draws = np.random.default_rng(RNG_SEED).integers(0, 4, size=(B, 4))
    result = dict(schema="stage9_mvsec_frozen_aggregation_v1", roi=roi,
        methods_order=list(METHODS), seeds_order=list(SEEDS), sequence_order=list(SEQUENCES),
        metrics={METRICS[0]: summarize(errors, draws), METRICS[1]: summarize(outliers, draws)},
        zero_flow_aee_px_per_interval=dict(per_sequence={s: float(v) for s, v in zip(SEQUENCES, reference)},
                                           equal_sequence_mean=float(reference.mean()), identical_across_all15_runs=True),
        interval_scope="percentile95% paired sequence-cluster CI, conditional on observed3 seeds and only4 clusters",
        all_four_comparisons_reported=True, significance_stars=False,
        multiple_comparison_significance_claim=False,
        input_values="frozen per-window-then-per-sequence metrics; no independent window bootstrap",
        numpy_version=np.__version__)
    return result


def fixture_map(function):
    return {method: {seed: {sequence: function(method, seed, index) for index, sequence in enumerate(SEQUENCES)}
                     for seed in SEEDS} for method in METHODS}


def self_test():
    checks = []
    zero = fixture_map(lambda m, s, q: float(q + 1))
    aee = fixture_map(lambda m, s, q: 1.0 if m == "eventgenesis" else 3.0)
    outlier = fixture_map(lambda m, s, q: 10.0 if m == "eventgenesis" else 12.0)
    result = aggregate_matrices(aee, outlier, zero)
    for metric in METRICS[:2]:
        for comp in METHODS[1:]:
            row = result["metrics"][metric]["paired_comparisons"][comp]
            assert row["mean_difference"] == -2.0 and row["percentile95_ci"] == [-2.0, -2.0]
    checks.append("constant_paired_difference_has_exact_constant_ci")
    aee = fixture_map(lambda m, s, q: float(s + 1))
    result = aggregate_matrices(aee, outlier, zero)
    assert result["metrics"][METRICS[0]]["methods"]["eventgenesis"]["mean"] == 2.0
    assert result["metrics"][METRICS[0]]["methods"]["eventgenesis"]["sample_sd"] == 1.0
    checks.append("three_seed_sampleSD_uses_ddof1")
    aee = fixture_map(lambda m, s, q: 100.0 if q == 3 else 0.0)
    result = aggregate_matrices(aee, outlier, zero)
    assert result["metrics"][METRICS[0]]["methods"]["eventgenesis"]["mean"] == 25.0
    checks.append("equal_sequence_mean_not_pixel_or_window_weighted")
    for label in ("missing_seed", "nonfinite_value", "different_zero_reference"):
        bad = copy.deepcopy(aee)
        bad_zero = copy.deepcopy(zero)
        if label == "missing_seed":
            del bad["eventgenesis"][2]
        elif label == "nonfinite_value":
            bad["eventgenesis"][0][SEQUENCES[0]] = float("nan")
        else:
            bad_zero["v2e"][0][SEQUENCES[0]] += 1
        try:
            aggregate_matrices(bad, outlier, bad_zero)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid fixture accepted: " + label)
    checks.append("missing_nonfinite_and_inconsistent_zero_rejected")
    return dict(status="passed", checks=checks, synthetic_fixture_only=True, real_scores_read=False,
                GPU_used=False, numpy_version=np.__version__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
