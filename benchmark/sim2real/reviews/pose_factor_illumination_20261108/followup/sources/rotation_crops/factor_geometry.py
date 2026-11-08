"""CPU-only, anchor-aware pose factor diagnostics. Never a deployable GT oracle.

Units: object/CAD translations and depths in metres, K in pixels, uv normalized
by [346,260]. ADD-S uses the native one-way CAD nearest-neighbour mean and the
strict success threshold 0.1 * diameter. Each rollout is equally weighted.
Saved rotation factors are crossed as-is; crossings do not rerun a crop/head.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PIXELS = np.array([346.0, 260.0])
NATIVE_METRICS = ("depth_mae_m", "depth_bias_m", "log_size_mae", "log_size_bias", "center_px", "success", "auc")
CANONICAL_ARMS = ("C0", "C1", "slow")
REGISTERED_COUNT_GAINS = (0.5, 1.0, 2.0)
SCHEMA = "pose_factor_geometry_diagnostic_v1"


def _array(value, shape=None, name="array"):
    a = np.asarray(value, dtype=np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError(f"invalid {name}: shape={a.shape}, expected={shape}, finite required")
    return a


def _rotation(value, name="rotation"):
    r = _array(value, (3, 3), name)
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-5, rtol=0) or abs(np.linalg.det(r) - 1) > 2e-5:
        raise ValueError(f"{name} is not SO(3)")
    return r


def _camera(value):
    k = _array(value, (3, 3), "K")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("K must have positive focal lengths and [0,0,1] bottom row")
    return k


def _transform(value):
    t = _array(value, name="T_gt")
    if t.shape == (4, 4):
        if not np.array_equal(t[3], [0, 0, 0, 1]):
            raise ValueError("bad homogeneous T_gt bottom row")
        t = t[:3]
    if t.shape != (3, 4):
        raise ValueError("T_gt must be 3x4 or 4x4, camera from object, metres")
    return _rotation(t[:, :3], "GT rotation"), t[:, 3]


@dataclass
class FactorAssets:
    vertices_m: np.ndarray
    keypoints_object_m: np.ndarray
    diameter_m: float

    def __post_init__(self):
        self.vertices_m = _array(self.vertices_m, name="vertices_m").copy()
        self.keypoints_object_m = _array(self.keypoints_object_m, name="keypoints_object_m").copy()
        for name in ("vertices_m", "keypoints_object_m"):
            a = getattr(self, name)
            if a.ndim != 2 or a.shape[1] != 3 or len(a) < 1:
                raise ValueError(f"{name} requires Nx3 nonempty")
        self.diameter_m = float(self.diameter_m)
        if not np.isfinite(self.diameter_m) or self.diameter_m <= 0:
            raise ValueError("positive finite diameter_m required")
        self.tree = cKDTree(self.vertices_m)


def decode_translation(K, anchor, uv, z_m, R):
    """Camera-from-object origin translation; z_m is the object ANCHOR depth."""
    k = _camera(K)
    anchor = _array(anchor, (3,), "anchor")
    uv = _array(uv, (2,), "normalized uv")
    r = _rotation(R)
    z = float(z_m)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("anchor z_m must be positive finite metres")
    ray = np.linalg.solve(k, np.r_[uv * PIXELS, 1.0])
    if not np.isclose(ray[2], 1.0, atol=1e-10, rtol=0):
        raise ValueError("camera ray must have z=1")
    return ray * z - r @ anchor


def _gt(record):
    k = _camera(record["K"])
    anchor = _array(record["anchor"], (3,), "anchor")
    r, t = _transform(record["T_gt"])
    p = r @ anchor + t
    if p[2] <= 0:
        raise ValueError("GT anchor behind camera")
    pix = k @ p
    uv = pix[:2] / pix[2] / PIXELS
    provided = _array(record["uv_gt"], (2,), "uv_gt normalized")
    if not np.allclose(provided, uv, atol=3e-7, rtol=0):
        raise ValueError("uv_gt does not match projected CAD anchor; unit/label mismatch")
    return {"uv": uv, "z_m": float(p[2]), "R": r, "t": t, "anchor_camera": p}


def _prediction(row):
    uv = _array(row["pred_uv"], (2,), "pred_uv normalized")
    z = float(row["z_m"])
    r = _rotation(row["R_pred"])
    size = float(row["size"])
    if not np.isfinite([z, size]).all() or min(z, size) <= 0:
        raise ValueError("predicted anchor depth and normalized size must be positive finite")
    if "log_size" in row and not np.isclose(math.log(size), float(row["log_size"]), atol=3e-6, rtol=1e-6):
        raise ValueError("size/log_size mismatch")
    if all(key in row for key in ("log_shape", "focal_log", "log_size")):
        expected = float(row["log_shape"]) + float(row["focal_log"]) - float(row["log_size"])
        if not np.isclose(math.log(z), expected, atol=5e-6, rtol=1e-6):
            raise ValueError("anchor depth/log-shape decode mismatch")
    p = {"uv": uv, "z_m": z, "R": r, "size": size}
    if "t_pred_m" in row:
        t = decode_translation(row["K"], row["anchor"], uv, z, r)
        if not np.allclose(t, _array(row["t_pred_m"], (3,), "t_pred_m"), atol=3e-6, rtol=1e-6):
            raise ValueError("native t_pred_m != anchor-aware decode; possible units mismatch")
    return p


def group_flat_predictions(rows, required_arms=CANONICAL_ARMS):
    """Align by explicit physical IDs; reject duplicates, missing arms or labels."""
    required_arms = tuple(required_arms)
    if not required_arms or len(set(required_arms)) != len(required_arms):
        raise ValueError("invalid required_arms")
    grouped = {}
    shared = ("K", "T_gt", "anchor", "uv_gt", "gt_log_size", "gt_log_shape")
    for row in rows:
        for key in ("method", "seed", "arm", "condition", "rid", "sample_id") + shared:
            if key not in row:
                raise ValueError(f"missing {key}")
        if row["arm"] not in required_arms:
            raise ValueError(f"unexpected arm {row['arm']}")
        if row["condition"] not in ("normal", "half", "eighth"):
            raise ValueError("unregistered illumination condition")
        identity = (str(row["method"]), int(row["seed"]), str(row["condition"]), str(row["rid"]), str(row["sample_id"]))
        if identity not in grouped:
            grouped[identity] = {key: row[key] for key in shared}
            grouped[identity].update(dict(method=identity[0], seed=identity[1], condition=identity[2], rid=identity[3], sample_id=identity[4], predictions={}))
            _gt(grouped[identity])
        record = grouped[identity]
        if row["arm"] in record["predictions"]:
            raise ValueError(f"duplicate prediction {identity} {row['arm']}")
        for key in shared:
            if not np.array_equal(np.asarray(row[key]), np.asarray(record[key])):
                raise ValueError(f"label mismatch for {identity}: {key}")
        record["predictions"][row["arm"]] = _prediction(row)
    if not grouped:
        raise ValueError("empty prediction input")
    for record in grouped.values():
        if set(record["predictions"]) != set(required_arms):
            raise ValueError(f"incomplete arms at {record['sample_id']}")
    # Matched physical windows must be present under each illumination, generator
    # and seed. No blind zip or intersection silently discards missing rows.
    inventories = defaultdict(set)
    labels = {}
    for identity, record in grouped.items():
        cell = identity[:3]
        sid = identity[3:]
        inventories[cell].add(sid)
        for key in ("K", "T_gt", "anchor", "uv_gt"):
            label_key = sid + (key,)
            if label_key in labels and not np.array_equal(np.asarray(record[key]), labels[label_key]):
                raise ValueError(f"physical label mismatch across conditions/methods/seeds: {sid} {key}")
            labels[label_key] = np.asarray(record[key])
    reference = next(iter(inventories.values()))
    if any(inventory != reference for inventory in inventories.values()):
        raise ValueError("physical window inventories differ across exported cells")
    return [grouped[key] for key in sorted(grouped)]


def score_factors(record, prediction, assets):
    """One diagnostic forward geometry score, including native validity policy."""
    gt = _gt(record)
    if not np.allclose(record["anchor"], assets.keypoints_object_m[0], atol=1e-10, rtol=0):
        raise ValueError("row anchor != keypoints[0]")
    uv = _array(prediction["uv"], (2,), "uv")
    r = _rotation(prediction["R"])
    z = float(prediction["z_m"])
    t = decode_translation(record["K"], record["anchor"], uv, z, r)
    pred_anchor = r @ np.asarray(record["anchor"]) + t
    delta_anchor = pred_anchor - gt["anchor_camera"]
    rel = r.T @ gt["R"]
    angle = math.degrees(math.acos(float(np.clip((np.trace(rel) - 1) / 2, -1, 1))))
    valid = bool(((assets.keypoints_object_m @ r.T + t)[:, 2] > 0).all())
    adds = None
    if valid:
        truth = assets.vertices_m @ gt["R"].T + gt["t"]
        # Exactly the original score's query convention, with an orthonormal R.
        adds = float(assets.tree.query((truth - t) @ r, workers=1)[0].mean())
    log_size_error = float(math.log(float(prediction["size"])) - float(record["gt_log_size"]))
    return {
        "depth_mae_m": abs(z - gt["z_m"]), "depth_bias_m": z - gt["z_m"],
        "log_size_mae": abs(log_size_error), "log_size_bias": log_size_error,
        # Preserve the native score's exported label arithmetic. The projected
        # anchor above independently validates this label; its floating-point
        # rounding need not be identical to a fresh double-precision projection.
        "center_px": float(np.linalg.norm((uv - np.asarray(record["uv_gt"])) * PIXELS)),
        "anchor_position_error_m": float(np.linalg.norm(delta_anchor)),
        "anchor_lateral_error_m": float(np.linalg.norm(delta_anchor[:2])),
        "translation_error_m": float(np.linalg.norm(t - gt["t"])),
        "rotation_error_deg": angle, "adds_m": adds, "valid_pose": int(valid),
        "success": float(adds is not None and adds < 0.1 * assets.diameter_m),
        "auc": max(1.0 - adds / 0.1, 0.0) if adds is not None else 0.0,
    }


def factor_oracles(record, arm, assets):
    """All 8 u/z/R replacement masks. GT factors never feed a trainable model."""
    base = record["predictions"][arm]
    gt = _gt(record)
    rows = []
    for bits in itertools.product((0, 1), repeat=3):
        p = dict(base)
        for key, bit in zip(("uv", "z_m", "R"), bits):
            if bit:
                p[key] = gt[key]
        rows.append({"variant": f"oracle/{arm}/u{bits[0]}z{bits[1]}r{bits[2]}",
                     "family": "oracle", "arm": arm, "uses_privileged_gt": bool(any(bits)),
                     "sources": dict(zip(("uv", "z_m", "R"), ["GT" if b else arm for b in bits])),
                     "metrics": score_factors(record, p, assets)})
    return rows


def prediction_crosses(record, assets, arms=("C0", "C1")):
    """All8 saved prediction crosses; no privileged GT and no new crop/head."""
    if len(arms) != 2 or len(set(arms)) != 2:
        raise ValueError("exactly two distinct cross arms required")
    rows = []
    for src in itertools.product(arms, repeat=3):
        p = dict(record["predictions"][src[0]])
        p["z_m"] = record["predictions"][src[1]]["z_m"]
        p["R"] = record["predictions"][src[2]]["R"]
        rows.append({"variant": f"cross/u{src[0]}_z{src[1]}_r{src[2]}", "family": "prediction_cross",
                     "uses_privileged_gt": False, "sources": dict(zip(("uv", "z_m", "R"), src)),
                     "metrics": score_factors(record, p, assets)})
    return rows


def evaluate_records(records, assets):
    for record in records:
        identity = {k: record[k] for k in ("method", "seed", "condition", "rid", "sample_id")}
        for arm in record["predictions"]:
            yield dict(identity, variant=f"actual/{arm}", family="actual", arm=arm,
                       uses_privileged_gt=False, diagnostic_only=True,
                       metrics=score_factors(record, record["predictions"][arm], assets))
            for row in factor_oracles(record, arm, assets):
                yield dict(identity, diagnostic_only=True, **row)
        for row in prediction_crosses(record, assets):
            yield dict(identity, diagnostic_only=True, **row)


def aggregate_rollout_equal(scored_rows):
    """Window->rollout->equal-rollout means, never pixel/window pooled macros.

    ADD-S mean is explicitly conditional on valid poses; success/AUC keep ALL
    windows, including invalid poses scored zero, exactly as the native score.
    """
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in scored_rows:
        key = tuple(row[k] for k in ("method", "seed", "condition", "variant"))
        sample = key + (row["rid"], row["sample_id"])
        if sample in seen:
            raise ValueError("duplicate scored row")
        seen.add(sample)
        groups[key][row["rid"]].append(row["metrics"])
    result = []
    for key, rollouts in sorted(groups.items()):
        by = {}
        for rid, vals in sorted(rollouts.items()):
            metric_keys = set(vals[0])
            if any(set(v) != metric_keys for v in vals):
                raise ValueError("metric schema mismatch")
            out = {"sample_count": len(vals), "valid_adds_count": sum(v["adds_m"] is not None for v in vals)}
            for name in sorted(metric_keys - {"adds_m"}):
                a = _array([v[name] for v in vals], name=name)
                out[name] = float(a.mean())
            valid_adds = [v["adds_m"] for v in vals if v["adds_m"] is not None]
            out["adds_valid_mean_m"] = float(np.mean(valid_adds)) if valid_adds else None
            by[rid] = out
        names = set(next(iter(by.values()))) - {"sample_count", "valid_adds_count", "adds_valid_mean_m"}
        summary = {name: float(np.mean([v[name] for v in by.values()])) for name in sorted(names)}
        summary["adds_valid_equal_rollout_mean_m"] = (float(np.mean([v["adds_valid_mean_m"] for v in by.values()]))
                                                      if all(v["adds_valid_mean_m"] is not None for v in by.values()) else None)
        result.append(dict(zip(("method", "seed", "condition", "variant"), key),
                           rollout_count=len(by), sample_count=sum(v["sample_count"] for v in by.values()),
                           summary=summary, by_rollout=by))
    if not result:
        raise ValueError("empty score input")
    return result


def native_parity_report(aggregated, native_rows, *, atol=2e-7, rtol=2e-6):
    actual = {(r["method"], int(r["seed"]), r["condition"], r["variant"].split("/")[1]): r
              for r in aggregated if r["variant"].startswith("actual/")}
    expected = {}
    for row in native_rows:
        key = (str(row["method"]), int(row["seed"]), str(row["condition"]), str(row["arm"]))
        if key in expected:
            raise ValueError("duplicate native score")
        expected[key] = row["score"]
    if set(actual) != set(expected):
        raise ValueError("native/reference score cell inventory mismatch")
    checks = []
    for key in sorted(actual):
        score, native = actual[key], expected[key]
        if set(score["by_rollout"]) != set(native["by_rollout"]):
            raise ValueError("native rollout inventory mismatch")
        for scope in ("summary", *sorted(score["by_rollout"])):
            lhs = score["summary"] if scope == "summary" else score["by_rollout"][scope]
            rhs = native["summary"] if scope == "summary" else native["by_rollout"][scope]
            for name in NATIVE_METRICS:
                a, b = float(lhs[name]), float(rhs[name])
                ok = math.isfinite(a) and math.isfinite(b) and (a == b if name == "success" else np.isclose(a, b, atol=atol, rtol=rtol))
                checks.append(dict(cell=list(key), scope=scope, metric=name, actual=a, native=b,
                                   abs_error=abs(a - b), passed=bool(ok)))
    return {"status": "passed" if all(c["passed"] for c in checks) else "failed",
            "atol": atol, "rtol": rtol, "checked_cells": len(actual), "checks": checks}


def controlled_count_gain(image3CHW, gain, count_scale=8):
    """Representation-only counterfactual, UNRESIZED consistent log counts only.

    Never apply to independently resized ON/OFF/total channels. Values clipped
    at count_scale are irreversible, so this cannot resimulate raw events.
    The real worker must scale original-resolution histogram counts before log
    and resize. Registered probes are 0.5,1,2; no automatic parameter search.
    """
    image = np.asarray(image3CHW)
    if image.ndim != 3 or image.shape[0] != 3 or image.dtype not in (np.float32, np.float64):
        raise ValueError("float32/64 3CHW unresized log-count representation required")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("raw representation must be finite [0,1]")
    gain, scale = float(gain), float(count_scale)
    if not np.isfinite([gain, scale]).all() or gain <= 0 or scale <= 0:
        raise ValueError("positive finite gain/count_scale required")
    denom = math.log1p(scale)
    counts = np.expm1(image[:2].astype(np.float64) * denom)
    expected_total = np.minimum(np.log1p(counts.sum(axis=0)) / denom, 1)
    if not np.allclose(image[2], expected_total, atol=7e-7, rtol=0):
        raise ValueError("inconsistent total count channel; resized inputs are unsupported")
    if gain == 1:
        return image.copy()  # Exact identity, no inverse/reencode roundoff.
    scaled = counts * gain
    result = np.concatenate((np.minimum(np.log1p(scaled) / denom, 1),
                             np.minimum(np.log1p(scaled.sum(axis=0))[None] / denom, 1)), axis=0).astype(image.dtype)
    if not np.array_equal(image > 0, result > 0):
        raise ValueError("gain lost nonzero support through numerical underflow")
    return result


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("output must be a fresh directory; existing results protected")
    inputs = {"predictions": Path(args.predictions), "vertices": Path(args.vertices), "keypoints": Path(args.keypoints)}
    if args.native_scores:
        inputs["native_scores"] = Path(args.native_scores)
    hashes = {name: {"path": str(path.resolve()), "sha256": _sha(path)} for name, path in inputs.items()}
    with inputs["predictions"].open(encoding="utf-8") as handle:
        flat = [json.loads(line) for line in handle if line.strip()]
    assets = FactorAssets(np.load(inputs["vertices"], allow_pickle=False), np.load(inputs["keypoints"], allow_pickle=False), args.diameter_m)
    records = group_flat_predictions(flat, tuple(args.arms))
    rows = list(evaluate_records(records, assets))
    aggregate = aggregate_rollout_equal(rows)
    parity = {"status": "not_checked", "reason": "native score reference not supplied"}
    if args.native_scores:
        with inputs["native_scores"].open(encoding="utf-8") as handle:
            parity = native_parity_report(aggregate, json.load(handle))
        if parity["status"] != "passed":
            # Keep explicit failed evidence in the NEW output, never silently
            # relabel a non-parity result as a completed independent analysis.
            out.mkdir(parents=True, exist_ok=False)
            (out / "native_parity.json").write_text(json.dumps(parity, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            raise ValueError("native score parity failed; explicit evidence preserved")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "per_window_diagnostic.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    (out / "macro_scores.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "native_parity.json").write_text(json.dumps(parity, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metrics = sorted({k for row in aggregate for k in row["summary"]})
    fields = ["method", "seed", "condition", "variant", "rollout_count", "sample_count"] + metrics
    with (out / "macro_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in aggregate:
            writer.writerow({k: row[k] for k in fields[:6]} | row["summary"])
    completion = {"schema": SCHEMA, "status": "complete_diagnostic_only", "inputs": hashes,
                  "diameter_m": args.diameter_m, "success_threshold_m": 0.1 * args.diameter_m,
                  "auc_cutoff_m": 0.1, "physical_records": len(records), "flat_predictions": len(flat),
                  "scored_variants": len(rows), "macro_cells": len(aggregate), "native_parity": parity["status"],
                  "optimizer_updates": 0, "real_data_used": False, "gt_oracles_deployable": False,
                  "aggregation": "windows -> per-rollout mean -> equal rollout mean",
                  "rotation_error_caveat": "ordinary geodesic label diagnostic, not symmetry-aware",
                  "prediction_cross_caveat": "saved R factor, no re-crop/re-head; diagnostic only"}
    (out / "completion.json").write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(completion, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("summarize")
    p.add_argument("--predictions", required=True)
    p.add_argument("--vertices", required=True)
    p.add_argument("--keypoints", required=True)
    p.add_argument("--diameter-m", required=True, type=float)
    p.add_argument("--out", required=True)
    p.add_argument("--native-scores")
    p.add_argument("--arms", nargs="+", default=list(CANONICAL_ARMS), choices=CANONICAL_ARMS)
    args = parser.parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
