"""Offline, fail-closed analysis of the registered 4800->5100 reconstruction trial.

Usage: py -3 analyze_results.py --input result_summary.json --output-dir analysis

No training, inference, network access, or changes to input evidence are performed.
The input embeds both methods' study/smoke/release JSON, as exported by the parent.
Training evidence is also required: methods[m]["training_metrics"][arm] may contain
all300 JSONL rows, or methods[m]["training_evidence"][arm] may contain the strict
summary described in validate_training(). Missing/incomplete evidence exits2
before any result directory or tables are created. Failed scientific adoption
gates are valid results and are reported without changing the preregistered rules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


METHODS = ("eventgenesis", "v2e")
ARMS = ("control", "photometric")
SUBSETS = ("all200", "tail195")
PRIMARY = "all200"
EXPECTED_PROTOCOL_SHA = "7f3b400205be412b6e281d6c7e4aa9c4a005b82c477f1c574bfddf41b3b63f68"
PHASES = ["smoke_eventgenesis", "smoke_v2e", "study_eventgenesis", "study_v2e"]
EXPECTED_UPDATES = list(range(4801, 5101))
REQUIRED_METRICS = (
    "lpips_alex", "lpips_vgg", "mse", "psnr_db", "ssim", "contrast_ratio",
    "correlation", "edge_mean", "target_edge_mean", "edge_ratio",
    "flat_fraction", "near_boundary_fraction", "negative_fraction",
    "above_one_fraction", "intensity_mae", "intensity_bias", "edge_mae",
)
LABELS = {"eventgenesis": "EventGenesis", "v2e": "V2E", "before": "源检查点4800",
          "control": "同期控制", "photometric": "加像素L1（权重0.25）", "detail": "加L1及边缘项"}


class EvidenceError(ValueError):
    """A missing or internally inconsistent evidence contract."""


def need(condition, message):
    if not condition:
        raise EvidenceError(message)


def finite(value, where):
    need(isinstance(value, (int, float)) and not isinstance(value, bool),
         f"{where}: expected a numeric value")
    need(math.isfinite(value), f"{where}: non-finite value")
    return float(value)


def finite_tree(value, where):
    if isinstance(value, dict):
        for key, val in value.items():
            finite_tree(val, f"{where}.{key}")
    elif isinstance(value, list):
        for index, val in enumerate(value):
            finite_tree(val, f"{where}[{index}]")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        finite(value, where)


def digest(value):
    need(isinstance(value, str) and len(value) == 64
         and all(c in "0123456789abcdef" for c in value.lower()),
         "Expected a64-character SHA256 digest")
    return value.lower()


def arm_map(rows, label):
    need(isinstance(rows, list) and len(rows) == len(ARMS), f"{label}: need exactly{len(ARMS)} arms")
    need(all(isinstance(row, dict) for row in rows), f"{label}: invalid arm record")
    result = {row.get("arm"): row for row in rows}
    need(set(result) == set(ARMS), f"{label}: missing/duplicate/unexpected arm")
    return result


def validate_metrics(metrics, where):
    need(isinstance(metrics, dict), f"{where}: missing metrics")
    for name in REQUIRED_METRICS:
        need(name in metrics, f"{where}: missing metric {name}")
    for name, val in metrics.items():
        finite(val, f"{where}.{name}")
    for name in ("mse", "lpips_alex", "lpips_vgg", "intensity_mae", "edge_mae",
                 "edge_mean", "target_edge_mean", "contrast_ratio"):
        need(metrics[name] >= 0, f"{where}.{name}: impossible negative value")
    for name in ("flat_fraction", "near_boundary_fraction", "negative_fraction", "above_one_fraction"):
        need(0 <= metrics[name] <= 1, f"{where}.{name}: outside[0,1]")
    need(-1.000001 <= metrics["correlation"] <= 1.000001, f"{where}: invalid correlation")


def validate_evaluation(evaluation, where):
    need(isinstance(evaluation, dict), f"{where}: missing evaluation")
    need(evaluation.get("real_data_used") is False, f"{where}: synthetic-only contract missing")
    need(evaluation.get("diagnostic_fixture") is False, f"{where}: a smoke fixture is not full evaluation")
    need(evaluation.get("frames_per_rollout") == 200, f"{where}: expected200 frames")
    rows = evaluation.get("rollouts")
    need(isinstance(rows, list) and len(rows) == 8, f"{where}: expected8 complete rollouts")
    ids = [row.get("rollout") for row in rows]
    need(all(isinstance(rid, str) and rid for rid in ids) and len(set(ids)) == 8,
         f"{where}: duplicate/missing rollout IDs")
    for row in rows:
        need(row.get("frame_count") == 200, f"{where}: rollout not200 frames")
        for subset in SUBSETS:
            validate_metrics(row.get(subset), f"{where}.{row['rollout']}.{subset}")
    for subset in SUBSETS:
        validate_metrics(evaluation.get(subset), f"{where}.{subset}")
        metric_names = set(evaluation[subset])
        need(all(set(row[subset]) == metric_names for row in rows),
             f"{where}.{subset}: inconsistent metric schema")
        for name, supplied in evaluation[subset].items():
            recomputed = statistics.mean(row[subset][name] for row in rows)
            need(math.isclose(supplied, recomputed, rel_tol=1e-9, abs_tol=1e-12),
                 f"{where}.{subset}.{name}: does not equal rollout-weighted mean")
    return ids


def validate_update(row, expected_update, where, smoke=False):
    need(isinstance(row, dict), f"{where}: missing optimizer record")
    need(row.get("optimizer_update") == expected_update, f"{where}: wrong optimizer step")
    need(row.get("finite_gradient_tensors") == 30, f"{where}: expected30 finite gradient tensors")
    need(row.get("tc_calls") == 76, f"{where}: expected76 TC calls")
    need(finite(row.get("gradient_l2"), f"{where}.gradient_l2") > 0,
         f"{where}: nonpositive gradient norm")
    losses = row.get("losses")
    need(isinstance(losses, dict) and bool(losses), f"{where}: missing losses")
    for name, val in losses.items():
        finite(val, f"{where}.losses.{name}")
    for name in ("labels", "events"):
        need(isinstance(row.get(name), list) and len(row[name]) == 2,
             f"{where}: expected2 sequence hashes in{name}")
        for value in row[name]:
            digest(value)
    if smoke:
        probes = row.get("tc_gradient_probes")
        need(isinstance(probes, list) and len(probes) == 2,
             f"{where}: missing temporal-gradient probes")
        need(all(finite(val, where + ".tc_gradient_probe") > 0 for val in probes),
             f"{where}: TC has no gradient")


def validate_training(method_data, arms, method):
    """Require the full log or an explicit, hash-bound full-log summary.

    A summary must contain rows=300, optimizer_updates=[4801,...,5100],
    all_losses_finite=True, all_gradients_finite=True,
    finite_gradient_tensors=[30], tc_calls=[76], gradient_norm_positive=True,
    and log_sha256. These booleans must be computed from the original log,
    not inferred from the one-step smoke. Raw rows permit independent replay
    of the ordered label/event sequence digest as well.
    """
    raw = method_data.get("training_metrics")
    summaries = method_data.get("training_evidence")
    evidence = {}
    for arm in ARMS:
        if isinstance(raw, dict) and arm in raw:
            rows = raw[arm]
            need(isinstance(rows, list) and len(rows) == 300,
                 f"{method}/{arm}: need all300 optimizer records")
            for row, update in zip(rows, EXPECTED_UPDATES):
                validate_update(row, update, f"{method}/{arm}/update{update}")
            for name, target in (("labels", "label_sequence_sha256"), ("events", "event_sequence_sha256")):
                computed = hashlib.sha256(json.dumps([r[name] for r in rows]).encode()).hexdigest()
                need(computed == arms[arm][target], f"{method}/{arm}: full-log {target} mismatch")
            evidence[arm] = {"source": "all300 raw training rows", "checked_updates": 300,
                             "finite_gradient_tensors": 30, "tc_calls_per_update": 76,
                             "sequence_digest_recomputed": True}
        else:
            need(isinstance(summaries, dict) and isinstance(summaries.get(arm), dict),
                 f"{method}/{arm}: missing complete training_metrics or training_evidence")
            row = summaries[arm]
            need(row.get("rows") == 300 and row.get("optimizer_updates") == EXPECTED_UPDATES,
                 f"{method}/{arm}: incomplete/noncontiguous update summary")
            for name in ("all_losses_finite", "all_gradients_finite", "gradient_norm_positive"):
                need(row.get(name) is True, f"{method}/{arm}: failed or missing{name}")
            need(row.get("finite_gradient_tensors") == [30] and row.get("tc_calls") == [76],
                 f"{method}/{arm}: wrong gradient/TC counts")
            digest(row.get("log_sha256"))
            evidence[arm] = {"source": "exported full-log summary", **row,
                             "sequence_digest_recomputed": False}
    return evidence


def validate_input(data):
    need(isinstance(data, dict), "Input must be an object")
    need(isinstance(data.get("methods"), dict) and set(data["methods"]) == set(METHODS),
         "Need exactly EventGenesis and V2E results")
    need(isinstance(data.get("context"), dict), "Missing context snapshot")
    need("failure_evidence" in data, "Missing failure_evidence; preserve prior failed attempts")
    finite_tree(data, "input")
    history = data.get("failure_evidence", {}).get("prior_l1_weight1_trial", {})
    need(history.get("path", "").endswith("20261108-recon-photometry-detail-results-v3/result_summary.json")
         and history.get("sha256") == "760a9890b107a52914ec51c9d01fa0e2ea485d2cd2a7c0c167524a5a98c5e830",
         "Missing immutable prior failed-adoption trial reference")
    for method in METHODS:
        raw = data["methods"][method].get("training_metrics")
        need(isinstance(raw, dict) and set(raw) == set(ARMS),
             f"{method}: quarter trial requires all raw logs for independent sequence replay")
    parent_protocol, registration = data.get("parent_protocol", {}), data.get("parent_registration", {})
    need(bool(registration) and all(parent_protocol.get(key) == value for key, value in registration.items()),
         "Parent runtime protocol disagrees with immutable registration")
    need(parent_protocol.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA
         and parent_protocol.get("worker_sequence") == PHASES,
         "Parent GPU2 protocol/worker sequence mismatch")
    phase_exits = data.get("parent_phase_exits", {})
    need(set(phase_exits) == set(PHASES) and all(
        receipt.get("exit_code") == 0 and receipt.get("phase") == phase and receipt.get("status") == "exited"
        for phase, receipt in phase_exits.items()), "Missing/failed parent phase exits")
    pairing = data.get("cross_generator_smoke_pairing", {})
    labels = pairing.get("labels", {})
    need(set(labels) == set(METHODS) and labels["eventgenesis"] == labels["v2e"],
         "Parent smoke label dictionary mismatch")
    need(pairing.get("labels_sha256") == hashlib.sha256(json.dumps(labels["eventgenesis"]).encode()).hexdigest()
         and pairing.get("completed_smoke_updates") == 4 and pairing.get("real_data_used") is False,
         "Parent independent smoke label digest mismatch")
    metadata, all_rollouts = {}, []
    for method in METHODS:
        item = data["methods"][method]
        need(isinstance(item, dict), f"{method}: missing result wrapper")
        completion, smoke, release = (item.get(key) for key in ("completion", "smoke", "release"))
        need(isinstance(completion, dict) and completion.get("status") == "complete",
             f"{method}: study incomplete")
        need(completion.get("method") == method and completion.get("real_data_used") is False,
             f"{method}: method/synthetic isolation mismatch")
        need(completion.get("pairing_passed") is True, f"{method}: study pairing not verified")
        need(isinstance(smoke, dict) and smoke.get("status") == "complete"
             and smoke.get("real_data_used") is False, f"{method}: smoke incomplete or real data used")
        need(isinstance(release, dict) and release.get("status") == "released",
             f"{method}: GPU release receipt missing")
        need(release.get("completed") == ["smoke_eventgenesis", "smoke_v2e", "study_eventgenesis", "study_v2e"]
             and release.get("gpu_index") == 2 and release.get("current_job") is None,
             f"{method}: parent manager did not complete/release all four GPU2 phases")
        need(release == data.get("parent_release"), f"{method}: shared release receipt mismatch")
        need(release.get("gpu_uuid") == "GPU-a02affe0-7598-579e-ca39-f551f35ad6e9"
             and release.get("run_id") == "20261108-recon-l1-quarter-paired-gpu2-v1",
             f"{method}: parent GPU2 UUID/run binding mismatch")
        manager = data.get("parent_completion", {})
        need(manager.get("status") == "complete" and manager.get("completed") == release["completed"],
             f"{method}: parent manager completion mismatch")
        need(manager.get("main_optimizer_updates") == 1200 and manager.get("smoke_optimizer_updates") == 4
             and manager.get("real_data_used") is False and manager.get("real_scoring_enabled") is False,
             f"{method}: parent total budget or synthetic isolation mismatch")
        protocol = item.get("protocol", {})
        need(protocol.get("arms") == list(ARMS) and protocol.get("arm_count") == 2
             and protocol.get("auxiliary_l1_weight") == .25
             and protocol.get("start_update") == 4800
             and protocol.get("additional_updates_per_arm") == 300
             and protocol.get("real_data_used") is False
             and protocol.get("real_scoring_enabled") is False,
             f"{method}: quarter-weight registered protocol mismatch")
        need(protocol.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA,
             f"{method}: worker uses a different resource protocol")
        need(item.get("cross_generator_smoke_pairing", {}).get("passed") is True,
             f"{method}: manager cross-generator smoke barrier failed")
        arms = arm_map(completion.get("arms"), method)
        smoke_arms = arm_map(smoke.get("results"), method + "/smoke")
        before_ids = validate_evaluation(completion.get("before"), method + "/before")
        all_rollouts.append(before_ids)
        for arm in ARMS:
            row = arms[arm]
            need(row.get("method") == method and row.get("seed") == 0
                 and row.get("absolute_update") == 5100 and row.get("additional_updates") == 300,
                 f"{method}/{arm}: wrong method/seed/budget")
            need(row.get("real_data_used") is False, f"{method}/{arm}: synthetic-only contract missing")
            for name in ("label_sequence_sha256", "event_sequence_sha256"):
                digest(row.get(name))
            all_rollouts.append(validate_evaluation(row.get("validation"), f"{method}/{arm}"))
            checkpoint = item.get("checkpoint_evidence", {}).get(arm, {})
            for flag in ("model_hash_match", "rng_roundtrip_match",
                         "independent_model_optimizer_rebuild", "optimizer_state_equal"):
                need(checkpoint.get(flag) is True, f"{method}/{arm}: checkpoint {flag} missing")
            need(checkpoint.get("optimizer_counters") == [5100],
                 f"{method}/{arm}: wrong rebuilt Adam counters")
            digest(checkpoint.get("sha256"))
            need(isinstance(checkpoint.get("size_bytes"), int) and checkpoint["size_bytes"] > 0,
                 f"{method}/{arm}: checkpoint size missing")
            matches = [r for r in data.get("external_checkpoints", [])
                       if r.get("path") == checkpoint.get("path")]
            need(len(matches) == 1 and matches[0].get("sha256") == checkpoint["sha256"]
                 and matches[0].get("size_bytes") == checkpoint["size_bytes"],
                 f"{method}/{arm}: independently hashed checkpoint binding mismatch")

            validate_update(smoke_arms[arm].get("update"), 4801, f"{method}/smoke/{arm}", smoke=True)
        for name in ("label_sequence_sha256", "event_sequence_sha256"):
            need(len({arms[arm][name] for arm in ARMS}) == 1, f"{method}: arm {name} mismatch")
        parents = [json.dumps(arms[arm].get("parent"), sort_keys=True) for arm in ARMS]
        need(len(set(parents)) == 1 and arms[ARMS[0]].get("parent") is not None,
             f"{method}: two arms do not share the same source checkpoint binding")
        parent = arms[ARMS[0]]["parent"]
        need(isinstance(parent, dict) and parent.get("optimizer_update") == 4800,
             f"{method}: source checkpoint is not the registered update4800")
        digest(parent.get("sha256"))
        need(isinstance(parent.get("size_bytes"), int) and parent["size_bytes"] > 0,
             f"{method}: source checkpoint size binding missing")
        for name in ("labels", "events"):
            need(len({json.dumps(smoke_arms[a]["update"][name]) for a in ARMS}) == 1,
                 f"{method}: smoke{name} pairing mismatch")
        parity = smoke.get("native_parity", {})
        need(parity.get("loss_exact") is True and parity.get("gradient_tensors") == 30
             and parity.get("control_uses_native_function_identity") is True,
             f"{method}: native C0 identity check missing")
        need(parity.get("strict_cuda_gradient_reproducibility_claimed") is False,
             f"{method}: unsupported strict CUDA reproducibility claim")
        need(isinstance(parity.get("losses"), dict) and bool(parity["losses"]),
             f"{method}: native loss evidence missing")
        for key, value in parity["losses"].items():
            finite(value, f"{method}.native_losses.{key}")
        repeats = parity.get("comparisons")
        need(isinstance(repeats, list) and len(repeats) == 2,
             f"{method}: native CUDA self-repeat evidence missing")
        for row in repeats:
            need(isinstance(row, dict) and isinstance(row.get("allclose"), bool),
                 f"{method}: invalid native repeat evidence")
            for key in ("max_absolute_gradient_error", "relative_gradient_l2", "cosine"):
                finite(row.get(key), f"{method}.native_repeat.{key}")
        oracle = smoke.get("auxiliary_cuda_oracle")
        need(isinstance(oracle, list) and len(oracle) == 2, f"{method}: auxiliary CUDA oracle incomplete")
        need(all(isinstance(row, dict) for row in oracle), f"{method}: invalid auxiliary oracle records")
        need({row.get("dtype") for row in oracle} == {"torch.float32", "torch.float64"},
             f"{method}: auxiliary oracle must cover float32 and float64")
        for row in oracle:
            for key in ("loss_error", "gradient_max_error"):
                need(finite(row.get(key), f"{method}.auxiliary_oracle.{key}") >= 0,
                     f"{method}: invalid auxiliary oracle error")
        metadata[method] = {"arms": arms, "training": validate_training(item, arms, method),
                            "label_sequence_sha256": arms["control"]["label_sequence_sha256"],
                            "event_sequence_sha256": arms["control"]["event_sequence_sha256"],
                            "native_parity": parity, "auxiliary_cuda_oracle": oracle}
    need(metadata["eventgenesis"]["label_sequence_sha256"] == metadata["v2e"]["label_sequence_sha256"],
         "Cross-generator label/augmentation sequence mismatch")
    need(all(sorted(ids) == sorted(all_rollouts[0]) for ids in all_rollouts),
         "Before/arms/methods do not evaluate the same8 rollouts")
    eg = data["methods"]["eventgenesis"]["smoke"]["results"]
    v2e = data["methods"]["v2e"]["smoke"]["results"]
    need({r["arm"]: r["update"]["labels"] for r in eg}
         == {r["arm"]: r["update"]["labels"] for r in v2e},
         "Cross-generator smoke label/augmentation mismatch")
    return metadata


def compare_candidate(candidate, control):
    """Exact preregistered comparisons, on all200 rollout-weighted means only."""
    checks = [
        ("contrast_ratio", candidate["contrast_ratio"], ">", 0.1, candidate["contrast_ratio"] > 0.1),
        ("flat_fraction", candidate["flat_fraction"], "<", 0.05, candidate["flat_fraction"] < 0.05),
        ("mse", candidate["mse"], "<=", 0.95 * control["mse"], candidate["mse"] <= 0.95 * control["mse"]),
        ("psnr_db", candidate["psnr_db"], ">=", control["psnr_db"] + 0.2, candidate["psnr_db"] >= control["psnr_db"] + 0.2),
        ("ssim", candidate["ssim"], ">=", control["ssim"], candidate["ssim"] >= control["ssim"]),
        ("lpips_alex", candidate["lpips_alex"], "<=", 1.01 * control["lpips_alex"], candidate["lpips_alex"] <= 1.01 * control["lpips_alex"]),
        ("lpips_vgg", candidate["lpips_vgg"], "<=", 1.01 * control["lpips_vgg"], candidate["lpips_vgg"] <= 1.01 * control["lpips_vgg"]),
        ("edge_mae", candidate["edge_mae"], "<=", control["edge_mae"], candidate["edge_mae"] <= control["edge_mae"]),
        ("correlation", candidate["correlation"], ">=", control["correlation"], candidate["correlation"] >= control["correlation"]),
    ]
    rows = [{"metric": key, "value": val, "operator": op, "threshold": threshold,
             "control": control[key], "delta_vs_control": val - control[key], "passed": good}
            for key, val, op, threshold, good in checks]
    return {"passed": all(row["passed"] for row in rows), "checks": rows,
            "mse_change_fraction": (candidate["mse"] - control["mse"]) / control["mse"]
                if control["mse"] else None,
            "psnr_change_db": candidate["psnr_db"] - control["psnr_db"],
            "lpips_alex_change_fraction": candidate["lpips_alex"] / control["lpips_alex"] - 1
                if control["lpips_alex"] else None,
            "lpips_vgg_change_fraction": candidate["lpips_vgg"] / control["lpips_vgg"] - 1
                if control["lpips_vgg"] else None}


def analyze(data, metadata):
    gates = {"primary_subset": PRIMARY, "secondary_subset": "tail195",
             "no_real_scoring": True, "single_seed_exploratory": True,
             "training_evidence_passed": True, "pairing_passed": True,
             "edge_gate_metric": "pointwise horizontal/vertical gradient MAE (edge_mae)",
             "lpips_relative_tolerance": 0.01, "methods": {}, "candidates": {}}
    tables = {subset: [] for subset in SUBSETS}
    rollout_rows = []
    for method in METHODS:
        comp = data["methods"][method]["completion"]
        for arm, evaluation in [("before", comp["before"])] + [(a, metadata[method]["arms"][a]["validation"]) for a in ARMS]:
            for subset in SUBSETS:
                tables[subset].append({"method": method, "arm": arm, "seed": 0,
                                      "optimizer_update": 4800 if arm == "before" else 5100,
                                      "rollouts": 8, "frames_per_rollout": 200 if subset == "all200" else 195,
                                      **evaluation[subset]})
                for row in evaluation["rollouts"]:
                    rollout_rows.append({"method": method, "arm": arm, "subset": subset,
                                         "rollout": row["rollout"], **row[subset]})
        control = metadata[method]["arms"]["control"]["validation"][PRIMARY]
        gates["methods"][method] = {a: compare_candidate(metadata[method]["arms"][a]["validation"][PRIMARY], control)
                                      for a in ARMS[1:]}
    for arm in ARMS[1:]:
        passed = all(gates["methods"][m][arm]["passed"] for m in METHODS)
        gates["candidates"][arm] = {"passed_both_generators": passed,
                                    "action": "eligible for explicitly authorized broader synthetic validation" if passed
                                    else "not adopted; preserve failed checks and do not launch long training"}
    gates["any_candidate_passed_both"] = any(v["passed_both_generators"] for v in gates["candidates"].values())
    gates["training_evidence"] = {m: metadata[m]["training"] for m in METHODS}
    gates["sequence_hashes"] = {m: {k: metadata[m][k] for k in ("label_sequence_sha256", "event_sequence_sha256")} for m in METHODS}
    return gates, tables, rollout_rows


def write_csv(path, rows):
    keys = list(rows[0])
    need(all(set(row) == set(keys) for row in rows), f"{path}: inconsistent CSV columns")
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report_text(data, gates, tables):
    lines = ["# 合成重建光度与细节：配对短训结果", "",
             "EventGenesis/V2E各seed0，均从各自update4800共同源权重续训300次；两臂同预算、同样本、同增强。结果仅用于研究开发，不是多seed论文最终结果；没有真实数据训练或评分。", "",
             "主结果按8条独立rollout等权平均，每条200帧。tail195仅为次要诊断，不用于替换all200主门槛。", "",
             "|生成器|训练臂|MSE↓|PSNR↑|SSIM↑|LPIPS-Alex↓|LPIPS-VGG↓|强度MAE↓|强度偏差|梯度MAE↓|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tables[PRIMARY]:
        lines.append(f"|{LABELS[row['method']]}|{LABELS[row['arm']]}|{row['mse']:.6f}|{row['psnr_db']:.3f}|{row['ssim']:.4f}|{row['lpips_alex']:.4f}|{row['lpips_vgg']:.4f}|{row['intensity_mae']:.5f}|{row['intensity_bias']:+.5f}|{row['edge_mae']:.5f}|")
    lines += ["", "## 预登记门槛", "",
              "全部300步的loss、30个梯度张量与每更新76次TC均已核对；方法内两臂事件和标签序列哈希相同，跨方法标签序列哈希相同。跨方法事件哈希不要求相同，事件生成方式正是实验变量。", ""]
    for arm in ARMS[1:]:
        both = gates["candidates"][arm]["passed_both_generators"]
        lines.append(f"**{LABELS[arm]}：{'两个生成器均通过，可讨论扩大合成验证' if both else '未同时通过，不采用、不启动长训'}。**")
        lines.append("")
        for method in METHODS:
            result = gates["methods"][method][arm]
            failed = [r["metric"] for r in result["checks"] if not r["passed"]]
            change = result["mse_change_fraction"]
            mse = f"MSE相对同期控制变化{change * 100:+.2f}%" if change is not None else "控制MSE为0，相对变化不定义"
            lines.append(f"- {LABELS[method]}：{mse}，PSNR变化{result['psnr_change_db']:+.3f}dB；" + ("全部主门槛通过。" if not failed else "失败项：" + "、".join(failed) + "。"))
        lines.append("")
    lines += ["边缘采用逐像素横纵梯度MAE门槛，不能用edge_ratio变大代替细节精度改善。LPIPS-Alex和VGG各允许相对控制最多1%恶化；强度MAE、带符号强度偏差与边缘幅度比保留为诊断，没有另设事后门槛。", "",
              "源权重before单独报告以显示续训变化，但采用判定比较的是相同步数的同期control。未根据真实集选损失、权重或后处理。", "",
              "## 数值复现与失败证据", "",
              "C0直接使用原Trainer函数；CUDA grid_sample反传的原生重复波动单独保留，不宣称梯度逐位确定性，也不将native self-repeat的allclose失败改记为通过。新增辅助损失的CUDA对照证据保存在输入smoke中。", "",
              "先前失败证据原样保留在result_summary.json的failure_evidence，分析不删改。完整门槛数值、差值和配对哈希见adoption_gates.json；所有指标见all200_primary.csv、tail195_secondary.csv与per_rollout.csv。", "",
              "本脚本不启动任何任务。即使单seed通过，也仅表示值得后续明确授权的扩大验证，不能自动升级成长训或真实评分。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("result_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results_analysis"))
    parser.add_argument("--validate-only", action="store_true", help="Validate all evidence without writing results")
    args = parser.parse_args(argv)
    try:
        need(args.input.is_file(), f"Input missing: {args.input}")
        raw = args.input.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"), parse_constant=lambda value: (_ for _ in ()).throw(EvidenceError(f"Non-finite JSON token: {value}")))
        metadata = validate_input(data)
        gates, tables, rollouts = analyze(data, metadata)
        gates["input_sha256"] = hashlib.sha256(raw).hexdigest()
        protocol = Path(__file__).with_name("PROTOCOL.md")
        need(protocol.is_file(), "Missing preregistered PROTOCOL.md")
        gates["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
        need(gates["protocol_sha256"] == EXPECTED_PROTOCOL_SHA, "Local PROTOCOL.md is not the GPU2 registered version")
        if args.validate_only:
            print(json.dumps({"status": "evidence_complete", "any_candidate_passed_both": gates["any_candidate_passed_both"]}))
            return 0
        need(not args.output_dir.exists(), f"Output already exists; refusing overwrite: {args.output_dir}")
        # No output is created until all input and pairing checks have passed.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for subset, rows in tables.items():
            write_csv(args.output_dir / ("all200_primary.csv" if subset == PRIMARY else "tail195_secondary.csv"), rows)
        write_csv(args.output_dir / "per_rollout.csv", rollouts)
        with (args.output_dir / "adoption_gates.json").open("x", encoding="utf-8") as stream:
            json.dump(gates, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        with (args.output_dir / "REPORT.md").open("x", encoding="utf-8") as stream:
            stream.write(report_text(data, gates, tables))
        print(json.dumps({"status": "analyzed", "output_dir": str(args.output_dir.resolve()),
                          "any_candidate_passed_both": gates["any_candidate_passed_both"]}))
        return 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"EVIDENCE_INCOMPLETE_OR_INVALID: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
