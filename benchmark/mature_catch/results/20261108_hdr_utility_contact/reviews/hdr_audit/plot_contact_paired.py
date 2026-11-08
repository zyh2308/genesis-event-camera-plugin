"""Plot audited paired event-catch results; no SSH or metric redefinition.

Expected JSON: rows with case_id, seed, method (ours/v2e/init_only),
verified_catch, raw_success, terminated, hand_contact_steps, final_contact,
first_touch_s, raw_events, output; and actual per-method scores.
Missing rows stay empty; their outcomes are never imputed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


METHODS = ("ours", "v2e", "init_only")
LABELS_ZH = ("Ours", "V2E", "停止后续事件更新")
LABELS_EN = ("Ours", "V2E", "Stop subsequent event updates")


def read_results(path, expected=6, require_complete=False):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = data["rows"]
    if expected <= 0 or not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of actual episode rows")
    seen = set(); seeds = {}
    for r in rows:
        if r["method"] not in METHODS:
            raise ValueError(f"Unknown method {r['method']}")
        if not isinstance(r["case_id"], str) or not r["case_id"]:
            raise ValueError("case_id must be a nonempty string")
        for field in ("verified_catch", "raw_success", "terminated", "final_contact"):
            if type(r[field]) is not bool:
                raise ValueError(f"{field} must be an actual bool")
        if type(r["seed"]) is not int or type(r["hand_contact_steps"]) is not int or r["hand_contact_steps"] < 0:
            raise ValueError("seed/contact steps must be integers; contact steps nonnegative")
        strict = bool(r["raw_success"] and not r["terminated"] and r["hand_contact_steps"] > 0 and r["final_contact"])
        if strict != r["verified_catch"]:
            raise ValueError(f"Strict fields conflict for {r['case_id']}/{r['method']}")
        key = (r["method"], r["case_id"])
        if key in seen:
            raise ValueError(f"Duplicate episode {key}")
        seen.add(key)
        previous = seeds.setdefault(r["case_id"], r["seed"])
        if previous != r["seed"]:
            raise ValueError(f"Paired seed differs across methods: {r['case_id']}")
        if r.get("raw_events") is not None and (type(r["raw_events"]) is not int or r["raw_events"] < 0):
            raise ValueError("raw_events must be a nonnegative integer or null")
    cases = data.get("planned_case_ids", sorted(seeds))
    if len(cases) != len(set(cases)) or len(cases) > expected or not set(seeds) <= set(cases):
        raise ValueError("Invalid planned_case_ids")
    summaries = {}
    for method in METHODS:
        measured = [r for r in rows if r["method"] == method]
        n = len(measured)
        if n > expected:
            raise ValueError("More measured episodes than the registered per-method budget")
        summary = dict(n=n, verified_catches=sum(r["verified_catch"] for r in measured),
            contact_episodes=sum(r["hand_contact_steps"] > 0 for r in measured),
            planned_episodes=expected, missing_episodes=expected-n)
        declared = data["scores"][method]
        if any(type(declared[k]) is not int or declared[k] != summary[k]
               for k in ("n", "verified_catches", "contact_episodes")):
            raise ValueError(f"Declared scores do not match actual rows: {method}")
        summaries[method] = summary
    if require_complete and (len(cases) != expected or any(s["n"] != expected for s in summaries.values())):
        raise ValueError("The paired bank has not completed all registered method/case combinations")
    return data, rows, cases, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-per-method", type=int, default=6)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    data, rows, cases, summaries = read_results(a.data, a.expected_per_method, a.require_complete)
    a.output.mkdir(parents=True, exist_ok=True)
    prefix = "25Hz事件闭环配对"
    csv_path = a.output / f"{prefix}_逐条.csv"
    by_key = {(r["method"], r["case_id"]): r for r in rows}
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["方法", "case_id", "seed", "严格接取成功", "真实手部接触控制步数", "最终仍接触",
            "原始成功标志", "提前终止", "首次接触时刻_s", "原始事件数", "结果目录", "记录状态"])
        for method, label in zip(METHODS, LABELS_ZH):
            for case in cases:
                r = by_key.get((method, case))
                if r is None:
                    w.writerow([label, case, "", "", "", "", "", "", "", "", "", "未完成"])
                else:
                    w.writerow([label, case, r["seed"], "是" if r["verified_catch"] else "否", r["hand_contact_steps"],
                        r["final_contact"], r["raw_success"], r["terminated"], r.get("first_touch_s"),
                        r.get("raw_events"), r.get("output"), "已实测"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.font_manager import FontProperties
    font = next((p for p in [Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")] if p.exists()), None)
    chinese = font is not None
    if font:
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    values = np.full((3, len(cases)), -1.)
    for i, method in enumerate(METHODS):
        for j, case in enumerate(cases):
            r = by_key.get((method, case))
            if r is not None: values[i, j] = int(r["verified_catch"])
    fig = plt.figure(figsize=(12.8, 5.1))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.15], wspace=.13)
    ax = fig.add_subplot(grid[0, 0]); bars = fig.add_subplot(grid[0, 1])
    cmap = ListedColormap(["#e9edf1", "#e5b8a6", "#75b9ad"])
    ax.imshow(values, cmap=cmap, norm=BoundaryNorm([-1.5, -.5, .5, 1.5], cmap.N), aspect="auto")
    labels = LABELS_ZH if chinese else LABELS_EN
    short_cases = [case.split("_")[0] for case in cases]
    if len(set(short_cases)) != len(cases): short_cases = cases
    ax.set_xticks(range(len(cases)), short_cases); ax.set_yticks(range(3), labels)
    for i, method in enumerate(METHODS):
        for j, case in enumerate(cases):
            r = by_key.get((method, case))
            if r is None:
                text = "未完成" if chinese else "Pending"
            else:
                text = (("成功" if r["verified_catch"] else "失败") if chinese else
                    ("Caught" if r["verified_catch"] else "Failed")) + "\n" + (
                    f"接触 {r['hand_contact_steps']}步" if chinese else f"Contact {r['hand_contact_steps']} steps")
            ax.text(j, i, text, ha="center", va="center", fontsize=10.5, color="#263544")
    ax.set_xticks(np.arange(-.5, len(cases), 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("逐回合结果｜成功与失败全部保留" if chinese else "Paired episodes | all successes and failures retained", pad=13)
    counts = [summaries[m]["verified_catches"] for m in METHODS]
    bars.barh(range(3), counts, height=.56, color=["#218c87", "#c17c48", "#758697"])
    bars.set_ylim(2.5, -.5); bars.set_yticks([])
    bars.set_xlim(0, a.expected_per_method + 1.5)
    bars.set_xticks(range(0, a.expected_per_method + 1, 2))
    bars.set_xlabel("严格成功回合数" if chinese else "Verified catches")
    bars.set_title("成功 / 已实测" if chinese else "Caught / measured", pad=13)
    for i, method in enumerate(METHODS):
        bars.text(counts[i] + .12, i, f"{counts[i]}/{summaries[method]['n']}", va="center", fontsize=13)
    bars.grid(axis="x", alpha=.15); bars.set_axisbelow(True)
    title = ("事件闭环接取｜25 Hz共同源图像、六条配对新来球" if chinese else
        "Event-based catch | shared 25 Hz source images, six new paired cases")
    fig.suptitle(title, fontsize=17, y=.98)
    note = ("共同接触配置、估计器、原权重和冻结事件核心固定；同一首帧初始化。接触步数为25 Hz控制步。\n"
        "当前六条条件的配对结果；停止更新组不再使用后续事件。不混入GT接取结果，不作总体排名。") if chinese else (
        "Fixed common contact profile, estimator, original policy and event core; shared first-frame initialization. Contact steps are 25 Hz control steps.\n"
        "Only these six paired cases. Ablation stops subsequent event updates; no GT success rates or overall generator ranking included.")
    fig.text(.025, .025, note, fontsize=10, color="#52606c")
    fig.subplots_adjust(left=.17, right=.98, bottom=.21, top=.82)
    png = a.output / f"{prefix}_结果.png"; pdf = a.output / f"{prefix}_结果.pdf"
    fig.savefig(png, dpi=190); fig.savefig(pdf); plt.close(fig)
    summary = dict(source=str(a.data.resolve()), source_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        actual_rows=len(rows), expected_per_method=a.expected_per_method, by_method=summaries,
        all_planned_cases_present=len(cases) == a.expected_per_method and all(s["missing_episodes"] == 0 for s in summaries.values()),
        missing_not_assumed_success_or_failure=True, GT_results_included=False)
    summary_path = a.output / "CONTACT_PAIRED_PLOT_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(csv=str(csv_path), png=str(png), pdf=str(pdf), summary=summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
