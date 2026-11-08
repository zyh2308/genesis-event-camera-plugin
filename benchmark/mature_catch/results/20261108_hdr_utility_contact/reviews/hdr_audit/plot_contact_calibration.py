"""Plot only supplied GT calibration episode rows, retaining missing cases.

Input: a list, or {"rows": [...]}, containing profile, case_id,
verified_catch (actual bool), hand_contact_steps (nonnegative integer).
Optional top-level planned_case_ids defines cells not measured yet.
No native success flag is interpreted here: caller must audit strict success.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROFILES = ("baseline", "hand_friction1p5", "substeps2", "combined")
ZH = ("原场景", "仅手部摩擦×1.5", "内部子步×2", "摩擦与子步共同调整")
EN = ("Baseline", "Hand friction x1.5", "Two internal substeps", "Combined")


def read_rows(path, stage="calibration"):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = data if isinstance(data, list) else data["rows"]
    seen = set()
    for r in rows:
        if r["profile"] not in PROFILES:
            raise ValueError(f"Unknown profile {r['profile']}")
        if not isinstance(r["case_id"], str) or not r["case_id"]:
            raise ValueError("case_id must be a nonempty string")
        if type(r["verified_catch"]) is not bool:
            raise ValueError("verified_catch must be an audited actual bool")
        n = r["hand_contact_steps"]
        if type(n) is not int or n < 0:
            raise ValueError("hand_contact_steps must be a nonnegative integer")
        key = (r["profile"], r["case_id"])
        if key in seen:
            raise ValueError(f"Duplicate episode {key}")
        seen.add(key)
        if r["verified_catch"] and n == 0:
            raise ValueError(f"Strict catch without physical hand contact: {key}")
        if all(k in r for k in ("raw_success", "terminated", "final_contact")):
            expected = bool(r["raw_success"] and not r["terminated"] and n > 0 and r["final_contact"])
            if r["verified_catch"] != expected:
                raise ValueError(f"Verified catch conflicts with raw strict fields: {key}")
        prefix = "h" if stage == "calibration" else "v"
        if not r["case_id"].startswith(prefix):
            raise ValueError("Do not mix calibration h-cases and new validation v-cases")
        if stage == "validation" and r["profile"] not in ("baseline", "combined"):
            raise ValueError("Validation compares only fixed baseline and selected combined profile")
    planned = data.get("planned_case_ids") if isinstance(data, dict) else None
    cases = sorted({r["case_id"] for r in rows}) if planned is None else list(planned)
    if len(cases) != len(set(cases)) or not all(isinstance(x, str) for x in cases):
        raise ValueError("Invalid planned_case_ids")
    if not {r["case_id"] for r in rows} <= set(cases):
        raise ValueError("Measured case not included in planned_case_ids")
    return rows, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-per-profile", type=int, default=6)
    p.add_argument("--expect-baseline", help="Audit optional known count, e.g. 3/6")
    p.add_argument("--stage", choices=("calibration", "validation"), default="calibration",
                   help="New validation is plotted separately, never pooled with calibration")
    a = p.parse_args()
    rows, cases = read_rows(a.data, a.stage)
    profiles = PROFILES if a.stage == "calibration" else ("baseline", "combined")
    if not rows or not cases:
        raise ValueError("No actual completed episode rows supplied")
    if a.expected_per_profile <= 0 or len(cases) > a.expected_per_profile:
        raise ValueError("Invalid expected case count")
    summaries = {}
    for name in profiles:
        measured = [r for r in rows if r["profile"] == name]
        summaries[name] = dict(
            actual_episodes=len(measured),
            verified_catches=sum(r["verified_catch"] for r in measured),
            physical_contact_episodes=sum(r["hand_contact_steps"] > 0 for r in measured),
            planned_episodes=a.expected_per_profile,
            missing_episodes=a.expected_per_profile - len(measured),
        )
    if a.expect_baseline:
        k, n = map(int, a.expect_baseline.split("/"))
        s = summaries["baseline"]
        if (s["verified_catches"], s["actual_episodes"]) != (k, n):
            raise ValueError("Supplied baseline rows do not match the known baseline count")

    a.output.mkdir(parents=True, exist_ok=True)
    prefix = "GT接触参数校准" if a.stage == "calibration" else "GT接触参数新来球验证"
    csv_path = a.output / f"{prefix}_逐条.csv"
    by_key = {(r["profile"], r["case_id"]): r for r in rows}
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["配置", "case_id", "严格接取成功", "真实手部接触控制步数", "记录状态"])
        for profile in profiles:
            label = ZH[PROFILES.index(profile)]
            for case in cases:
                r = by_key.get((profile, case))
                w.writerow([label, case, ("是" if r["verified_catch"] else "否") if r else "",
                            r["hand_contact_steps"] if r else "", "已实测" if r else "未完成"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import ListedColormap, BoundaryNorm
    font = next((p for p in [Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")] if p.exists()), None)
    chinese = font is not None
    if font:
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    labels = [(ZH if chinese else EN)[PROFILES.index(p)] for p in profiles]
    values = np.full((len(profiles), len(cases)), -1.)
    for i, profile in enumerate(profiles):
        for j, case in enumerate(cases):
            r = by_key.get((profile, case))
            if r is not None:
                values[i, j] = int(r["verified_catch"])
    fig = plt.figure(figsize=(12.6, 5.5 if a.stage == "calibration" else 4.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.2], wspace=.13)
    ax = fig.add_subplot(grid[0, 0]); bars = fig.add_subplot(grid[0, 1])
    cmap = ListedColormap(["#e9edf1", "#e5b8a6", "#75b9ad"])
    ax.imshow(values, cmap=cmap, norm=BoundaryNorm([-1.5, -.5, .5, 1.5], cmap.N), aspect="auto")
    ax.set_xticks(range(len(cases)), cases); ax.set_yticks(range(len(profiles)), labels)
    for i, profile in enumerate(profiles):
        for j, case in enumerate(cases):
            r = by_key.get((profile, case))
            text = ("未完成" if chinese else "Pending") if r is None else (
                ("成功" if r["verified_catch"] else "失败") if chinese else
                ("Caught" if r["verified_catch"] else "Failed")) + "\n" + (
                    f"接触 {r['hand_contact_steps']}步" if chinese else f"Contact {r['hand_contact_steps']} steps")
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color="#263544")
    ax.set_xticks(np.arange(-.5, len(cases), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(profiles), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("逐回合结果｜缺失数据保留为空" if chinese else "Per-episode results | missing data retained", pad=13)
    counts = [summaries[p]["verified_catches"] for p in profiles]
    bars.barh(range(len(profiles)), counts, color="#218c87", height=.57)
    bars.set_ylim(len(profiles)-.5, -.5); bars.set_yticks([])
    bars.set_xlim(0, a.expected_per_profile + 1.5)
    bars.set_xticks(range(0, a.expected_per_profile + 1, 2))
    bars.set_xlabel("严格成功回合数" if chinese else "Verified catches")
    bars.set_title("成功 / 已实测" if chinese else "Caught / measured", pad=13)
    for i, profile in enumerate(profiles):
        s = summaries[profile]
        bars.text(counts[i] + .12, i, f"{counts[i]}/{s['actual_episodes']}", va="center", fontsize=13)
    bars.grid(axis="x", alpha=.15); bars.set_axisbelow(True)
    title = (("共同接触场景校准｜六条GT校准来球" if chinese else "Common contact calibration | six GT calibration cases")
        if a.stage == "calibration" else ("共同接触场景验证｜六条新来球、固定已选配置" if chinese else
        "Common contact validation | six new cases, fixed selected profile"))
    fig.suptitle(title, fontsize=17, y=.98)
    scope_zh = ("本图为六条GT校准来球的物理可行性检查，不是事件感知结果或生成器排名。" if a.stage == "calibration" else
        "新来球未用于选择配置；本图检验物理可行性泛化，不是事件感知结果或生成器排名。")
    scope_en = ("Six GT calibration cases: physical feasibility only, not event perception or generator ranking." if a.stage == "calibration" else
        "New cases did not select the profile: physical generalization only, not event perception or generator ranking.")
    note = ("所有事件方法共用物理参数；原权重、冻结事件核心与严格计分不变。接触步数为25 Hz控制步。\n" + scope_zh) if chinese else (
            "Common physics for all event sources; original policy, event core and strict scoring unchanged. Contact steps are 25 Hz control steps.\n"
            + scope_en)
    fig.text(.025, .025, note, fontsize=10, color="#52606c")
    fig.subplots_adjust(left=.17, right=.98, bottom=.20, top=.83)
    png = a.output / f"{prefix}_结果.png"
    pdf = a.output / f"{prefix}_结果.pdf"
    fig.savefig(png, dpi=190); fig.savefig(pdf); plt.close(fig)
    summary = dict(stage=a.stage, source=str(a.data.resolve()), source_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        actual_rows=len(rows), by_profile=summaries, all_planned_cases_present=all(
            s["missing_episodes"] == 0 for s in summaries.values()),
        missing_not_assumed_success_or_failure=True)
    summary_name = "CONTACT_CALIBRATION_PLOT_SUMMARY.json" if a.stage == "calibration" else "CONTACT_VALIDATION_PLOT_SUMMARY.json"
    (a.output / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(csv=str(csv_path), png=str(png), pdf=str(pdf), summary=summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
