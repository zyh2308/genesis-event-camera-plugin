"""Compare measured original/shared-flight-fit estimators on identical cases.

This plot includes Ours and V2E only, and never pools GT or initialization-only
outcomes into their denominators. Both input files remain hashed in full.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


METHODS = ("ours", "v2e")


def load(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data["status"] != "PASS" or data["all_planned_successes_and_failures_retained"] is not True:
        raise ValueError("A completed audited source retaining all outcomes is required")
    rows = [r for r in data["rows"] if r["method"] in METHODS]
    keys = set()
    for r in rows:
        for name in ("verified_catch", "raw_success", "terminated", "final_contact"):
            if type(r[name]) is not bool: raise ValueError(f"Invalid boolean {name}")
        if type(r["hand_contact_steps"]) is not int or r["hand_contact_steps"] < 0:
            raise ValueError("Invalid contact count")
        strict = r["raw_success"] and not r["terminated"] and r["hand_contact_steps"] > 0 and r["final_contact"]
        if strict != r["verified_catch"]: raise ValueError("Conflicting strict result")
        key = (r["method"], r["case_id"])
        if key in keys: raise ValueError("Duplicate actual episode")
        keys.add(key)
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        actual = dict(n=len(selected), verified_catches=sum(r["verified_catch"] for r in selected),
                      contact_episodes=sum(r["hand_contact_steps"] > 0 for r in selected))
        if actual["n"] != 6 or any(actual[k] != data["scores"][method][k] for k in actual):
            raise ValueError("Expected six actual episodes with matching declared scores")
    return data, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    base, base_rows = load(a.base); fit, fit_rows = load(a.fit)
    bp = dict(base["fixed_parameters"]); fp = dict(fit["fixed_parameters"])
    if bp.pop("causal_flight_fit_enabled", False) is not False or fp.pop("causal_flight_fit_enabled", False) is not True:
        raise ValueError("Expected original estimator versus enabled shared causal flight fit")
    if bp != fp: raise ValueError("Other declared fixed experiment parameters differ")
    cases = sorted({r["case_id"] for r in base_rows})
    if len(cases) != 6 or set(cases) != {r["case_id"] for r in fit_rows}:
        raise ValueError("The two experiments must use the same six cases")
    sequences = [("base", "ours", base_rows), ("base", "v2e", base_rows),
                 ("shared_fit", "ours", fit_rows), ("shared_fit", "v2e", fit_rows)]
    indexed = []
    paired_seeds = {}
    for stage, method, rows in sequences:
        by_case = {r["case_id"]: r for r in rows if r["method"] == method}
        if set(by_case) != set(cases): raise ValueError("Unpaired case bank")
        for case, row in by_case.items():
            previous = paired_seeds.setdefault(case, row["seed"])
            if previous != row["seed"]: raise ValueError("A paired case has different seeds")
        indexed.append(by_case)
    counts = [sum(r["verified_catch"] for r in rows.values()) for rows in indexed]
    a.output.mkdir(parents=True, exist_ok=True)
    prefix = "25Hz共同估计器对照"
    csv_path = a.output / f"{prefix}_逐条.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["共同估计器", "事件方法", "case_id", "seed", "严格成功", "真实接触控制步数",
                    "原始成功", "提前终止", "最终接触", "首次接触时刻_s", "原始事件数", "结果目录"])
        for (stage, method, _), by_case in zip(sequences, indexed):
            for case in cases:
                r = by_case[case]
                w.writerow(["原估计器" if stage == "base" else "共同因果轨迹拟合", method, case, r["seed"],
                    "是" if r["verified_catch"] else "否", r["hand_contact_steps"], r["raw_success"],
                    r["terminated"], r["final_contact"], r.get("first_touch_s"), r.get("raw_events"), r.get("output")])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.font_manager import FontProperties
    font = next((f for f in [Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")] if f.exists()), None)
    zh = font is not None
    if font: plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 5.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.15], wspace=.13)
    ax = fig.add_subplot(grid[0, 0]); bar = fig.add_subplot(grid[0, 1])
    matrix = np.array([[int(by_case[c]["verified_catch"]) for c in cases] for by_case in indexed])
    cmap = ListedColormap(["#e5b8a6", "#75b9ad"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=BoundaryNorm([-.5,.5,1.5],2))
    labels = (["原估计器 · Ours", "原估计器 · V2E", "共同轨迹拟合 · Ours", "共同轨迹拟合 · V2E"] if zh else
        ["Original estimator · Ours", "Original estimator · V2E", "Shared flight fit · Ours", "Shared flight fit · V2E"])
    ax.set_yticks(range(4), labels)
    short = [c.split("_")[0] for c in cases]
    ax.set_xticks(range(6), short if len(set(short))==6 else cases)
    for i, by_case in enumerate(indexed):
        for j, case in enumerate(cases):
            r = by_case[case]
            text = (("成功" if r["verified_catch"] else "失败") if zh else
                ("Caught" if r["verified_catch"] else "Failed")) + "\n" + (
                f"接触 {r['hand_contact_steps']}步" if zh else f"Contact {r['hand_contact_steps']} steps")
            ax.text(j,i,text,ha="center",va="center",fontsize=10.5,color="#263544")
    ax.set_xticks(np.arange(-.5,6,1),minor=True); ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=3);ax.tick_params(which="minor",bottom=False,left=False)
    ax.axhline(1.5,color="#35424f",linewidth=1.6)
    ax.set_title("同一批六条开发来球｜所有成功与失败保留" if zh else
        "Same six development cases | all successes and failures retained",pad=13)
    bar.barh(range(4),counts,color=["#218c87","#c17c48","#218c87","#c17c48"],height=.57)
    bar.set_ylim(3.5,-.5);bar.set_yticks([]);bar.set_xlim(0,7.4);bar.set_xticks([0,2,4,6])
    bar.axhline(1.5,color="#35424f",linewidth=1.6);bar.grid(axis="x",alpha=.15);bar.set_axisbelow(True)
    for i,k in enumerate(counts):bar.text(k+.12,i,f"{k}/6",va="center",fontsize=13)
    bar.set_title("严格成功 / 实测回合" if zh else "Verified catches / measured",pad=13)
    bar.set_xlabel("严格接取成功回合数" if zh else "Verified catches")
    fig.suptitle("共同估计器改进｜25 Hz源图像、原权重与冻结仿真器" if zh else
        "Shared estimator update | 25 Hz images, original policy and frozen simulator",fontsize=17,y=.98)
    note = ("双方共同启用因果轨迹拟合；共同接触配置、原策略权重和冻结事件核心不变。接触步数为25 Hz控制步。\n"
        f"当前开发对照：Ours {counts[0]}/6 → {counts[2]}/6，V2E {counts[1]}/6 → {counts[3]}/6；不据此宣称Ours整体领先。") if zh else (
        "The same causal flight fit is enabled for both sources; contact profile, policy weights and event core stay fixed.\n"
        f"Development comparison: Ours {counts[0]}/6 to {counts[2]}/6, V2E {counts[1]}/6 to {counts[3]}/6; no overall superiority claim.")
    fig.text(.025,.025,note,fontsize=10,color="#52606c")
    fig.subplots_adjust(left=.19,right=.98,bottom=.20,top=.82)
    png=a.output/f"{prefix}_结果.png";pdf=a.output/f"{prefix}_结果.pdf"
    fig.savefig(png,dpi=190);fig.savefig(pdf);plt.close(fig)
    sources={stage:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for stage,path in [("base",a.base),("shared_fit",a.fit)]}
    manifest=dict(sources=sources,actual_plotted_rows=24,paired_cases=cases,paired_seeds=paired_seeds,
        fixed_parameters=bp,changed_only_declared_parameter="causal_flight_fit_enabled, common to both methods",
        outcomes=[dict(stage=s,method=m,n=6,verified_catches=k) for (s,m,_),k in zip(sequences,counts)],
        GT_or_init_only_pooled=False,all_ours_v2e_successes_and_failures_retained=True)
    for path in (png,pdf,csv_path):
        manifest.setdefault("outputs",{})[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path=a.output/f"{prefix}_SOURCE_SHA256.json"
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(png=str(png),pdf=str(pdf),csv=str(csv_path),manifest=str(manifest_path),counts=counts),ensure_ascii=False))


if __name__=="__main__":main()
