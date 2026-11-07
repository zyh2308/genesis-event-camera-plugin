"""Export a synthetic-only activation/budget diagnosis; never starts training.

Run with the same-folder ``result_summary.json`` present. All derived files are
written to ``analysis/`` next to this script. No real dataset is opened.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
METHODS = ("eventgenesis", "v2e")
ARMS = ("original", "sigmoid")
LABELS = {"eventgenesis": "EventGenesis", "v2e": "V2E", "original": "线性输出", "sigmoid": "sigmoid"}
METRICS = (
    "lpips_alex", "lpips_vgg", "raw_lpips_vgg", "mse", "ssim", "psnr_db",
    "clipped_std", "target_std", "contrast_ratio", "edge_ratio", "correlation",
    "near_boundary_fraction", "sigmoid_derivative_mean", "logit_std", "flat_fraction",
    "edge_mean", "target_edge_mean", "raw_mean", "raw_std", "negative_fraction", "above_one_fraction",
)
OPTIONAL = {"near_boundary_fraction", "sigmoid_derivative_mean", "logit_std", "edge_mean", "target_edge_mean", "raw_mean", "raw_std", "negative_fraction", "above_one_fraction"}
REQUIRED = set(METRICS) - OPTIONAL
LOWER = {"lpips_alex", "lpips_vgg", "mse", "flat_fraction"}
HIGHER = {"ssim", "psnr_db", "correlation"}
WINDOWS = ("all200", "tail195")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    def clean(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and not math.isfinite(v):
            if math.isnan(v):
                return "undefined_nonfinite"
            return "positive_infinity" if v > 0 else "negative_infinity"
        return v
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert rows, f"No rows for {path.name}"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def fmt(value: Any, places: int = 5, signed: bool = False) -> str:
    if value is None:
        return "未记录"
    if math.isinf(float(value)):
        return "+∞" if float(value) > 0 else "−∞"
    return format(float(value), ("+" if signed else "") + "." + str(places) + "f")


def metric_dict(value: dict[str, Any], label: str) -> dict[str, Any]:
    missing = REQUIRED - value.keys()
    assert not missing, (label, "missing metrics", sorted(missing))
    out = {k: value.get(k) for k in METRICS}
    for k, v in out.items():
        if v is None:
            assert k not in REQUIRED, (label, k, "required metric is null")
            continue
        assert isinstance(v, (float, int)) and not isinstance(v, bool), (label, k, v)
        assert not math.isnan(float(v)), (label, k, "NaN")
        assert math.isfinite(float(v)) or (k == "psnr_db" and v > 0), (label, k, "nonfinite")
    return out


def validate_evaluation(e: dict[str, Any], label: str) -> set[str]:
    for window in WINDOWS:
        metric_dict(e[window], label + "/" + window)
    rollouts = e["rollouts"]
    ids = [r["rollout"] for r in rollouts]
    assert len(ids) == 8 and len(set(ids)) == 8, (label, "requires all8 distinct rollouts", ids)
    for r in rollouts:
        for window in WINDOWS:
            metric_dict(r[window], label + "/" + r["rollout"] + "/" + window)
    # Verify equal-rollout aggregation independently; frames are not replicates.
    for window in WINDOWS:
        for metric in REQUIRED:
            vals = [float(r[window][metric]) for r in rollouts]
            if all(math.isfinite(x) for x in vals):
                mean = math.fsum(vals) / len(vals)
                assert math.isclose(float(e[window][metric]), mean, rel_tol=2e-6, abs_tol=2e-8), (
                    label, window, metric, "not equal-rollout mean", e[window][metric], mean)
    return set(ids)


def digest_valid(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def change(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k in METRICS:
        x, y = a.get(k), b.get(k)
        out[k] = None if x is None or y is None or (not math.isfinite(float(x)) and x == y) else float(y) - float(x)
    return out


def main() -> None:
    source = HERE / "result_summary.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    out = HERE / "analysis"
    out.mkdir(exist_ok=True)
    arms: dict[tuple[str, str], dict[str, Any]] = {}
    refs: dict[tuple[str, str], dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for method in METHODS:
        completed = data["workers"][method]["completion"]
        assert completed["status"] == "complete", (method, "worker incomplete")
        records = completed["arms"]
        assert len(records) == 2 and {a["arm"] for a in records} == set(ARMS)
        for a in records:
            assert a["method"] == method and a["seed"] == 0 and a["updates"] == 600
            assert {"0", "300", "600"}.issubset(a["validation"]), (method, a["arm"], "missing timepoints")
            arms[method, a["arm"]] = a
            for key in ("initial_state_sha256", "label_sequence_sha256", "event_sequence_sha256"):
                checks.append(dict(name=f"{method}/{a['arm']}/{key}_valid", passed=digest_valid(a.get(key)), evidence=a.get(key)))
        for key in ("initial_state_sha256", "label_sequence_sha256", "event_sequence_sha256"):
            values = [a[key] for a in records]
            checks.append(dict(name=f"{method}/paired_{key}", passed=len(set(values)) == 1, evidence=values))
        for r in completed.get("references", []):
            assert r["name"] in {"sigmoid_4800", "sigmoid_stop", "linear_historical"}
            assert (method, r["name"]) not in refs
            assert isinstance(r["updates"], int) and r["updates"] > 0 and digest_valid(r["checkpoint_sha256"])
            refs[method, r["name"]] = r
        assert (method, "sigmoid_4800") in refs and (method, "sigmoid_stop") in refs
        assert refs[method, "sigmoid_4800"]["updates"] == 4800
        assert refs[method, "sigmoid_stop"]["updates"] > 4800
    for key in ("initial_state_sha256", "label_sequence_sha256"):
        values = [a[key] for a in arms.values()]
        checks.append(dict(name="all_four/" + key, passed=len(set(values)) == 1, evidence=values))
    pairing = dict(passed=all(c["passed"] for c in checks), checks=checks,
                   input_pairing_record=data.get("pairing"),
                   policy="Initial and label/augmentation digests equal across all4 arms; event digests equal only within a generator.")
    write_json(out / "pairing_checks.json", pairing)
    assert pairing["passed"], "Pairing failed; comparisons were not exported. See analysis/pairing_checks.json"

    rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    ids_expected: set[str] | None = None
    for (method, arm), a in arms.items():
        for step, e in sorted(a["validation"].items(), key=lambda x: int(x[0])):
            ids = validate_evaluation(e, f"{method}/{arm}/{step}")
            ids_expected = ids if ids_expected is None else ids_expected
            assert ids == ids_expected, "Validation rollout identities differ"
            info = dict(method=method,arm=arm,source_kind="matched_short_training",seed=0,updates=int(step),
                        initial_state_sha256=a["initial_state_sha256"],label_sequence_sha256=a["label_sequence_sha256"],
                        event_sequence_sha256=a["event_sequence_sha256"],checkpoint_sha256="")
            for window in WINDOWS:
                rows.append(dict(**info,window=window,**metric_dict(e[window],str(info))))
            for r in e["rollouts"]:
                for window in WINDOWS:
                    rollout_rows.append(dict(**info,window=window,rollout=r["rollout"],**metric_dict(r[window],str(info))))
    for (method, name), r in refs.items():
        e = r["validation"];ids = validate_evaluation(e, method + "/" + name)
        assert ids == ids_expected
        info = dict(method=method,arm=name,source_kind="existing_checkpoint_reference",seed=0,updates=r["updates"],
                    initial_state_sha256="",label_sequence_sha256="",event_sequence_sha256="",checkpoint_sha256=r["checkpoint_sha256"])
        for window in WINDOWS:
            rows.append(dict(**info,window=window,**metric_dict(e[window],str(info))))
        for rr in e["rollouts"]:
            for window in WINDOWS:
                rollout_rows.append(dict(**info,window=window,rollout=rr["rollout"],**metric_dict(rr[window],str(info))))
    write_csv(out / "all_timepoints.csv", rows)
    write_csv(out / "per_rollout.csv", rollout_rows)
    mechanism_rows = []
    for r in rows:
        mechanism_rows.append({k:r[k] for k in ("method","arm","source_kind","seed","updates","window","clipped_std","target_std","contrast_ratio","edge_ratio","correlation","flat_fraction","near_boundary_fraction","sigmoid_derivative_mean","logit_std","edge_mean","target_edge_mean","raw_mean","raw_std","negative_fraction","above_one_fraction")} | {
            "raw_minus_clipped_lpips_vgg":float(r["raw_lpips_vgg"]) - float(r["lpips_vgg"])})
    write_csv(out / "mechanisms.csv", mechanism_rows)

    deltas=[]; learning=[]; summaries={}
    for method in METHODS:
        linear=arms[method,"original"];sigmoid=arms[method,"sigmoid"]
        assert set(linear["validation"]) == set(sigmoid["validation"])
        summaries[method] = dict(activation={},short_learning={},late_learning={})
        for step in sorted(linear["validation"],key=int):
            for window in WINDOWS:
                l=linear["validation"][step][window];s=sigmoid["validation"][step][window]
                diff=change(l,s)
                if window=="all200":summaries[method]["activation"][step]=diff
                for metric in METRICS:
                    deltas.append(dict(method=method,updates=int(step),window=window,metric=metric,
                        sigmoid=s.get(metric),linear=l.get(metric),sigmoid_minus_linear=diff[metric],
                        better_direction="lower" if metric in LOWER else "higher" if metric in HIGHER else "diagnostic_only"))
        for arm in ARMS:
            for window in WINDOWS:
                a=arms[method,arm]["validation"]["300"][window];b=arms[method,arm]["validation"]["600"][window]
                diff=change(a,b)
                if window=="all200":summaries[method]["short_learning"][arm]=diff
                for metric in METRICS:
                    learning.append(dict(method=method,arm=arm,source_kind="matched_short_training",from_updates=300,to_updates=600,
                        window=window,metric=metric,before=a.get(metric),after=b.get(metric),after_minus_before=diff[metric]))
        r0=refs[method,"sigmoid_4800"];r1=refs[method,"sigmoid_stop"]
        for window in WINDOWS:
            a=r0["validation"][window];b=r1["validation"][window];diff=change(a,b)
            if window=="all200":summaries[method]["late_learning"]=dict(from_updates=4800,to_updates=r1["updates"],delta=diff)
            for metric in METRICS:
                learning.append(dict(method=method,arm="sigmoid_existing_trajectory",source_kind="existing_continuous_trajectory",
                    from_updates=4800,to_updates=r1["updates"],window=window,metric=metric,
                    before=a.get(metric),after=b.get(metric),after_minus_before=diff[metric]))
    write_csv(out / "paired_activation_deltas.csv", deltas)
    write_csv(out / "learning_changes.csv", learning)
    # Native continuous-training logs contextualize the selected best point.
    # Define the tail by chronological position (last5 points), not quality.
    history_rows=[];history_trends=[];histories={}
    for method in METHODS:
        history=data["workers"][method].get("formal_validation_history",[])
        history=sorted(history,key=lambda r:int(r["optimizer_update"]))
        assert len({int(r["optimizer_update"]) for r in history})==len(history), (method,"duplicate formal-validation steps")
        histories[method]=history
        for h in history:
            value=float(h["lpips_vgg_equal_rollout_mean"])
            assert math.isfinite(value)
            history_rows.append(dict(method=method,optimizer_update=int(h["optimizer_update"]),epoch=h.get("epoch"),
                lpips_vgg_equal_rollout_mean=value,selected_best_checkpoint=int(h["optimizer_update"])==4800,
                source_kind="native_continuous_sigmoid_training_log",selection_bias_warning="4800 was chosen as a VGG minimum; later rebound is not evidence of a permanent plateau"))
        tail=history[-5:]
        if len(tail)>=2:
            xs=[float(h["optimizer_update"]) for h in tail];ys=[float(h["lpips_vgg_equal_rollout_mean"]) for h in tail]
            mx=math.fsum(xs)/len(xs);my=math.fsum(ys)/len(ys)
            slope=math.fsum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.fsum((x-mx)**2 for x in xs)
            history_trends.append(dict(method=method,total_validation_records=len(history),tail_points=len(tail),
                tail_from_updates=int(xs[0]),tail_to_updates=int(xs[-1]),first_native_vgg=ys[0],last_native_vgg=ys[-1],
                last_minus_first=ys[-1]-ys[0],descriptive_ols_slope_per100updates=slope*100,
                tail_definition="last5 chronological validation points or all available if fewer; fixed without using scores",
                inference="descriptive only; repeated validations are correlated, not independent samples"))
    if history_rows:write_csv(out/"formal_validation_history.csv",history_rows)
    if history_trends:write_csv(out/"formal_history_tail_trends.csv",history_trends)
    write_json(out / "analysis_summary.json",dict(captured_utc=data["captured_utc"],input_sha256=sha(source),main_window="all200",sensitivity_window="tail195",pairing_passed=True,methods=summaries))

    lines=["# 重建：训练进度与输出激活的合成短程配对诊断","",
        f"输入证据时间：{data['captured_utc']}。全部结果来自合成验证；本脚本只读结果、导出表格和图，不启动训练或真实测试。","",
        "主比较使用全部200帧；首5帧只预热的tail195同时完整导出。每项指标先在rollout内按帧平均，再对同一8条独立验证rollout等权平均；没有把帧当成独立重复。","",
        "配对检查：两方法各有线性/sigmoid两个seed0训练臂；四臂初始参数与实际标签/增强序列摘要一致，每种生成器内部事件序列摘要一致。跨生成器不要求事件相同。原损失、采样和预算保持共同配置，激活是短程对照中的实验变量。","",
        "## 600步同预算结果","",
        "| 事件来源 | 激活 | LPIPS-Alex ↓ | clipped LPIPS-VGG ↓ | SSIM ↑ | PSNR/dB ↑ | MSE ↓ | 对比度比 | 边缘幅值比 | 平坦帧比例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for arm in ARMS:
            m=arms[method,arm]["validation"]["600"]["all200"]
            lines.append("| "+" | ".join([LABELS[method],LABELS[arm],*(fmt(m[k]) for k in ("lpips_alex","lpips_vgg","ssim","psnr_db","mse","contrast_ratio","edge_ratio")),f"{m['flat_fraction']:.2%}"])+" |")
    lines += ["","## 同步数的激活效应","","差值统一为sigmoid减线性；LPIPS/MSE正值表示sigmoid误差更高，SSIM正值表示sigmoid更高。对比度与边缘比例只作机制诊断，不能一概认为越大越好。","",
        "| 事件来源 | 更新数 | ΔLPIPS-Alex | Δclipped VGG | ΔSSIM | Δ对比度比 | Δ边缘幅值比 |","|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for step in ("300","600"):
            d=summaries[method]["activation"][step]
            lines.append("| "+" | ".join([LABELS[method],step,*(fmt(d[k],signed=True) for k in ("lpips_alex","lpips_vgg","ssim","contrast_ratio","edge_ratio"))])+" |")
    lines += ["","## 300到600步的学习变化","","差值为600步减300步，只描述本次有限训练区间。","",
        "| 事件来源 | 激活 | ΔLPIPS-Alex | Δclipped VGG | ΔSSIM | Δ对比度比 |","|---|---|---:|---:|---:|---:|"]
    for method in METHODS:
        for arm in ARMS:
            d=summaries[method]["short_learning"][arm]
            lines.append("| "+" | ".join([LABELS[method],LABELS[arm],*(fmt(d[k],signed=True) for k in ("lpips_alex","lpips_vgg","ssim","contrast_ratio"))])+" |")
    lines += ["","## 既存sigmoid连续轨迹的局部变化","",
        "以下没有新增续训：统一评估既存4800步与安全停止权重。两方法停止步数不同，只能各自在原轨迹内比较，不能作为相同增量的跨方法因果对照。","",
        "| 事件来源 | 原轨迹步数 | ΔLPIPS-Alex | Δclipped VGG | ΔSSIM | Δ对比度比 |","|---|---|---:|---:|---:|---:|"]
    for method in METHODS:
        r=summaries[method]["late_learning"];d=r["delta"]
        lines.append("| "+" | ".join([LABELS[method],f"4800→{r['to_updates']}",*(fmt(d[k],signed=True) for k in ("lpips_alex","lpips_vgg","ssim","contrast_ratio"))])+" |")
    lines += ["","## 数据支持的有限判断",""]
    for method in METHODS:
        s=summaries[method];d=s["activation"]["600"];late=s["late_learning"]["delta"]
        if d["lpips_alex"]>0 and d["lpips_vgg"]>0:
            conclusion="600步时sigmoid的两项clipped感知误差均高于线性，支持短程激活/优化交互存在质量代价的可能。"
        elif d["lpips_alex"]<0 and d["lpips_vgg"]<0:
            conclusion="600步时sigmoid的两项clipped感知误差均低于线性，本次短程结果不支持sigmoid必然降低画质。"
        else:
            conclusion="600步时两项感知指标方向不完全一致，不给出统一的激活优劣判断。"
        linear=arms[method,"original"]["validation"]["600"]["all200"]
        if linear["flat_fraction"]>.05:
            conclusion+=" 线性臂存在较多平坦输出，不能把它当稳定可用方案。"
        if late["lpips_alex"]<0 and late["lpips_vgg"]<0:
            conclusion+=" 4800步后两项感知误差仍下降，说明该局部区间还在改善；这不是训练步数不足解释全部HQF差距的证明。"
        elif late["lpips_alex"]>=0 and late["lpips_vgg"]>=0:
            conclusion+=" 相对被选中的4800步最佳点，后续有限区间未见两项感知误差改善；这受最佳点选择影响，不能据此判断平台。"
        else:
            conclusion+=" 4800步后的两项感知误差变化方向不一致，局部趋势仍需保留。"
        lines.append("- "+LABELS[method]+"："+conclusion)
    lines += ["","sigmoid在[0,1]内原则上能表示目标亮度；固定学习率、损失与激活的交互可能影响学习速度。平均导数≤0.25是sigmoid的基本性质，不单独构成饱和证据。近边界比例、对比度与边缘幅值只能与实际误差变化共同解释。","",
        "原linear历史最终权重只作为带真实步数的非等预算参考，见完整CSV；不能与sigmoid4800或600步直接相减来估计纯激活效应。raw LPIPS-VGG仅解释历史评价口径差异，主要质量比较始终使用clipped Alex/VGG。","",
        "**最佳点选择偏差：4800步是此前按合成VGG选出的最佳检查点，并非任意时间点。** 从已选的最小值出发，后续验证回升带有选择偏差；4800→停止点的局部变差不能当成平台或步数无效的证据。下面的原连续训练完整日志和按时间预定的末5点描述，为这一局部比较补充背景。","",
        "本轮只有seed0和8条合成验证轨迹，不提供p值或显著性声明。短程曲线不能证明永久容量上限，也不能单独证明HQF退步的全部成因；训练进度与激活/优化影响可以同时存在。不根据这些结果自动启动长训、选择真实检查点或调整真实后处理。","",
        "[所有时间点](all_timepoints.csv) · [逐rollout结果](per_rollout.csv) · [同预算激活差值](paired_activation_deltas.csv) · [学习变化](learning_changes.csv) · [机制指标](mechanisms.csv) · [配对检查](pairing_checks.json)","",
        "![短程学习曲线](short_learning_curves.png)","","![原sigmoid轨迹局部变化](sigmoid_late_trajectory.png)",""]
    if history_trends:
        lines += ["## 原连续训练的完整VGG日志与尾段背景","",
            "日志采用原生验证口径，与新评估器重评的指标分开列出，不静默拼成一条曲线。末段固定取最后5个按时间排序的验证点；斜率只是描述，不作显著性检验或未来收敛外推。","",
            "| 事件来源 | 完整日志点数 | 末段更新区间 | 末点−首点VGG | 描述性斜率/100步 |","|---|---:|---|---:|---:|"]
        for h in history_trends:
            lines.append("| "+" | ".join([LABELS[h["method"]],str(h["total_validation_records"]),f"{h['tail_from_updates']}→{h['tail_to_updates']}",fmt(h["last_minus_first"],signed=True),fmt(h["descriptive_ols_slope_per100updates"],signed=True)])+" |")
        lines += ["","[完整原生日志CSV](formal_validation_history.csv) · [按时间预定末段的描述CSV](formal_history_tail_trends.csv)","","![完整原生验证日志](formal_validation_history.png)",""]
    else:
        lines += ["输入没有提供足够的原连续验证日志；不据4800最佳点与单个后续点判断平台。",""]
    (out/"README.md").write_text("\n".join(lines).rstrip()+"\n",encoding="utf-8")
    make_figures(out,arms,refs,histories)
    index={str(p.relative_to(out)):dict(size_bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="artifact_index.json"}
    write_json(out/"artifact_index.json",dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),input_sha256=sha(source),export_script_sha256=sha(Path(__file__)),files=index))
    print(json.dumps(dict(status="export_complete",output=str(out),files=len(index),pairing_passed=True),ensure_ascii=False))


def make_figures(out: Path, arms: dict, refs: dict, histories: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path=next((p for p in (Path("/mnt/c/Windows/Fonts/msyh.ttc"),Path("C:/Windows/Fonts/msyh.ttc")) if p.is_file()),None)
    font=FontProperties(fname=str(font_path)) if font_path else FontProperties()
    colors={"original":"#A45A40","sigmoid":"#187F91"}
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,method in enumerate(METHODS):
        for col,(metric,title) in enumerate((("lpips_alex","clipped LPIPS-Alex ↓"),("lpips_vgg","clipped LPIPS-VGG ↓"),("contrast_ratio","预测/GT对比度比"))):
            ax=axes[row,col]
            for arm in ARMS:
                vals=arms[method,arm]["validation"];steps=sorted(map(int,vals))
                ax.plot(steps,[vals[str(s)]["all200"][metric] for s in steps],"o-",color=colors[arm],label=LABELS[arm],markersize=4)
            ax.set_title(LABELS[method]+" · "+title,fontproperties=font,fontsize=10)
            ax.set_xlabel("optimizer更新数",fontproperties=font);ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
            ax.set_ylim(bottom=0)
            if col==0:ax.legend(prop=font,frameon=False,fontsize=9)
    fig.suptitle("合成短程配对：相同初始化与训练样本\n全部200帧、8条独立rollout等权平均；仅seed 0",fontproperties=font,fontsize=13)
    for suffix in ("png","pdf"):fig.savefig(out/("short_learning_curves."+suffix),dpi=190)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6.8),constrained_layout=True)
    for row,method in enumerate(METHODS):
        a=refs[method,"sigmoid_4800"];b=refs[method,"sigmoid_stop"]
        for col,(metric,title) in enumerate((("lpips_alex","clipped LPIPS-Alex ↓"),("lpips_vgg","clipped LPIPS-VGG ↓"))):
            ax=axes[row,col];xs=[a["updates"],b["updates"]];ys=[r["validation"]["all200"][metric] for r in (a,b)]
            ax.plot(xs,ys,"o-",color=colors["sigmoid"],markersize=5)
            for x,y in zip(xs,ys):ax.annotate(f"{y:.5f}",(x,y),xytext=(0,8),textcoords="offset points",ha="center",fontsize=9)
            spread=max(abs(ys[1]-ys[0]),.005);ax.set_ylim(min(ys)-spread*.7,max(ys)+spread*.9)
            ax.set_xticks(xs);ax.set_xlabel("原连续轨迹更新数",fontproperties=font)
            ax.set_title(LABELS[method]+" · "+title,fontproperties=font,fontsize=11)
            ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("既存sigmoid权重的局部趋势：没有新增续训\n两方法停止步数不同；连接线仅辅助阅读，不外推最终收敛",fontproperties=font,fontsize=12)
    for suffix in ("png","pdf"):fig.savefig(out/("sigmoid_late_trajectory."+suffix),dpi=190)
    plt.close(fig)
    if any(histories.values()):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
        for ax,method in zip(axes,METHODS):
            h=histories[method]
            if h:
                xs=[r["optimizer_update"] for r in h];ys=[r["lpips_vgg_equal_rollout_mean"] for r in h]
                ax.plot(xs,ys,"o-",color="#536779",markersize=3,label="原连续验证日志")
                chosen=[r for r in h if r["optimizer_update"]==4800]
                if chosen:ax.scatter([4800],[chosen[0]["lpips_vgg_equal_rollout_mean"]],marker="*",s=110,color="#B56D39",zorder=4,label="此前选中的最佳点")
                ax.set_ylim(bottom=0);ax.legend(prop=font,frameon=False)
            ax.set_title(LABELS[method],fontsize=11);ax.set_xlabel("原连续轨迹更新数",fontproperties=font)
            ax.set_ylabel("原生合成验证 LPIPS-VGG ↓",fontproperties=font);ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("4800为已选VGG最优点：后续回升不能单独证明平台\n原日志仅提供历史背景，不与不同口径重评曲线混合",fontproperties=font,fontsize=12)
        for suffix in ("png","pdf"):fig.savefig(out/("formal_validation_history."+suffix),dpi=190)
        plt.close(fig)


if __name__=="__main__":
    main()
