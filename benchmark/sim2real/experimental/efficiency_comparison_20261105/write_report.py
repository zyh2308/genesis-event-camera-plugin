"""Generate human conclusions from the final raw measurements, with all limits."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
analysis=json.loads((ROOT/'matched_suite_static_analysis.json').read_text())
release=json.loads((ROOT/'FROZEN_RELEASE.json').read_text())
controls=ROOT/'controls'
auto=json.loads((controls/'auto_sampling.json').read_text())['rows']
fine=json.loads((controls/'factor400.json').read_text())['rows']
long=json.loads((controls/'batch16_long.json').read_text())['rows']
sensor=json.loads((controls/'sensor_only.json').read_text())['rows']
cli=json.loads((controls/'stock_cli_receipt.json').read_text())
native=json.loads((ROOT/'genesis_smoke_accepted.json').read_text())
long_regression=json.loads((ROOT/'static_long_observables.json').read_text())
precision_review=json.loads((ROOT/'PRECISION_REVIEW.json').read_text())
ci=analysis['scene_bootstrap_95ci'];gm=analysis['geometric_mean_speedup']
lines=['# EventGenesis 与 v2e 效率对照及限定配置冻结：2026-11-05','',
f"最终静态发布包在五个固定场景、相同中间采样密度下，相对 v2e + SuperSloMo 的时间提速几何平均为 **{gm:.3f} 倍**，场景重采样 95% 区间为 **[{ci[0]:.3f}, {ci[1]:.3f}]**。达到预先登记的 1.20 倍实用优势门槛；完成实际入口验收后冻结为 `{release['release_id']}`。本轮不再增加算法优化。",'',
'**这个优势只适用于本报告的完整传感器配置、固定高时间密度和硬件。v2e 自动采样另行列出，不宣称对所有速度/质量设置全面领先。显存占用仍显著高于 v2e。**','',
'**完整原始float64逐位门禁与本轮3ns诊断线均未通过，失败状态保留。冻结限于既定数值精度/噪声/遮挡语义与已验证微秒写出、计数输入协议，不发布为严格逐位确定性的通用版本。**','',
'## 比较设置','',
'- 同一 g1 GPU 2：V100S 32GB，独占预约；两 Torch/OMP 线程、一 OpenBLAS 线程。未改变正式训练、共享环境、物理缓存或原冻结核心。',
'- 四类 rich 场景各一条 + pose 一条，每条连续 200ms，100Hz 相同物理端点；rich 索引100，pose索引0。输入哈希、相机内参与分割ID哈希随每场景保存。',
'- 保留 Stage9 的阈值、300Hz带宽、阈值失配、泄漏、shot noise、不应期等完整配置。EventGenesis 使用原线性HDR/深度/分割/位姿；v2e 使用从同一物理帧导出的既有灰度输入及官方 SuperSloMo。没有给下游模型增加特权输入。',
'- 原始图像/几何精度与 FP64 传感器状态不变；v2e 保留其原生 FP32 实现，不擅自改成同一个传感器模型。不同生成器输出事件数量和数值不要求相同。',
'- 每10ms固定377子步，约26.525us；v2e batch8。三次重复、轮换顺序、CUDA同步、模型预热。另以400子步和可用batch16作控制。',
'- v2e 保留官方网络、PIL转换和resize，只将无损PNG传输换成等值内存数组，真实PNG解码及对应事件逐位核对通过。使用官方 API 的实际中点时间戳；原CLI额外的时间重标定另行审计。',
'- 主表计时：已有CPU缓存到排序后的CPU事件；不含Genesis渲染、输入文件加载、模型/进程启动和输出保存。双方使用同一事件文件写入器单独计时。RTF定义为模拟秒/墙钟秒，1表示实时。',
'- 主表只来自静态实际入口的最后30次测量。此前30次原型计时全部保留，不混用最优值。v2e每次cleanup并注销atexit回调，避免跨试验保留模型/状态。EventGenesis显存排除测量进程中无关的v2e模型；worker总峰值也保留。','',
'## 最终结果','',
'| 场景 | EventGenesis / 秒 | v2e / 秒 | 提速 | EventGenesis峰值 / GiB | v2e峰值 / GiB |',
'|---|---:|---:|---:|---:|---:|']
for x in analysis['cases']:lines.append(f"| {x['case']} | {x['ours_s']:.3f} | {x['v2e_s']:.3f} | {x['speedup']:.3f}× | {x['ours_peak_gib']:.3f} | {x['v2e_peak_gib']:.3f} |")
lines+=['',
'均为每条200ms模拟序列的三次中位数。显存为Torch `max_memory_allocated`，不含CUDA上下文/缓存；EventGenesis列剔除同一测量进程中无关的v2e模型，worker总峰值也保存。范围/事件数/RTF/逐次耗时/保存耗时见 CSV 与 JSON。区间只反映五个固定场景在本设备上的差异，不等于全部场景总体、训练效果或物理真实性的置信区间。双方均未达到实时。',
'', '## 不同运行点与额外控制','',
f"- v2e 自动采样：中位数 **{statistics.median(x['wall_s'] for x in auto):.3f}s/200ms**，实际最大帧间隔约 **{statistics.median(x['max_dt_s'] for x in auto)*1e6:.1f}us**，生成中间帧 **{[x['input_frames'] for x in auto]}**。这组保留相同噪声参数，但时间采样不同，不能直接用于同精度速度排名；也不能仅凭速度认为精度/噪声动力学完全等价。",
f"- v2e原25us设置、400子步：中位数 **{statistics.median(x['wall_s'] for x in fine):.3f}s/200ms**。",
f"- 400ms更长序列、v2e batch16：EventGenesis / v2e中位数分别为 **{statistics.median(x['wall_s'] for x in long if x['variant']=='ours_resident'):.3f} / {statistics.median(x['wall_s'] for x in long if x['variant']=='v2e_ssm_memory'):.3f}s**，时间优势仍为 **{release['batch16_long_speedup']:.3f}倍**。",
f"- 同一组377个亮度帧，仅传感器：EventGenesis / v2e中位数分别为 **{statistics.median(x['wall_s'] for x in sensor if x['variant']=='ours'):.4f} / {statistics.median(x['wall_s'] for x in sensor if x['variant']=='v2e'):.4f}s**。这组不是相同公式/精度的逐位比较，保留两者原生实现。",
f"- stock CLI冷启动、文件路径：**{cli['wall_s']:.3f}s**，退出码{cli['exit_code']}；包括启动/权重加载/临时图像/原生写出，不与暖启动主表直接相除。原始时间范围/数据集接口见收据。",'',
'官方代码/说明：[v2e](https://github.com/SensorsINI/v2e/tree/cac638444557a224ba6ee5c29260916f152c00bd)。关于分开报告插帧/事件模块的学术参照：[ICRA 2023 Real-time event simulation](https://cogsys-tuebingen.github.io/realtime_event_simulator/)。本轮没有采用其简化噪声或降低采样来获取加速。','',
'## 全流程检查与冻结边界','',
'1. 输入端：沿用原生HDR、米制深度、分割ID、CV相机位姿及秒级时间戳；完整原包的utils/evaluation依赖一起归档，17个冻结文件canonical-LF哈希不变。',
'2. 几何端：GPU常驻图像/掩码/诊断，保留覆盖、unknown和严格拒绝语义；HDR空洞的NumPy回退保持。生产入口不再使用运行时exec/inspect或替换进程全局几何函数。',
'3. 传感器端：FP64状态机、原RNG顺序、完整噪声/不应期、多事件、异常回退与新进程恢复检查通过。逐样本检查明确断言快路径启用、每例两个真实调用，防止以参考回退空跑通过。',
'4. 输出端：默认NumPy事件/图像接口、延迟诊断、原AER recorder、事件数量/边界/排序检查通过。最终30组保存原生浮点事件，冻结前重新验证大小/哈希及时间范围。',
f"   最终实际源码连续1秒复检：**{long_regression['events']}**个事件，地址/极性/顺序与RNG一致，最大原始时间差 **{long_regression['max_time_error_s']*1e9:.4f}ns**；实际微秒recorder逐位一致。固定计数体素/定位检查 **{long_regression['fixed_model_queries']}**个窗口，最大定位变化0px；不外推到全部下游表示。",
f"5. 原生Genesis：实际HDR adapter → 静态GPU入口 → 原recorder检查通过，产生 **{native['event_count']}** 个事件，微秒事件写出逐位一致。它是接入冒烟检查，不能代替五场景Genesis渲染效率基准。",
'6. **完整原始浮点链路的严格逐位门禁仍未通过。** 保留上轮同一冻结scatter后端的纳秒级时间波动及原失败证据；同帧传感器逐位一致、微秒写出一致，不替代完整float64消费者的契约。没有降低dtype、改噪声或放宽原门禁。',
f"   独立补跑未改动的参考版本后，参考之间最大时间差 **{precision_review['reference_repeat_max_s']*1e9:.4f}ns**，原版自身也超过本轮3ns诊断线。静态版相对新参考的最大差为 **{precision_review['static_vs_fresh_reference_max_s']*1e9:.4f}ns**；四份实际AER写出均逐位一致。完整比较保留，不把一次参考重复的最大值当成通用误差上界，也不将失败诊断改记通过。",
'7. 加速范围仅为归档的Stage9配置及分辨率，其他设置明确回到原参考路径。并非所有preset/相机margin/额外状态provider/多环境都已获得同等加速验收。',
'', '首次打包缺少utils依赖、配置整数/浮点比较误拒绝快路径的失败记录全部保留。它们由独立入口门禁拦截，未用于最终计时或冻结结论；修复后针对实际启用的快路径重新核验。', '',
'## 决策与使用','',
'本轮同采样精度下已达到实用时间优势，因此停止追加算法优化，冻结这个限定配置的精确GPU版本。它适合下一轮离线数据生产；不据此重生成或替换正在训练的任何数据。若后续目标转为低显存、其他preset或实时在线仿真，应作为新的受控优化任务，不能把本表推广过去。',
'',f"包内容SHA256：`{release['package_sha256']}`。原始服务器目录：`{ROOT}`。",'',
'[逐次计时CSV](../../evidence/20261105-efficiency-comparison/matched_suite_static_timings.csv) · [统计JSON](../../evidence/20261105-efficiency-comparison/matched_suite_static_analysis.json) · [限定配置冻结描述](../../../../frozen_baselines/eventgenesis-gpu-stage9-20261105/BASELINE.json) · [版本使用说明](../../../../frozen_versions/eventgenesis-gpu-stage9-20261105/README.md) · [复现步骤](../../experimental/efficiency_comparison_20261105/REPRODUCE.md)', '']
(ROOT/'EFFICIENCY_COMPARISON_REPORT.md').write_text('\n'.join(lines))
print('REPORT_WRITTEN',len(lines))
