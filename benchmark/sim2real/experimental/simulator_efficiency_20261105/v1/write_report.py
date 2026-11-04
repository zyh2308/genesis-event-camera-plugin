"""Generate human-readable tables directly from retained machine evidence."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent;v2=root/'final_v2'
analysis=json.loads((v2/'analysis.json').read_text());v1=json.loads((root/'analysis.json').read_text());audit=json.loads((root/'final_audit.json').read_text())
attribution=json.loads((root/'attribution_audit.json').read_text())
assert attribution['passed'],attribution['checks']
worker=json.loads((v2/'worker_throughput/results.json').read_text())['groups']
rates={(g['variant'],g['workers']):statistics.median(m['intervals_per_s'] for m in g['measurements']) for g in worker}
lines=['# EventGenesis 计算优化实测：g1 GPU 2，2026-11-05','',
'第一轮实现与验证完成。在保持当前质量配置的缓存回放测试中，速度优先版汇总提速 **%.3f 倍**，耗时降低 **%.1f%%**；五个固定场景各自提速约 %.2f–%.2f 倍。省显存版的代表场景峰值由约 10.6 GiB 降至 4.6 GiB。'%(analysis['aggregate']['fused_early']['weighted_speedup'],100*analysis['aggregate']['fused_early']['time_reduction_fraction'],min(r['stats']['fused_early']['speedup'] for r in analysis['cases']),max(r['stats']['fused_early']['speedup'] for r in analysis['cases'])),
'', '**完整长序列的严格逐位门禁未通过，失败状态保留。** 追加的相同中间帧验证、原版自身重复和下游输入检查已完成，支持传感器算术与本轮下游输入等价；不能称整链路逐位确定性通过。具体差异见下文。',
'', '这些是独立候选实现的实测，不是新冻结版本。现有三线训练、训练数据、检查点及真实测试协议未接入候选。17 个冻结核心文件复核通过。','',
'## 测量范围与可复现口径','',
'- g1 GPU 2：Tesla V100S-PCIE-32GB，GPU UUID `GPU-a02affe0-7598-579e-ca39-f551f35ad6e9`。Torch 2.8，CUDA 扩展使用已存在的私有 CUDA 12.6/GCC 12 工具链，目标 sm70，禁用 fast-math 与 FMA 合并；未修改共享环境。',
'- 冻结 release：`'+audit['frozen_core_release']+'`。使用 Stage 9 的完整实际配置；FP64 几何输入/状态、阈值不一致、300 Hz 低通、泄漏、散粒噪声、不应期及 unknown/support 规则保持原值。不是关闭噪声后的理想模型。',
'- 四个 rich bank 场景固定 endpoint 100，加一个 pose bank 场景固定 endpoint 10。每个区间 10 ms，每区间实际仍为 377 个传感器时间点。输入来自合成训练物理缓存；没有真实数据或标签驱动调优。',
'- 计时包含 public capture、中间几何、传感器、CPU 事件输出以及共同的输入哈希核验开销。缓存读取、初始 attach、Genesis 渲染和保存测量产物不计入；不能把本表当作完整 Genesis 仿真实时率。',
'- v1：两方案、五场景、各五次，50 次测量；v2：三方案、五场景、各三次，45 次测量。每次测量前另有一次预热，CUDA 同步，执行顺序交替。所有原始数值保留，不用最优单次代替中位数。',
'- 另外完成单/多 worker 吞吐测试、连续 20 区间回归、独立进程恢复、传感器边界样例和固定模型回归。结果不外推为其他 GPU 或任意场景上的倍率。','',
'## 完整 capture 的计时结果','',
'下表为最终 v2 每方案三次测量中位数。省显存版也运行完整 377 个时间点。','',
'| 固定场景 | 原版 / 秒 | 速度版 / 秒 | 速度倍率 | 省显存版 / 秒 | 原版 → 省显存 / GiB |',
'|---|---:|---:|---:|---:|---:|']
for row in analysis['cases']:
    s=row['stats'];b=s['baseline'];f=s['fused_early'];m=s['fused_chunk']
    lines.append('| %s | %.4f | %.4f | %.3f× | %.4f | %.2f → %.2f |'%(row['case'],b['median_wall_s'],f['median_wall_s'],f['speedup'],m['median_wall_s'],b['peak_allocated_bytes']/2**30,m['peak_allocated_bytes']/2**30))
lines+=['',
'汇总速度倍率按五场景原版中位数之和除以候选中位数之和计算；几何平均倍率为 %.3f×。v1 独立首轮重复所得汇总倍率为 %.3f×，与 v2 一致。速度版的显存没有下降，单区间峰值略增约 0.13 GiB；连续 20 区间的原版/速度版峰值为 %.2f / %.2f GiB。省显存版以分块换取较低峰值，汇总倍率为 %.3f×。'%(analysis['aggregate']['fused_early']['geometric_mean_speedup'],v1['aggregate']['weighted_speedup'],audit['long_trace']['long_baseline']['peak_allocated_bytes']/2**30,audit['long_trace']['long_fused']['peak_allocated_bytes']/2**30,analysis['aggregate']['fused_chunk']['weighted_speedup']),
'', '原始每次计时、最小/最大值、DVS/几何分项和显存见 [CSV](../../evidence/20261105-simulator-efficiency/final_v2/timings.csv) 与 [完整分析](../../evidence/20261105-simulator-efficiency/final_v2/analysis.json)。这是固定窗口的重复计时，没有把窗口当作数据集总体的独立统计样本。','',
'## 传感器部分与实现内容','',
'| 固定场景 | 原版 DVS / 秒 | 融合 DVS / 秒 | DVS 倍率 |',
'|---|---:|---:|---:|']
for row in analysis['cases']:
    b=row['stats']['baseline']['median_dvs_s'];f=row['stats']['fused_early']['median_dvs_s']
    lines.append('| %s | %.4f | %.4f | %.2f× |'%(row['case'],b,f,b/f))
lines+=['',
'主要实现是像素并行、时间顺序递推的 CUDA 传感器内核。Torch RNG 的调用顺序、形状、dtype 不变；预生成噪声后，CUDA 连续完成低通状态、阈值穿越、不应期、噪声事件和状态更新。CPU 保持原始事件追加与排序语义。几何端只做端点 log 缓存、批量输出搬运、已无候选时结束空迭代，或调用原函数按时间分块。',
'', '特殊零斜率与缓冲区溢出回退到完整原版，并恢复 RNG 后再执行，不丢事件。v2 的缓冲区按当前区间长度/不应期推导上界，避免安静像素的旧时间戳导致分配随 episode 年龄增长；含旧时间戳与新进程恢复的专门回归。',
'', '## 同一卡的部署吞吐量','',
'固定 s2_00，三个测量区间，使用进程间屏障同时开始。每个 worker 限两个 Torch/OMP 线程，NumPy BLAS 一线程；增加 worker 同时增加 CPU 配额，不能称为等 CPU 成本的算法倍率。','',
'| 方案 | worker 数 | 总吞吐量 / 区间·秒⁻¹ | 相对原版双 worker |',
'|---|---:|---:|---:|']
base_rate=rates[('baseline',2)]
for g in worker:
    value=rates[(g['variant'],g['workers'])]
    lines.append('| %s | %d | %.4f | %.3f× |'%(g['variant'],g['workers'],value,value/base_rate))
lines+=['',
'当前这张卡的实测首选是 **两个速度优先 worker**：相对原版双 worker 吞吐提高 %.1f%%。四个省显存 worker 能运行，但总吞吐低于两个速度优先 worker，不能将“可多开”直接当作加速。原始每个 worker 的事件都另行核验。'%(100*(rates[('fused_early',2)]/base_rate-1)),
'', '## 正确性与连续运行','',
'- 同一组中间帧送入原版和 CUDA 候选：五类实际合成输入，各连续两个区间、754 帧；原始事件、全部张量状态和 RNG 逐位一致。两轮实现都通过。',
'- 静止、上/下降亮度、多事件、不应期、强噪声、零斜率回退、溢出回退、状态序列化、不支持功能拒绝、NaN 输入拒绝及 RNG 不变检查均通过。',
'- 独立 OS 进程读取保存状态后续接：25,409 个事件与参考逐位一致，全部保存状态亦一致。这是仿真器状态恢复检查。',
'- 控制投影下的恒等、亚像素移动、缺失区域、遮挡表面、多深度层几何用例均通过，所有记录浮点差为零，mask/surface ID 相同。',
'- v2 所有配对计时窗口中，两个候选的原始事件逐位一致，unknown/support 等布尔掩码与采样参数一致。完整几何重复仍存在 FP32 scatter 原有的微小状态变化；候选与原版最大状态差分别为 %.3g / %.3g，不能把完整几何状态称为逐位确定性。'%(analysis['aggregate']['fused_early']['max_state_difference'],analysis['aggregate']['fused_chunk']['max_state_difference']),
'- 固定定位网络：v1 50 次、v2 30 次查询，输入和位置预测未变化，最大位置差 0 px。只检查回归，不是 E-POSE/机器人准确率或 Sim-to-Real 结论。']
long=audit['long_trace']
lines+=['- 连续 20 个区间、200 ms：原版与速度版各产生 %d 个事件，位置、极性、顺序和 RNG 一致；%d 个时间戳有差异，最大 %.3g 秒（%.3f 纳秒）。单条轨迹耗时 %.3f / %.3f 秒，该差异导致原始严格门禁失败。'%(long['events_baseline'],attribution['candidate']['changed_timestamps'],attribution['candidate']['max_time_error_s'],attribution['candidate']['max_time_error_s']*1e9,long['long_baseline']['wall_s'],long['long_fused']['wall_s']),
'- 原版自身重复同一长序列：%d 个时间戳变化，最大 %.3g 秒；也存在非零状态变化。冻结几何本身有重复波动。'%(attribution['reference_repeat']['changed_timestamps'],attribution['reference_repeat']['max_time_error_s']),
'- 同一组中间帧按区间分别喂给原版/候选、连续 20 个区间：事件与状态严格逐位一致。完整长序列另检查 %d 个固定 25ms 体素窗口，候选体素与固定定位预测均未变化。'%len(attribution['queries']),
'', '原始失败证据见 [严格门禁](../../evidence/20261105-simulator-efficiency/final_audit.json)，追加归因见 [分层核验](../../evidence/20261105-simulator-efficiency/attribution_audit.json)。最初严格门禁与依赖它的初版报告退出码为 1，均未删除或改写；后续核验回答传感器与下游输入等价问题，不把原门禁重标为通过。','',
'## 没有收益或存在代价的方案','',
'- 仅批量预处理/上传的初探：2.994 → 3.046 秒，未获得提速；仍保留日志。',
'- 单独端点缓存/批量搬运的初探：约 2.945 秒，收益小，不能作为主要加速结论；初探每方案仅两次。',
'- 时间分块显存下降明显，但单 worker 比速度优先版本慢。四 worker 的实际总吞吐也未超过两个速度版 worker。',
'- 没有使用降低分辨率、减少 377 个采样点、关闭噪声、改成 FP32 几何或丢弃事件来获得上述结果。','',
'## 仍然昂贵的部分与后续边界','',
'代表窗口的 CUDA event span 中，两个端点投影约 79 + 80 ms，双向覆盖合成约 663 ms。CPU profile 中 `Tensor.cpu()` 累计阻塞约 1.0 秒，包含等待 GPU，不能与 CUDA span 直接相加，也不能全部叫作 PCIe 传输。几何层、诊断数据搬运与 CPU/GPU 同步是下一轮主要目标。',
'', '保持当前设置，完整区间耗时仍约 1.9–2.4 秒/10 ms，尚不能实时运行。优先考虑中间张量常驻 GPU、减少同步/重复分配和更专用的覆盖合成内核。多速率几何与传感器步进分离可能获得更大收益，但本轮未实施，必须另行检查遮挡、像素定位、事件时序和新的冻结版本，不能用未测倍率作承诺。',
'', '此次优化加速后续合成事件生成。现有三线模型读取已经生成的离线数据，optimizer 每步耗时不会因此直接改变。尚未运行整套物理 bank 重生成、跨设备基准或正式模型重训；本轮也未改变任何生成器间的实验协议。','',
'## 代码与证据位置','',
'- [候选实现与复现说明](../../experimental/simulator_efficiency_20261105/README.md)',
'- [最终 v2 源码](../../experimental/simulator_efficiency_20261105/v2/candidates.py)',
'- [首轮 50 次计时](../../evidence/20261105-simulator-efficiency/paired_suite/progress.json) 与 [最终 45 次计时](../../evidence/20261105-simulator-efficiency/final_v2/paired_suite/progress.json)',
'- [多 worker 实测](../../evidence/20261105-simulator-efficiency/final_v2/worker_throughput/results.json)',
'- [外部大产物与 SHA256](../../evidence/20261105-simulator-efficiency/external_artifacts.json)',
'- 服务器原始工作目录：`'+str(root)+'`。大 NPZ、仿真器恢复检查点、编译二进制、profiler 原始文件保留服务器；轻量报告、代码、运行收据和哈希索引入仓。','']
(root/'BENCHMARK_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('REPORT_WRITTEN',root/'BENCHMARK_REPORT.md',flush=True)
