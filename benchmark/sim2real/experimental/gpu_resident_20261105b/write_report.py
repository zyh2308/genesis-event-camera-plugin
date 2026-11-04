"""Generate final tables from machine results, including all qualification limits."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
audit=json.loads((root/'final_audit.json').read_text());assert audit['scope_passed']
analysis=json.loads((root/'paired_suite_final/analysis.json').read_text());core_v2=json.loads((root/'paired_suite_v2/analysis.json').read_text())
base=json.loads((root/'long_baseline/summary.json').read_text());long=json.loads((root/'long_resident_final/summary.json').read_text());model=json.loads((root/'model_regression_final.json').read_text())
agg=analysis['aggregate'];timings=[]
for folder in ['worker_throughput','worker_throughput_final']:
    for group in json.loads((root/folder/'results.json').read_text())['groups']:
        timings.append(dict(folder=folder,variant=group['variant'],workers=group['workers'],rate=statistics.median(x['intervals_per_s'] for x in group['measurements'])))
rate_base=next(r['rate'] for r in timings if r['folder']=='worker_throughput' and r['variant']=='baseline')
copies={}
for condition in ['baseline','resident_events','resident_full']:
    rows=json.loads((root/('profile_'+condition)/'copy_summary.json').read_text())['copies']
    copies[condition]={kind:dict(bytes=sum(r['bytes'] for name,r in rows.items() if kind in name),calls=sum(r['calls'] for name,r in rows.items() if kind in name)) for kind in ['HtoD','DtoH']}
lines=['# GPU 常驻管线实测：g1 GPU 2，2026-11-05','',
'最终接口版本在五个固定合成场景上的汇总提速为 **%.3f×**，缓存到事件的耗时降低 **%.1f%%**；相对上一轮传感器融合路径再提速 **%.3f×**。强制导出完整诊断时仍提速 **%.3f×**。'%(agg['resident_events']['weighted_speedup'],agg['resident_events']['time_reduction_fraction']*100,agg['resident_events']['speedup_vs_prior'],agg['resident_full']['weighted_speedup']),
'', '核心、数据、初始化和三线训练配置没有替换。当前结果是隔离候选的计算开销及回归证据；正式三线继续沿用原冻结实现。',
'', '**原始 float64 秒级完整事件数组的严格逐位状态：%s。** 相同输入的传感器/回退、遮挡掩码、实际 AER 写出和固定回归检查分别报告，不能将它们合称为完整浮点链路逐位确定性通过。'%('通过' if audit['strict_raw_bitwise_passed'] else '未通过，已保留'),
'', '## 数据流改变','',
'原版：GPU 几何 → CPU 中间图像与诊断 → GPU 传感器 → CPU 稀疏事件。最终版：GPU 几何/中间图像/掩码 → GPU 传感器 → CPU 最终稀疏事件。诊断仍完整计算，保留显存并在访问时导出；全导出成本另行计入。',
'', 'CPU 端点预处理、自适应采样和少量标量控制仍保留。真正缺失投影的像素继续使用原 NumPy 回退并上传稀疏值，以保持参考数值；纯 GPU 回退的 HDR 最大差为约 4.77e-6，未进入推荐路径。没有减少采样、精度、噪声或不可靠区域检查。',
'', '## 计时口径','',
'- g1 GPU 2，Tesla V100S 32GB；固定 UUID `GPU-a02affe0-7598-579e-ca39-f551f35ad6e9`，现有 Torch 2.8 环境与上一轮 CUDA 扩展。',
'- 冻结核心 `'+audit['frozen_release']+'`，17 文件 canonical-LF SHA256 复核通过。Stage 9 完整传感器配置不变，FP64 几何输入与状态、每 10ms 的 377 个采样点、RNG 顺序、不应期及遮挡规则保持。',
'- 四类 rich bank + 一类 pose bank，固定训练物理缓存 endpoint 100 / 10，未用真实数据。每方案每场景三次，交替顺序，每次前有预热并同步 CUDA。',
'- 两轮核心实现各 60 次，最终公开接口再做 60 次，共 180 次单窗口测量。最终下表来自最后 60 次，不混用初探最优值。',
'- 包含 public capture、CPU 端点预处理、几何、传感器与最终事件排序。缓存读取、Genesis 渲染、初始化、保存报告与计时后的核验不计入；因此不是整个 Genesis 仿真实时率。',
'', '## 最终五场景结果','',
'| 场景 | 原版 / s | 上一轮融合 / s | 常驻显存、事件输出 / s | 提速 | 常驻显存、完整诊断 / s |',
'|---|---:|---:|---:|---:|---:|']
for row in analysis['cases']:
    s=row['stats'];lines.append('| %s | %.4f | %.4f | %.4f | %.3f× | %.4f |'%(row['case'],s['baseline']['median_s'],s['prior_fused']['median_s'],s['resident_events']['median_s'],s['resident_events']['speedup_vs_baseline'],s['resident_full']['median_s']))
lines+=['', '汇总按五个场景中位数之和计算。第二版核心路径独立测得 %.3f×，最终接口封装复测如上。原始每次、最小/最大、分项、显存和复制计数见 [最终 CSV](../../evidence/20261105-gpu-resident/paired_suite_final/timings.csv)。'%core_v2['aggregate']['resident_events']['weighted_speedup'],
'', '常驻显存会把部分诊断占用从 CPU 移到 GPU，峰值显存未下降。单窗口各场景的原版/候选峰值范围为 %.2f–%.2f / %.2f–%.2f GiB。不能把速度提升解释为显存降低。'%(min(r['stats']['baseline']['peak_bytes'] for r in analysis['cases'])/2**30,max(r['stats']['baseline']['peak_bytes'] for r in analysis['cases'])/2**30,min(r['stats']['resident_events']['peak_bytes'] for r in analysis['cases'])/2**30,max(r['stats']['resident_events']['peak_bytes'] for r in analysis['cases'])/2**30),
'', '## 硬件复制追踪','',
'CUPTI 追踪固定 s2_00 窗口；追踪开销不进入上面的计时表。下行同时包含 pageable 数组和 pinned 标量，不把 Python 的 `.cpu()` 次数冒充全部硬件复制次数。','',
'| 模式 | CPU→GPU / bytes | GPU→CPU / bytes | 下行复制次数 |',
'|---|---:|---:|---:|']
for condition,value in copies.items():lines.append('| %s | %d | %d | %d |'%(condition,value['HtoD']['bytes'],value['DtoH']['bytes'],value['DtoH']['calls']))
lines+=['', '在仅输出事件的路径中，上行约 275 MB → 3.6 MB，下行约 2.14 GB → 52 KB。仍有小标量同步和最终输出，不是零 CPU 往返。完整诊断请求会重新产生相应大数组导出；这部分没有被删除或隐藏。追踪对应第二版核心路径，最终接口封装复用同一计算/复制实现。',
'', '## 连续一秒与并行部署','',
'连续 100 个 10ms 区间：原版 %.3f 秒，最终版 %.3f 秒，单条轨迹约 %.3f×。各自产生 %d 个事件。此项各运行一次，只作连续状态/吞吐回归，不代替重复场景统计。'%(base['median_wall_s'],long['median_wall_s'],base['median_wall_s']/long['median_wall_s'],audit['long_comparisons']['long_resident_final']['events']),
'', '| 实现 | worker 数 | 固定窗口总吞吐 / 区间·s⁻¹ | 相对原版双 worker |', '|---|---:|---:|---:|']
for row in timings:
    label=('最终常驻显存' if row['folder'].endswith('_final') else row['variant']+'（核心阶段）')
    lines.append('| %s | %d | %.4f | %.3f× |'%(label,row['workers'],row['rate'],row['rate']/rate_base))
lines+=['', '同一卡的该场景上，两个常驻 worker 没有明显超过单 worker；默认单 worker 可节省 CPU 和显存。每 worker 两个 Torch/OMP 线程，OpenBLAS 一线程；测试为独立进程、屏障同步、各三次，不能外推所有场景的并行最优配置。',
'', '## 正确性与明确边界','',
'- 两版完整噪声传感器边界测试、强噪声、多事件、不应期、零斜率和容量异常回退、非法输入拒绝、RNG 恢复检查通过。',
'- 独立新进程恢复后，与原传感器的事件及状态逐位相同。五类实际输入各连续两区间，原版读取候选的同一组 GPU 帧，事件、状态和标量计数逐位一致；最终 API 封装再次通过。',
'- 受支持、条件不可靠但有限、部分/全部空洞、HDR 空洞及严格拒绝样例中，推荐回退路径的亮度、请求字段与掩码均与原版逐位一致。纯 GPU 数学分支的末位误差明确保留为拒绝理由。',
'- 诊断数组访问、延迟导出、不同掩码互不污染、张量返回属性和默认 NumPy 插值接口均检查通过。GPU 通道独立命名，未将原默认接口改成隐式 CUDA 返回。',
'- 最终 60 次配对窗口的原始事件、可见/未知等布尔掩码与采样参数一致。少量完整几何浮点状态存在冻结 scatter 后端自身的重复波动。']
candidate=audit['long_comparisons']['long_resident_final'];repeat=audit['long_comparisons']['long_baseline_repeat']
lines+=['- 一秒完整原始事件：位置、极性、顺序、数量与 RNG 一致；%d 个浮点时间戳变化，最大 %.3g s（%.3f ns）。原版自身重复有 %d 个变化，最大 %.3g s（%.3f ns）。这支持与冻结几何重复波动的归因；严格原始数组状态仍单独保留。'%(candidate['changed_timestamps'],candidate['max_time_error_s'],candidate['max_time_error_s']*1e9,repeat['changed_timestamps'],repeat['max_time_error_s'],repeat['max_time_error_s']*1e9),
'- 实际调用原冻结 AER recorder 后，两个 uint64 微秒事件数据集逐位一致。这里只证明该写出接口；不将其替代 float64 秒级消费者的独立契约。',
'- 最终固定计数体素/定位回归 %d 个窗口，体素逐位相同，最大定位变化 0 px。没有做训练、GT 评分、真实测试或宣称所有下游表示均逐位相同。'%model['queries'],
'', '[完整范围验收](../../evidence/20261105-gpu-resident/final_audit.json) 与 [严格原始数组门禁](../../evidence/20261105-gpu-resident/strict_raw_gate.json) 分开保存，未以放宽阈值或改写失败状态来代替原门禁。',
'', '## 交付和下一步范围','',
'代码：[最终 GPU 入口](../../experimental/gpu_resident_20261105b/resident_final.py)、[核心常驻实现](../../experimental/gpu_resident_20261105b/resident2.py)、[复现说明](../../experimental/gpu_resident_20261105b/README.md)。原始复制追踪、NPZ、HDF5 和恢复检查点留在服务器，路径/大小/SHA256 见 [外部产物](../../evidence/20261105-gpu-resident/external_artifacts.json)。',
'', '当前仅验证完整 Stage 9 配置、无 margin/额外状态 provider 的缓存回放；还不是所有传感器预设或多线程调用的通用发布版。保持冻结核心不动。主要剩余成本是几何投影/覆盖合成、端点采样和少量同步。端点缓存从磁盘/Genesis 进入 GPU，以及最终文件保存，仍是边界工作；未声称整个 Genesis 渲染器零拷贝或实时运行。',
'', '这次加速影响后续合成事件生成。已有三线训练读取离线事件，不会因此直接改变 optimizer 速度或实验输入。未启动、恢复或修改正式训练。',
'', '服务器原始目录：`'+str(root)+'`。']
(root/'GPU_RESIDENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('GPU_RESIDENT_REPORT_WRITTEN',flush=True)
