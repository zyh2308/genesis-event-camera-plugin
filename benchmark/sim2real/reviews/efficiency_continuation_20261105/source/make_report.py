"""Build the first-delivery report from completed g1 receipts only."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    r=args.run.resolve();z=r/'zbuf_reuse'
    measured=read(z/'benchmark_deterministic_20/summary.json')
    if measured['status']!='complete':raise RuntimeError('Metrology incomplete')
    if measured['mode']!='deterministic' or measured['intervals']!=20 or measured['trials_per_package_case']!=3 or len(measured['cases'])!=1 or len(measured['rows'])!=6:
        raise RuntimeError('Unexpected metrology mode, duration or repetitions')
    source=read(z/'candidate_release/SOURCE_MANIFEST.json')
    selected_sha=source['delta_sha256']
    row=measured['cases'][0]
    if row['case']!='s2_00':raise RuntimeError('Unexpected representative case')
    expected_reference='e74793c4805eabfa2a19d9db02b5d829c73a12a61d3fed6b4cbeb8e5995a060b'
    for item in measured['rows']:
        expected=selected_sha if item['label']=='candidate' else expected_reference
        if item['core_source_sha256']['genesis_event_plugin/gpu_geometry.py']!=expected or item['case']!='s2_00' or not item['deterministic_algorithms']:
            raise RuntimeError('Measured source/config does not match selected candidate')
    required=['s0_00','s1_00','s2_00','s3_00','train_000']
    gates=[]
    for case in required:
        path=z/'comparisons'/('p0_s2_00.json' if case=='s2_00' else 'shorts_'+case+'.json')
        result=read(path)
        if not result['all_physical_observables_bitwise_equal']:raise RuntimeError('Exact short gate failed')
        trace=read(z/'short_gates/candidate'/case/'trace.json')
        if trace['case']!=case or trace['intervals']!=2 or trace['core_source_sha256']['genesis_event_plugin/gpu_geometry.py']!=selected_sha:
            raise RuntimeError('Short gate source/fixture mismatch')
        gates.append({'case':case,'events':result['raw_events']['shape_left'][0],'all_physical_observables_bitwise_equal':True})
    long=read(z/'comparisons/loader_s3_00.json')
    if not long['all_physical_observables_bitwise_equal']:raise RuntimeError('200ms continuous gate failed')
    long_trace=read(z/'loader_inputs/candidate/s3_00/trace.json')
    if long_trace['case']!='s3_00' or long_trace['intervals']!=20 or long_trace['core_source_sha256']['genesis_event_plugin/gpu_geometry.py']!=selected_sha:
        raise RuntimeError('Continuous gate source/fixture mismatch')
    loader=read(z/'formal_loader_gate/formal_loader_results.json')
    if not loader['passed']:raise RuntimeError('Actual loader gate failed')
    formal_cases=read(z/'formal_cases.json')
    if set(formal_cases['required_tasks'])!={'reconstruction','flow','pose'} or formal_cases['cases'][0]['candidate']!=str(z/'formal_events/candidate.npy'):
        raise RuntimeError('Actual loader manifest does not bind this candidate')
    stress=read(z/'coverage_check.json')
    if not stress['all_passed'] or stress['candidate_sha256']!=selected_sha:raise RuntimeError('Stress gate failed or source mismatch')
    profile=read(z/'profile/profile.json')
    if profile['status']!='complete' or profile['core_source_sha256']['genesis_event_plugin/gpu_geometry.py']!=selected_sha:raise RuntimeError('Profile incomplete or source mismatch')
    if source['actual_changed_files']!=['genesis_event_plugin/gpu_geometry.py']:raise RuntimeError('Unexpected source delta')
    pilot={label:[read(z/'pilot'/(label+'_'+str(i))/'trial0/result.json') for i in range(3)] for label in ('reference','candidate')}
    normal={label:statistics.median(x['initialized_pipeline_s'] for x in values) for label,values in pilot.items()}
    base=read(r/'profiles/reference/profile.json')
    counters=[]
    for key in ('aten::nonzero','aten::index_select','aten::scatter_reduce_','cudaStreamSynchronize'):
        counters.append({'operator':key,'reference':sum(x['count'] for x in base['operators'] if x['key']==key),'candidate':sum(x['count'] for x in profile['operators'] if x['key']==key)})
    qualification={'status':'first_delivery_complete_not_production_frozen','candidate':'first_layer_zbuf_reuse','source_sha256':source['delta_sha256'],'candidate_manifest':source,'stress_cases':stress['case_count'],'five_class_short_exact':gates,'continuous_s3_200ms_events':long['raw_events']['shape_left'][0],'continuous_events_state_rng_intermediates_exact':True,'actual_three_loader_passed':True,'actual_loader_report':str(z/'formal_loader_gate/formal_loader_results.json'),'deterministic_configuration':{'torch.use_deterministic_algorithms':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8'},'default_cuda_strict_qualification':{'status':'failed','inherited_failure_retained':True,'new_complete_default_mode_qualification_run':False,'time_quantization_or_tolerance_override':False},'not_rerun_for_this_geometry_only_delta':['100-interval long fixture','independent-process sensor resume','sensor exceptional/overflow fallback','formal training and real scoring'],'unchanged_sensor_qualification_not_automatically_promoted_to_new_production_release':True,'deterministic_metrology':measured,'normal_20ms_pilot_medians_s':normal,'profile_operator_counts':counters,'rejected_candidate':read(r/'projection_storage_rejected.json'),'production_data_modified':False,'shared_environment_modified':False}
    (r/'FIRST_DELIVERY.json').write_text(json.dumps(qualification,indent=2)+'\n')
    profile_sha=measured['rows'][0]['profile_sha256']
    package=r/'zbuf_reuse_candidate.tar.gz'
    with tarfile.open(package,'w:gz') as archive:archive.add(z/'candidate_release',arcname='candidate_release')
    package_sha=hashlib.sha256(package.read_bytes()).hexdigest()
    release=read(r/'resource_release.json') if (r/'resource_release.json').exists() else None
    lease_text=('已释放，状态 '+release['status']) if release else '本报告生成时仍预约；最终资源收据另附'
    text=f'''# 插件效率支线首次交付

2026-11-05，日本时间。范围：在交接的 index-reuse v1 上减少等价工程开销；未启动正式训练或真实评分。

## 结果

选中首层 z-buffer 复用候选。相同 V100S GPU2、原生 auto、完整 Stage9 信号/噪声/遮挡规则、指定确定性配置下，s2_00 的 200 ms 缓存片段每版独立新进程三次、预热后轮换顺序：

| 指标 | 原 index-reuse v1 | 新候选 |
|---|---:|---:|
| 初始化加生成中位数，秒 | {row['reference_median_s']:.6f} | {row['candidate_median_s']:.6f} |
| 峰值已分配显存，GiB | {row['allocated_peak_gib']['reference']:.6f} | {row['allocated_peak_gib']['candidate']:.6f} |
| 峰值预留显存，GiB | {row['reserved_peak_gib']['reference']:.6f} | {row['reserved_peak_gib']['candidate']:.6f} |
| 每次输出事件数 | {row['events']['reference']} | {row['events']['candidate']} |

耗时减少 {100*row['time_reduction_fraction']:.3f}%，倍率 {row['speedup']:.6f}。已分配峰值减少 {100*(1-row['allocated_peak_gib']['candidate']/row['allocated_peak_gib']['reference']):.3f}%，预留峰值减少 {100*(1-row['reserved_peak_gib']['candidate']/row['reserved_peak_gib']['reference']):.3f}%。这是一条代表片段的实测，不宣称五场景总体倍率或端到端 Genesis 加速。计入 fresh plugin setup、生成、必要搬运和最终排序后的 CPU FP64 事件；不计渲染、缓存读盘、序列化、进程启动或首次 JIT。没有把 profile 时间用于此表。

耗时采用三次中位数；显存采用三次运行各自峰值的最大值。

正常 CUDA 模式另测 s2_00 的 20 ms 短片段，三次中位数为 v1 {normal['reference']:.6f}s、新版 {normal['candidate']:.6f}s。片段长度和数值模式不同，不与上表相除；正常模式完整严格资格仍为失败。

## 精确版本与入口

- 交接仓库锚点：23761f2f5e3e68e0f50b3d7c5664be5690d93822；22 个 EVIDENCE_INDEX 文件大小/SHA 全通过。
- 原候选几何 SHA：e74793c4805eabfa2a19d9db02b5d829c73a12a61d3fed6b4cbeb8e5995a060b。
- 新候选几何 SHA：{source['delta_sha256']}；仅 genesis_event_plugin/gpu_geometry.py 改变，其他完整基包清单文件逐字节相同。
- 候选构建提交：{source['implementation_commit']}；实际计时代码和源哈希保存在 FIRST_DELIVERY.json 与各 result.json。
- profile SHA：{profile_sha}；实际 N=377/10ms、无 cap/fallback；CUDA_VISIBLE_DEVICES 为 GPU-a02affe0-7598-579e-ca39-f551f35ad6e9。
- 锁定 V2E 源码快照的 107 个文件与 SuperSloMo39 权重均通过完整大小/SHA 复核，详见 v2e_pin_verification.json；快照没有 .git 元数据，身份依据完整文件清单。
- 入口：candidate_release/genesis_event_plugin/optimized.py 的 OptimizedGenesisPhysicsEventPlugin.attach/capture；准备与核验命令见 source/README.md 及 source/run_checks.py。
- 完整可独立解包代码：zbuf_reuse_candidate.tar.gz，SHA256 {package_sha}。包内保留 stage9_sensor_profile.json、许可证和 SOURCE_MANIFEST.json；它是候选，不是新冻结版。
- 指定 torch.use_deterministic_algorithms(True)、CUBLAS_WORKSPACE_CONFIG=:4096:8；FP64/FP32 边界与原包相同。私有 sm70 编译维持 --fmad=false、无 fast-math；MAX_JOBS=2。工具链从原安装收据读取，未安装/升级依赖。

## 改动与剩余瓶颈

覆盖合成构造候选时已经按原候选顺序计算 zbuf。首轮 unresolved 全 True，旧代码又以相同 idx/z 重算 nearest。新代码首轮只读复用 zbuf，后续层原式不变；没有跳过贡献、提前退出、改变容差或更换采样/噪声过程。不得再对这份 nearest 别名做原地修改，否则会破坏返回的 z-buffer 诊断。

正常模式独立代表区间 profile 中，剩余前三个阶段成本为覆盖合成、投影/对象坐标变换、完整传感器调用。覆盖阶段主要涉及 gather/index、逐层归约及 nonzero 引发的同步；投影涉及 FP64 bmm；传感器仍含原 RNG 与完整状态递推。profile 中 GPU annotation 是区间跨度，含等待且层级重叠，不能当作纯核执行时间相加，也不能与 .cpu() 等待重复计费。原算子自耗时排名及新旧计数完整保存在 profile.json。

本轮没有重测 CUPTI 传输字节或 CPU 进程树峰值，因此新增收益限于上述计时、GPU 显存及算子计数。工程等价验收不产生新的传感器物理真实性结论。

```json
{json.dumps(counters,ensure_ascii=False,indent=2)}
```

## 验收和保留的失败

- 35 个严格压力例通过，含 HDR/低照度、灰度/RGB、空支持、部分覆盖、遮挡翻转和异常边界；返回数组/诊断按原始字节比较。
- 五类各两区间实际插件短检通过：{gates}。actual attach/capture、每区间一次 sensor、N=377、零回退；事件、中间亮度/深度/覆盖/unknown、状态与 RNG 全等。
- s3_00 连续 20 区间/200 ms 通过，{long['raw_events']['shape_left'][0]} 个事件；实际重建/光流/位姿 loader 的所测窗口通过严格核对。只恢复物理时间原点，没有时间量化。
- 默认 CUDA 完整原始事件/loader 严格失败继续保留；本轮正常模式短计时不提升这一资格。没有改变相等门禁、噪声、遮挡规则、分辨率或事件数预算。
- 首个“共享齐次坐标存储”候选虽然精确通过，正常短测中位数 1.849194s vs v1 1.818727s，峰值也未下降，已淘汰；源码及测量保留在 candidates/projection_storage 和 evidence。
- 首次队列因前一任务结束后利用率采样尚未归零而退出，已释放；记录保留为 attempt1。随后有界等待并仍要求实际零负载、无外部进程，再取得原子预约。
- 本轮没有重跑 100 区间长程、独立进程恢复及传感器异常/容量回退；传感器/内核源码未变，既有结果作为引用。未满足新的生产冻结完整门槛，未修改或提升任何冻结目录。

## 资源与产物

g1 GPU2：{lease_text}。使用全局预约锁和 UUID flock，检查真实进程、显存、利用率、CPU、内存及 NFS 磁盘；原预约和新预约均有收据。未连接 g3，未操作其他任务。

WSL worktree：/home/科研/Eventbased_WAM/repositories/gec-efficiency-continuation-20261105。

g1 worktree：/home/zhaoyuhan/Eventbased_WAM/repositories/gec-efficiency-continuation-20261105。

原始 run：{r}。大数组、Chrome trace 和 loader 输出留在该目录，文件大小/SHA 由 external_artifacts.json 登记。原始逐次计时、失败、profile、完整包清单和资源收据随本报告归档。共享 bank、标签、checkpoint、冻结核心和共享环境保持原样。生成器加速不会直接加快读取离线事件的 optimizer 更新。
'''
    (r/'FIRST_DELIVERY_ZH.md').write_text(text)
    print('REPORT_COMPLETE',row['reference_median_s'],row['candidate_median_s'],row['speedup'],package_sha,flush=True)


if __name__=='__main__':main()
