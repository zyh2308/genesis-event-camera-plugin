"""Write a Chinese review from a finalized delivery, without inventing results.

Only reads JSON/source receipts and writes a new Markdown file. Missing evidence
is an error. Failed normal-mode checks are reported as failures, never relabeled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from urllib.parse import quote


CASES = ('s0_00', 's1_00', 's2_00', 's3_00', 'train_000')
METHODS = ('frozen_gpu', 'candidate_gpu', 'v2e_auto')
SELECTED = 'histfree+deadmix+indexreuse_v1'
GEOMETRY_SHA = 'e74793c4805eabfa2a19d9db02b5d829c73a12a61d3fed6b4cbeb8e5995a060b'
FROZEN17 = '62da4f35ce3c5c96f04268bd1687a27657af9238'
FAST_TAG = 'freeze/eventgenesis-gpu-stage9-20261105'
FAST_COMMIT = '3e68e3b727ec2c21d9aa947a7989ebc2efd14e75'
FAST_PACKAGE = '21480a95fe934bc385d37f075c4fbb366b9ccb9d0c33654ebddac83902f82ce9'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def field(value, key, where):
    require(isinstance(value, dict) and key in value, f'{where}: missing {key}')
    return value[key]


def number(value, where):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f'{where}: expected finite numeric value')
    return value


def yes(value):
    require(type(value) is bool, 'A recorded pass flag is missing or not Boolean')
    return '通过' if value else '**未通过**'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    d, out = a.delivery.resolve(strict=True), a.out.resolve()
    require(not out.exists(), 'Refusing to overwrite an existing report; choose a fresh --out')
    require(not out.with_suffix(out.suffix+'.sources.json').exists(), 'Refusing to replace an existing report source receipt')
    used = []

    def read(relative):
        path = d/relative
        require(path.is_file(), 'Missing required report evidence: '+str(path))
        used.append(path)
        return json.loads(path.read_text(encoding='utf-8'))

    def link(label, relative):
        target = Path(os.path.relpath(d/relative, out.parent)).as_posix()
        return '['+label+']('+quote(target, safe='/._-')+')'

    b = read('evidence/benchmark_summary.json')
    q = read('evidence/full_quality_summary.json')
    v = read('evidence/normal_mode_event_variability.json')
    n = read('evidence/normal_consumer_summary.json')
    probe = read('evidence/normal_reconstruction_predictions_gpu/reconstruction_prediction_probe.json')
    component = read('evidence/component_results.json')
    index = read('evidence/index_component_results.json')
    tile = read('evidence/tile_component_results.json')
    index2 = read('evidence/index_v2_component_results.json')
    selection = read('evidence/candidate_selection.json')
    acceptance = read('evidence/acceptance_receipt.json')
    contract = read('evidence/numerical_contract.json')
    manifest = read('release/manifest.json')
    scope = read('DELIVERY_SCOPE.json')
    require(b['status'] == 'complete' and b['validation']['passed'], 'Benchmark evidence has not passed validation')
    for name, data in [('quality', q), ('normal variability', v), ('normal consumer', n), ('checkpoint probe', probe)]:
        require(data.get('status') == 'complete', name+' is incomplete')
    require(selection.get('selected') == SELECTED, 'Unexpected selected candidate')
    require(selection.get('candidate_sha256') == GEOMETRY_SHA, 'Selected geometry digest is not the tested v1')
    geometry = d/'release/genesis_event_plugin/gpu_geometry.py'
    require(geometry.is_file() and hashlib.sha256(geometry.read_bytes()).hexdigest() == GEOMETRY_SHA,
            'Embedded geometry does not match the tested selected receipt')
    used.append(geometry)
    file_maps = {}
    for key in ('base_files', 'candidate_files'):
        values = field(manifest, key, 'release manifest')
        require(isinstance(values, list), 'Release file inventory is not a list')
        file_maps[key] = {item['path']: item['sha256'] for item in values}
    old_files, new_files = file_maps['base_files'], file_maps['candidate_files']
    require(old_files.keys() == new_files.keys(), 'Base/candidate static file sets differ')
    changed = [k for k in old_files if old_files[k] != new_files[k]]
    require(changed == ['genesis_event_plugin/gpu_geometry.py'], 'Only geometry may change: '+repr(changed))
    require(new_files[changed[0]] == GEOMETRY_SHA, 'Manifest candidate digest differs')
    for key in ('training_modified_or_started', 'frozen_release_modified', 'production_promotion_performed'):
        require(field(scope, key, 'delivery scope') is False, 'Unexpected operation recorded: '+key)
    inventory_sha = hashlib.sha256(json.dumps(new_files, sort_keys=True).encode()).hexdigest()
    run_root = field(scope, 'run_root', 'delivery scope')
    jobs = {(j['case'], j['method']): j for j in b['jobs']}
    comparisons = {r['case']: r for r in b['per_case_comparisons']}
    require(all((case, method) in jobs for case in CASES for method in METHODS), 'Main timing table lacks a registered job')
    require(all(jobs[(case, method)]['validation_passed'] for case in CASES for method in METHODS), 'A main benchmark job failed validation')
    require(('s2_00', 'v2e_auto_min63') in jobs, 'Missing minimum63 control')
    for case in CASES:
        for method, expected_sha in (('frozen_gpu', old_files[changed[0]]), ('candidate_gpu', GEOMETRY_SHA)):
            require(jobs[(case, method)]['core_source_sha256']['gpu_geometry.py'] == expected_sha,
                    'Timing process used a different geometry: '+case+'/'+method)

    def metric(case, method, key='initialized_pipeline_s', stat='median'):
        return number(jobs[(case, method)]['metrics'][key][stat], case+'/'+method+'/'+key+'/'+stat)

    primary = b['aggregate_ratios']['initialized_pipeline_s']
    secondary = b['aggregate_ratios']['genwall_s']
    require(primary['all_five_cases_validated'] and secondary['all_five_cases_validated'], 'Aggregate ratios include unvalidated/missing cases')
    gain = number(primary['geometric_mean_old_over_new_speedup'], 'primary speedup')
    v2e_ratio = number(primary['geometric_mean_candidate_over_v2e_slowdown'], 'primary V2E ratio')
    secondary_gain = number(secondary['geometric_mean_old_over_new_speedup'], 'generation-only speedup')
    repeats = b['expected']['trials_per_job']
    require(repeats == 3, 'The registered timing design requires three repeats')
    gpu_uuid = b['expected']['gpu_uuid']
    runtime = b['expected']['runtime']
    noise = b['expected']['noise_config_kwargs']
    cutoff = number(noise['cutoff_hz'], 'cutoff')
    dt_values = [number(b['physical_inputs'][case]['simulated_s'], 'duration')/
                 b['physical_inputs'][case]['intervals'] for case in CASES]
    require(all(math.isclose(x, dt_values[0], abs_tol=1e-12, rel_tol=0) for x in dt_values), 'Endpoint interval differs across cases')
    dt, tau = dt_values[0], 1/(2*math.pi*cutoff)
    require(contract.get('status') == 'complete' and acceptance.get('status') == 'complete', 'Actual configuration/acceptance receipts are incomplete')
    numerical_cases = contract['cases']
    require({r['case'] for r in numerical_cases} == set(CASES), 'Resolved numerical contract lacks a registered case')
    eps_values = [number(row['actual_noise_config']['sensor_eps_target'], 'actual sensor_eps_target')
                  for row in numerical_cases]
    require(len(set(eps_values)) == 1 and eps_values[0] > 0, 'Resolved sensor epsilon differs across cases')
    eps = eps_values[0]
    for case in numerical_cases:
        require(case['actual_noise_config']['cutoff_hz'] == cutoff, 'Resolved cutoff differs from profile')
        require(len(case['samples']) == 2, 'Resolved numerical short trace has unexpected length')
        for sample in case['samples']:
            require(math.isclose(sample['dt_s'], dt, abs_tol=1e-12, rel_tol=0), 'Resolved physical interval differs')
            derived = math.ceil(sample['dt_s']*2*math.pi*cutoff/eps)
            require(derived == sample['derived_sensor_samples'] == sample['actual_sampling']['sensor_samples']
                    == sample['actual_sampling']['samples'] == 377, 'Resolved377-sample contract does not match actual trace')
    sensor_bound = math.ceil(dt/tau/eps)
    formula_detail = f'从实际传感器初始化配置读取 ε_target={eps:g}，该项给出 {sensor_bound} 步，已与每次实际采样记录核对。'
    require(sensor_bound == 377, 'Recorded numerical contract does not reproduce the registered377 samples')

    # Interval counts must come from recorded JSON, never inferred from folder names.
    pairs = field(q, 'pairs', 'quality summary')
    require(len(pairs) == 6, 'Expected five short cases and exactly one long case')
    expected_pairs = {('gate', case): 2 for case in CASES}
    expected_pairs[('gate_long', 's3_00')] = 100
    require({(r['folder'], r['case']) for r in pairs} == set(expected_pairs), 'Unexpected deterministic test coverage')
    accepted_pairs = {(r['folder'], r['case']): r for r in acceptance['pairs']}
    require(set(accepted_pairs) == set(expected_pairs), 'Acceptance receipt lacks an actual trace pair')
    exact_rows, comparison_paths = [], []
    for pair in pairs:
        key = pair['folder'], pair['case']
        report_name = Path(pair['report']).name
        relative = 'evidence/gate_comparison/'+report_name
        comparison = read(relative)
        receipt = accepted_pairs[key]
        intervals = receipt['intervals']
        require(intervals == expected_pairs[key], 'JSON must record actual interval count for '+str(key))
        flags = receipt['deterministic_algorithms']
        require(flags == [True, True], 'Deterministic runtime flags missing for '+str(key))
        require(receipt['cublas_workspace_config'] == [':4096:8', ':4096:8'], 'Deterministic CUBLAS contract differs')
        require(receipt['geometry_source_sha256'][1] == GEOMETRY_SHA, 'Gate used a different candidate geometry')
        require(receipt['geometry_source_sha256'][0] == old_files[changed[0]], 'Gate used a different frozen baseline geometry')
        frames = receipt['actual_sensor_frames_per_interval']
        require(len(frames) == 2 and all(len(side) == intervals and set(side) == {377} for side in frames),
                'Actual sensor frame counts changed in a deterministic gate')
        exact_rows.append((pair, intervals, comparison))
        comparison_paths.append(relative)
        require(comparison['status'] == 'complete', 'An actual-entry physical comparison is incomplete')
    index_stress = field(index, 'stress', 'index component')
    stress_count = field(index_stress, 'case_count', 'index stress')
    require(stress_count == 35, 'Registered35-case stress suite missing/incomplete')
    selected_stress = field(index_stress['candidate_summary'], 'indexreuse', 'index stress candidates')
    require(selected_stress['passed']+selected_stress['failed'] == stress_count, 'Selected stress counts disagree')
    require(probe['continuous_windows'] == 20, 'Normal-mode fixed-checkpoint probe must contain20 continuous windows')
    require(probe['checkpoint_parameters_frozen'] is True and probe['training_started'] is False
            and probe['real_data_used'] is False, 'Checkpoint probe changed model/training or used real data')
    for key in ('normal_loader_inputs_all_exact', 'no_tolerance_acceptance', 'no_training_or_real_test'):
        require(type(field(n, key, 'normal consumer summary')) is bool, 'Normal-consumer flags missing')
    require(n['normal_loader_check_exit_code'] in (0, 2), 'Normal formal-loader check did not finish normally')
    require(len(v['cases']) == len(CASES) and {r['case'] for r in v['cases']} == set(CASES), 'Normal variability lacks a registered case')

    headline = (f'新版相对上一 GPU 版本的完整管线加速比为 **{gain:.3f}×**（五类片段比值的几何平均）。'
                if gain >= 1 else
                f'新版相对上一 GPU 版本的完整管线耗时为 **{1/gain:.3f}×**；本轮不能声称整体提速。')
    v2e_text = (f'相对 V2E 原生自动采样，新版耗时仍为 **{v2e_ratio:.3f}×**，不能声称总体快于 V2E。'
                if v2e_ratio > 1 else
                f'相对 V2E 原生自动采样，五片段耗时比的几何平均为 **{v2e_ratio:.3f}×**；该结果不构成同质量速度排名。')
    lines = ['# EventGenesis 自适应采样计算优化核验报告', '', headline+' '+v2e_text, '',
        '本轮保留全部传感器时刻、数值精度、噪声随机调用、覆盖与遮挡规则；优化重复选行、未使用的混合计算和仅用于循环上界的直方图。结果属于隔离候选验收；没有修改训练、覆盖冻结版本或自动替换生产仿真器。', '',
        '## 计时协议与主要结果', '',
        f'同一组固定登记的五类合成片段、同源物理/显示输入哈希；每片段20个物理区间、每方法{repeats}次计时重复。共享服务器 g1 的 GPU2（`{gpu_uuid}`），Torch {runtime["num_threads"]}个CPU线程、interop {runtime["interop_threads"]}线程，TF32关闭。重复计时不是训练seed，也不是置信区间。', '',
        '主指标从每片段初始化开始，包含 setup 和生成、各自传感器首帧初始化，终点为按时间排序的CPU原始事件。预训练SSM网络与私有CUDA算子已预热；双方均不包含 Genesis 物理渲染、输入磁盘读取和结果序列化。', '',
        f'V2E使用官方SuperSloMo原生自动插帧、最低倍率2，传感器沿用预先登记的{cutoff:g}Hz带宽和全噪声配置；这不等同于V2E所有参数均为默认值。双方使用各自原生自动采样，没有强制V2E采用377倍插帧。', '',
        '| 片段 | 上一GPU版/秒 | 新版/秒 | V2E自动/秒 | 旧版÷新版 | 新版÷V2E |',
        '|---|---:|---:|---:|---:|---:|']
    for case in CASES:
        ratio = comparisons[case]['initialized_pipeline_s']
        lines.append(f'| {case} | {metric(case,"frozen_gpu"):.4f} | {metric(case,"candidate_gpu"):.4f} | {metric(case,"v2e_auto"):.4f} | {ratio["old_over_new_speedup"]:.3f}× | {ratio["candidate_over_v2e_slowdown"]:.3f}× |')
    lines += ['', '表中为三次中位数；min/max/IQR及setup、纯生成耗时均保留在JSON/CSV。跨片段只汇总比值的几何平均，不用不同片段平均耗时制造排名。', '',
        f'完整管线加速比几何平均 **{gain:.3f}×**；去掉初始化的纯生成加速比几何平均 **{secondary_gain:.3f}×**。'+link('计时与配置核查', 'evidence/benchmark_summary.json')+'；'+link('逐项统计CSV', 'evidence/benchmark_summary.csv')+'。', '',
        '## 显存与V2E采样范围', '',
        '| 片段 | 旧版峰值/GiB | 新版峰值/GiB | V2E峰值/GiB | V2E生成帧数/次 | V2E首末支持/秒 | V2E最大步长/毫秒 |',
        '|---|---:|---:|---:|---|---|---:|']
    memory_ratios = []
    for case in CASES:
        peaks = [metric(case, method, 'peak_worker_allocated_bytes')/(1024**3) for method in METHODS]
        info = jobs[(case, 'v2e_auto')]['v2e_temporal_information']
        require(len(info) == repeats, 'V2E timing-range evidence missing')
        memory_ratios.append(peaks[1]/peaks[0])
        counts = '/'.join(str(r['input_frames']) for r in info)
        max_dt = max(number(r['max_dt_s'], 'V2E maxdt') for r in info)*1000
        first = min(number(r['first_frame_s'], 'V2E first timestamp') for r in info)
        last = max(number(r['last_frame_s'], 'V2E last timestamp') for r in info)
        lines.append(f'| {case} | {peaks[0]:.3f} | {peaks[1]:.3f} | {peaks[2]:.3f} | {counts} | {first:.6g}–{last:.6g} | {max_dt:.5g} |')
    lines += ['', f'峰值为Torch整进程已分配显存峰值的重复中位数，含V2E持久网络权重，不是设备总占用；新版/旧版峰值比范围为{min(memory_ratios):.3f}–{max(memory_ratios):.3f}。显存变化与速度收益必须一起看。V2E各次首/末有效时间、实际帧数和步长完整记录于计时JSON，未拉伸时间轴，也未拿事件数当质量指标。', '']
    control = jobs[('s2_00', 'v2e_auto_min63')]
    require(control['validation_passed'], 'Minimum63 control failed validation')
    native_time = metric('s2_00', 'v2e_auto')
    control_time = metric('s2_00', 'v2e_auto_min63')
    control_frames = '/'.join(str(r['input_frames']) for r in control['v2e_temporal_information'])
    control_maxdt = max(number(r['max_dt_s'], 'minimum63 maxdt') for r in control['v2e_temporal_information'])*1000
    lines += [f'补充控制仅在s2_00上运行：V2E最低倍率2为{native_time:.4f}秒，最低倍率63为{control_time:.4f}秒，后者每次生成{control_frames}帧、最大步长{control_maxdt:.5g}毫秒。63是独立数值步长控制，不是V2E默认设置，也没有证明与EventGenesis质量相同。', '',
        '## 为什么仍保留377步', '',
        f'当前物理间隔Δt={dt*1000:g}毫秒、带宽f_c={cutoff:g}Hz，τ=1/(2πf_c)={tau*1000:.6g}毫秒。冻结采样器的一项下界为 `N_sensor = ceil((Δt/τ)/ε_target)`；最终仍与其他运动、几何及辐照度约束取最大值。'+formula_detail, '',
        '该约束限制离散低通更新系数，不是最终事件时刻误差的证明，也不是相机标定精度。直接减少子步会改变RNG调用序列、漏电抖动累计方差及shot噪声与不应期的联合分布；即使噪声率参数相同，也不能认为噪声概率分布不变。本轮因此没有把377硬改为更小值。', '',
        '## 正确性门禁：确定性范围与默认模式分开', '',
        '| 确定性实际入口片段 | 物理区间数 | 原始事件数 | 物理可观测量逐位核验 |',
        '|---|---:|---:|---|']
    for pair, intervals, comparison in exact_rows:
        lines.append(f'| {pair["case"]}（{pair["folder"]}） | {intervals} | {pair["events"]} | {yes(comparison["all_physical_observables_bitwise_equal"])} |')
    lines += ['',
        f'这是五类各2区间短检查，加上**仅s3_00一类100区间连续检查**；没有宣称五类都做过长检查。检查覆盖{acceptance["total_compared_intermediate_frames"]:,}个中间帧位置、逐帧图像、可见性/覆盖/表面/未知区诊断、未量化原始事件、完整传感器状态与RNG。只排除明确的执行耗时遥测字段，不排除物理诊断。完整门禁：{yes(q["all_exact"])}；实际重建/光流/位姿三个loader：{yes(q["actual_formal_loaders_passed"])}。'+link('完整门禁汇总', 'evidence/full_quality_summary.json')+'；'+link('实际配置与区间收据', 'evidence/acceptance_receipt.json')+'。', '',
        f'选中index-reuse v1的严格压力检查为{stress_count}项，记录通过{selected_stress["passed"]}、失败{selected_stress["failed"]}项，覆盖灰度/RGB、HDR/低照度、空支持、部分覆盖、真实遮挡翻转、异常值与容差边界。确定性门禁使用CUBLAS `:4096:8`；正常计时记录为 `{b["expected"]["cublas_workspace_config"]}`。该结论限于已测试的确定性、FP32无梯度事件生成路径，不能推广到所有输入、CUDA默认模式或反向梯度。'+link('组件与压力证据', 'evidence/index_component_results.json')+'。', '',
        '| 默认CUDA模式片段 | 旧版自身逐位相等/比较数 | 旧新版逐位相等/比较数 | 旧新版最大时间差/秒 |',
        '|---|---:|---:|---:|']
    for case in CASES:
        row, = [r for r in v['cases'] if r['case'] == case]
        own, cross = row['old_self'], row['cross']
        require(len(own) == 3 and len(cross) == 9, 'Normal variability comparison matrix incomplete')
        diffs = [r['max_time_delta_s'] for r in cross if 'max_time_delta_s' in r]
        max_delta = f'{max(diffs):.6g}' if diffs else '形状不同，不能逐事件配对'
        if diffs and len(diffs) != len(cross):
            max_delta += f'（仅{len(diffs)}/{len(cross)}组可配对）'
        lines.append(f'| {case} | {sum(r["bitwise_equal"] for r in own)}/{len(own)} | {sum(r["bitwise_equal"] for r in cross)}/{len(cross)} | {max_delta} |')
    lines += ['',
        f'默认模式跨版本原始字节全相等：{yes(v["all_cross_raw_byte_equal"])}；地址/极性全相等：{yes(v["all_cross_event_address_polarity_equal"])}。保留旧版自身重复作为对照，区分原有CUDA浮点原子归约波动与跨版本变化；没有时间戳量化或容差放行，也不能由该对照证明所有波动都无害。'+link('正常模式原始事件比较', 'evidence/normal_mode_event_variability.json')+'。', '',
        f'正常模式正式loader输入全相等：{yes(n["normal_loader_inputs_all_exact"])}（实际检查退出码{n["normal_loader_check_exit_code"]}）。固定重建checkpoint在{probe["continuous_windows"]}个连续窗口上实际推理，checkpoint SHA256为 `{probe["checkpoint_sha256"]}`；下表如实列出旧新版与旧版自身对照。', '',
        '| 固定权重预测比较 | 逐位相等 | 最大绝对差 | RMS差 | 平均MSE变化 |',
        '|---|---|---:|---:|---:|']
    for key, label in [('reference_vs_candidate', '旧版→新版'), ('reference_vs_reference_repeat', '旧版→旧版重复')]:
        result = field(probe['pairs'], key, 'checkpoint prediction pairs')
        diff = result['whole_sequence_prediction_difference']
        lines.append(f'| {label} | {yes(diff["bitwise_equal"])} | {number(diff["max_abs"],key):.6g} | {number(diff["rms"],key):.6g} | {number(result["mean_mse_delta"],key):.6g} |')
    lines += ['', '该检查没有训练或真实数据评分，没有改权重、重设每个窗口的循环状态、缩放或美化模型输出；它用于观察正常模式差异如何传到真实consumer，不构成下游质量提升或Sim-to-Real结论。'+link('正常模式consumer汇总', 'evidence/normal_consumer_summary.json')+'；'+link('固定checkpoint逐窗口探针', 'evidence/normal_reconstruction_predictions_gpu/reconstruction_prediction_probe.json')+'。', '',
        '## 选择与淘汰依据', '',
        '| 同输入组件候选 | 核验逐位相等 | 耗时中位数/秒 | 对本文件reference加速比 |',
        '|---|---|---:|---:|']
    comparisons_components = [(component, ['deadmix','histfree','histfree+deadmix','prune-full+deadmix']),
                              (index, ['indexreuse']), (tile, ['tile16','tile32']),
                              (index2, ['indexreuse_v2'])]
    for document, names in comparisons_components:
        rows = field(document, 'rows', 'component result')
        refs = [x['wall_s'] for row in rows if row['candidate'] == 'reference' for x in row['timings']]
        require(refs, 'Component evidence lacks its own reference timing')
        ref = statistics.median(refs)
        for name in names:
            picked = [r for r in rows if r['candidate'] == name]
            require(picked, 'Component evidence lacks candidate '+name)
            times = [number(x['wall_s'], name) for row in picked for x in row['timings']]
            passed = all(r['bitwise_equal'] for r in picked)
            if times:
                median = statistics.median(times)
                require(median > 0, 'Invalid component timing')
                lines.append(f'| {name} | {yes(passed)} | {median:.4f} | {ref/median:.3f}× |')
            else:
                lines.append(f'| {name} | {yes(passed)} | 未计时 | — |')
    lines += ['',
        '组件计时只用于定位重复工作，不能替代上述实际入口基准。最终选中 `'+selection['selected']+'`：'+str(field(selection,'reason','candidate selection'))+'。prune、时间分块和index-reuse v2即使在小样本上有等价或内存收益，也不能自动继承v1完整门禁；本轮未将其混入最终候选。每项实测速度见表，未测的收益不作承诺。'+link('选型收据', 'evidence/candidate_selection.json')+'。', '',
        '最终复现以实际归档的'+link('v1静态源码', 'release/genesis_event_plugin/gpu_geometry.py')+'及其SHA256为准；探索builder后来包含v2，不应再次运行它覆盖已测v1。统计规则见'+link('聚合脚本', 'source/summarize_results.py')+'。', '',
        '## 版本、边界与复现', '',
        f'- 本次只有 `gpu_geometry.py` 改变，实际源码SHA256：`{GEOMETRY_SHA}`；其他静态包文件逐字节相同。候选完整文件清单的SHA256（按path→SHA256排序JSON）为 `{inventory_sha}`。',
        f'- frozen17参考提交 `{FROZEN17}` 未修改。上一高速GPU基线为 `{FAST_TAG}`，源码证据提交 `{FAST_COMMIT}`，已登记发布包SHA256 `{FAST_PACKAGE}`。上述标识用于固定比较对象，本次没有将新候选自动升为该冻结版本。',
        f'- g1完整原始运行目录：`{run_root}`。轻量交付保留实际被测源码、门禁、失败尝试与哈希；大数组和完整trace按服务器路径及SHA256索引。',
        '- 本轮没有启动、恢复或改动三线训练，也没有进行真实数据模型选择、参数调优或正式性能排名。下一步若要用于论文或替换数据生产入口，仍需根据本文速度/显存与正常模式consumer证据作明确决定。', '',
        link('发布差异清单', 'release/manifest.json')+'；'+link('交付范围', 'DELIVERY_SCOPE.json')+'；'+link('原始产物索引', 'evidence/external_artifacts.json')+'。', '',
        '本报告由交付JSON生成；已读取文件的SHA256和报告生成器SHA256随附于同名 `.sources.json`，不改写实验记录。', '']
    text = '\n'.join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    provenance = dict(delivery=str(d), report=str(out), generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs=[dict(path=str(path.relative_to(d)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(set(used))],
        report_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), training_or_server_actions=False)
    with out.with_suffix(out.suffix+'.sources.json').open('x', encoding='utf-8') as stream:
        json.dump(provenance, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', report=str(out), report_sha256=provenance['report_sha256'],
                         evidence_files=len(provenance['inputs'])), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
