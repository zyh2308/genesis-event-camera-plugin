"""Summarize actual receipts, preserving failed strict gates and all raw evidence."""
from pathlib import Path
import datetime, gzip, hashlib, json, shutil, statistics, tarfile
W=Path(__file__).resolve().parent;R=W.parent/'sim2real/20261103-stage9-direct-v1';P=W.parents[1]
read=lambda p:json.loads(Path(p).read_text())
native=read(W/'native_auto/summary.json');det=read(W/'deterministic/results.json')
initial=read(W/'formal_existing/formal_loader_results.json');extended=read(W/'formal_extended/formal_loader_results.json')
det_inputs=read(W/'formal_deterministic/formal_loader_results.json')
pred=read(W/'reconstruction_predictions_retry1/reconstruction_prediction_probe.json')
pred_gpu=read(W/'reconstruction_predictions_gpu/reconstruction_prediction_probe.json')
pos=read(W/'reconstruction_positive_control/reconstruction_positive_control.json')
assert native['status']=='complete' and det['status']=='complete'
assert initial['status']==extended['status']==pred['status']=='complete'
assert pred_gpu['status']=='complete'
assert det_inputs['status']=='complete' and det_inputs['passed']
assert pos['positive_control_passed']
assert read(W/'source_reverification.json')['mismatches']==[]
assert read(W/'01_native_auto_suite.done.json')['exit_code']==0
assert read(W/'02_deterministic_isolation.done.json')['exit_code'] in [0,2]

def comparisons(value):
    if isinstance(value,dict):
        if 'bitwise_equal' in value:yield value
        for item in value.values():yield from comparisons(item)
    elif isinstance(value,list):
        for item in value:yield from comparisons(item)

def task_stats(case,task):
    values=list(comparisons(case[task]));bad=[x for x in values if not x['bitwise_equal']]
    assert all(x.get('finite_a',True) and x.get('finite_b',True) for x in values)
    return {'physical_windows_or_combinations':len(case[task]),'array_comparisons':len(values),
            'non_bitwise_equal_arrays':len(bad),'max_abs_difference':max([x.get('max_abs',0) for x in bad] or [0])}

case_summaries=[]
for group,data in [('one_second',initial),('five_scenes',extended)]:
    for case in data['cases']:
        assert case['status']=='checked'
        case_summaries.append({'group':group,'id':case['id'],'raw':case['raw'],
            'event_addresses_polarities_order_exact':case['address_polarity_order']['bitwise_equal'],
            'all_physical_window_memberships_equal':all(w['membership_index_equal'] and w['window_address_polarity']['bitwise_equal'] for w in case['windows_10ms']),
            'tasks':{t:task_stats(case,t) for t in ['flow','pose','reconstruction']}})

table=[]
for case in ['s0_00','s1_00','s2_00','s3_00','train_000']:
    row={'case':case,'physical_duration_s':0.2}
    for method,label in [('ours_resident','eventgenesis'),('v2e_native_auto','v2e')]:
        trials=[x for x in native['rows'] if x['case']==case and x['method']==method]
        assert len(trials)==3
        row[label+'_median_wall_s']=statistics.median(x['wall_s'] for x in trials)
        row[label+'_median_peak_GiB']=statistics.median(x['peak_bytes'] for x in trials)/1024**3
        row[label+'_event_counts']=[x['events']['count'] for x in trials]
        if label=='eventgenesis':assert all(x['substeps']==[377]*20 and x['fallbacks']==0 for x in trials)
        else:row['v2e_actual_intermediate_frame_counts']=[x['input_frames'] for x in trials]
    table.append(row)
control=read(W/'native_auto/v2e_auto_min63_control.json')
control_summary={'median_wall_s':statistics.median(x['wall_s'] for x in control['rows']),
    'actual_max_gap_s':max(x['max_dt_s'] for x in control['rows']),
    'frames':[x['input_frames'] for x in control['rows']], 'quality_matched':False}

summary={'schema':'formal_consumer_and_native_auto_acceptance_v1','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'run':str(W),'native_auto_table':table,'v2e_auto_min63':control_summary,'formal_cases':case_summaries,
    'default_raw_float64_bitwise_gate_passed':all(x['raw']['bitwise_equal'] for x in case_summaries),
    'default_all_formal_loader_inputs_bitwise_equal':initial['passed'] and extended['passed'],
    'deterministic_execution_all_tested_raw_bitwise_equal':det['all_raw_bitwise_equal'],
    'deterministic_formal_loader_inputs_bitwise_equal':det_inputs['passed'],
    'reconstruction_prediction_pairs':{k:{a:b for a,b in v.items() if a!='rows'} for k,v in pred['pairs'].items()},
    'reconstruction_gpu_prediction_pairs':{k:{a:b for a,b in v.items() if a!='rows'} for k,v in pred_gpu['pairs'].items()},
    'reconstruction_positive_control':{k:v for k,v in pos.items() if k!='rows'},
    'full_production_bank_universal_equivalence_claimed':False,'same_quality_speed_ranking_claimed':False,
    'current_training_restarted_or_modified':False,'formal_bank_or_labels_changed':False,'real_data_used':False,
    'automatic_production_promotion_performed':False,
    'decision':'existing training continues; native-auto speed has no universal advantage claim; default exact-input gate remains failed; deterministic conformance is a separate tested runtime configuration'}
(W/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

lines=['# 自动采样、正式loader与加速质量核查：2026-11-05','',
'本轮实际在g1 GPU2完成原生自动采样对照和确定性隔离；CPU完成正式loader、连续重建推理及清零输入正控。未启动/恢复训练，未修改正在运行的训练、物理bank、标签、真实评分开关或原冻结17文件。','',
'## 1. 采样与效率结论','',
'EventGenesis使用冻结PhysicsAdaptiveSampler原生自动规则，未固定U；300Hz与sensor_eps_target=0.05使本组每10ms原生返回377步。V2E本轮启用官方SSM auto/minfactor2，四个rich场景实际生成40张中间图像/0.2秒，匹配物体场景为88张，说明原生auto确实随运动变化。此前给V2E强制377步仅为高密度工作量控制，不能保留其自适应效率优势，不能作为默认使用效率排名。','',
'下表为同一V100S GPU2、同一已加载物理缓存、三个重复的中位数，单位为处理0.2秒片段的墙钟秒。不含Genesis渲染、启动和磁盘写出；插帧/几何、传感器、必要搬运及排序均计入。两者保持登记传感器参数和各自原生公式，并非同质量证明。','',
'|场景|EventGenesis原生auto / s|V2E原生auto / s|','|---|---:|---:|']
lines += [f"|{x['case']}|{x['eventgenesis_median_wall_s']:.4f}|{x['v2e_median_wall_s']:.4f}|" for x in table]
lines += ['',
'**当前原生auto运行点，V2E明显更快；不宣称EventGenesis整体更快。** V2E在登记300Hz配置下出现官方IIR步长警告；它的自动运动采样不独立保证传感器低通积分精度。不能据警告直接宣称真实泛化差，也不能称该组为全部参数的官方默认配置。','',
f"另测V2E auto/minfactor63：中位数{control_summary['median_wall_s']:.4f}s，实际最大间隔{control_summary['actual_max_gap_s']*1e6:.3f}us，1260张中间图像。它是额外最小密度控制，不是原始auto点，也没有证明与EventGenesis同等质量。实际最大间隔已记录，不能只凭名义factor作合格声明。",'',
'## 2. 正式输入：严格门禁保留失败','',
'1秒s3_00片段及独立原版重复，分别检查32个正式30ms光流窗、100个10ms计数窗、100个重建物理窗×5种原生/验证/训练增强设置。固定5场景另检查24个光流窗、96个计数窗（含匹配木块16窗）、400个重建窗/增强组合。组合数不是独立场景数，两批rich片段存在重叠。','',
'原版与新版的事件数、坐标/极性/顺序、已检查物理窗的事件归属均相同；本轮光流Gaussian及其正式增强/验证输入、位姿全帧计数输入均逐位一致。','',
'重建原版/新版有64/500个组合不逐位一致，原版独立重复为68/500；最大voxel差均1.1920928955078125e-6。扩展五场景中s2/s3也有少量末位差，其余已测输入相同。没有量化原始时间戳、修改H5时间dtype或采用容差让门禁通过。默认raw及所有正式输入逐位门禁仍为失败。','',
'## 3. 冻结checkpoint连续重建','',
'使用登记的同一synthetic_best.pt，读取正式验证loader、160中心裁剪，在相同1秒片段开头重置状态一次，然后连续100窗保留循环状态。原版、新版、独立原版重复的全部float32预测图像逐位一致，图像差max/RMS及每帧MSE差均为0。该探针从片段起点冷启动，不冒充完整episode或公共真实测试。','',
'清零相同事件输入后预测发生明显改变，正控通过：正常与零事件结果最大图像差约0.4657、RMS约0.06114。说明上述相同输出不是因为该checkpoint完全不响应事件。该正控不能证明对所有微扰都敏感。','',
f"追加CUDA实际前向比较：原版/新版最大图像差{pred_gpu['pairs']['reference_vs_candidate']['whole_sequence_prediction_difference']['max_abs']:.9g}，RMS {pred_gpu['pairs']['reference_vs_candidate']['whole_sequence_prediction_difference']['rms']:.9g}，平均MSE变化{pred_gpu['pairs']['reference_vs_candidate']['mean_mse_delta']:.9g}。原版独立重复对照也完整保留。它是固定合成片段的数值探针，不是新增真实测试结论。",'',
'## 4. 确定性计算隔离','',
f"独立子进程设置torch.use_deterministic_algorithms(True)和CUBLAS_WORKSPACE_CONFIG=:4096:8，保持物理、噪声、dtype、采样及遮挡配置不变。4个20ms短片段加s3_00连续1秒的原版/新版原始事件全等状态为：**{det['all_raw_bitwise_equal']}**。具体每例数量、时间与哈希见机器收据。",'',
'该实验检验指定确定性执行配置下的等价性，支持差异与原有并行浮点非确定性相关；开关影响多个受支持算子，不作“每项差异均排他地来自scatter”的过度归因。不得据此把默认非确定性路径的raw失败改为通过，也不能将确定性耗时与原普通路径混合。','',
'确定性版本的1秒原始事件另外经过正式三loader：32个光流窗、100个计数窗、100个重建物理窗×5种输入/增强设置，全部逐位一致。它与默认路径的失败是两份独立收据，不相互覆盖。此设置下该1秒片段的原版/新版单次生成耗时约393.709/195.435秒，数值检查中仍约有2倍自加速；这不是多次重复的正式性能汇总。','',
'## 5. 是否重训以及后续资格','',
'原生自动采样、正式数据场景和本次工程加速是不同事项。本轮保持原正式场景/划分/标签不变，不新增数据扩展任务。现有三项EventGenesis seed0目录均有完整更新数和检查点完成记录：光流8910、重建12000、位姿30000。它们读取离线事件，并不在optimizer循环中调用生成器，因此换更快生成器不会直接缩短剩余网络训练。','',
'如果新版本的剩余生成加训练时间确实更短，可以在隔离目录采用；但已经完成的数据生产与训练不应仅为换生成代码而重新开始。本轮没有据此终止作业。','',
'本轮证据没有发现事件地址、极性、噪声配置或已测重建结果的退化；它仍不是全生产bank、所有物体、所有preset的无损保证。默认输入严格等价资格保留失败，确定性路径只按本轮覆盖范围确认。若后续要求逐位固定的新数据生产，应单独登记确定性运行设置、完整任务输入验收和相应效率，不自动切换正式入口，不放宽门禁或降低质量换速度。','',
'## 6. 证据与执行记录','',
'机器汇总见validation_summary.json；完整逐窗记录以gzip无损压缩归档；大型原始npy/H5/预测数组保留服务器并列入external_artifacts.json。源码、配置和外部产物哈希随本轮保存。v2e归档107个源码文件再次核对一致。','',
'首次连续推理因人工抄写checkpoint SHA多一个字符被门禁拦截；已保留失败日志，改为从原登记metadata读取后复跑。实际检查点与登记哈希一致，未替换模型或降低检查标准。','']
(W/'RESULTS_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')

delivery=W/'delivery_v2';delivery.mkdir(exist_ok=False);e=delivery/'evidence';s=delivery/'source';e.mkdir();s.mkdir()
shutil.copyfile(W/'RESULTS_REVIEW.md',delivery/'RESULTS_REVIEW.md')
for rel in ['validation_summary.json','existing_training_readonly_snapshot.json','source_reverification.json','ending_receipt.json','launch_receipt.json','prediction_retry1_receipt.json','cases_existing.json','cases_extended.json','cases_deterministic.json','01_native_auto_suite.done.json','02_deterministic_isolation.done.json','03_reconstruction_gpu_probe.done.json','formal_extended.done.json','native_auto/summary.json','native_auto/v2e_auto_min63_control.json','deterministic/results.json','reconstruction_predictions_retry1/reconstruction_prediction_probe.json','reconstruction_predictions_gpu/reconstruction_prediction_probe.json','reconstruction_positive_control/reconstruction_positive_control.json']:
    src=W/rel;target=e/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
for rel in ['formal_existing/formal_loader_results.json','formal_extended/formal_loader_results.json','formal_deterministic/formal_loader_results.json','01_native_auto_suite.log','02_deterministic_isolation.log','03_reconstruction_gpu_probe.log','formal_existing.log','formal_extended_driver.log','formal_deterministic.log','reconstruction_predictions.log','reconstruction_predictions_retry1.log','reconstruction_positive_control.log']:
    target=e/(rel+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
    with (W/rel).open('rb') as inp,gzip.open(target,'wb') as out:shutil.copyfileobj(inp,out)
for name in ['native_auto_suite.py','deterministic_isolation.py','check_formal_loaders.py','check_formal_loaders_v2.py','check_reconstruction_predictions.py','check_reconstruction_predictions_device.py','check_reconstruction_positive_control.py','prepare_extended_cases.py','extended_cpu_driver.py','finalize_validation.py']:
    shutil.copyfile(W/name,s/name)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
external=[]
for path in W.rglob('*'):
    if path.is_file() and path.suffix in ['.npy','.npz','.h5'] and 'delivery' not in path.parts:
        external.append({'path':str(path),'size_bytes':path.stat().st_size,'sha256':digest(path)})
(e/'external_artifacts.json').write_text(json.dumps({'root':str(W),'artifacts':external},indent=2)+'\n')
index=[{'path':str(path.relative_to(delivery)),'size_bytes':path.stat().st_size,'sha256':digest(path)} for path in delivery.rglob('*') if path.is_file()]
(delivery/'artifact_index.json').write_text(json.dumps({'files':index},indent=2)+'\n')
with tarfile.open(W/'validation_delivery_v2.tar.gz','w:gz') as tar:tar.add(delivery,arcname='validation_delivery')
print('DELIVERY_COMPLETE',len(index),len(external),digest(W/'validation_delivery_v2.tar.gz'),flush=True)
