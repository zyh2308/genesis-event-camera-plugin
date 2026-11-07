"""Build the research-facing review from machine snapshots, never model scores invented in prose."""
import csv,hashlib,json,statistics
from pathlib import Path

root=Path('/home/科研/Eventbased_WAM/repositories/gec-mainline-resume-20261105/benchmark/sim2real/reviews/three_line_recovery_20261107')
s=json.loads((root/'server_snapshot.json').read_text())
pose=s['pose'];checks=pose['gate']['checks']
assert len(pose['results'])==6 and len(checks)==30 and sum(c['passed'] for c in checks)==22
assert sum(c['passed'] for c in checks if not c['condition'].endswith('_pose'))==13
rows=[]
for r in pose['results']:
    for arm,conditions in r['arms'].items():
        for condition,v in conditions.items():
            rows.append(dict(method=r['method'],seed=r['seed'],arm=arm,condition=condition,
                depth_mae_cm=100*v['depth_mae_m'],depth_bias_cm=100*v['depth_bias_m'],
                center_error_px=v['center_px'],add_s_success_pct=100*v['success'],auc_pct=100*v['auc']))
def write_csv(name,values):
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values)
write_csv('pose_synthetic_by_seed.csv',rows);write_csv('pose_acceptance_checks.csv',checks)
aggregates=[]
for method in ('eventgenesis','v2e'):
    for arm in ('clean_control','physical_supervised','paired_consistency'):
        for condition in ('clean','half','eighth'):
            selected=[r for r in rows if (r['method'],r['arm'],r['condition'])==(method,arm,condition)]
            assert len(selected)==3
            a=dict(method=method,arm=arm,condition=condition,n_seeds=3)
            for k in ('depth_mae_cm','center_error_px','add_s_success_pct','auc_pct'):
                a[k+'_mean']=statistics.mean(r[k] for r in selected)
                a[k+'_sample_sd']=statistics.stdev(r[k] for r in selected)
            aggregates.append(a)
write_csv('pose_synthetic_mean_sd.csv',aggregates)
def get(method,arm,c,key):
    return next(r[key+'_mean'] for r in aggregates if (r['method'],r['arm'],r['condition'])==(method,arm,c))
table=[]
labels={'clean':'正常','half':'半亮度','eighth':'八分之一亮度'}
for method in ('eventgenesis','v2e'):
    for c,label in labels.items():
        values=[]
        for key in ('depth_mae_cm','add_s_success_pct'):
            values.append(' → '.join(f'{get(method,arm,c,key):.2f}' for arm in ('clean_control','paired_consistency')))
        table.append(f'| {method} | {label} | {values[0]} | {values[1]} |')
report=f'''# 三线恢复、独立深度诊断与下一步

本次服务器快照从 `{s['captured_utc']}` 开始采集，表内训练日志分别有自己的时间戳。SSH 已通过单次较长等待恢复；没有改全局 SSH、代理或服务器服务配置。

## 现状

| 路线 | 已核验的成果 | 还缺什么 |
|---|---|---|
| 重建 | 5种生成方式×3seed，15/15正式训练完成；HQF14序列共210次真实推理全部完成 | 评分进程仍在运行，尚无最终score.json，不能给出本轮真实排名；HS-ERGB本轮未新增结果 |
| 光流 | 12/15正式训练完成；Vid2E、FILM、RIFE的seed2已同时运行 | 三组满8,910更新；MVSEC完整数据、最终真实评分与汇总表 |
| 位姿 | 原始15/15正式训练已完成；本轮另做2生成器×3seed×3设置，共18次修复短训 | 新候选未通过合成稳定门槛，因此本轮未新增真实E-POSE评分；开始补齐原训练轨迹的光照覆盖 |

光流快照的更新数为 Vid2E {s['flow']['vid2e']['optimizer_update']:,}、FILM {s['flow']['film_esim']['optimizer_update']:,}、RIFE {s['flow']['rife_esim']['optimizer_update']:,} / 8,910。RIFE从未启动的GPU4队列移至空闲GPU2；原argv、训练数据、初始权重核验与预算保持不变，仅GPU分配和CUDA缓存目录改变。其20步训练、21步恢复检查已经经过原wrapper，再进入正式训练。三个日志均记录70个有限梯度张量。运行记录、原队列spec和源码SHA见server_snapshot.json的flow_launch。没有中断其他训练。

## 位姿：最暗深度改善，但还不能采用

本轮使用固定最终定位分支、独立可训练深度分支、相同冻结旋转参考。三种设置为正常辅助对照、物理降亮度监督、物理降亮度加配对正常视图teacher一致性。后两者使用完全相同的低光照窗口；一致性分支的teacher输入是同一物理窗口、同一仿射变换的正常视图。每次6epoch、288个optimizer更新，最后一轮权重固定，不按真实结果选择。

下表为三seed均值，对照→配对一致性。它是**合成开发诊断，不能作为真实E-POSE论文结果**。逐seed及样本标准差见CSV。

| 训练事件 | 合成条件 | 深度MAE/cm ↓ | ADD-S@0.1D成功率/% ↑ |
|---|---|---:|---:|
{chr(10).join(table)}

EventGenesis最暗条件的深度误差降低约39%，成功率增加5.73个百分点；V2E最暗成功率仍更高。正常亮度六组成功率保持检查均通过，定位分支的预测在各设置间保持相同。不过EventGenesis seed0/1半亮度深度和成功率退步，V2E部分seed深度或成功率也未保持。

**原18项检查13/18通过；加上12项低光照成功率保持检查后22/30通过，整体失败。** 没有放宽原检查。额外12项中有9项通过；增加这些保护不改变“原门槛也失败”的结论。合成最暗条件只是0.125倍线性辐照度，未经lux标定；不能直接等同真实数据某档照度。

最暗条件中固定定位误差仍约EventGenesis30.19px、V2E15.65px。因此深度修好一部分后，定位仍是明显短板。这不是证明全部差距都由定位造成。验证仅含4条降亮度轨迹、每档64窗口，均已被开发过程查看，不作总体显著性或独立盲测宣称。

完整权重、逐轨迹结果与运行日志在 `{s['pose_root']}`，源码SHA在server_snapshot.json。CPU预检使用程序生成输入，检查配对变换、内参、深度不变性、实际梯度更新、冻结定位和严格checkpoint往返；服务器实际配对smoke及18次完整短训随后通过。CPU预检本身不冒充训练效果。

## 正在做的下一步

先补齐**现有96条原训练轨迹**的半亮度和八分之一亮度事件。此前适应短训仅覆盖其中8条，存在覆盖不足的可能；尚未证明它就是根因。这不是重新设计物理场景，也不改变仿真器、噪声、遮挡或标签。两种生成器均使用全部96条、两档亮度，384个generator/trajectory/light产物；采用已验证单worker流程，避免此前双worker OOM。准备任务在GPU5登记、实际运行，18小时上限，结束或失败自动释放，不自动串接未经核验的新训练。

目录：`{s['full_light']['root']}`。下轮在完整训练光照覆盖上验证定位与深度能否共同保持稳定，再考虑真实E-POSE开发复测。任何最终论文对比需将采用的同一数据/模型配方用于所有列入表格的生成器，不能只给EventGenesis增加训练覆盖。此前木块真实序列已被查看，后续结果仍需标为开发诊断，不能恢复其盲测身份。

## 数据与SSH的真实限制

用户说明手工连接也需3–5分钟。此前20秒ConnectTimeout不足；本轮用一次360秒等待成功建立g1连接并复用。原因尚未定位到代理、TCP、SSH握手或认证中的某一环节，不把“连接成功”冒充根因修复。g3本轮没有新的成功核验。没有关闭主机密钥校验或改全局配置。

MVSEC室内已建立的续传不受新连接超时影响，继续保留；新的顺序接续器只会等原传输退出，在确认为SSH超时后冷却并做一次长等待核验，再按原SHA/前缀/写锁要求接续。不并发写同一文件，不修改正在运行的旧传输脚本。

MVSEC室外：4096字节Range探测返回有效HDF5头，但完整大文件请求返回2,040字节的Google配额超限HTML。两份大文件均未取得，未登记、未传给loader或评分。校准ZIP已完整下载且通过CRC检查，SHA见local_data_snapshot.json；此SHA是本地记录，不是独立官方校验值。服务器直连及临时回环代理尝试的失败证据保留，失败代理隧道已关闭，不继续反复重试配额。参照[官方MVSEC下载页](https://daniilidis-group.github.io/mvsec/download/)，不能把不同格式输入静默替代已冻结HDF5协议。

## 时间判断

HQF还需完成现有评分循环，预计数小时，以完整score.json/CSV和退出状态为准。光流三个剩余作业已并行；按目前约6秒/更新的另两组速度估计，最后启动的RIFE还约15小时，另计验证与最终评测。原8轨迹两档光照准备实测约63分钟；96条简单线性估计约12.6小时，仅作规划，不是保证。随后仍需完整数据loader预检、配对训练和稳定检查。

因此周一前完成主要训练有实际希望，但完整MVSEC表受官方数据配额与传输制约，位姿受稳定门槛制约；目前不能承诺全部论文表格已经可以按时交付，更不能保证略胜V2E。冻结核心 `{s['core']}` 和噪声规则保持不变。
'''
(root/'README.md').write_text(report)
index={str(p.relative_to(root)):dict(size_bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_index.json'}
(root/'artifact_index.json').write_text(json.dumps(index,indent=2)+'\n')
print('Review generated:',len(rows),'per-seed condition rows;',len(aggregates),'mean/SD rows;',len(index),'indexed artifacts')
