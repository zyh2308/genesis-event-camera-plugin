# Stage9光流主线审核和评估桥接

本目录由光流审核子代理准备，没有SSH、GPU调用、真实数据评分或训练变更。远端当前训练数由主会话另行核查。

## 已确认的研究合同

- 新版核心`62da4f35ce3c5c96f04268bd1687a27657af9238`，Stage9正式根`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1`。
- 5种管线×3seed，STE-FlowNet dt1每组45epoch/8910updates，有效batch8，从同seed共享随机初始化；合成验证EPE选择`synthetic_best.pt`，非last/旧Stage4权重。
- 共享ReplicaCAD32条2s轨迹，24train/8val；30ms直接位移标签，训练1584窗、验证528窗。GT仅合成验证或独立诊断，不作optimizer监督。
- 真MVSEC按实际APS端点区间位移px评分，不把GT/prediction任意缩放成30ms速度；native260×346事件中心pad352，输出按(y46,x3)裁回原传感器。
- 7700个indoor窗口已事前注册：2165/2624/2911；注册SHA`46e0141bde84b8c3c3cc96615cec247f44bb8488ae33bdaf626d4f23d8dca02f`。outdoor_day1保留必需范围，当前缺失不能静默删除。

## 不可沿用的旧入口假设

仓库`sim2real_status.md`的新版段仍是11月3日部署快照，不证明训练仍是0。旧Stage4 protocol绑定两种生成器、seed0与旧epoch33/20检查点，不可直接冒用到新版15组。`run_mvsec_candidate_inference.py`本身会计算候选分数，不能将它称为无评分loader审计。

相关性CUDA只允许保留参考前向和有序CUDA反向的共同seed1切换。3090的直接全CUDA方案Adam更新偏差1.4424%超过1%门槛，已拒绝；有序方案通过。主会话已远端确认V100batch8门禁exit0、seed1共同有序方案正在运行，应保持现有配置。历史g3记录不可视作连接许可。

ESIM三插帧基线的时间戳修复为stable row排序，最大逆序rich约128ns、pose约4ns，保留所有事件数值/计数；不能将效率支线的V2E API时间分析直接当作Stage9官方CLI数据已替换或有错。

## 新桥接脚本

使用`stage9_mvsec_bridge_v2.py`的prepare、evaluate、self-test。旧`stage9_mvsec_bridge.py`仅保留开发记录，不应用于真实评分：最后审查发现既有Stage3 adapter没有应用Stage9训练的内部模型warp兼容层。v2显式绑定并复用原`steflow_align_corners.patch_ste`，在任何模型前向之前恢复与训练相同的`align_corners=True`行为，不修改正在训练的模型或原源码。

prepare不读真实HDF5、不加载模型、不评分；核对15个completion的8910预算、recipe和隔离字段、selected/last实际文件hash、同seed初始化，以及Stage9实际`training_bank_manifest.json`。核对冻结核心62da与batch8合同，并绑定adapter、metrics及实际训练model_compat三个脚本SHA。缺少完成记录会输出明确missing清单。生成协议和manifest保持评分关闭，required4/available3/missingoutdoor明确记录。

evaluate先调用成熟`stage4_mvsec_dt1_metrics.authorize_real_evaluation`，需要研究冻结后的protocol、manifest及其SHA绑定authorization。它不生成或伪造授权。再要求完整15权重矩阵锁定，复用已有MVSEC loader、官方GT传播、共有mask和AEE/outlier函数；每窗核对注册event/GT/mask哈希。native full ROI主表和固定legacy256副表同时报告，空support不变成零分。outdoor未到位时结果永久标`authorized_indoor_partial_score`，不生成四序列汇总。

## 主会话部署命令示例

以下是现有g1环境和新隔离目录的示例，不代表已执行。实际adapter/metric路径先按源hash核对，不能指向缺少相邻manifest的孤立archive副本。

```bash
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
R=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1
S="$R/sources/benchmark/sim2real/scripts"
$PY stage9_mvsec_bridge_v2.py prepare \
  --campaign "$R" \
  --registration "$R/real_evaluation_inputs/stage9_mvsec_indoor_unscored_registration.json" \
  --adapter-script "$S/run_mvsec_candidate_inference.py" \
  --metric-script "$S/stage4_mvsec_dt1_metrics.py" \
  --compat-script "$R/sources/benchmark/sim2real/compat/steflow_align_corners.py" \
  --out /new/isolated/stage9_mvsec_preparation
```

**注册文件的服务器位置是建议目标，并非已存在事实。** 本地已核对来源为`/home/科研/Eventbased_WAM/data/sim2real/stage5/real/stage9_mvsec_indoor_unscored_registration.json`，22,075,595字节，SHA为本页所列46e014开头哈希。主会话必须先以不可覆盖的传输流程复制至新目标并核对大小/完整SHA，且核验JSON中六个真实数据/GT文件的实际位置与内容。没有这份注册文件绝不能评分；未完成或零字节传输日志不能证明已同步。prepare找不到文件会直接失败，不自动重新挑窗口。

若registration更新以包含outdoor，需要单独核验并显式给`--registration-sha256`；不能默默读任意同名注册文件。上述`model_compat`源脚本在g1实际存在，仍须主会话记录其SHA。

评分命令在完整训练、数据/source核验和研究冻结后才能运行：

```bash
$PY stage9_mvsec_bridge_v2.py evaluate \
  --protocol /reviewed/frozen_protocol.json \
  --manifest /reviewed/frozen_manifest.json \
  --authorization /reviewed/hash_bound_authorization.json \
  --method eventgenesis --seed 0 \
  --model-root /home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/STE-FlowNet \
  --device cuda:0 --out /new/isolated/mvsec_eventgenesis_seed0
```

调用方负责实时资源核对和GPU UUID原子预约；evaluate要求只暴露一个明确GPU UUID。正常执行不接管或中断训练，不修改checkpoint、模型或真实数据。

## CPU自测

已在WSL退出0，11项检查通过：15权重元数据与hash；错误checkpointhash、seed、不足预算、last/旧路径、真实训练泄漏、错误recipe、错误注册SHA均拒绝；成熟向量/zero-flow指标oracle；旧评分门禁拒绝未冻结协议；新增实际冻结model warp的零流恒等及旧坐标约定负向控制。全部为临时合成夹具，无模型或真实数据加载。最新机器收据见`cpu_selftest_v3.json`，前两次9/10项记录也保留。

真实任务仍需要主会话部署后做一次prepare dryrun，核对实际目录和回执。本脚本不声称已有MVSEC实测结果。
