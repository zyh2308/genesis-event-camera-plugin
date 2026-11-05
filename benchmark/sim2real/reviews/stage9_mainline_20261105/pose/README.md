# Stage9 E-POSE正式评分入口

正式评估请使用`stage9_epose_v5.py`；旧版保留，不覆盖已部署文件。v2修正角点直接EPE辅助列名称；v3处理官方非严格刚体GT结构，始终保留原始T用于主几何；v4修正原始事件时间戳边界并列行匹配；v5仅修正JSONL输出关闭后再登记SHA。正式授权必须填v5源码SHA256，不能沿用旧hash。

v5脚本SHA256：`77cb84aca862bc62c9ada8f3dfed074b02dca4526641fef8d09e3163d0b47786`。部署时必须同目录包含`native_pose_audit_policy.json`和原始`pose_rotation_native_diagnostic.json`。分析见`NATIVE_GT_ANALYSIS.md`、`PACKET_ALIGNMENT_ANALYSIS.md`、`MANIFEST_HASH_FIX.md`，本地验证见`v5_validation_receipt.json`。

真实预检历史必须保留：首次v1被官方非正交旋转块挡住；v3随后因重复时间戳packet边界被挡住；v4服务器全18序列、20916packet内容准备成功，但随后在任何模型加载前因manifest记录SHA早于输出关闭被拦下。v5修复该实现错误，正式prepare/评分状态仍以父会话新收据为准；这些记录不能混称“真实评分已完成”。

此目录只增加独立评估工具，不改仿真器、数据生成或训练。父会话负责服务器部署、15组checkpoint核验和最终授权。本次代码没有加载真实候选权重、没有计算真实分数。

## 已验证

- `metric_fixture_local.json`：恒等/1mm位移、非对称有限点云两种有向最近邻、阈值边界、全失败分母、两种AUC精确积分通过。
- `interface_fixture_local.json`：本地人造HDF5/MAT数据的全部packet登记、原始事件精确对应、完整分母、seen/untouched分组和错配拒绝通过。
- 真实数据接口需父会话在g1运行；本地夹具通过不等于真实接口已经通过。

## 固定口径

范围是官方YCB036木块全部有标注完整序列，不是E-POSE全物体。目录文件身份决定序列全集；不按效果删帧/删序列。r_01_gl、r_05_gl已暴露，单独列为exploratory，其余为untouched_primary。

输入为原始HDF5事件，完整MAT packet在其首尾时间戳raw envelope内匹配唯一连续行区间；逐行x/y/极性/相对时间必须全部一致。仅允许去除首尾相同边界时间戳的并列行，歧义和中间不一致必须拒绝；这不等于inclusive时间极值包含的所有行。全帧公开标定去畸变后生成ON/OFF/total通道，不使用GT crop、mask、box或confidence cell。GT只在输出后算误差。PnP使用固定8角点EPnP，不按GT挑解；失败和无事件packet留在成功率/AUC分母。

主指标ADD-S@0.1D（严格小于），最近邻方向固定为GT→estimated，与[PoseCNN官方ADI](https://github.com/yuxng/PoseCNN/blob/master/lib/utils/pose_error.py)一致。反方向仅另存legacy诊断，旧pilot不可直接混表。AUC同时输出：

1. `adds_auc_0_to_0_1m_normalized`：0到0.1米，沿用[PoseCNN论文Sec.V-F](https://arxiv.org/pdf/1711.00199)的固定范围。
2. `adds_auc_0_to_0_1D_normalized`：0到0.1倍CAD直径，明确为补充。

积分均为经验成功率CDF的精确解析面积 `mean(max(1-error/L,0))`；失败贡献0。结果是0到1的比例，出表可乘100。固定范围遵循既有惯例，不声称逐位复现特定MATLAB评估器或E-POSE官方榜单。平移、原始旋转角、9点投影误差和直接关键点EPE为辅助；旋转未声明消除物体对称性。

各序列先独立评分，主汇总对untouched序列等权，然后对3个seed报告均值与样本标准差。序列配对置信区间可由父会话按统一方案处理；不要把上万个相邻帧当独立重复实验。

## 服务器路径

```bash
P=/home/zhaoyuhan/Eventbased_WAM
R=$P/runs/sim2real/20261103-stage9-direct-v1
T=$R/mainline_pose_evaluation_20261105
# 将本目录脚本复制到新的T目录；不要覆盖运行中的代码。
python "$T/stage9_epose_v5.py" prepare \
  --physical-sample "$R/pose_bank/physical/train_000.h5" \
  --interface-only --out "$T/interface_preflight_v1"
# 通过后完整登记，仍不加载模型、不评分：
python "$T/stage9_epose_v5.py" prepare \
  --physical-sample "$R/pose_bank/physical/train_000.h5" \
  --out "$T/prepared_full_v1"
python "$T/stage9_epose_v5.py" model-source --out "$T/model_source_files.json"
```

默认真实路径是：

```text
P/data/real/epose/figshare_28106717_v1/extracted/block/block/<sequence>/*_data.mat
P/data/real/epose/figshare_28106717_v1/extracted/original_data/original_data/block/<sequence>/data.h5
P/runs/sim2real/20261101-epose-wood-short-research-v1/cad_vertices.npy
P/runs/sim2real/20261101-epose-wood-short-research-v1/cad_keypoints.npy
```

CAD关键点必须与Stage9物理bank精确同序。若校验失败，保留失败证据，先定位原始文件/约定，不能静默放宽为通过。

## 最终评分门禁

父会话独立核实全部5方法×3seed完成30000更新，按已冻结的合成关键点EPE准则选`synthetic_best.pt`，生成`pose_frozen_checkpoints.json`。按模板写新的授权文件，逐项填真实SHA256，`model_source_files`填`model_source_files.json`的整个对象。interface-only结果不能进入score。脚本、模型源码/cfg、protocol、15checkpoint任一变化都将拒绝旧授权。

```bash
# 仅在父会话批准并填妥authorization.json后执行：
CUDA_VISIBLE_DEVICES=GPU-... python "$T/stage9_epose_v5.py" score \
  --prepared "$T/prepared_full_v1" \
  --frozen-checkpoints "$R/recovery_20261103/pose_frozen_checkpoints.json" \
  --authorization "$T/authorization.json" \
  --device cuda:0 --expected-gpu-uuid GPU-... --batch-size 8 \
  --out "$T/results_full_v1"
```

输出15组逐帧预测JSONL、逐序列指标JSON、长格式`per_sequence_scores.csv`和完成收据。每个输出目录必须新建，不覆盖。真实指标只能用于最终报告，不能反过来选择训练epoch或调参。
