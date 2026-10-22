# Public-data Sim-to-Real status — Stage 5

**最新 WSL 观测：2026-10-22 16:04 UTC / 2026-10-23 01:04 JST。** 三线详细证据与历史快照见 [Stage 5 双线交接（续）](STAGE5_DUAL_LINE_HANDOFF_20261021_CONTINUATION.md)。Frozen core `e79e656ac00f7847b98db9e12bf43b093f3afff1` 未改；训练/模型选择遵守 synthetic-only，真实评分关闭；位姿依用户要求暂停。

| 线路 | 状态 | 最新可核验进度 | 阻塞 / 下一交付 |
|---|---|---|---|
| 重建 E2VID+ | **合成训练完成；真实评分未做** | g1 summary（10月20日20:57 UTC）记录两组 seed 0 各 200 epoch / 12,000 updates、exit 0。合成 event-only 递归推理和产物 hash 核验完成。HQF 14 序列 manifest 已校验，HQF 主测 / HS-ERGB 扩展已按研究审查确定。 | 无 HQF 候选模型推理或分数。g1 状态本次无法刷新。等待正式真实测试冻结。 |
| 光流 STE-FlowNet dt1 | **旧训练完成；新版有效性诊断未完成** | 旧配对训练两组各 45 epoch / 8,910 updates。固定窗监督拟合各 200 updates 证明单窗可记忆；MVSEC indoor_flying1/2/3 固定 APS 21 时间/事件/GT 接口审计通过，没有模型或评分。 | 八 rollout bank 最后一次确认 4/8（10月20日21:09:41 UTC），当前未知；两次 SSH 超时后停止远程访问。无可核验的新自监督短训。需恢复连接后核查 bank/V2E 配对，再做短训与独立 rollout 验证。 |
| 位姿 | **按用户要求暂停** | 没有本阶段 E-POSE 访问、下载或训练。 | 等用户通知。 |

## MVSEC 数据可用性

本地 `MVSEC-download-v2` manifest 为 `failed`，7 个室内文件（3 序列 data/GT + indoor calibration）完成校验；outdoor_day1 HDF5 data/GT 与 outdoor 标定未齐。10月22日间隔超过24小时后的官方 HDF5 重试仍因 Google Drive 下载配额失败，退出码1、0 bytes。官方 ROS bag 是不同格式，本轮未替换当前协议。详见[下载审计](evidence/S5/20261021-recon-flow-localization/mvsec_download_route_audit_v1.json)。三条 indoor APS index 21 未评分文件审计见[运行证据](evidence/S5/20261021-recon-flow-localization/mvsec_real_file_window_audit_v2/)和 v1 indoor_flying1 结果。

## 机器与仓库同步

- WSL 仓库：`/home/科研/Eventbased_WAM/repositories/gec-sim2real`
- 分支：`exp/public-sim2real-v1`；更新前 HEAD：`b0f05e7ebc34ab9b2c96f728a3823a15b6ad220f`
- Remote：`https://github.com/zyh2308/genesis-event-camera-plugin.git`
- 本次交接更新和新证据待本地提交/推送；g1 本次两次 SSH 均 ConnectTimeout，远端状态未知，不声称已同步。
- 最新机器快照：[live_observation_20261022T160417Z.json](evidence/S5/20261021-recon-flow-localization/live_observation_20261022T160417Z.json)，其 SHA256 sidecar 相邻保存。
