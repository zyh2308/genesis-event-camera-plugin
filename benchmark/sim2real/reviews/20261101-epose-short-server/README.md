# 2026-11-01 E-POSE 服务器短训初测

**结果：两组 ADD-S≤0.1d 成功率均为 0/200。已有真实分数，但没有形成可用位姿能力。**

先读 `EPOSE_短训结果说明.md`，数字见 `EPOSE_初测指标.csv`；表格 PNG 和真实预测 MP4 可直接用于组会。展示固定片段全 200 个窗口，海报为预定 MAT 0153，不是最佳帧。

每组 16 个合成训练窗、8 个合成验证窗、496 次更新；g1 RTX3090，匹配实际木块 CAD。真实 test=block/r_01_gl 的 MAT 0003:3:0600。相同初始化，固定最终权重，真实评分前冻结。所有修改保持独立 run，不改 frozen simulator core。

`evidence/evidence_manifest.json` 含 267 个小文件哈希和服务器大产物路径。`evidence/real_eval/independent_verification.json` 是独立复算结果；`audit_summary.json` 汇总本地 267 文件验证、有效训练更新及合成定位诊断。原始 checkpoint、共享 bank 保存在 g1 run 目录，未放入 Git。

服务器 run：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-epose-wood-short-research-v1/`。

在 WSL 从本目录重新生成图表/视频：`python3 sources/build_deliverables.py --output . --evidence evidence`。该脚本读取冻结结果，不重新训练或评分。服务器训练和评分脚本见 evidence/sources；基础代码快照来源提交 7b8a6d4f22e99df9af8d134000cebdfc0f30289b，快照 tar 哈希见本次归档收据。此前 PLY 兼容性失败和场景复用前的取消记录保留。

该真实片段已见。后续改进仅依赖合成训练/验证，新增 run 独立登记，最终确认使用预先保留的其他真实序列。不可使用 GT 选网格、GT crop/mask/box 获取有利分数。
