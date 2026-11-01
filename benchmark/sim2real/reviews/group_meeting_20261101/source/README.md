# 复现脚本

原始构建目录为本机工作区的 `artifacts/meeting_build/`，输出目录为 `outputs/Sim2Real_20261101/`。本目录是脚本归档；重建时请恢复上述相对目录结构。

- `export_demos_remote.py`：在 g1 的 `Eventbased_WAM_torch28` 环境中运行。读取已保存的 HQF 预测和 E-POSE MAT；对固定光流权重做推理；用 ffmpeg 编码。CPU 推理设置 `CUDA_VISIBLE_DEVICES=''`，不占用训练 GPU。当前文件底部的任务列表是补导出的光流分支；首次重建全部材料时按需列入 `reconstruction`、`pose`，并使用一个新的输出目录。
- `make_figures.py`：从已审计的逐序列 CSV 独立汇总数据，用 matplotlib 导出 PNG/SVG/PDF。
- `build_deck.mjs`：使用 `@oai/artifact-tool` 创建原生表格和图表；需要本机 bundled Node runtime 和 `RUNTIME_NODE_MODULES` 环境变量。按 presentation skill 的流程运行并检查输出。
- `package_materials.py`：归档审阅范围、数据来源、离线展示页和讲稿。
- `qa_materials.py`：逐帧解码 MP4、核对服务器 SHA256、从光流数组复算 EPE、检查 PPTX 原生表格/图表/内嵌工作簿和视频链接。

不需要重训、重新选择 checkpoint 或读取真实标签作为模型输入。原数据、权重、配置和版本均在 `../provenance/` 和 `../SOURCE_AUDIT.md` 登记。绝对路径仅用于本项目现有环境，不是发布用通用安装器。
