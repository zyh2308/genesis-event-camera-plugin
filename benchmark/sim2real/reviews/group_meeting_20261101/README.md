# 2026-11-01 三线组会材料

研究侧亲自导出了3段demo、8页PPT和3张结果图。主线是实验画面与配对结果；实际证据边界见 `SOURCE_AUDIT.md`。原始训练、模型与冻结core未修改。

## 从这里开始

- `EventGenesis_Sim2Real_组会汇报.pptx`：8页，3张原生统计图、3个原生数值表，含可编辑的内嵌Excel数据。备注附来源和讲述边界。
- `组会讲稿.md`：约5分钟讲稿。
- `figure_*.svg / .png / .pdf`：论文排版可用的独立图。
- `result_tables.csv`、`reconstruction_all14_sequences.csv`：可填改/复算的结果表。
- `provenance/`：源文件哈希、实际光流预测/GT数组和逐帧demo溯源。
- `source/`：导出与核验代码。

## 三段视频与离线播放页

遵守仓库现有 `*.mp4` 忽略规则，视频与完整离线播放包不放入Git。

g1完整压缩包：
`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-research-meeting-materials-v1/meeting_materials.zip`

SHA256：`e67892b412c14f9f87096d4162cbeb50aebd7e6b808efa5487d477876e8de23b`

原始视频和导出manifest：
`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-research-meeting-export-v1/`

本机完整材料：
`C:/Users/zhaoyuhan/Documents/ChatGPT/New project/outputs/Sim2Real_20261101/`

解压完整包后打开 `index.html`。PPT底部链接指向同目录独立MP4；视频没有嵌入PPT。三段依次为真实重建16秒、合成光流诊断12秒、真实位姿任务示例6.7秒。完整包带逐文件SHA256SUMS。

## 核验

三段MP4在本机逐帧解码无错误，哈希与服务器一致；光流从固定最终checkpoint独立CPU推理，四项EPE复算与报告差值均小于1e-5。PPT通过结构、布局、字体、原生表/图/工作簿和导入检查，并从最终PPT重新渲染检查8页。没有在原生PowerPoint应用中做播放兼容性测试；独立MP4可由常规播放器或浏览器播放。

重建是真实HQF14结果；光流是仅一个留出区间的合成诊断；位姿视频的框是GT，不是模型输出。三者不能合并讲成“真实泛化全面领先”。
