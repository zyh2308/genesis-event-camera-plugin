# Stage9训练数据图表

来源是2026-11-05 12:46:12 UTC只读训练快照，冻结core62da。源文件大小分别129098和720577bytes已匹配，SHA记录在chart_provenance.json。没有引入旧Stage4/5结果，也没有真实测试分数。

- training_completion_matrix.csv：45组训练的完成收据、权重hash、预算、最新状态及日志步数；最新步数取两者较大值，原数均保留。未观察到训练的步数留空，不写0。状态文件不能独立证明服务器进程仍运行。
- training_task_summary.csv：重建11/15完成，每组12000更新；光流6/15完成，每组8910更新；位姿15/15完成，每组30000更新。
- synthetic_validation_points.csv：450个位姿验证时刻×3指标；重建11个末端LPIPS-VGG观测；光流8个时刻×模型/零流EPE。仅收验证值，没有把训练损失当验证曲线。
- pose_validation_3seed_summary.csv：同一更新时刻的3seed均值、样本SD。位姿具有全部1000到30000更新的记录。
- latest_synthetic_validation.csv：现有每run每metric最后一次验证。
- real_results_NOT_EVALUATED.csv：真实评估空模板，status全部NOT_EVALUATED，value全部为空。光流及位姿指标名仍待绑定正式冻结评价协议，不擅自填指标。

重建和光流尚未下载完整验证历史，所以脚本只展示观测散点，不连线冒充完整学习曲线。生成的图用于训练进展和合成诊断，不能代表最终Sim-to-Real结论。

当前Windows现有Python均没有matplotlib，WSL入口返回E_ACCESSDENIED；未安装包、改权限或访问服务器。CSV及标准matplotlib脚本已可用。主会话可在已有matplotlib的CPU环境绘图，无需权重或原始训练数据：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python build_training_charts.py --mode plot --out <本charts目录>
```

生成training_completion_matrix、pose_synthetic_validation_curves、reconstruction_synthetic_validation_observations和flow_synthetic_validation_observations四组PNG/PDF。仅需本目录CSV与chart_provenance.json；不会访问源服务器、开启真实评分或修改训练。文件已存在时拒绝覆盖。

## 补齐重建和光流验证历史

stream_validation_curves.py仅顺序读取campaign/training/reconstruction与flow下的30个training_metrics.jsonl，严格接受event=synthetic_validation。绝不读取pose训练日志。每源单遍计算打开时精确文件前缀的SHA；文件若继续增长会明确标注。最后未完成的一行纳入前缀hash但不解析。完整行损坏、错身份、非有限指标或重复冲突都会拒绝；缺文件只记MISSING_LOG，不补0。9项假日志检查已通过，结果在stream_fixture_v1/results.json。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python stream_validation_curves.py --campaign <正式campaign根目录> --out <全新stream目录>

# 在尚无PNG/PDF的charts部署目录，一次生成基础4对及完整历史2对图
CUDA_VISIBLE_DEVICES='' python build_training_charts.py --mode plot --out <charts部署目录> \
  --validation-csv <stream目录>/validation_points.csv
```

补齐图为reconstruction_synthetic_validation_curves及flow_synthetic_validation_curves，每任务3个seed面板、每方法独立曲线。训练未完成的线在最后一次实际验证处截断，不把不同预算或不齐的seed合成误导性均值。源hash、行号及前缀范围都保留在CSV与source_manifest.json中。此脚本没有访问真实数据、checkpoint、GPU或网络。
