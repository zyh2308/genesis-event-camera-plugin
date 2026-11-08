# Endpoint-only：独立的128回合机制消融

这是新增 companion，不修改正在运行的主实验、analysis_v1、figures 或 watch。主盲测仍须原来的完整320回合；这里另验32个相同来球×四档输入帧率×一套固定Ours前端，共128新回合。主实验Ours/V2E的256条事件闭环原样引用，不重跑，64个主参考也不复制成新样本。

## 注册与真实性约束

- 新EVALUATION_PROTOCOL顶层 `event_reconstruction_mode="endpoint_only"`；cases恰128条，method_order均为 `["ours"]`。内部source仍是ours，显式variant为endpoint_only。
- 每条physical_case_id、seed、输入Hz、launch及lighting对应原主盲测的Ours；首帧RGB-D实际检测及实际reset状态也必须相同。
- 固定原Ours B前端 `w10_fit0_gap100`，不重新选型；17个core、原权重、profile、锁文件与bank哈希保持相同。
- 共享runtime只允许sparse_event_sensor.py、run_original.py这两个已审benchmark hook变化，并只新增endpoint_event_worker.py。run_original的原控制/action/终止循环须字节一致。
- source delta receipt必须绑定旧/新文件SHA及实际unified diff；根审查确认只涉及env/manifest、worker选择和证据记录，不改变控制、评分或策略。
- 每个实际actor protocol必须声明endpoint_only；worker_start和每个packet都声明中间帧0。第i包的endpoint_reads必须i+1、dvs_update_calls必须i，证明reset后每个实际端点只更新一次DVS。
- 初始化失败保留为0步失败，无worker/事件时明确记不适用；不删除失败，不把未完成或基础设施错误当作任务失败。

source receipt通过新spec的endpoint_source_delta_receipt及endpoint_source_delta_sha256引用。JSON最小字段为changed_files、added_files、files{name:{old_sha256|null,new_sha256}}、unified_diff_file、unified_diff_sha256、review.allowed_hunks_only与review.control_policy_scoring_unchanged。

## Collector

```text
python collect_endpoint_ablation.py --root <独立ablation根目录> --main-blind-root <完整主blind32目录> --analysis-dir <原analysis_v1目录> --wait
```

只调用旧完成器的expected_rows、finished_progress、verify_inputs、actor_record及原raw/window审计。不会把缺V2E绕过成原blind PASS。新输出FAIR_STAGE_RESULTS.metadata.stage为ablation，rows内部method=ours、variant=endpoint_only；FAIR_COLLECTION_COMPLETE在全部128实际回合和raw审计通过后才写。另有逐actor模式证据、source delta证据及所有原始文件SHA清单。

## 三曲线与机制指标

```text
python plot_endpoint_ablation.py --main-blind-root <本地主blind轻量凭据目录> --root <本地完整ablation轻量凭据目录> --output <新的图表目录> --check-only
python plot_endpoint_ablation.py --main-blind-root <主blind目录> --root <ablation目录> --output <新的图表目录> [--main-media-plan <原盲测DEMO_DOWNLOAD_PLAN.json>]
```

输出全四档的Full Ours/V2E/endpoint-only成功曲线及Wilson区间，Full Ours与endpoint-only每档的成对胜/负/平和exact McNemar/Holm表。分母始终32个物理来球，不把四档当128个独立样本。

机制图复用原控制器接触前实际查询的状态可用性、位置/速度误差，沿用缺失/无效状态0.5m/2m/s惩罚，先在episode内平均，再对32个来球等权。完整384个实际事件actor都写表，包括所有失败。这些是闭环实际轨迹上的诊断；不同动作之后轨迹可能分叉，不能称为同一物理状态下的图像或事件精度证明。

## Demo接口

plot同时输出ENDPOINT_DEMO_JOIN_PLAN.json，仅准备接口，不新写renderer。若给原盲测media plan，就完全复用其已固定的代表case/rate；不根据新增endpoint结果挑例。不提供时固定第一个注册block、每档各一例。

每例列出三种输入的实际actor目录、完整genesis.mp4（或真实0步初始化图）、trace/result/protocol及原完成SHA。新endpoint actor内部method仍ours，显示名明确endpoint-only。后续小renderer可复用已有validate_panel、字体、时间和定格逻辑，增加第三列即可；完整录像保留，失败不裁切。原主demo不变。

原watch已封版，不能把它重定向到128数据或改掉320 guard。后续独立交付companion可复用只读queue/download函数，调用这两个新CLI；不用另起通用框架。
