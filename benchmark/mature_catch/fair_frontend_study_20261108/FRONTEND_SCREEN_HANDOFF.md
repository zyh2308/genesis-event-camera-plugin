# 开发缓存前端筛选交接

工具已锁定，未运行新的真实缓存筛选。七项本地CPU契约检查通过；真实归档的默认精确复现仍须在服务器先做smoke。

部署最小文件：`fair_frontend_replay.py`、`CACHE_CORPUS.json`。记录`FRONTEND_SCREEN_SHA256.json`。运行必须使用已有含NumPy/SciPy/OpenCV/h5py的物理Python环境，不使用基础`/usr/bin/python3`直接跑数值工具；不安装或修改环境。

`CACHE_CORPUS.json`锁定已有五组全部60个事件缓存：每source30条，18个seed×cadence单元。只含已查看的31–36/41–46开发数据，保留所有轨迹/成功/失败。没有触及新盲测数据。每候选的GT评分在推理之后执行。

候选ID与生产`frontend_config.py`一致：`w0_fit0_gap50`等八种。shortlist每项同时给出`frontend_id`与三个生产env键。双方每个window家族选一个，以这些选择的并集加`w10_fit1_gap50`作为完全相同的实时开发交叉候选，最多五个。

以`$PY`表示`/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot/.venv-robot/bin/python`；`$SOURCE`为`FAIR_RUNTIME.json`中的sealed source；`$TOOLS`为部署工具目录；`$R`为本轮独立输出目录。下面变量仅解释参数，不要求新增shell环境。

先用corpus第一项做smoke，输出放未来worker0目录：

```text
$PY fair_frontend_replay.py --source $SOURCE --source-manifest FAIR_RUNTIME.json --corpus CACHE_CORPUS.json --output $R/cache_screen/shard0 --cache-id <corpus.caches[0].cache_id> --shard 0 --shards 4
```

smoke只有一个cache时`SUMMARY.complete=false`，不会输出shortlist；应核对该cache的`native_parity.status=PASS`和八候选的状态。默认归档样本逐值、公开状态在首个原归档假设抓住之前的prefix逐值比较，容差1e-8；任何parity失败停止该worker，不把它变成模型错误罚分。

随后四个CPU worker分别以`--shard 0..3 --shards 4`运行，每个输出各自目录；worker0加`--resume`沿用smoke，不重复已完成cache。每个worker计划15个缓存、每缓存八候选。单进程顺序运行避免env并发混用；四process可由root独立安排CPU资源。每个worker只有本shard齐全才`status=PASS`，全语料尚未齐不会输出shortlist。

四shard齐后：

```text
$PY fair_frontend_replay.py --source $SOURCE --source-manifest FAIR_RUNTIME.json --corpus CACHE_CORPUS.json --output $R/cache_screen/merged --merge-shards $R/cache_screen/shard0 $R/cache_screen/shard1 $R/cache_screen/shard2 $R/cache_screen/shard3
```

merge验证语料、工具SHA、source和评分规则相同，全部60个cache恰好一次、原默认parity全部PASS，才输出`SHORTLIST.json`。不复制原raw/HDR，合并摘要引用各shard原cache结果路径。

注意：这是固定原轨迹的flight估计重放，完整预接触窗口评分与可精确公开prefix并列。源轨迹原有停止采集、假设抓住后的手部传播尾部/提前终止等删失均明确记录。没有重放hand控制，不应把缓存分数写成抓取成功率。最终source选择只能由后续相同并集、配对开发来球的真实闭环成功率决定。
