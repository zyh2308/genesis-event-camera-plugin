# MVSEC：固定输入核验与四序列 CPU 准备

本目录只准备代码、固定来源 payload 和小型 CPU 自测。未 SSH、未占 GPU、未训练、未运行真实预测或评分。父会话通过已有 RPC 审核并部署，不能重新建立 SSH。

## 已固定的来源与边界

- 物理核心：`62da4f35ce3c5c96f04268bd1687a27657af9238`。
- 原室内注册：`46e0141bde84b8c3c3cc96615cec247f44bb8488ae33bdaf626d4f23d8dca02f`，原7700行不改写。
- 左目投影注册：`56297b67edde57f6d8780550919bd190ac319c1a03e6d0db16e5ae92d61e9614`。payload由WSL原始JSON字节生成并嵌入原始base64，服务器重算其SHA并从原文重建metadata，不依赖手抄值。
- 固定adapter为原注册的`0247d1d3…`版本。g1原R/sources中generic `3668931a…`保持原样；原注册adapter、verifier、downstream manifest和compat从本地锁定归档复制到全新source_snapshot，文件内容未改。
- 室外候选区间仍为APS `[9200,9600)`、`[10500,10900)`。仅按原完整时间覆盖/序列边界排除，并登记排除原因；不看模型分数选窗。
- 四序列为indoor_flying1/2/3与outdoor_day1；室内分别2165/2624/2911窗。

## 文件

- `mvsec_verified_inputs.py`：`build-payload`本地生成；`verify`在服务器执行，使用原`outdoor_day/.stage9_mvsec_transfer.lock`非阻塞独占。检查3个完整容器SHA、6个required数组的shape/dtype/SHA、标定ZIP CRC及前后文件身份。只有全部通过才以`x`写新receipt；遇损坏不移动、不修补、不覆盖文件。
- `prepare_four_sequences.py`：`inspect`核对来源/收据/合同；`prepare`再新核六个室内文件SHA，运行既有bridge11项CPU自测，注册室外、保留原7700行合并，再运行v2 bridge prepare核对15个权重。所有候选协议仍`real_scores_authorized=false`。
- `watch_cpu_preparation.py`：供断开VPN后仍独立完成上述准备。只等待已授权的现有传输，不启动任何传输。每15秒检查尺寸；尺寸到位后尝试同writer锁。锁忙只等，SHA/数组/owner错误立即失败并停止。包含数据等待、核验、CPU准备在内总时限最多2小时。
- `source_snapshot.zip`：10份固定原字节源码/manifest，保留`scripts/`、`manifests/`、`compat/`、`protocol/`依赖布局。`source_manifest.json` SHA为`1e7ba8df1b4ca335750a87a33282af3c6cbd37e864bb3f5d1ca19f8dcdd73776`。
- `test_input_validation.py`、`test_preparation_contract.py`：小型临时合成夹具，不是公共数据评分。

## 服务器部署布局

父会话用已有RPC把`cpu_preparation_bundle.zip`解入新独立目录，先核包及deployment_manifest，再启动。不得覆盖R或已有结果。

```text
/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-mvsec-final-preparation-v1/
  control/mvsec_verified_inputs.py
  control/prepare_four_sequences.py
  control/watch_cpu_preparation.py
  source_snapshot/source_manifest.json
  source_snapshot/scripts/...
  source_snapshot/manifests/...
  source_snapshot/compat/...
  source_snapshot/protocol/...
  validation_payload.json
  deployment_manifest.json
```

包中仅源码和metadata，不含数据、权重或伪造完成收据。解压只能指向全新目录；审查ZIP相对路径无绝对路径、`..`或链接。watcher不会自行解压。

服务器命令（由父会话在既有RPC中启动，不代表本目录已执行）：

```bash
PY=/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python
NEW=/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-mvsec-final-preparation-v1
"$PY" -B "$NEW/control/watch_cpu_preparation.py" --root "$NEW"
```

为了不依附终端，父会话可用既有RPC的`subprocess.Popen`，`start_new_session=True`，stdout/stderr写`NEW/watcher_stdout.log`的新文件，stdin=DEVNULL。无需新SSH、tmux或GPU租约。

watcher入口会隐藏CUDA并将OMP/MKL/OpenBLAS设为2线程。总时限触发时只结束自己刚创建的独立CPU子进程组，永不查询或终止无关作业。

## 新旧receipt

默认新收据：

```text
/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-EINexus-left-20261107-v1/verified_input_receipt_20261108-v1.json
```

若它已存在，不覆盖：先核其来源合同，再对实际文件/数组完整复验，写独立`_reverified_<watcher-root>.json`，两份观察记录必须一致才可复用旧receipt。错误或不完整receipt会失败保留；不会用“存在同名文件”冒充验证成功。

已保留的坏GT及其mismatch收据不作为任何输入，也不修改。

## 准备产物与自然下一步

只有`completion.json.status=complete_unscored`和`cpu_preparation/PREPARATION_COMPLETE_NO_SCORING.json`存在并核验后，才能说输入注册与15权重准备完整。`status.json`为运行中的机器状态；失败时保留`failure.json`及对应日志，不把部分产物写成完成。

完成时包含：

- 新验证收据与原室内6文件SHA核验；原7700注册保持原SHA。
- 室外固定窗注册及四序列合并注册，精确绑定服务器left-only实际容器SHA，非full-native源容器SHA。
- 15个正式synthetic_best/last/数据清单/初始化/配方的hash绑定。
- `matrix_preparation/protocol_candidate.json`与`manifest_candidate.json`，评分保持关闭。

父研究会话随后按已确定的四序列、指标和模型选择合同完成研究核验及hash绑定冻结，即可自然推进统一真实评分；本CPU watcher不会生成评分授权、启动evaluate、选权重或增加用户审批。

## 已验证的限制

原adapter的APS saved-index核验仍按原审核规则执行。若室外实际边界不满足，注册会保存失败，不会静默放宽；原时间戳事件窗口定义不变。

检查点/输入缺失、错误SHA、原训练compat或downstream manifest不一致都会停止。不能为了让流程完成而换旧权重、最后轮权重、其他数据或改评分口径。
